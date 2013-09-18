import codecs
import copy
import inspect
import json
import logging
import pickle
import os.path
import numpy as np

from . import core
from . import objects
from . import simulator


logger = logging.getLogger(__name__)


class Model(object):

    def __init__(self, name, seed=None, fixed_seed=None):
        self.signals = []
        self.probes = []

        #
        # -- Build stuff --
        #
        self._operators = []

        self.objs = {}
        self.aliases = {}
        self.probed = {}
        self.signal_probes = []

        self.name = name + ''  # -- make self.name a string, raise error otw
        self.seed = np.random.randint(2**31-1) if seed is None else seed
        self.rng = np.random.RandomState(self.seed)
        self.fixed_seed = fixed_seed

        self.t = self.add(core.Signal(name='t'))
        self.steps = self.add(core.Signal(name='steps'))
        self.one = self.add(core.Constant([1.0], name='one'))

        # Automatically probe these
        self.probe(self.t)
        self.probe(self.steps)

        # -- steps counts by 1.0
        self._operators += [simulator.ProdUpdate(core.Constant(1), self.one, core.Constant(1), self.steps)]

    def _get_new_seed(self):
        return (self.rng.randint(2**31-1) if self.fixed_seed is None
                else self.fixed_seed)

    def __str__(self):
        return "Model: " + self.name

    ### I/O

    def save(self, fname, format=None):
        """Save this model to a file.

        So far, JSON and Pickle are the possible formats.

        """
        if format is None:
            format = os.path.splitext(fname)[1]

        if format in ('json', '.json'):
            with codecs.open(fname, 'w', encoding='utf-8') as f:
                json.dump(self.to_json(), f, sort_keys=True, indent=2)
                logger.info("Saved %s successfully.", fname)
        else:
            # Default to pickle
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
                logger.info("Saved %s successfully.", fname)

    def to_json(self):
        d = {
            '__class__': self.__module__ + '.' + self.__class__.__name__,
            'name': self.name,
            # 'simulator': ?? We probably don't want to serialize this
        }

        d['signals'] = [sig.to_json() for sig in self.signals]
        d['transforms'] = [trans.to_json() for trans in self.transforms]
        d['filters'] = [filt.to_json() for filt in self.filters]
        d['probes'] = [pr.to_json() for pr in self.probes]

        # d['aliases'] = self.aliases
        # d['objs'] = {k: v.to_json() for k, v in self.objs.items()}
        # d['probed'] = ?? Deal with later!
        # d['data'] = ?? Do we want to serialize this?
        return d

    @staticmethod
    def load(self, fname, format=None):
        """Load this model from a file.

        So far, JSON and Pickle are the possible formats.

        """
        if format is None:
            format = os.path.splitext(fname)[1]

        if format == 'json':
            with codecs.open(fname, 'r', encoding='utf-8') as f:
                return Model.from_json(json.load(f))
        else:
            # Default to pickle
            with open(fname, 'rb') as f:
                return pickle.load(f)

        raise IOError("Could not load {}".format(fname))

    ### Simulation methods

    @staticmethod
    def prep_for_simulation(model, dt):
        model.name = model.name + ", dt=%f" % dt
        model.dt = dt
        model._operators += [simulator.ProdUpdate(core.Constant(dt), model.one,
                                                  core.Constant(1), model.t)]

        # Sort all objects by name
        all_objs = sorted(model.objs.values(), key=lambda o: o.name)

        # 1. Build objects first
        logger.info("Building objects")
        for o in all_objs:
            o.build(model=model, dt=dt)

        # 2. Then probes
        logger.info("Building probes")
        for target in model.probed:
            if not isinstance(model.probed[target], core.Probe):
                model.probed[target].build(model=model, dt=dt)
                model.probed[target] = model.probed[target].probe

    def simulator(self, dt=0.001, sim_class=simulator.Simulator, **sim_args):
        """Get a new simulator object for the model.

        Parameters
        ----------
        dt : float, optional
            Fundamental unit of time for the simulator, in seconds.
        sim_class : child class of `Simulator`, optional
            The class of simulator to be used.
        **sim_args : optional
            Arguments to pass to the simulator constructor.

        Returns
        -------
        simulator : `sim_class`
            A new simulator object, containing a copy of the model in its
            current state.
        """
        logger.info("Copying model")
        memo = {}
        modelcopy = copy.deepcopy(self, memo)
        modelcopy.memo = memo
        self.prep_for_simulation(modelcopy, dt)
        return sim_class(model=modelcopy, **sim_args)

    ### Model manipulation

    def add(self, obj):
        try:
            obj.add_to_model(self)
            return obj
        except AttributeError:
            raise TypeError("Error in %s.add_to_model."%obj)

    def get(self, target, default=None):
        if isinstance(target, str):
            if self.aliases.has_key(target):
                return self.aliases[target]
            elif self.objs.has_key(target):
                return self.objs[target]
            logger.error("Cannot find %s in model %s.", target, self.name)
            return default

        return target

    def get_string(self, target, default=None):
        if isinstance(target, str):
            if self.aliases.has_key(target):
                obj = self.aliases[target]
            elif self.objs.has_key(target):
                return target

        for k, v in self.objs.iteritems():
            if v == target:
                return k

        logger.warning("Cannot find %s in model %s.", str(target), self.name)
        return default

    # def data(self, target):
    #     target = self.get_string(target, target)
    #     if not isinstance(target, str):
    #         target = target.name
    #     return self._data[target]

    def remove(self, target):
        obj = self.get(target)
        if obj is None:
            logger.warning("%s is not in model %s.", str(target), self.name)
            return

        if 'core' in obj.__module__:
            obj.remove_from_model(self)
        else:
            for k, v in self.objs.iteritems():
                if v == obj:
                    del self.objs[k]
                    logger.info("%s removed.", k)
            for k, v in self.aliases.iteritem():
                if v == obj:
                    del self.aliases[k]
                    logger.info("Alias '%s' removed.", k)

        return obj

    def alias(self, alias, target):
        obj_s = self.get_string(target)
        if obj_s is None:
            raise ValueError(target + " cannot be found.")
        self.aliases[alias] = obj_s
        logger.info("%s aliased to %s", obj_s, alias)
        return self.get(obj_s)

    # Model creation methods

    def probe(self, target, sample_every=0.001, filter=None):
        if core.is_signal(target):
            p = core.Probe(target, sample_every)
            self.add(p)

        self.probed[target] = p
