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


class ModelFrozenError(RuntimeError):
    msg = "Model has been built and cannot be modified further."


class Model(object):

    def __init__(self, name, simulator=simulator.Simulator,
                 seed=None, fixed_seed=None):
        self.signals = set()
        self.transforms = set()
        self.filters = set()
        self.probes = set()

        self.objs = {}
        self.aliases = {}
        self.probed = {}
        self.data = {}
        self.signal_probes = []

        self.name = name
        self.simulator = simulator

        self.seed = np.random.randint(2**31-1) if seed is None else seed
        self.rng = np.random.RandomState(self.seed)
        self.fixed_seed = fixed_seed

        self.t = self.add(core.Signal(name='simtime'))
        self.steps = self.add(core.Signal(name='steps'))
        self.one = self.add(core.Constant(1, value=[1.0], name='one'))

        # Automatically probe these
        self.probe(self.t)
        self.probe(self.steps)

        # -- steps counts by 1.0
        self.add(core.Filter(1.0, self.one, self.steps))
        self.add(core.Filter(1.0, self.steps, self.steps))

        self.built = False

    def _get_new_seed(self):
        return self.rng.randint(2**31-1) if self.fixed_seed is None \
            else self.fixed_seed

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
            'dt': self.dt,
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

    @property
    def built(self):
        return self._frozen

    @built.setter
    def built(self, frozen):
        self._frozen = frozen

        # If built, stub out all methods but reset and run
        def stub(*args, **kwargs):
            raise ModelFrozenError(ModelFrozenError.msg)
        if frozen:
            for k, v in inspect.getmembers(self, predicate=inspect.isroutine):
                if k not in ('reset', 'run'):
                    setattr(self, k, stub)

    def build(self, dt=0.001):
        logger.info("Copying model")
        modelcopy = copy.deepcopy(self)
        modelcopy.name += ", dt=%f" % dt
        modelcopy.dt = dt
        modelcopy.add(core.Filter(dt, modelcopy.one, modelcopy.t))
        modelcopy.add(core.Filter(dt, modelcopy.steps, modelcopy.t))

        # Sort all objects by name
        all_objs = sorted(modelcopy.objs.values(), key=getattr(o, 'name'))

        # 1. Build objects first
        logger.info("Building objects")
        for o in all_objs:
            o.build(model=modelcopy, dt=dt)

        # 2. Then probes
        logger.info("Building probes")
        for o in all_objs:
            for p in o.probes:
                p.build(model=modelcopy, dt=dt)
        for p in self.signal_probes:
            p.build(model=modelcopy, dt=dt)

        # Collect raw probes
        for target in self.probed:
            if not isinstance(self.probed[target], core.Probe):
                self.probed[target] = self.probed[target].probe

        modelcopy.built = True
        logger.info("Finished. New model is %s.", modelcopy.name)
        return modelcopy

    def reset(self):
        logger.debug("Resetting simulator for %s", self.name)
        try:
            self.sim_obj.reset()
        except AttributeError:
            logger.warning("Tried to reset %s, but had never been run.",
                           self.name)

    def run(self, time, dt=0.001, output=None, stop_when=None):
        if not self.built:
            builtmodel = self.build(dt=dt)
            return builtmodel.run(dt=dt, output=output, stop_when=stop_when)

        if dt != self.dt:
            raise ModelFrozenError(
                "Model previously built with dt=%f. Rebuild model to use "
                "different dt." % self.dt)

        if getattr(self, 'sim_obj', None) is None:
            logger.debug("Creating simulator for %s", self.name)
            self.sim_obj = self.simulator(self)

        steps = int(time // dt)
        logger.debug("Running %s for %f seconds, or %d steps",
                     self.name, time, steps)
        self.sim_obj.run_steps(steps)

        for k in self.probed:
            self.data[k] = self.sim_obj.probe_data(self.probed[k])

        return self

    ### Model manipulation

    def add(self, obj):
        if hasattr(obj, 'name') and self.objs.has_key(obj.name):
            raise ValueError("Something called " + obj.name + " already exists."
                             " Please choose a different name.")

        if 'core' in obj.__module__:
            obj.add_to_model(self)
        else:
            raise TypeError("Object not recognized as a Nengo object. ")

        return obj

    def get(self, target, default=None):
        if isinstance(target, str):
            if self.aliases.has_key(target):
                return self.aliases[target]
            elif self.objs.has_key(target):
                return self.objs[target]
            logger.error("Cannot find %s in model %s.", target, self.name)
            return default

        if not target in self.objs.values():
            logger.error("Cannot find %s in model %s.", str(target), self.name)
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
        if sample_every is None:
            sample_every = self.dt

        if isinstance(target, core.Signal):
            p = core.Probe(target, sample_every)

        self.probed[target] = p
