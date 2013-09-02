import codecs
import json
import logging
import pickle
import os.path
import numpy as np

from . import core
from . import objects
from . import probes
from . import simulator


logger = logging.getLogger(__name__)

class Model(object):

    def __init__(self, name, seed=None, fixed_seed=None,
                 simulator=simulator.Simulator, dt=0.001):
        self.dt = dt

        self.signals = set()
        self.transforms = set()
        self.filters = set()
        self.probes = set()

        self.objs = {}
        self.aliases = {}
        self.probed = {}
        self.data = {}

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

        # simtime <- dt * steps
        self.add(core.Filter(dt, self.one, self.t))
        self.add(core.Filter(dt, self.steps, self.t))

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

    def reset(self):
        logger.debug("Resetting simulator for %s", self.name)
        self.sim_obj.reset()

    def run(self, time, dt=0.001, output=None, stop_when=None):
        if getattr(self, 'sim_obj', None) is None:
            logger.debug("Creating simulator for %s", self.name)
            self.sim_obj = self.simulator(self)

        steps = int(time // self.dt)
        logger.debug("Running %s for %f seconds, or %d steps",
                     self.name, time, steps)
        self.sim_obj.run_steps(steps)

        for k in self.probed:
            self.data[k] = self.sim_obj.probe_data(self.probed[k])

        return self.data

    ### Model manipulation

    def add(self, obj):
        if hasattr(obj, 'name') and self.objs.has_key(obj.name):
            raise ValueError("Something called " + obj.name + " already exists."
                             " Please choose a different name.")
        obj.add_to_model(self)
        if hasattr(obj, 'name') and not obj.__module__ == 'core':
            self.objs[obj.name] = obj
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

        obj.remove_from_model(self)

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

    def probe(self, target, sample_every=None, filter=None):
        if sample_every is None:
            sample_every = self.dt

        if isinstance(target, core.Signal):
            p = core.Probe(target, sample_every)

        self.probed[target] = p
        self.add(p)
        return p
