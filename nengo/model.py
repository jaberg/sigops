import math

from .objects import *
from . import simulator

class Model(object):

    BACKENDS = {
        'numpy': 'nengo.simulator',
    }

    def __init__(self, name, seed=None, fixed_seed=None, backend='numpy',
                 dt=0.001):
        self.dt = dt

        self.signals = set()
        self.transforms = set()
        self.filters = set()
        self.probes = set()

        self.objs = {}
        self.aliases = {}
        self.probed = {}
        self.data = {}

        self.simtime = self.add(Signal(name='simtime'))
        self.steps = self.add(Signal(name='steps'))
        self.one = self.add(Constant(1, value=[1.0], name='one'))

        # Automatically probe these
        self.probe(self.simtime)
        self.probe(self.steps)

        # -- steps counts by 1.0
        self.add(Filter(1.0, self.one, self.steps))
        self.add(Filter(1.0, self.steps, self.steps))

        # simtime <- dt * steps
        self.add(Filter(dt, self.one, self.simtime))
        self.add(Filter(dt, self.steps, self.simtime))

        self.name = name
        self.backend = backend

        if seed is None:
            self.seed = 123
        else:
            self.seed = seed

        if fixed_seed is not None:
            raise NotImplementedError()

    def __str__(self):
        return "Model: " + self.name

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if hasattr(self, '_backend') and backend == self._backend:
            return

        try:
            toimport = Model.BACKENDS[backend]
            self.simulator = __import__(toimport, globals(), locals(),
                                        ['simulator'], -1)
            self.sim_obj = None
            self._backend = backend

        except KeyError:
            print (backend + " not a registered backend. "
                   "Falling back to numpy.")
            self.backend = 'numpy'

        except ImportError:
            if backend == 'numpy':
                raise ImportError("Cannot import numpy backend!")
            print (backend + " cannot be imported. "
                   "Falling back to numpy.")
            self.backend = 'numpy'

    @property
    def objects(self):
        return self.objs.values()

    ### Simulation methods

    def reset(self):
        logger.debug("Resetting simulator")
        self.sim_obj.reset()

    def run(self, time, dt=0.001, output=None, stop_when=None):
        if self.sim_obj is None:
            self.sim_obj = self.simulator.Simulator(self)

        steps = int(time // self.dt)
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
        if hasattr(obj, 'name'):
            self.objs[obj.name] = obj
        return obj

    def get(self, target, default=None):
        if isinstance(target, str):
            if self.aliases.has_key(target):
                return self.aliases[target]
            elif self.objs.has_key(target):
                return self.objs[target]
            print "Cannot find " + target + " in this model."
            return default

        if not target in self.objs.values():
            print "Cannot find " + str(target) + " in this model."
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

        print "Cannot find " + str(target) + " in this model."
        return default

    def remove(self, target):
        obj = self.get(target)
        if obj is None:
            print target + " not in this model."
            return

        obj.remove_from_model(self)

        for k, v in self.objs.iteritems():
            if v == obj:
                del self.objs[k]
        for k, v in self.aliases.iteritem():
            if v == obj:
                del self.aliases[k]

        return obj

    def alias(self, alias, target):
        obj = self.get(target)
        if obj is None:
            raise ValueError(target + " cannot be found.")
        self.aliases[alias] = obj
        return obj


    # Model creation methods

    def probe(self, target, sample_every=None, pstc=None):
        def _filter_coefs(pstc, dt):
            pstc = max(pstc, dt)
            decay = math.exp(-dt / pstc)
            return decay, (1.0 - decay)

        if sample_every is None:
            sample_every = self.dt

        probe_type = ''
        if isinstance(target, str):
            s = target.split('.')
            if len(s) > 1:
                target, probe_type = s[0], s[1]
        obj = self.get(target)

        if type(obj) == Ensemble:
            obj_s = self.get_string(target)
            p = obj.probe(probe_type, sample_every, self)
            self.probed[obj_s] = p
            return p

        if type(obj) != Signal:
            obj = obj.signal

        if pstc is None:
            obj_s = self.get_string(target)
        else:
            obj_s = "%s,pstc=%f" % (self.get_string(target), pstc)

        if pstc is not None and pstc > self.dt:
            fcoef, tcoef = _filter_coefs(pstc=pstc, dt=self.dt)
            probe_sig = self.add(Signal(obj.n))
            self.add(Filter(fcoef, probe_sig, probe_sig))
            self.add(Transform(tcoef, obj, probe_sig))
            p = self.add(Probe(probe_sig, sample_every))
        else:
            p = self.add(Probe(obj, sample_every))

        self.probed[obj_s] = p
        return p
