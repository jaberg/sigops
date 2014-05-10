import math

from .simulator_objects import *


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
        self.probe_data = {}

        self.simtime = self.add(Signal(name='simtime'))
        self.steps = self.add(Signal(name='steps'))
        self.one = self.add(Constant(1, value=[1.0], name='one'))

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
    def time(self):
        if self.sim_obj is None:
            return None
        return self.sim_obj.simulator_time

    @property
    def objects(self):
        return self.objs.values()

    ### Simulation methods

    def reset(self):
        logger.debug("Resetting simulator")
        self.sim_obj.reset()

    def run(self, time, dt=0.001, output=None, stop_when=None):
        if self.sim_obj is None:
            logger.debug("No simulator object yet. Building.")
            self.sim_obj = self.simulator.Simulator(self)
        if stop_when is not None:
            raise NotImplementedError()
        if output is not None:
            raise NotImplementedError()

        steps = int(time / dt)
        logger.debug("Running simulator for " + str(steps) + " steps")
        self.sim_obj.run_steps(steps)

        for k in self.probed:
            self.probe_data[k] = self.sim_obj.probe_data(self.probed[k])

        return self.probe_data

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

    def probe(self, target, sample_every=None, pstc=None, static=False):
        def _filter_coefs(pstc, dt):
            pstc = max(pstc, dt)
            decay = math.exp(-dt / pstc)
            return decay, (1.0 - decay)

        if sample_every is None:
            sample_every = self.dt

        obj = self.get(target)
        obj_s = self.get_string(target)

        if pstc is not None and pstc > self.dt:
            fcoef, tcoef = _filter_coefs(pstc=pstc, dt=self.dt)
            probe_sig = self.signal(obj.sig.n)
            self.filter(fcoef, probe_sig, probe_sig)
            self.transform(tcoef, obj.sig, probe_sig)
            p = SimModel.probe(self, probe_sig, sample_every)
        else:
            p = SimModel.probe(self, obj.sig, sample_every)

        i = 0
        while self.probed.has_key(obj_s):
            i += 1
            obj_s = self.get_string(target) + "_" + str(i)

        self.probed[obj_s] = p
        return p
