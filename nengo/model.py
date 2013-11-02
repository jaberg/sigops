from collections import OrderedDict
import copy
import logging
import pickle
import os.path
import numpy as np

from . import objects
from . import simulator


logger = logging.getLogger(__name__)


class Model(object):

    def __init__(self, name, seed=None):
        self.objs = OrderedDict()
        self.probed = OrderedDict()
        self.signal_probes = []

        self.name = name + ''  # -- make self.name a string, raise error otw
        self.seed = seed

        self._rng = None

        # Some automatic stuff
        self.t = self.make_node('t', output=0)
        self.steps = self.make_node('steps', output=0)
        self.one = self.make_node('one', output=[1.0])

        # Automatically probe time
        self.probe(self.t)

    def __str__(self):
        return "Model: " + self.name

    def _get_new_seed(self):
        if self._rng is None:
            # never create rng without knowing the seed
            assert self.seed is not None
            self._rng = np.random.RandomState(self.seed)
        return self._rng.randint(np.iinfo(np.int32).max)

    ### I/O

    def save(self, fname, format=None):
        """Save this model to a file.

        So far, Pickle is the only implemented format.

        """
        if format is None:
            format = os.path.splitext(fname)[1]

        with open(fname, 'wb') as f:
            pickle.dump(self, f)
            logger.info("Saved %s successfully.", fname)

    @staticmethod
    def load(self, fname, format=None):
        """Load this model from a file.

        So far, JSON and Pickle are the possible formats.

        """
        # if format is None:
        #     format = os.path.splitext(fname)[1]

        # Default to pickle
        with open(fname, 'rb') as f:
            return pickle.load(f)

        raise IOError("Could not load {}".format(fname))

    ### Simulation methods

    def simulator(self, dt=0.001, sim_class=simulator.Simulator,
                  seed=None, **sim_args):
        """Get a new simulator object for the model.

        Parameters
        ----------
        dt : float, optional
            Fundamental unit of time for the simulator, in seconds.
        sim_class : child class of `Simulator`, optional
            The class of simulator to be used.
        seed : int, optional
            Random number seed for the simulator's random number generator.
            This random number generator is responsible for creating any random
            numbers used during simulation, such as random noise added to
            neurons' membrane voltages.
        **sim_args : optional
            Arguments to pass to the simulator constructor.

        Returns
        -------
        simulator : `sim_class`
            A new simulator object, containing a copy of the model in its
            current state.
        """
        return sim_class(model=self, dt=dt, seed=seed, **sim_args)

    ### Model manipulation

    def add(self, obj):
        try:
            obj.add_to_model(self)
            return obj
        except AttributeError:
            raise TypeError("Error in %s.add_to_model."%obj)

    def get(self, target, default=None):
        if isinstance(target, str):
            return self.objs.get(target, default)
        return target

    def get_string(self, target, default=None):
        if isinstance(target, str) and self.objs.has_key(target):
            return target
        for k, v in self.objs.iteritems():
            if v == target:
                return k
        return default

    def remove(self, target):
        obj = self.get(target)
        if obj is None:
            logger.warning("%s is not in model %s.", str(target), self.name)
            return

        for k, v in self.objs.iteritems():
            if v == obj:
                del self.objs[k]
                logger.info("%s removed.", k)

        return obj

    # Model creation methods

    def probe(self, target, sample_every=0.001, filter=None):
        if builder.is_signal(target):
            p = core.Probe(target, sample_every)
            self.add(p)

        self.probed[target] = p
