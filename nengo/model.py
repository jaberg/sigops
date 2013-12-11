try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict
import logging
import pickle
import os.path

import numpy as np


logger = logging.getLogger(__name__)


class Model(object):

    def __init__(self, label="Model", seed=None):
        self.objs = []
        self.probed = OrderedDict()
        self.signal_probes = []

        self.label = label + ''  # -- make self.name a string, raise error otw
        self.seed = seed

        self._rng = None

    def __str__(self):
        return "Model: " + self.label

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

    ### Model manipulation

    def add(self, obj):
        try:
            obj.add_to_model(self)
            return obj
        except AttributeError, ae:
            raise TypeError("Error in %s.add_to_model.\n%s" % (obj, ae))

    def remove(self, target):
        if not target in self.objs:
            logger.warning("%s is not in model %s.", str(target), self.label)
            return

        self.objs = [o for o in self.objs if o != target]
        logger.info("%s removed.", target)
