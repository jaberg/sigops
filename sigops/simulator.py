from __future__ import print_function

import logging
import itertools
from collections import defaultdict

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class SignalDict(dict):
    """
    Map from Signal -> ndarray

    This dict subclass ensures that the ndarray values aren't overwritten,
    and instead data are written into them, which ensures that
    these arrays never get copied, which wastes time and space.

    Use ``init`` to set the ndarray initially.
    """

    def __getitem__(self, obj):
        """SignalDict overrides __getitem__ for two reasons.

        1. so that scalars are returned as 0-d ndarrays
        2. so that a SignalView lookup returns a views of its base
        """
        if obj in self:
            return dict.__getitem__(self, obj)
        elif obj.base in self:
            # look up views as a fallback
            # --work around numpy's special case behaviour for scalars
            base_array = self[obj.base]
            try:
                # for some installations, this works
                itemsize = int(obj.dtype.itemsize)
            except TypeError:
                # other installations, this...
                itemsize = int(obj.dtype().itemsize)
            byteoffset = itemsize * obj.offset
            bytestrides = [itemsize * s for s in obj.elemstrides]
            view = np.ndarray(shape=obj.shape,
                              dtype=obj.dtype,
                              buffer=base_array.data,
                              offset=byteoffset,
                              strides=bytestrides)
            return view
        else:
            raise KeyError("%s has not been initialized. Please call "
                           "SignalDict.init first." % (str(obj)))

    def __setitem__(self, key, val):
        """Ensures that ndarrays stay in the same place in memory.

        Unlike normal dicts, this means that you cannot add a new key
        to a SignalDict using __setitem__. This is by design, to avoid
        silent typos when debugging Simulator. Every key must instead
        be explicitly initialized with SignalDict.init.
        """
        self.__getitem__(key)[...] = val

    def __str__(self):
        """Pretty-print the signals and current values."""
        sio = StringIO()
        for k in self:
            sio.write("%s %s\n" % (repr(k), repr(self[k])))
        return sio.getvalue()

    def init(self, signal, ndarray):
        """Set up a permanent mapping from signal -> ndarray."""
        dict.__setitem__(self, signal, ndarray)


class Simulator(object):
    """Reference simulator for models. """

    def __init__(self, operators, dt=0.001):
        self.dt = dt
        self.operators = operators

        # -- map from Signal.base -> ndarray
        self.signals = SignalDict()
        for op in self.operators:
            op.init_signals(self.signals, dt)

        self.dg = self._init_dg()
        self._step_order = [node
                            for node in nx.topological_sort(self.dg)
                            if hasattr(node, 'make_step')]
        self._steps = [node.make_step(self.signals, self.dt)
                       for node in self._step_order]

        self.n_steps = 0

    def _init_dg(self, verbose=False):
        operators = self.operators
        dg = nx.DiGraph()

        for op in operators:
            dg.add_edges_from(itertools.product(op.reads + op.updates, [op]))
            dg.add_edges_from(itertools.product([op], op.sets + op.incs))

        # -- all views of a base object in a particular dictionary
        by_base_writes = defaultdict(list)
        by_base_reads = defaultdict(list)
        reads = defaultdict(list)
        sets = defaultdict(list)
        incs = defaultdict(list)
        ups = defaultdict(list)

        for op in operators:
            for node in op.sets + op.incs:
                by_base_writes[node.base].append(node)

            for node in op.reads:
                by_base_reads[node.base].append(node)

            for node in op.reads:
                reads[node].append(op)

            for node in op.sets:
                sets[node].append(op)

            for node in op.incs:
                incs[node].append(op)

            for node in op.updates:
                ups[node].append(op)

        # -- assert that only one op sets any particular view
        for node in sets:
            assert len(sets[node]) == 1, (node, sets[node])

        # -- assert that only one op updates any particular view
        for node in ups:
            assert len(ups[node]) == 1, (node, ups[node])

        # --- assert that any node that is incremented is also set/updated
        for node in incs:
            assert len(sets[node] + ups[node]) > 0, (node)

        # -- assert that no two views are both set and aliased
        if len(sets) >= 2:
            for node, other in itertools.combinations(sets, 2):
                assert not node.shares_memory_with(other), \
                    ("%s shares memory with %s" % (node, other))

        # -- assert that no two views are both updated and aliased
        if len(ups) >= 2:
            for node, other in itertools.combinations(ups, 2):
                assert not node.shares_memory_with(other), (node, other)

        # -- Scheduling algorithm for serial evaluation:
        #    1) All sets on a given base signal
        #    2) All incs on a given base signal
        #    3) All reads on a given base signal
        #    4) All updates on a given base signal

        # -- incs depend on sets
        for node, post_ops in incs.items():
            pre_ops = list(sets[node])
            for other in by_base_writes[node.base]:
                pre_ops += sets[other]
            dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

        # -- reads depend on writes (sets and incs)
        for node, post_ops in reads.items():
            pre_ops = sets[node] + incs[node]
            for other in by_base_writes[node.base]:
                pre_ops += sets[other] + incs[other]
            dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

        # -- updates depend on reads, sets, and incs.
        for node, post_ops in ups.items():
            pre_ops = sets[node] + incs[node] + reads[node]
            for other in by_base_writes[node.base]:
                pre_ops += sets[other] + incs[other] + reads[other]
            for other in by_base_reads[node.base]:
                pre_ops += sets[other] + incs[other] + reads[other]
            dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

        return dg

    def step(self):
        """Advance the simulator by one timestep.
        """
        for step_fn in self._steps:
            step_fn()
        self.n_steps += 1

    def run_steps(self, steps):
        """Simulate for the given number of `dt` steps."""
        for i in range(steps):
            if i % 1000 == 0:
                logger.debug("Step %d", i)
            self.step()
