import logging
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np


logger = logging.getLogger(__name__)


def is_base(sig):
    return sig.base == sig


def is_view(sig):
    return not is_base(sig)

class SignalDict(dict):
    def __getitem__(self, obj):
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
                              strides=bytestrides,
                             )
            return view
        else:
            raise KeyError(obj)


class collect_operators_into(object):
    lists = []
    """
    Within this context, operators that are constructed
    are, by default, appended to an `operators` list.

    For example:

    >>> operators = []
    >>> with collect_operators_into(operators):
    >>>    Reset(foo)
    >>>    Copy(foo, bar)
    >>> assert len(operators) == 2

    After the context exits, `operators` contains the Reset
    and the Copy instances.

    """
    def __init__(self, operators):
        if operators is None:
            operators = []
        self.operators = operators

    def __enter__(self):
        self.lists.append(self.operators)

    def __exit__(self, exc_type, exc_value, tb):
        self.lists.remove(self.operators)

    @staticmethod
    def collect_operator(op):
        for lst in collect_operators_into.lists:
            lst.append(op)


class Operator(object):
    # automatic staticmethod
    def __new__(cls, *args, **kwargs):
        rval = super(Operator, cls).__new__(cls, *args, **kwargs)
        collect_operators_into.collect_operator(rval)
        return rval

    #
    # The lifetime of a Signal during one simulator timestep:
    # 0) at most one set operator (optional)
    # 1) any number of increments
    # 2) any number of reads
    # 3) at most one update
    #
    # A signal that is only read can be considered a "constant"
    #
    # A signal that is both set *and* updated can be a problem: since
    # reads must come after the set, and the set will destroy
    # whatever were the contents of the update, it can be the case
    # that the update is completely hidden and rendered irrelevant.
    # There are however at least two reasons to use both a set and an update:
    # (a) to use a signal as scratch space (updating means destroying it)
    # (b) to use sets and updates on partly overlapping views of the same
    #     memory.
    #

    # -- Signals that are read and not modified by this operator
    reads = []
    # -- Signals that are only assigned by this operator
    sets = []
    # -- Signals that are incremented by this operator
    incs = []
    # -- Signals that are updated to their [t + 1] value.
    #    After this operator runs, these signals cannot be
    #    used for reads until the next time step.
    updates = []


    def init_signals(self, signals, dt):
        """
        Install any buffers into the signals view that
        this operator will need. Classes for nonlinearities
        that use extra buffers should create them here.
        """


class Reset(Operator):
    """
    Assign a constant value to a Signal.
    """
    def __init__(self, dst, value=0):
        self.dst = dst
        self.value = float(value)

        self.sets = [dst]

    def __str__(self):
        return 'Reset(%s)' % str(self.dst)

    def make_step(self, signals, dt):
        target = signals[self.dst]
        value = self.value
        def step():
            target[...] = value
        return step


class Copy(Operator):
    """
    Assign the value of one signal to another
    """
    def __init__(self, dst, src, as_update=False, tag=None):
        self.dst = dst
        self.src = src
        self.tag = tag

        self.reads = [src]
        self.sets = [] if as_update else [dst]
        self.updates = [dst] if as_update else []

    def __str__(self):
        return 'Copy(%s -> %s)' % (str(self.src), str(self.dst))

    def make_step(self, dct, dt):
        dst = dct[self.dst]
        src = dct[self.src]
        def step():
            dst[...] = src
        return step


class DotInc(Operator):
    """
    Increment signal Y by dot(A, X)
    """
    def __init__(self, A, X, Y, xT=False, tag=None):
        self.A = A
        self.X = X
        self.Y = Y
        self.xT = xT
        self.tag = tag

        self.reads = [self.A, self.X]
        self.incs = [self.Y]

    def __str__(self):
        return 'DotInc(%s, %s -> %s "%s")' % (
                str(self.A), str(self.X), str(self.Y), self.tag)

    def make_step(self, dct, dt):
        X = dct[self.X]
        A = dct[self.A]
        Y = dct[self.Y]
        X = X.T if self.xT else X
        def step():
            # -- we check for size mismatch,
            #    because incrementing scalar to len-1 arrays is ok
            #    if the shapes are not compatible, we'll get a
            #    problem in Y[...] += inc
            try:
                inc =  np.dot(A, X)
            except Exception, e:
                e.args = e.args + (A.shape, X.shape)
                raise
            if inc.shape != Y.shape:
                if inc.size == Y.size == 1:
                    inc = np.asarray(inc).reshape(Y.shape)
                else:
                    raise ValueError('shape mismatch', (inc.shape, Y.shape))
            Y[...] += inc

        return step


class BaseSimulator(object):
    def __init__(self, operators, signals, dt):
        self._signals = signals
        self.dt = dt

        self.dg = self._init_dg(operators)
        self._step_order = [node
            for node in nx.topological_sort(self.dg)
            if hasattr(node, 'make_step')]
        self._steps = [node.make_step(self._signals, self.dt)
            for node in self._step_order]

        self.n_steps = 0

    def _init_dg(self, operators, verbose=False):
        dg = nx.DiGraph()

        for op in operators:
            dg.add_edges_from(itertools.product(op.reads, [op]))
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

        # -- assert that no two views are both set and aliased
        for node, other in itertools.combinations(sets, 2):
            assert not node.shares_memory_with(other)

        # -- Scheduling algorithm for serial evaluation:
        #    1) All sets on a given base signal
        #    2) All incs on a given base signal
        #    3) All reads on a given base signal

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

        # -- assert that only one op updates any particular view
        for node in ups:
            assert len(ups[node]) == 1, (node, ups[node])

        # -- assert that no two views are both updated and aliased
        for node, other in itertools.combinations(ups, 2):
            assert not node.shares_memory_with(other), (
                    node, other)

        # -- updates depend on reads, sets, and incs.
        for node, post_ops in ups.items():
            pre_ops = sets[node] + incs[node] + reads[node]
            for other in by_base_writes[node.base]:
                pre_ops += sets[other] + incs[other] + reads[other]
            for other in by_base_reads[node.base]:
                pre_ops += sets[other] + incs[other] + reads[other]
            dg.add_edges_from(itertools.product(set(pre_ops), post_ops))

        return dg

    @property
    def signals(self):
        class Accessor(object):
            def __getitem__(_, item):
                return self._signals[item]

            def __setitem__(_, item, val):
                self._signals[item][...] = val

            def __iter__(_):
                return self._signals.__iter__()

            def __len__(_):
                return self._signals.__len__()

            def __str__(_):
                import StringIO
                sio = StringIO.StringIO()
                for k in self._signals:
                    print >> sio, k, self._signals[k]
                return sio.getvalue()

        return Accessor()

    def step(self):
        for step_fn in self._steps:
            step_fn()

        self.n_steps += 1

    def run_steps(self, steps):
        """Simulate for the given number of steps."""
        for i in xrange(steps):
            if i % 1000 == 0:
                logger.debug("Step %d", i)
            self.step()
