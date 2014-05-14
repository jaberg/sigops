import numpy as np

import itertools
from collections import defaultdict

import numpy as np
import networkx as nx


def is_op(thing):
    try:
        return thing._is_sigops_operator
    except AttributeError:
        return False


class Operator(object):
    """Base class for operator instances understood by nengo.Simulator.

    The lifetime of a Signal during one simulator timestep:
    0) at most one set operator (optional)
    1) any number of increments
    2) any number of reads
    3) at most one update

    A signal that is only read can be considered a "constant".

    A signal that is both set *and* updated can be a problem:
    since reads must come after the set, and the set will destroy
    whatever were the contents of the update, it can be the case
    that the update is completely hidden and rendered irrelevant.
    There are however at least two reasons to use both a set and an update:
    (a) to use a signal as scratch space (updating means destroying it)
    (b) to use sets and updates on partly overlapping views of the same
        memory.

    N.B.: It is done on purpose that there are no default values for
    reads, sets, incs, and updates.

    Each operator should explicitly set each of these properties.
    """

    _is_sigops_operator = True

    @property
    def reads(self):
        """Signals that are read and not modified"""
        return self._reads

    @reads.setter
    def reads(self, val):
        self._reads = val

    @property
    def sets(self):
        """Signals assigned by this operator

        A signal that is set here cannot be set or updated
        by any other operator.
        """
        return self._sets

    @sets.setter
    def sets(self, val):
        self._sets = val

    @property
    def incs(self):
        """Signals incremented by this operator

        Increments will be applied after this signal has been
        set (if it is set), and before reads.
        """
        return self._incs

    @incs.setter
    def incs(self, val):
        self._incs = val

    @property
    def updates(self):
        """Signals assigned their value for time t + 1

        This operator will be scheduled so that updates appear after
        all sets, increments and reads of this signal.
        """
        return self._updates

    @updates.setter
    def updates(self, val):
        self._updates = val

    @property
    def all_signals(self):
        return self.reads + self.sets + self.incs + self.updates

    def init_signals(self, signals):
        """
        Install any buffers into the signals view that
        this operator will need. Classes for nonlinearities
        that use extra buffers should create them here.
        """
        for sig in self.all_signals:
            if sig.base not in signals:
                signals[sig.base] = np.asarray(
                    np.zeros(sig.base.shape, dtype=sig.base.dtype)
                    + sig.base.value)


def depgraph(operators, verbose=False):
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
