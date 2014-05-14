import logging

import numpy as np
import networkx as nx

from .signaldict import SignalDict
from .operator import depgraph, is_op

logger = logging.getLogger(__name__)


class Engine(object):

    def _make_step(self, node):
        return node.make_step(self.signals)

    def _init_signals(self, node):
        node.init_signals(self.signals)

    def __init__(self, operators, signals=None):
        """

        operators: list of Operator instances

        signals: SignalDict instance

        """
        self.signals = SignalDict() if signals is None else signals
        self._operators = operators
        map(self._init_signals, operators)
        self.dg = depgraph(operators)
        self._topo_all = nx.topological_sort(self.dg)
        self._topo_ops = filter(is_op, self._topo_all)
        self._steps = [self._make_step(step) for step in self._topo_ops
                       if step is not None]
        self.n_steps = 0

    def step(self, N=1):
        """Simulate for the given number of `dt` steps."""
        for ii in range(N):
            for step_fn in self._steps:
                step_fn()
            self.n_steps += 1

        # post-condition:
        # self.signals[signal] reveals current value of signal in model
        # as read-only numpy ndarray (see SigDict for details)



class Simulator(Engine):
    """Reference simulator for Nengo models."""

    def _make_step(self, node):
        try:
            make_step = node.make_step_dt
            args = (self.signals, self.dt)
        except AttributeError:
            make_step = node.make_step
            args = (self.signals, )
        return make_step(*args)

    def _init_signals(self, node):
        node.init_signals(self.signals)
        try:
            init_signals = node.init_signals_dt
            args = (self.signals, self.dt)
        except AttributeError:
            init_signals = node.init_signals
            args = (self.signals, )
        return init_signals(*args)

    def __init__(self, operators, dt=0.001):
        self.dt = dt
        Engine.__init__(self,
                        operators,
                        SignalDict(__time__=np.asarray(0.0, dtype=np.float64)))
