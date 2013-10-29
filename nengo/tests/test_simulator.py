try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo
import nengo.simulator as simulator
from nengo.builder import Builder
from nengo.builder import Signal, Constant
from nengo.builder import DotInc, ProdUpdate, Reset, Copy


def testbuilder(model, dt):
    model.dt = dt
    model.seed = 0
    if not hasattr(model, 'probes'):
        model.probes = []
    return model


class TestSimulator(unittest.TestCase):
    Simulator = simulator.Simulator

    def test_steps(self):
        m = nengo.Model("test_signal_indexing_1")
        sim = m.simulator(sim_class=self.Simulator)
        self.assertEqual(0, sim.signals[sim.model.steps.signal])
        sim.step()
        self.assertEqual(1, sim.signals[sim.model.steps.signal])
        sim.step()
        self.assertEqual(2, sim.signals[sim.model.steps.signal])

    def test_time(self):
        m = nengo.Model("test_signal_indexing_1")
        sim = m.simulator(sim_class=self.Simulator)
        self.assertEqual(0.00, sim.signals[sim.model.t.signal])
        sim.step()
        self.assertEqual(0.001, sim.signals[sim.model.t.signal])
        sim.step()
        self.assertEqual(0.002, sim.signals[sim.model.t.signal])

    def test_signal_indexing_1(self):
        m = nengo.Model("test_signal_indexing_1")

        one = Signal(n=1, name='a')
        two = Signal(n=2, name='b')
        three = Signal(n=3, name='c')
        tmp = Signal(n=3, name='tmp')
        m.signals = [one, two, three, tmp]

        m.operators = [
            ProdUpdate(Constant(1), three[:1], Constant(0), one),
            ProdUpdate(Constant(2.0), three[1:], Constant(0), two),
            Reset(tmp),
            DotInc(Constant([[0,0,1],[0,1,0],[1,0,0]]), three, tmp),
            Copy(src=tmp, dst=three, as_update=True),
        ]

        sim = m.simulator(sim_class=self.Simulator, builder=testbuilder)
        sim.signals[three] = np.asarray([1, 2, 3])
        sim.step()
        self.assertTrue(np.all(sim.signals[one] == 1))
        self.assertTrue(np.all(sim.signals[two] == [4, 6]))
        self.assertTrue(np.all(sim.signals[three] == [3, 2, 1]))
        sim.step()
        self.assertTrue(np.all(sim.signals[one] == 3))
        self.assertTrue(np.all(sim.signals[two] == [4, 2]))
        self.assertTrue(np.all(sim.signals[three] == [1, 2, 3]))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
