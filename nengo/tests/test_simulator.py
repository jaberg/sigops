try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo
import nengo.simulator as simulator
from nengo.builder import Builder
from nengo.builder import Signal
from nengo.builder import DotInc, ProdUpdate, Reset, Copy


def testbuilder(model, dt):
    model.dt = dt
    model.seed = 0
    if not hasattr(model, 'probes'):
        model.probes = []
    return model


class TestSimulator(unittest.TestCase):
    Simulator = simulator.Simulator

    def test_signal_init_values(self):
        """Tests that initial values are not overwritten."""
        m = nengo.Model("test_signal_init_values")
        zero = Signal([0])
        one = Signal([1])
        five = Signal([5.0])
        zeroarray = Signal([[0,0,0]])
        array = Signal([1,2,3])
        m.operators = [ProdUpdate(zero, zero, one, five),
                       ProdUpdate(one, zeroarray, one, array)]

        sim = m.simulator(sim_class=simulator.Simulator, builder=testbuilder)
        self.assertEqual(0, sim.signals[sim.get(zero)][0])
        self.assertEqual(1, sim.signals[sim.get(one)][0])
        self.assertEqual(5.0, sim.signals[sim.get(five)][0])
        self.assertTrue(np.all(
            np.array([1,2,3]) == sim.signals[sim.get(array)]))
        sim.step()
        self.assertEqual(0, sim.signals[sim.get(zero)][0])
        self.assertEqual(1, sim.signals[sim.get(one)][0])
        self.assertEqual(5.0, sim.signals[sim.get(five)][0])
        self.assertTrue(np.all(
            np.array([1,2,3]) == sim.signals[sim.get(array)]))

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

        one = Signal(np.zeros(1), name='a')
        two = Signal(np.zeros(2), name='b')
        three = Signal(np.zeros(3), name='c')
        tmp = Signal(np.zeros(3), name='tmp')

        m.operators = [
            ProdUpdate(Signal(1), three[:1], Signal(0), one),
            ProdUpdate(Signal(2.0), three[1:], Signal(0), two),
            Reset(tmp),
            DotInc(Signal([[0,0,1],[0,1,0],[1,0,0]]), three, tmp),
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
    nengo.log(debug=True, path='log.txt')
    unittest.main()
