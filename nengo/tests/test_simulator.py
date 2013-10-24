try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo
import nengo.simulator as simulator
from nengo.builder import Signal, Constant
from nengo.builder import DotInc, ProdUpdate, Reset, Copy


class TestSimulator(unittest.TestCase):
    def test_steps(self):
        m = nengo.Model("test_signal_indexing_1")
        sim = m.simulator(sim_class=simulator.Simulator)
        self.assertEqual(0, sim.signals[m.steps])
        sim.step()
        self.assertEqual(1, sim.signals[m.steps])
        sim.step()
        self.assertEqual(2, sim.signals[m.steps])

    def test_time(self):
        m = nengo.Model("test_signal_indexing_1")
        sim = m.simulator(sim_class=simulator.Simulator)
        self.assertEqual(0.00, sim.signals[m.t])
        sim.step()
        self.assertEqual(0.001, sim.signals[m.t])
        sim.step()
        self.assertEqual(0.002, sim.signals[m.t])

    def test_signal_indexing_1(self):
        m = nengo.Model("test_signal_indexing_1")
        one = m.add(Signal(n=1, name='a'))
        two = m.add(Signal(n=2, name='b'))
        three = m.add(Signal(n=3, name='c'))

        tmp = m.add(Signal(n=3, name='tmp'))

        m._operators += [ProdUpdate(
            Constant(1), three[0:1], Constant(0), one)]
        m._operators += [ProdUpdate(
            Constant(2.0), three[1:], Constant(0), two)]
        m._operators += [
            Reset(tmp),
            DotInc(
                Constant([[0,0,1],[0,1,0],[1,0,0]]),
                three,
                tmp),
            Copy(src=tmp, dst=three, as_update=True),
            ]

        sim = m.simulator(sim_class=simulator.Simulator)
        memo = sim.model.memo
        sim.signals[sim.get(three)] = np.asarray([1, 2, 3])
        sim.step()
        self.assertTrue(np.all(sim.signals[sim.get(one)] == 1))
        self.assertTrue(np.all(sim.signals[sim.get(two)] == [4, 6]))
        self.assertTrue(np.all(sim.signals[sim.get(three)] == [3, 2, 1]))
        sim.step()
        self.assertTrue(np.all(sim.signals[sim.get(one)] == 3))
        self.assertTrue(np.all(sim.signals[sim.get(two)] == [4, 2]))
        self.assertTrue(np.all(sim.signals[sim.get(three)] == [1, 2, 3]))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
