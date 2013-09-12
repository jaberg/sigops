try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo
import nengo.simulator as simulator
import nengo.core as core


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
        one = m.add(core.Signal(n=1, name='a'))
        two = m.add(core.Signal(n=2, name='b'))
        three = m.add(core.Signal(n=3, name='c'))

        m._operators += [simulator.ProdUpdate(core.Constant(1), three[0:1], core.Constant(0), one)]
        m._operators += [simulator.ProdUpdate(core.Constant(2.0), three[1:], core.Constant(0), two)]
        m._operators += [simulator.DotInc(core.Constant([[0,0,1], [0,1,0], [1,0,0]]), three, m._get_output_view(three))]

        sim = m.simulator(sim_class=simulator.Simulator)
        memo = sim.model.memo
        sim.signals[sim.copied(three)] = np.asarray([1, 2, 3])
        sim.step()
        self.assertTrue(np.all(sim.signals[sim.copied(one)] == 1))
        self.assertTrue(np.all(sim.signals[sim.copied(two)] == [4, 6]))
        self.assertTrue(np.all(sim.signals[sim.copied(three)] == [3, 2, 1]))
        sim.step()
        self.assertTrue(np.all(sim.signals[sim.copied(one)] == 3))
        self.assertTrue(np.all(sim.signals[sim.copied(two)] == [4, 2]))
        self.assertTrue(np.all(sim.signals[sim.copied(three)] == [1, 2, 3]))


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
