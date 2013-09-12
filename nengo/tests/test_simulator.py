try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo.simulator as simulator
import nengo.core as core


class TestSimulator(unittest.TestCase):
    def test_signal_indexing_1(self):
        one = core.Signal(n=1, name='a')
        two = core.Signal(n=2, name='b')
        three = core.Signal(n=3, name='c')
        three_out = core.Signal(n=3, name='c-out')

#        m.add(core.Filter(1, three[0:1], one))
#        m.add(core.Filter(2.0, three[1:], two))
#        m.add(core.Filter([[0, 0, 1], [0, 1, 0], [1, 0, 0]], three, three))
        operators = []
        operators += [simulator.Reset(three_out)]
        operators += [simulator.Copy(src=three_out, dst=three, as_update=True)]
        operators += [simulator.ProdUpdate(core.Constant(1), three[0:1], core.Constant(0), one)]
        operators += [simulator.ProdUpdate(core.Constant(2.0), three[1:], core.Constant(0), two)]
        operators += [simulator.DotInc(core.Constant([[0,0,1], [0,1,0], [1,0,0]]), three, three_out)]

        sim = simulator.Simulator(operators)
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
    unittest.main()
