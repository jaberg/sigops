try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo.simulator as simulator
import nengo.base as base


class TestSimulator(unittest.TestCase):
    def test_signal_indexing_1(self):
        one = base.Signal(n=1, name='a')
        two = base.Signal(n=2, name='b')
        three = base.Signal(n=3, name='c')

        tmp = base.Signal(n=3, name='tmp')

        operators = []
        operators += [simulator.ProdUpdate(base.Constant(1), three[0:1], base.Constant(0), one)]
        operators += [simulator.ProdUpdate(base.Constant(2.0), three[1:], base.Constant(0), two)]
        operators += [
            simulator.Reset(tmp),
            simulator.DotInc(
                base.Constant([[0,0,1],[0,1,0],[1,0,0]]),
                three,
                tmp),
            simulator.Copy(src=tmp, dst=three, as_update=True),
            ]

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
