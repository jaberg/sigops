try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

from nengo.base import Filter, Signal
from nengo.simulator import Simulator


class TestSimulator(unittest.TestCase):
    def test_signal_indexing_1(self):
        one = Signal(n=1, name='a')
        two = Signal(n=2, name='b')
        three = Signal(n=3, name='c')

        f1 = Filter(1, three[0:1], one)
        f2 = Filter(2.0, three[1:], two)
        f3 = Filter([[0, 0, 1], [0, 1, 0], [1, 0, 0]], three, three)

        sim = Simulator(m)
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
