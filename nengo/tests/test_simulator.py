try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

import nengo.simulator as simulator
from nengo.base import Signal
from nengo.base import DotInc, ProdUpdate, Reset, Copy


class TestSimulator(unittest.TestCase):
    def test_signal_init_values(self):
        """Tests that initial values are not overwritten."""
        zero = Signal(value=[0])
        one = Signal(value=[1])
        five = Signal(value=[5.0])
        zeroarray = Signal(value=np.array([[0,0,0]]))
        array = Signal(value=np.array([1,2,3]))
        signals = [zero, one, five, array]
        operators = [ProdUpdate(zero, zero, one, five),
                     ProdUpdate(one, zeroarray, one, array)]

        sim = simulator.Simulator(operators)
        self.assertEqual(1, sim.signals[one])
        self.assertEqual(5.0, sim.signals[five])
        self.assertTrue(np.all(
            np.array([1,2,3]) == sim.signals[array]))
        sim.step()
        self.assertEqual(1, sim.signals[one])
        self.assertEqual(5.0, sim.signals[five])
        self.assertTrue(np.all(
            np.array([1,2,3]) == sim.signals[array]))

    def test_signal_indexing_1(self):
        one = Signal(shape=1, name='a')
        two = Signal(shape=2, name='b')
        three = Signal(shape=3, name='c')
        tmp = Signal(shape=3, name='tmp')

        operators = [
            ProdUpdate(Signal(value=1), three[0:1], Signal(value=0), one),
            ProdUpdate(Signal(value=2.0), three[1:], Signal(value=0), two),
            Reset(tmp),
            DotInc(Signal(value=[[0,0,1],[0,1,0],[1,0,0]]), three, tmp),
            Copy(src=tmp, dst=three, as_update=True),
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
