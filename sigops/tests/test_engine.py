try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
from sigops import Signal, ProdUpdate, Reset, DotInc, Copy, Engine


class TestEngine(unittest.TestCase):
    def test_signal_init_values(self):
        """Tests that initial values are not overwritten."""
        zero = Signal([0])
        one = Signal([1])
        five = Signal([5.0])
        zeroarray = Signal([[0], [0], [0]])
        array = Signal([1, 2, 3])
        operators = [ProdUpdate(zero, zero, one, five),
                     ProdUpdate(zeroarray, one, one, array)]

        sim = Engine(operators)
        self.assertEqual(0, sim.signals[zero][0])
        self.assertEqual(1, sim.signals[one][0])
        self.assertEqual(5.0, sim.signals[five][0])
        self.assertTrue(np.all(
            np.array([1, 2, 3]) == sim.signals[array]))
        sim.step()
        self.assertEqual(0, sim.signals[zero][0])
        self.assertEqual(1, sim.signals[one][0])
        self.assertEqual(5.0, sim.signals[five][0])
        self.assertTrue(np.all(
            np.array([1, 2, 3]) == sim.signals[array]))

    def test_steps(self):
        sim = Engine([])
        self.assertEqual(0, sim.n_steps)
        sim.step()
        self.assertEqual(1, sim.n_steps)
        sim.step()
        self.assertEqual(2, sim.n_steps)

    def test_signal_indexing_1(self):
        one = Signal(np.zeros(1), name='a')
        two = Signal(np.zeros(2), name='b')
        three = Signal(np.zeros(3), name='c')
        tmp = Signal(np.zeros(3), name='tmp')

        operators = [
            ProdUpdate(Signal(1), three[:1], Signal(0), one),
            ProdUpdate(Signal(2.0), three[1:], Signal(0), two),
            Reset(tmp),
            DotInc(Signal([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), three, tmp),
            Copy(src=tmp, dst=three, as_update=True),
        ]

        sim = Engine(operators)
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
