try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np

from sigops import Signal, ProdUpdate, Reset, DotInc, Copy, Simulator


class TestSimulator(unittest.TestCase):
    def test_signal_init_values(self):
        """Tests that initial values are not overwritten."""
        zero = Signal([0])
        one = Signal([1])
        five = Signal([5.0])
        zeroarray = Signal([[0], [0], [0]])
        array = Signal([1, 2, 3])
        operators = [ProdUpdate(zero, zero, one, five),
                     ProdUpdate(zeroarray, one, one, array)]

        sim = Simulator(operators)
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
        sim = Simulator([])
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

        sim = Simulator(operators)
        sim.signals[three] = np.asarray([1, 2, 3])
        sim.step()
        self.assertTrue(np.all(sim.signals[one] == 1))
        self.assertTrue(np.all(sim.signals[two] == [4, 6]))
        self.assertTrue(np.all(sim.signals[three] == [3, 2, 1]))
        sim.step()
        self.assertTrue(np.all(sim.signals[one] == 3))
        self.assertTrue(np.all(sim.signals[two] == [4, 2]))
        self.assertTrue(np.all(sim.signals[three] == [1, 2, 3]))


    def test_signaldict(self):
        """Tests simulator.SignalDict's dict overrides."""
        signaldict = simulator.SignalDict()

        scalar = Signal(1)

        # Both __getitem__ and __setitem__ raise KeyError
        with self.assertRaises(KeyError):
            signaldict[scalar]
        with self.assertRaises(KeyError):
            signaldict[scalar] = np.array(1.)

        signaldict.init(scalar, scalar.value)
        self.assertTrue(np.allclose(signaldict[scalar], np.array(1.)))
        # __getitem__ handles scalars
        self.assertTrue(signaldict[scalar].shape == ())

        one_d = Signal([1])
        signaldict.init(one_d, one_d.value)
        self.assertTrue(np.allclose(signaldict[one_d], np.array([1.])))
        self.assertTrue(signaldict[one_d].shape == (1,))

        two_d = Signal([[1], [1]])
        signaldict.init(two_d, two_d.value)
        self.assertTrue(np.allclose(signaldict[two_d], np.array([[1.], [1.]])))
        self.assertTrue(signaldict[two_d].shape == (2, 1))

        # __getitem__ handles views
        two_d_view = two_d[0, :]
        self.assertTrue(np.allclose(signaldict[two_d_view], np.array([1.])))
        self.assertTrue(signaldict[two_d_view].shape == (1,))

        # __setitem__ ensures memory location stays the same
        memloc = signaldict[scalar].__array_interface__['data'][0]
        signaldict[scalar] = np.array(0.)
        self.assertTrue(np.allclose(signaldict[scalar], np.array(0.)))
        self.assertTrue(signaldict[scalar].__array_interface__['data'][0]
                        == memloc)

        memloc = signaldict[one_d].__array_interface__['data'][0]
        signaldict[one_d] = np.array([0.])
        self.assertTrue(np.allclose(signaldict[one_d], np.array([0.])))
        self.assertTrue(signaldict[one_d].__array_interface__['data'][0]
                        == memloc)

        memloc = signaldict[two_d].__array_interface__['data'][0]
        signaldict[two_d] = np.array([[0.], [0.]])
        self.assertTrue(np.allclose(signaldict[two_d], np.array([[0.], [0.]])))
        self.assertTrue(signaldict[two_d].__array_interface__['data'][0]
                        == memloc)

        # __str__ pretty-prints signals and current values
        # Order not guaranteed for dicts, so we have to loop
        for k in signaldict:
            self.assertTrue("%s %s" % (repr(k), repr(signaldict[k]))
                            in str(signaldict))

    def test_signal(self):
        # Make sure assert_named_signals works
        Signal(np.array(0.))
        Signal.assert_named_signals = True
        with self.assertRaises(AssertionError):
            Signal(np.array(0.))

        # So that other tests that build signals don't fail...
        Signal.assert_named_signals = False


if __name__ == "__main__":
    unittest.main()
