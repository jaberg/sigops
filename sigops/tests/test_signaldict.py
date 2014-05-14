try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
from sigops import Signal, SignalDict


class TestSignalDict(unittest.TestCase):
    def test_signaldict(self):
        """Tests simulator.SignalDict's dict overrides."""
        signaldict = SignalDict()

        scalar = Signal(1)

        with self.assertRaises(KeyError):
            signaldict[scalar]

        signaldict[scalar] = scalar.value
        self.assertTrue(np.allclose(signaldict[scalar], np.array(1.)))
        # __getitem__ handles scalars
        self.assertTrue(signaldict[scalar].shape == ())

        one_d = Signal([1])
        signaldict[one_d] = one_d.value
        self.assertTrue(np.allclose(signaldict[one_d], np.array([1.])))
        self.assertTrue(signaldict[one_d].shape == (1,))

        two_d = Signal([[1], [1]])
        signaldict[two_d] = two_d.value
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


if __name__ == "__main__":
    unittest.main()
