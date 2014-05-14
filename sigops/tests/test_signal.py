try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
from sigops import Signal


class TestSignalDict(unittest.TestCase):
    def test_assert_named_signals(self):
        Signal(np.array(0.))
        Signal.assert_named_signals = True
        with self.assertRaises(AssertionError):
            Signal(np.array(0.))

        # So that other tests that build signals don't fail...
        Signal.assert_named_signals = False


if __name__ == "__main__":
    unittest.main()
