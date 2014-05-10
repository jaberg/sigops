import numpy as np
from nengo import Model
from nengo.objects import Filter, Signal
from nengo.simulator import Simulator


def test_signal_indexing_1():
    m = Model("test_signal_indexing_1")
    one = m.add(Signal(1))
    two = m.add(Signal(2))
    three = m.add(Signal(3))

    m.add(Filter(1, three[0:1], one))
    m.add(Filter(2.0, three[1:], two))
    m.add(Filter([[0, 0, 1], [0, 1, 0], [1, 0, 0]], three, three))

    sim = Simulator(m)
    sim.signals[three] = np.asarray([1, 2, 3])
    sim.step()
    assert np.all(sim.signals[one] == 1)
    assert np.all(sim.signals[two] == [4, 6])
    assert np.all(sim.signals[three] == [3, 2, 1])
    sim.step()
    assert np.all(sim.signals[one] == 3)
    assert np.all(sim.signals[two] == [4, 2])
    assert np.all(sim.signals[three] == [1, 2, 3])
