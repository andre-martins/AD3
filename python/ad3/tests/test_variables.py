import pytest
from ad3 import factor_graph as fg
import numpy as np


def test_binary_variable():
    graph = fg.PFactorGraph()
    var = graph.create_binary_variable()
    var.set_log_potential(0.5)
    assert var.get_log_potential() == 0.5


def test_multi_variable():
    graph = fg.PFactorGraph()
    five = graph.create_multi_variable(5)
    assert len(five) == 5
    vals = np.arange(5).astype(np.double)

    five[1] = vals[1]
    assert five[1] == vals[1]

    five.set_log_potentials(vals)
    for i in range(5):
        assert five[i] == vals[i]

    state = five.get_state(1)
    assert state.get_log_potential() == vals[1]

    with pytest.raises(IndexError):
        five.get_state(6)

    with pytest.raises(IndexError):
        five[6]

    with pytest.raises(IndexError):
        five[6] = 1.1
