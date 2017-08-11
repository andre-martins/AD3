import numpy as np
from ad3 import factor_graph as fg


def test_primal_dual_vars():
    g = fg.PFactorGraph()
    g.solve()
    assert g.get_dual_variables() == []
    assert g.get_local_primal_variables() == []
    assert g.get_global_primal_variables() == []

    g = fg.PFactorGraph()
    a = g.create_binary_variable()
    a.set_log_potential(0)
    b = g.create_binary_variable()
    b.set_log_potential(1)
    g.create_factor_pair([a, b], -2)
    g.create_factor_pair([a, b], -10)
    g.solve()
    assert len(g.get_dual_variables()) == 4
    assert len(g.get_local_primal_variables()) == 4
    assert len(g.get_global_primal_variables()) == 2
