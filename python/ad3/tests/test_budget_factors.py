import pytest
from numpy.testing import assert_array_almost_equal

from ad3 import factor_graph as fg


def test_knapsack_wrong_cost_size():
    graph = fg.PFactorGraph()
    n_vars = 50
    variables = [graph.create_binary_variable() for _ in range(n_vars)]
    budget = 1

    with pytest.raises(ValueError):
        small_cost = [17]
        graph.create_factor_knapsack(variables, costs=small_cost, budget=budget)

    with pytest.raises(ValueError):
        big_cost = [17] * (n_vars + 1)
        graph.create_factor_knapsack(variables, costs=big_cost, budget=budget)

    with pytest.raises(TypeError):
        graph.create_factor_knapsack(variables, costs=42, budget=budget)

    with pytest.raises(TypeError):
        graph.create_factor_knapsack(variables, costs=None, budget=budget)


def test_budget():
    graph = fg.PFactorGraph()

    potentials = [100, 1, 100, 1, 100]

    for val in potentials:
        var = graph.create_binary_variable()
        var.set_log_potential(val)

    _, assign, _, _ = graph.solve()
    assert sum(assign) == 5

    budget = 3

    graph = fg.PFactorGraph()

    variables = [graph.create_binary_variable() for _ in potentials]
    for var, val in zip(variables, potentials):
        var.set_log_potential(val)

    graph.create_factor_budget(variables, budget=budget)
    _, assign, _, status = graph.solve()
    assert_array_almost_equal(assign, [1, 0, 1, 0, 1])


def test_knapsack():
    graph = fg.PFactorGraph()

    potentials = [100, 1, 100, 1, 100]
    costs = [3, 5, 5, 5, 2]

    for val in potentials:
        var = graph.create_binary_variable()
        var.set_log_potential(val)

    _, assign, _, _ = graph.solve()
    assert sum(assign) == 5

    budget = 5

    graph = fg.PFactorGraph()
    variables = [graph.create_binary_variable() for _ in potentials]
    for var, val in zip(variables, potentials):
        var.set_log_potential(val)

    graph.create_factor_knapsack(variables, costs, budget)
    _, assign, _, status = graph.solve(branch_and_bound=True)
    assert_array_almost_equal(assign, [1, 0, 0, 0, 1])
