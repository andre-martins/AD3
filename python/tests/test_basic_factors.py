import pytest
import numpy as np
from numpy.testing import assert_array_equal

from .. import factor_graph as fg
from .. import solve


def test_pair():
    graph = fg.PFactorGraph()
    a = graph.create_binary_variable()
    b = graph.create_binary_variable()

    assert a.get_degree() == b.get_degree() == 0

    val = 100.0

    pair = graph.create_factor_pair([a, b], val)
    assert a.get_degree() == b.get_degree() == 1

    pair = graph.create_factor_pair([a, b], -val)
    assert a.get_degree() == b.get_degree() == 2


def test_dense():
    graph = fg.PFactorGraph()

    n_a = 2
    a = graph.create_multi_variable(n_a)

    n_b = 3
    b = graph.create_multi_variable(n_b)

    vals = np.arange(n_a * n_b)

    def check_degree_all(degree):

        for i in range(n_a):
            assert a.get_state(i).get_degree() == degree

        for i in range(n_b):
            assert b.get_state(i).get_degree() == degree

    check_degree_all(0)
    graph.create_factor_dense([a, b], vals)
    check_degree_all(1)
    graph.create_factor_dense([a, b], vals)
    check_degree_all(2)


def test_dense_at_most_one():
    # check that dense factor enforces one-of-k

    graph = fg.PFactorGraph()
    n = 3
    a = graph.create_multi_variable(n)
    b = graph.create_multi_variable(n)

    a.set_log_potentials(-100 * np.ones(3))
    b.set_log_potentials(np.ones(3))

    dense_vals = np.zeros((3, 3))
    dense_vals[0, 0] = 1
    dense_vals[1, 2] = 10
    # (0, 0, 0) / (0, 0, 0) would be highest if allowed
    # (1, 1, 0) / (1, 0, 1) would be highest otherwise, if allowed
    # (0, 1, 0) / (0, 0, 1) is the highest allowable.

    graph.create_factor_dense([a, b], dense_vals.ravel())
    _, assignments, _, _ = solve(graph)
    expected = np.array([0, 1, 0, 0, 0, 1])
    assert_array_equal(expected, assignments)


_logic = ['XOR', 'OR', 'XOROUT', 'ATMOSTONE', 'OROUT', 'ANDOUT', 'IMPLY']

@pytest.mark.parametrize('factor_type', _logic)
def test_smoke_logic(factor_type):
    graph = fg.PFactorGraph()
    variables = [graph.create_binary_variable() for _ in range(3)]
    negated = [False for _ in variables]

    for var in variables:
        assert var.get_degree() == 0

    graph.create_factor_logic(factor_type, variables, negated)

    for var in variables:
        assert var.get_degree() == 1

    # check that we find a feasible solution
    val, _, _, _ = solve(graph)
    assert val == 0


def test_logic_validate():
    graph = fg.PFactorGraph()
    variables = [graph.create_binary_variable() for _ in range(3)]
    negated = [False for _ in variables]
    with pytest.raises(NotImplementedError) as e:
        graph.create_factor_logic('foo', variables, negated)

    assert 'unknown' in str(e.value).lower()


def test_logic_constraints():
    graph = fg.PFactorGraph()
    n_states = 3

    var_a = graph.create_multi_variable(n_states)
    var_b = graph.create_multi_variable(n_states)

    var_a.set_log_potential(0, 1)  # A = 0
    var_b.set_log_potential(1, 10) # B = likely 1, possibly 2
    var_b.set_log_potential(2, 9.5)

    graph.fix_multi_variables_without_factors()

    value_unconstrained, marginals, _, _ = solve(graph)

    expected = [0, 1]
    obtained = np.array(marginals).reshape(2, -1).argmax(axis=1)
    assert_array_equal(expected, obtained)

    # add logic constraint: (A = 0) -> not (B = 1)
    graph.create_factor_logic('IMPLY', [var_a.get_state(0),
                                        var_b.get_state(1)],
                              [False, True])

    graph.fix_multi_variables_without_factors()  # need to run this again

    value_constrained, marginals, _, _ = solve(graph)

    expected = [0, 2]
    obtained = np.array(marginals).reshape(2, -1).argmax(axis=1)
    assert_array_equal(expected, obtained)

    # total score must be lower
    assert value_constrained < value_unconstrained
