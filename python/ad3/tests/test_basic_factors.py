import pytest
import numpy as np
from numpy.testing import assert_array_equal

from ad3 import factor_graph as fg


def test_pair():
    graph = fg.PFactorGraph()
    a = graph.create_binary_variable()
    b = graph.create_binary_variable()

    assert a.get_degree() == b.get_degree() == 0

    val = 100.0

    graph.create_factor_pair([a, b], val)
    assert a.get_degree() == b.get_degree() == 1

    graph.create_factor_pair([a, b], -val)
    assert a.get_degree() == b.get_degree() == 2

    with pytest.raises(TypeError):
        graph.create_factor_pair([a, b], None)

    with pytest.raises(TypeError):
        graph.create_factor_pair(None, val)

    with pytest.raises(TypeError):
        graph.create_factor_pair(-1, val)

    with pytest.raises(TypeError):
        graph.create_factor_pair([a, -1], val)

    with pytest.raises(AttributeError):
        graph.create_factor_pair([a, None], val)

    with pytest.raises(ValueError):
        graph.create_factor_pair([a, b, a], val)


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

    with pytest.raises(TypeError):
        graph.create_factor_dense(None, vals)

    with pytest.raises(TypeError):
        graph.create_factor_dense(-1, vals)

    with pytest.raises(TypeError):
        graph.create_factor_dense([a, -1], vals)

    with pytest.raises(AttributeError):
        graph.create_factor_dense([a, None], vals)

    with pytest.raises(ValueError):
        graph.create_factor_dense([a, b], vals[:-1])

    with pytest.raises(ValueError):
        graph.create_factor_dense([a, b], np.concatenate([vals, vals]))

    with pytest.raises(TypeError):
        graph.create_factor_dense([a, b], None)


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
    _, assignments, _, _ = graph.solve()
    expected = np.array([0, 1, 0, 0, 0, 1])
    assert_array_equal(expected, assignments)


_logic = ['XOR', 'OR', 'XOROUT', 'ATMOSTONE', 'OROUT', 'ANDOUT', 'IMPLY']


@pytest.mark.parametrize('factor_type', _logic)
def test_smoke_logic(factor_type):
    graph = fg.PFactorGraph()
    variables = [graph.create_binary_variable() for _ in range(3)]

    for var in variables:
        assert var.get_degree() == 0

    graph.create_factor_logic(factor_type, variables)

    for var in variables:
        assert var.get_degree() == 1

    # check that we find a feasible solution
    val, _, _, _ = graph.solve()
    assert val == 0


def test_logic_validate():
    graph = fg.PFactorGraph()
    variables = [graph.create_binary_variable() for _ in range(3)]
    with pytest.raises(NotImplementedError) as e:
        graph.create_factor_logic('foo', variables)

    assert 'unknown' in str(e.value).lower()

    with pytest.raises(ValueError):
        graph.create_factor_logic('OR', variables, negated=[True])

    with pytest.raises(ValueError):
        graph.create_factor_logic('OR', variables, negated=[True] * 10)

    with pytest.raises(TypeError):
        graph.create_factor_logic('OR', variables, negated=42)


def test_logic_negate():
    # not a & not b is equiv to not(a or b)
    rng = np.random.RandomState()

    for _ in range(10):
        potentials = rng.randn(3)
        graph = fg.PFactorGraph()
        variables = [graph.create_binary_variable() for _ in range(3)]
        for var, val in zip(variables, potentials):
            var.set_log_potential(val)
        graph.create_factor_logic('ANDOUT', variables, [True, True, False])
        val_1, (_, _, out_1), _, _ = graph.solve()

        graph = fg.PFactorGraph()
        variables = [graph.create_binary_variable() for _ in range(3)]
        for var, val in zip(variables, potentials):
            var.set_log_potential(val)
        graph.create_factor_logic('OROUT', variables, [False, False, True])
        val_2, (_, _, out_2), _, _ = graph.solve()

        assert val_1 == val_2
        assert out_1 == out_2


def test_logic_constraints():
    graph = fg.PFactorGraph()
    n_states = 3

    var_a = graph.create_multi_variable(n_states)
    var_b = graph.create_multi_variable(n_states)

    var_a.set_log_potential(0, 1)  # A = 0
    var_b.set_log_potential(1, 10)  # B = likely 1, possibly 2
    var_b.set_log_potential(2, 9.5)

    graph.fix_multi_variables_without_factors()

    value_unconstrained, marginals, _, _ = graph.solve()

    expected = [0, 1]
    obtained = np.array(marginals).reshape(2, -1).argmax(axis=1)
    assert_array_equal(expected, obtained)

    # add logic constraint: (A = 0) -> not (B = 1)
    graph.create_factor_logic('IMPLY', [var_a.get_state(0),
                                        var_b.get_state(1)],
                              [False, True])

    graph.fix_multi_variables_without_factors()  # need to run this again

    value_constrained, marginals, _, _ = graph.solve()

    expected = [0, 2]
    obtained = np.array(marginals).reshape(2, -1).argmax(axis=1)
    assert_array_equal(expected, obtained)

    # total score must be lower
    assert value_constrained < value_unconstrained
