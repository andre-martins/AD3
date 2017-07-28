from __future__ import print_function
import numpy as np
import ad3.factor_graph as fg
from ad3 import solve
import time


def test_random_instance(n):
    costs = np.random.rand(n)
    budget = np.sum(costs) * np.random.rand()
    scores = np.random.randn(n)

    tic = time.clock()
    x = solve_lp_knapsack_ad3(scores, costs, budget)
    toc = time.clock()
    print('ad3: {:.2f}'.format(toc - tic))

    try:
        tic = time.clock()
        x_gold = solve_lp_knapsack_lpsolve(scores, costs, budget)
        toc = time.clock()
        print('lpsolve: {:.2f}'.format(toc - tic))
        res = x - x_gold
        assert np.linalg.norm(res) < 1e-6

    except ImportError:
        print('lpsolve not available')


def solve_lp_knapsack_ad3(scores, costs, budget):
    factor_graph = fg.PFactorGraph()
    binary_variables = []
    for i in range(len(scores)):
        binary_variable = factor_graph.create_binary_variable()
        binary_variable.set_log_potential(scores[i])
        binary_variables.append(binary_variable)

    negated = [False] * len(binary_variables)
    factor_graph.create_factor_knapsack(binary_variables, negated, costs,
                                        budget)

    # Run AD3.
    _, posteriors, _, _ = solve(factor_graph)
    return posteriors


def solve_lp_knapsack_gurobi(scores, costs, budget):
    from gurobipy import Model, LinExpr, GRB

    n = len(scores)

    # Create a new model.
    m = Model("lp_knapsack")

    # Create variables.
    for i in range(n):
        m.addVar(lb=0.0, ub=1.0)
    m.update()
    vars = m.getVars()

    # Set objective.
    obj = LinExpr()
    for i in range(n):
        obj += scores[i] * vars[i]
    m.setObjective(obj, GRB.MAXIMIZE)

    # Add constraint.
    expr = LinExpr()
    for i in range(n):
        expr += costs[i] * vars[i]
    m.addConstr(expr, GRB.LESS_EQUAL, budget)

    # Optimize.
    m.optimize()
    assert m.status == GRB.OPTIMAL
    x = np.zeros(n)
    for i in range(n):
        x[i] = vars[i].x

    return x


def solve_lp_knapsack_lpsolve(scores, costs, budget):
    import lpsolve55 as lps

    relax = True
    n = len(scores)

    lp = lps.lpsolve('make_lp', 0, n)
    # Set verbosity level. 3 = only warnings and errors.
    lps.lpsolve('set_verbose', lp, 3)
    lps.lpsolve('set_obj_fn', lp, -scores)

    lps.lpsolve('add_constraint', lp, costs, lps.LE, budget)

    lps.lpsolve('set_lowbo', lp, np.zeros(n))
    lps.lpsolve('set_upbo', lp, np.ones(n))

    if not relax:
        lps.lpsolve('set_int', lp, [True] * n)
    else:
        lps.lpsolve('set_int', lp, [False] * n)

    # Solve the ILP, and call the debugger if something went wrong.
    ret = lps.lpsolve('solve', lp)
    assert ret == 0

    # Retrieve solution and return
    x, _ = lps.lpsolve('get_variables', lp)
    x = np.array(x)

    return x


if __name__ == "__main__":
    n_tests = 100
    n = 100
    for i in range(n_tests):
        test_random_instance(n)
