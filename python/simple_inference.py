import itertools
import numpy as np

from . import factor_graph as fg


def simple_grid(unaries, pairwise, verbose=1):
    height, width, n_states = unaries.shape

    graph = fg.PFactorGraph()

    multi_variables = []
    for i in range(height):
        multi_variables.append([])
        for j in range(width):
            new_variable = graph.create_multi_variable(n_states)
            for state in range(n_states):
                new_variable.set_log_potential(state, unaries[i, j, state])
            multi_variables[i].append(new_variable)

    for i, j in itertools.product(range(height), range(width)):
        if j > 0:
            # horizontal edge
            edge_variables = [multi_variables[i][j - 1], multi_variables[i][j]]
            graph.create_factor_dense(edge_variables, pairwise.ravel())

        if i > 0:
            # vertical edge
            edge_variables = [multi_variables[i - 1][j], multi_variables[i][j]]
            graph.create_factor_dense(edge_variables, pairwise.ravel())

    value, marginals, edge_marginals, status = graph.solve(verbose=verbose)
    marginals = np.array(marginals).reshape(unaries.shape)
    edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)

    return marginals, edge_marginals, value, status


def general_graph(unaries, edges, edge_weights, verbose=1, n_iterations=1000,
                  eta=.1, exact=False):
    if unaries.shape[1] != edge_weights.shape[1]:
        raise ValueError("incompatible shapes of unaries"
                         " and edge_weights.")
    if edge_weights.shape[1] != edge_weights.shape[2]:
        raise ValueError("Edge weights need to be of shape "
                         "(n_edges, n_states, n_states)!")
    if edge_weights.shape[0] != edges.shape[0]:
        raise ValueError("Number of edge weights different from number of"
                         "edges")

    factor_graph = fg.PFactorGraph()
    n_states = unaries.shape[-1]

    multi_variables = []
    for u in unaries:
        new_variable = factor_graph.create_multi_variable(n_states)
        for state, cost in enumerate(u):
            new_variable.set_log_potential(state, cost)
        multi_variables.append(new_variable)

    for i, e in enumerate(edges):
            edge_variables = [multi_variables[e[0]], multi_variables[e[1]]]
            factor_graph.create_factor_dense(edge_variables,
                                             edge_weights[i].ravel())

    value, marginals, edge_marginals, solver_status = factor_graph.solve(
        eta=eta,
        adapt=True,
        max_iter=n_iterations,
        branch_and_bound=exact,
        verbose=verbose)

    marginals = np.array(marginals).reshape(unaries.shape)

    edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)

    return marginals, edge_marginals, value, solver_status
