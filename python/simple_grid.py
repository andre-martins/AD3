import itertools
import numpy as np
import ad3


def simple_grid(unaries, pairwise, return_value=False):
    height, width, num_states = unaries.shape

    factor_graph = ad3.PFactorGraph()

    multi_variables = []
    for i in xrange(height):
        multi_variables.append([])
        for j in xrange(width):
            new_variable = factor_graph.create_multi_variable(num_states)
            for state in xrange(num_states):
                new_variable.set_log_potential(state, unaries[i, j, state])
            multi_variables[i].append(new_variable)

    for i, j in itertools.product(xrange(height), xrange(width)):
        if (j > 0):
            #horizontal edge
            edge_variables = [multi_variables[i][j - 1], multi_variables[i][j]]
            factor_graph.create_factor_dense(edge_variables, pairwise.ravel())

        if (i > 0):
            #horizontal edge
            edge_variables = [multi_variables[i - 1][j], multi_variables[i][j]]
            factor_graph.create_factor_dense(edge_variables, pairwise.ravel())

    factor_graph.set_eta_ad3(.1)
    factor_graph.adapt_eta_ad3(True)
    factor_graph.set_max_iterations_ad3(1000)
    value, marginals, edge_marginals = factor_graph.solve_lp_map_ad3()
    marginals = np.array(marginals).reshape(unaries.shape)

    if return_value:
        return marginals, value
    return marginals
