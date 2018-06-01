from __future__ import print_function
import os
import itertools
from time import time
import numpy as np

from ad3 import PFactorGraph
from ad3.extensions import PFactorSequence

plot = False if os.environ.get('NOPLOT') else True
if plot:
    import matplotlib.pyplot as plt

grid_size = 20
num_states = 5

factor_graph = PFactorGraph()

multi_variables = []
random_grid = np.random.uniform(size=(grid_size, grid_size, num_states))
for i in range(grid_size):
    multi_variables.append([])
    for j in range(grid_size):
        new_variable = factor_graph.create_multi_variable(num_states)
        for state in range(num_states):
            new_variable.set_log_potential(state, random_grid[i, j, state])
        multi_variables[i].append(new_variable)

alpha = .3
potts_matrix = alpha * np.eye(num_states)
potts_potentials = potts_matrix.ravel().tolist()

for i, j in itertools.product(range(grid_size), repeat=2):
    if j > 0:
        # horizontal edge
        edge_variables = [multi_variables[i][j - 1], multi_variables[i][j]]
        factor_graph.create_factor_dense(edge_variables, potts_potentials)

    if i > 0:
        # vertical edge
        edge_variables = [multi_variables[i - 1][j], multi_variables[i][j]]
        factor_graph.create_factor_dense(edge_variables, potts_potentials)

tic = time()
value, marginals, edge_marginals, solver_status = factor_graph.solve()
toc = time()

res = np.array(marginals).reshape(grid_size, grid_size, num_states)

unary = np.argmax(random_grid, axis=-1)
out = np.argmax(res, axis=-1)
if plot:
    plt.matshow(unary, vmin=0, vmax=4)
    plt.title("Unary potentials")
    plt.matshow(out, vmin=0, vmax=4)
    plt.title("Result of inference with dense factors ({:.2f}s)".format(
                toc - tic))
else:
    print("unary potentials: \n", unary)
    print("result with dense factors: \n", out)


use_sequence_factors = True

if use_sequence_factors:

    # Now do the same with sequence factors.
    # Create a factor graph using sequence-factors which is equivalent to the
    # previous one.

    sequential_factor_graph = PFactorGraph()

    # Create a binary variable for each state at each position in the grid.
    binary_variables = []
    for i in range(grid_size):
        binary_variables.append([])
        for j in range(grid_size):
            binary_variables[i].append([])
            for k in range(num_states):
                # Assign a random log-potential to each state.
                var = sequential_factor_graph.create_binary_variable()
                log_potential = multi_variables[i][j].get_log_potential(k)
                var.set_log_potential(log_potential)
                binary_variables[i][j].append(var)

    # Design the edge log-potentials.
    # Right now they are diagonal and favoring smooth configurations, but
    # that needs not be the case.
    additional_log_potentials = []
    for i in range(grid_size + 1):
        if i == 0:
            num_previous_states = 1
        else:
            num_previous_states = num_states
        if i == grid_size:
            num_current_states = 1
        else:
            num_current_states = num_states
        for k in range(num_previous_states):
            for l in range(num_current_states):
                if k == l and i != 0 and i != grid_size:
                    additional_log_potentials.append(alpha)
                else:
                    additional_log_potentials.append(0.0)

    # Create a sequential factor for each row in the grid.
    # NOTE: need to keep a list of factors, otherwise the Python garbage
    # collector will destroy the factor objects...
    factors = []
    for i in range(grid_size):
        variables = []
        col_num_states = []
        for j in range(grid_size):
            variables.extend(binary_variables[i][j])
            col_num_states.append(len(binary_variables[i][j]))
        factor = PFactorSequence()
        # Set True below to let the factor graph own the factor so that we
        # don't need to delete it.
        sequential_factor_graph.declare_factor(factor, variables, False)
        factor.initialize(col_num_states)
        factor.set_additional_log_potentials(additional_log_potentials)
        factors.append(factor)

    # Create a sequential factor for each column in the grid.
    for j in range(grid_size):
        variables = []
        col_num_states = []
        for i in range(grid_size):
            variables.extend(binary_variables[i][j])
            col_num_states.append(len(binary_variables[i][j]))
        factor = PFactorSequence()
        # Set True below to let the factor graph own the factor so that we
        # don't need to delete it.
        sequential_factor_graph.declare_factor(factor, variables, False)
        factor.initialize(col_num_states)
        factor.set_additional_log_potentials(additional_log_potentials)
        factors.append(factor)

    tic = time()
    _, marginals, _, _ = sequential_factor_graph.solve()
    toc = time()

    res = np.array(marginals).reshape(grid_size, grid_size, num_states)
    res = np.argmax(res, axis=-1)
    if plot:
        plt.matshow(res, vmin=0, vmax=4)
        plt.title("Results of inference with sequence factors ({:.2f}s)".format(
            toc - tic))
    else:
        print("inference result with sequence:\n", res)

if plot:
    plt.show()
