import itertools
import numpy as np
import matplotlib.pyplot as plt
import pdb

import ad3

grid_size = 20
num_states = 5

factor_graph = ad3.PFactorGraph()

multi_variables = []
random_grid = np.random.uniform(size=(grid_size, grid_size, num_states))
for i in xrange(grid_size):
    multi_variables.append([])
    for j in xrange(grid_size):
        new_variable = factor_graph.create_multi_variable(num_states)
        for state in xrange(num_states):
            new_variable.set_log_potential(state, random_grid[i, j, state])
        multi_variables[i].append(new_variable)

alpha = .3
potts_matrix = alpha * np.eye(num_states)
potts_potentials = potts_matrix.ravel().tolist()

for i, j in itertools.product(xrange(grid_size), repeat=2):
    if (j > 0):
        #horizontal edge
        edge_variables = [multi_variables[i][j - 1], multi_variables[i][j]]
        factor_graph.create_factor_dense(edge_variables, potts_potentials)

    if (i > 0):
        #vertical edge
        edge_variables = [multi_variables[i - 1][j], multi_variables[i][j]]
        factor_graph.create_factor_dense(edge_variables, potts_potentials)


factor_graph.set_eta_ad3(.1)
factor_graph.adapt_eta_ad3(True)
factor_graph.set_max_iterations_ad3(1000)
value, marginals, edge_marginals, solver_status =\
    factor_graph.solve_lp_map_ad3()

res = np.array(marginals).reshape(20, 20, 5)
plt.matshow(np.argmax(random_grid, axis=-1), vmin=0, vmax=4)
plt.matshow(np.argmax(res, axis=-1), vmin=0, vmax=4)
plt.show()

use_sequence_factors = True

if use_sequence_factors:

    # Now do the same with sequence factors.
    # Create a factor graph using sequence-factors which is equivalent to the
    # previous one.

    sequential_factor_graph = ad3.PFactorGraph()

    # Create a binary variable for each state at each position in the grid.
    binary_variables = []
    for i in xrange(grid_size):
        binary_variables.append([])
        for j in xrange(grid_size):
            binary_variables[i].append([])
            for k in xrange(num_states):
                # Assign a random log-potential to each state.
                state_variable = sequential_factor_graph.create_binary_variable()
                log_potential = multi_variables[i][j].get_log_potential(k)
                state_variable.set_log_potential(log_potential)
                binary_variables[i][j].append(state_variable)

    # Design the edge log-potentials.
    # Right now they are diagonal and favoring smooth configurations, but
    # that needs not be the case.
    additional_log_potentials = []
    for i in xrange(grid_size+1):
        if i == 0:
            num_previous_states = 1
        else:
            num_previous_states = num_states
        if i == grid_size:
            num_current_states = 1
        else:
            num_current_states = num_states
        for k in xrange(num_states): # CHECK: num_previous_states?
            for l in xrange(num_states): # CHECK: num_current_states?
                if i != 0 and i != grid_size: # CHECK: delete this if?
                    if k == l:
                        additional_log_potentials.append(alpha)
                    else:
                        additional_log_potentials.append(0.0)

    # Create a sequential factor for each row in the grid.
    # NOTE: need to keep a list of factors, otherwise the Python garbage
    # collector will destroy the factor objects...
    factors = []
    for i in xrange(grid_size):
        variables = []
        num_states = []
        for j in xrange(grid_size):
            variables.extend(binary_variables[i][j])
            num_states.append(len(binary_variables[i][j]))
        #pdb.set_trace()
        factor = ad3.PFactorSequence()
        # Set True below to let the factor graph own the factor so that we
        # don't need to delete it.
        sequential_factor_graph.declare_factor(factor, variables, False)
        factor.initialize(num_states)
        factor.set_additional_log_potentials(additional_log_potentials)
        factors.append(factor)
            
    # Create a sequential factor for each column in the grid.
    for j in xrange(grid_size):
        variables = []
        num_states = []
        for i in xrange(grid_size):
            variables.extend(binary_variables[i][j])
            num_states.append(len(binary_variables[i][j]))
        factor = ad3.PFactorSequence()
        # Set True below to let the factor graph own the factor so that we
        # don't need to delete it.
        sequential_factor_graph.declare_factor(factor, variables, False)
        factor.initialize(num_states)
        factor.set_additional_log_potentials(additional_log_potentials)
        factors.append(factor)
                
    factor_graph.set_eta_ad3(.1)
    factor_graph.adapt_eta_ad3(True)
    factor_graph.set_max_iterations_ad3(1000)
    value, marginals, edge_marginals = factor_graph.solve_lp_map_ad3()

    res = np.array(marginals).reshape(20, 20, 5)
    plt.matshow(np.argmax(res, axis=-1), vmin=0, vmax=4)
    plt.show()
  
