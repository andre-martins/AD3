import itertools
import numpy as np
import matplotlib.pyplot as plt
import pdb

import ad3

grid_size = 10
num_states = 5
num_diverse_outputs = 4 # Number of diverse outputs to generate.
min_hamming_cost = 8 # Minimum Hamming cost between any pair of outputs.

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

# Plot the evidences.
plt.matshow(np.argmax(random_grid, axis=-1), vmin=0, vmax=4)

for t in xrange(num_diverse_outputs):
  factor_graph.set_eta_ad3(.1)
  factor_graph.adapt_eta_ad3(True)
  factor_graph.set_max_iterations_ad3(1000)
  value, marginals, edge_marginals = factor_graph.solve_lp_map_ad3()
  #pdb.set_trace()

  res = np.array(marginals).reshape(grid_size, grid_size, num_states)
  output = np.argmax(res, axis=-1)
  print output
  plt.matshow(output, vmin=0, vmax=4)
  #plt.show()

  # Add another budget constraint.
  variables = []
  for i, j in itertools.product(xrange(grid_size), repeat=2):
      v = output[i][j]
      variables.append(multi_variables[i][j].get_state(v))
      # Budget factor imposing a minimum Hamming cost w.r.t. this output.
      negated = [False] * len(variables)
      factor_graph.create_factor_budget(variables, negated,
                                        len(variables) - min_hamming_cost)
      
plt.show()


