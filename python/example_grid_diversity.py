import itertools
import numpy as np
import matplotlib.pyplot as plt
import pdb

import ad3

grid_size = 10
num_states = 5
num_diverse_outputs = 4 # Number of diverse outputs to generate.
min_hamming_cost = 32 # Minimum Hamming cost between any pair of outputs.

factor_graph = ad3.PFactorGraph()

description = ''
num_factors = 0

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

        # Print factor to string.
        num_factors += 1    
        description += 'DENSE ' + str(2*num_states)
        for k in xrange(num_states):
            description += ' ' + str(1 + multi_variables[i][j - 1].get_state(k).get_id())
        for k in xrange(num_states):
            description += ' ' + str(1 + multi_variables[i][j].get_state(k).get_id())
        description += ' ' + str(2)
        description += ' ' + str(num_states)
        description += ' ' + str(num_states)
        t = 0
        for k in xrange(num_states):
            for l in xrange(num_states):
                description += ' ' + str(potts_matrix[k][l])
                t += 1
        description += '\n'

    if (i > 0):
        #vertical edge
        edge_variables = [multi_variables[i - 1][j], multi_variables[i][j]]
        factor_graph.create_factor_dense(edge_variables, potts_potentials)

        # Print factor to string.
        num_factors += 1    
        description += 'DENSE ' + str(2*num_states)
        for k in xrange(num_states):
            description += ' ' + str(1 + multi_variables[i - 1][j].get_state(k).get_id())
        for k in xrange(num_states):
            description += ' ' + str(1 + multi_variables[i][j].get_state(k).get_id())
        description += ' ' + str(2)
        description += ' ' + str(num_states)
        description += ' ' + str(num_states)
        t = 0
        for k in xrange(num_states):
            for l in xrange(num_states):
                description += ' ' + str(potts_matrix[k][l])
                t += 1
        description += '\n'

# Plot the evidences.
plt.matshow(np.argmax(random_grid, axis=-1), vmin=0, vmax=4)

for t in xrange(num_diverse_outputs):

  # Write factor graph to file.
  f = open('example_diversity_' + str(t) + '.fg', 'w')
  f.write(str(num_states * grid_size * grid_size) + '\n')
  f.write(str(num_factors) + '\n')
  for i in xrange(grid_size):
    for j in xrange(grid_size):
        for k in xrange(num_states):
          f.write(str(multi_variables[i][j].get_log_potential(k)) + '\n')
  f.write(description)
  f.close()  

  # Solve with AD3.
  factor_graph.set_eta_ad3(.1)
  factor_graph.adapt_eta_ad3(True)
  factor_graph.set_max_iterations_ad3(1000)
  value, marginals, edge_marginals = factor_graph.solve_lp_map_ad3()

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
                                        
  # Print factor to string.
  num_factors += 1    
  description += 'BUDGET ' + str(len(variables))
  for var in variables:
      description += ' ' + str(1 + var.get_id())
  description += ' ' + str(len(variables) - min_hamming_cost)
  description += '\n'
                                        
      
plt.show()


