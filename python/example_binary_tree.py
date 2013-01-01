import itertools
import numpy as np
import matplotlib.pyplot as plt
import pdb

import ad3

num_nodes = 100
lower_bound = 30 #5 # Minimum number of zeros.
upper_bound = 60 #10 # Maximum number of zeros.
counting_state = 1

# Decide whether each position counts for budget.
counts_for_budget = []
for i in xrange(num_nodes):
    value = np.random.uniform()
    if value < 0.2:
        counts_for_budget.append(False)
    else:
        counts_for_budget.append(True)
print counts_for_budget

# Create a random tree.
max_num_children = 5
parents = [-1] * num_nodes
available_nodes = range(1, num_nodes)
nodes_to_process = [0]
while len(nodes_to_process) > 0:
    i = nodes_to_process.pop()
    num_children = 1 + np.floor(np.random.uniform() * max_num_children)
    if num_children > len(available_nodes):
        num_children = len(available_nodes)
    ind_children = np.random.permutation(len(available_nodes))[0:num_children]
    children = []
    for ind in ind_children:
        children.append(available_nodes[ind])
    for j in children:
        parents[j] = i
        nodes_to_process.insert(0, j)
        available_nodes.remove(j)        
#parents = range(-1, num_nodes-1)
print parents


# 1) Build a factor graph using DENSE factors.
pairwise_factor_graph = ad3.PFactorGraph()
multi_variables = []
for i in xrange(num_nodes):
    multi_variable = pairwise_factor_graph.create_multi_variable(2)
    value = np.random.normal()
    multi_variable.set_log_potential(0, 0.0)
    multi_variable.set_log_potential(1, value)
    multi_variables.append(multi_variable)

description = ''
num_factors = 0

edge_log_potentials = []
edge_log_potentials.append([])
for i in xrange(1, num_nodes):
    p = parents[i]
    edge_log_potentials.append([])
    for k in xrange(2):
        for j in xrange(2):
            value = np.random.normal()
            edge_log_potentials[i].append(value)
    edge_variables = []
    edge_variables.append(multi_variables[p])
    edge_variables.append(multi_variables[i])
    pairwise_factor_graph.create_factor_dense(edge_variables, edge_log_potentials[i])
    num_factors += 1
    
    # Print factor to string.
    description += 'DENSE ' + str(4)
    for k in xrange(2):
        description += ' ' + str(1 + multi_variables[p].get_state(k).get_id())
    for j in xrange(2):
        description += ' ' + str(1 + multi_variables[i].get_state(j).get_id())
    description += ' ' + str(2)
    description += ' ' + str(2)
    description += ' ' + str(2)
    t = 0
    for k in xrange(2):
        for j in xrange(2):
            description += ' ' + str(edge_log_potentials[i][t])
            t += 1
    description += '\n'
    
# If there are upper/lower bounds, add budget factors.
if upper_bound >= 0 or lower_bound >= 0:
    variables = []
    for i in xrange(num_nodes):
        if counts_for_budget[i]:
            variables.append(multi_variables[i].get_state(counting_state))
    # Budget factor for upper bound.
    negated = [False] * len(variables)
    pairwise_factor_graph.create_factor_budget(variables, negated, upper_bound)
    num_factors += 1

    # Print factor to string.
    num_counting_nodes = len([i for i in xrange(num_nodes) if counts_for_budget[i]])
    description += 'BUDGET ' + str(num_counting_nodes)
    for i in xrange(num_nodes):
        if counts_for_budget[i]:    
            description += ' ' + str(1 + multi_variables[i].get_state(counting_state).get_id())
    description += ' ' + str(upper_bound)
    description += '\n'
    
    # Budget factor for lower bound.
    negated = [True] * len(variables)
    pairwise_factor_graph.create_factor_budget(variables, negated, len(variables) - lower_bound)
    num_factors += 1

    # Print factor to string.
    description += 'BUDGET ' + str(num_counting_nodes)
    for i in xrange(num_nodes):
        if counts_for_budget[i]:    
            description += ' ' + str(-(1 + multi_variables[i].get_state(counting_state).get_id()))
    description += ' ' + str(num_counting_nodes - lower_bound)
    description += '\n'
    
      
# Write factor graph to file.
f = open('example_binary_tree_dense.fg', 'w')
f.write(str(2*num_nodes) + '\n')
f.write(str(num_factors) + '\n')
for i in xrange(num_nodes):
  for j in xrange(2):
      f.write(str(multi_variables[i].get_log_potential(j)) + '\n')
f.write(description)
f.close()  

# Run AD3.        
pairwise_factor_graph.set_eta_ad3(.1)
pairwise_factor_graph.adapt_eta_ad3(True)
pairwise_factor_graph.set_max_iterations_ad3(1000)
value, posteriors, additional_posteriors = pairwise_factor_graph.solve_exact_map_ad3()
  
# Print solution.
t = 0
best_states = []
for i in xrange(num_nodes):
    local_posteriors = posteriors[t:(t+2)]
    j = np.argmax(local_posteriors)
    best_states.append(j)
    t += 2
print best_states


# 2) Build a factor graph using a BINARY_TREE factor.
factor_graph = ad3.PFactorGraph()

variable_log_potentials = []
additional_log_potentials = []
num_current_states = 2
value = multi_variables[0].get_log_potential(1)
variable_log_potentials.append(value)
for i in xrange(1, num_nodes):
    p = parents[i]
    num_previous_states = 2
    num_current_states = 2
    value = multi_variables[i].get_log_potential(1)
    variable_log_potentials.append(value)
    count = 0
    for k in xrange(2):
        for j in xrange(2):
            value = edge_log_potentials[i][count]
            count += 1
            additional_log_potentials.append(value)
            
            
binary_variables = []
factors = []
for i in xrange(len(variable_log_potentials)):
    binary_variable = factor_graph.create_binary_variable()
    binary_variable.set_log_potential(variable_log_potentials[i])
    binary_variables.append(binary_variable)
    
    
#pdb.set_trace()

if upper_bound >= 0 or lower_bound >= 0:
    for i in xrange(num_nodes + 1):
        if i < lower_bound or i > upper_bound:
            additional_log_potentials.append(-1000.0)
        else:
            additional_log_potentials.append(0.0)
    
    factor = ad3.PFactorBinaryTreeCounts()
    variables = binary_variables
    factor_graph.declare_factor(factor, variables, True)
    factor.initialize(parents, counts_for_budget)
    factor.set_additional_log_potentials(additional_log_potentials)
    factors.append(factor)
else:    
    factor = ad3.PFactorBinaryTree()
    variables = binary_variables
    factor_graph.declare_factor(factor, variables, True)
    factor.initialize(parents)
    factor.set_additional_log_potentials(additional_log_potentials)
    factors.append(factor)

# Run AD3.        
factor_graph.set_eta_ad3(.1)
factor_graph.adapt_eta_ad3(True)
factor_graph.set_max_iterations_ad3(1000)
value, posteriors, additional_posteriors = factor_graph.solve_lp_map_ad3()

# Print solution.
t = 0
best_states = []
for i in xrange(num_nodes):
    if posteriors[t] > 0.5:
        j = 1
    else:
        j = 0
    best_states.append(j)
    t += 1
print best_states

#pdb.set_trace()


