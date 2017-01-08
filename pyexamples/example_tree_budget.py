import numpy as np

import ad3.factor_graph as fg

num_nodes = 5
max_num_states = 2
lower_bound = 3  # Minimum number of zeros.
upper_bound = 4  # Maximum number of zeros.

# Create a random tree.
max_num_children = 5
parents = [-1] * num_nodes
available_nodes = list(range(1, num_nodes))
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

print("Randomly picked tree:", parents)

# Design number of states for each node.
num_states_array = 1 + np.floor(
    np.random.uniform(size=num_nodes) * max_num_states)
num_states = [int(x) for x in num_states_array]
print("States per node:", num_states)

# 1) Build a factor graph using DENSE factors.
pairwise_factor_graph = fg.PFactorGraph()
multi_variables = []
for i in range(num_nodes):
    multi_variable = pairwise_factor_graph.create_multi_variable(num_states[i])
    for state in range(num_states[i]):
        value = np.random.normal()
        multi_variable.set_log_potential(state, value)
    multi_variables.append(multi_variable)

description = ''
num_factors = 0

edge_log_potentials = []
edge_log_potentials.append([])
for i in range(1, num_nodes):
    p = parents[i]
    edge_log_potentials.append([])
    for k in range(num_states[p]):
        for j in range(num_states[i]):
            value = np.random.normal()
            edge_log_potentials[i].append(value)
    edge_variables = []
    edge_variables.append(multi_variables[p])
    edge_variables.append(multi_variables[i])
    pairwise_factor_graph.create_factor_dense(edge_variables,
                                              edge_log_potentials[i])
    num_factors += 1

    # Print factor to string.
    description += 'DENSE ' + str(num_states[p] + num_states[i])
    for k in range(num_states[p]):
        description += ' ' + str(1 + multi_variables[p].get_state(k).get_id())
    for j in range(num_states[i]):
        description += ' ' + str(1 + multi_variables[i].get_state(j).get_id())
    description += ' ' + str(2)
    description += ' ' + str(num_states[p])
    description += ' ' + str(num_states[i])
    t = 0
    for k in range(num_states[p]):
        for j in range(num_states[i]):
            description += ' ' + str(edge_log_potentials[i][t])
            t += 1
    description += '\n'

if upper_bound >= 0 or lower_bound >= 0:
    variables = []
    for i in range(num_nodes):
        variables.append(multi_variables[i].get_state(0))
    # Budget factor for upper bound.
    negated = [False] * num_nodes
    pairwise_factor_graph.create_factor_budget(variables, negated, upper_bound)
    num_factors += 1

    # Print factor to string.
    description += 'BUDGET ' + str(num_nodes)
    for i in range(num_nodes):
        description += ' ' + str(1 + multi_variables[i].get_state(0).get_id())
    description += ' ' + str(upper_bound)
    description += '\n'

    # Budget factor for lower bound.
    negated = [True] * num_nodes
    pairwise_factor_graph.create_factor_budget(variables, negated,
                                               num_nodes - lower_bound)
    num_factors += 1

    # Print factor to string.
    description += 'BUDGET ' + str(num_nodes)
    for i in range(num_nodes):
        description += ' ' + str(
            -(1 + multi_variables[i].get_state(0).get_id()))
    description += ' ' + str(num_nodes - lower_bound)
    description += '\n'

    # Write factor graph to file.
    f = open('example_budget.fg', 'w')
    f.write(str(sum(num_states)) + '\n')
    f.write(str(num_factors) + '\n')
    for i in range(num_nodes):
        for j in range(num_states[i]):
            f.write(str(multi_variables[i].get_log_potential(j)) + '\n')
    f.write(description)
    f.close()

    # Run AD3.
    pairwise_factor_graph.set_eta_ad3(.1)
    pairwise_factor_graph.adapt_eta_ad3(True)
    pairwise_factor_graph.set_max_iterations_ad3(1000)
    value, posteriors, additional_posteriors, status = pairwise_factor_graph.solve_lp_map_ad3()

    # Print solution.
    t = 0
    best_states = []
    for i in range(num_nodes):
        local_posteriors = posteriors[t:(t + num_states[i])]
        j = np.argmax(local_posteriors)
        best_states.append(j)
        t += num_states[i]
    print("Solution using DENSE and BUDGET factors:", best_states)

# 2) Build a factor graph using a GENERAL_TREE factor.
factor_graph = fg.PFactorGraph()

variable_log_potentials = []
additional_log_potentials = []
num_current_states = num_states[0]
for j in range(num_current_states):
    value = multi_variables[0].get_log_potential(j)
    variable_log_potentials.append(value)
for i in range(1, num_nodes):
    p = parents[i]
    num_previous_states = num_states[p]
    num_current_states = num_states[i]
    for j in range(num_current_states):
        value = multi_variables[i].get_log_potential(j)
        variable_log_potentials.append(value)
    count = 0
    for k in range(num_previous_states):
        for j in range(num_current_states):
            value = edge_log_potentials[i][count]
            count += 1
            additional_log_potentials.append(value)

if upper_bound >= 0 or lower_bound >= 0:
    for b in range(num_nodes + 1):
        if b >= lower_bound and b <= upper_bound:
            variable_log_potentials.append(0.0)
        else:
            variable_log_potentials.append(-1000.0)

binary_variables = []
factors = []
for i in range(len(variable_log_potentials)):
    binary_variable = factor_graph.create_binary_variable()
    binary_variable.set_log_potential(variable_log_potentials[i])
    binary_variables.append(binary_variable)

if upper_bound >= 0 or lower_bound >= 0:
    factor = fg.PFactorGeneralTreeCounts()

    f = open('example_general_tree_counts.fg', 'w')
    f.write(str(len(binary_variables)) + '\n')
    f.write(str(1) + '\n')
    for i in range(len(binary_variables)):
        f.write(str(variable_log_potentials[i]) + '\n')
    f.write('GENERAL_TREE_COUNTS ' + str(len(binary_variables)))
    for i in range(len(binary_variables)):
        f.write(' ' + str(i + 1))
    f.write(' ' + str(num_nodes))
    for i in range(num_nodes):
        f.write(' ' + str(num_states[i]))
    for i in range(num_nodes):
        f.write(' ' + str(parents[i]))
    for i in range(len(additional_log_potentials)):
        f.write(' ' + str(additional_log_potentials[i]))
    f.write('\n')
    f.close()

else:
    factor = fg.PFactorGeneralTree()
variables = binary_variables
factor_graph.declare_factor(factor, variables, True)
factor.initialize(parents, num_states)
factor.set_additional_log_potentials(additional_log_potentials)
factors.append(factor)

# Run AD3.        
factor_graph.set_eta_ad3(.1)
factor_graph.adapt_eta_ad3(True)
factor_graph.set_max_iterations_ad3(1000)
value, posteriors, additional_posteriors, status = factor_graph.solve_lp_map_ad3()

# Print solution.
t = 0
best_states = []
for i in range(num_nodes):
    local_posteriors = posteriors[t:(t + num_states[i])]
    j = np.argmax(local_posteriors)
    best_states.append(j)
    t += num_states[i]
print("Solution using GENERAL_TREE_COUNTS factor:", best_states)
