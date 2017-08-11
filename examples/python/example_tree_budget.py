from __future__ import print_function
import numpy as np

from ad3 import PFactorGraph
from ad3.extensions import PFactorGeneralTree, PFactorGeneralTreeCounts

def var_len_argmax(posteriors, num_states):
    t = 0
    best_states = []
    for i in range(len(num_states)):
        local_posteriors = posteriors[t:(t + num_states[i])]
        j = np.argmax(local_posteriors)
        best_states.append(j)
        t += num_states[i]
    return best_states


num_nodes = 5
max_num_states = 2
lower_bound = 3  # Minimum number of zeros.
upper_bound = 4  # Maximum number of zeros.

rng = np.random.RandomState(0)

# Create a random tree.
max_num_children = 5
parents = [-1] * num_nodes
available_nodes = list(range(1, num_nodes))
nodes_to_process = [0]

while len(nodes_to_process) and len(available_nodes):
    i = nodes_to_process.pop()
    num_available = len(available_nodes)
    max_available = min(max_num_children, num_available)
    num_children = rng.randint(1, max_available + 1)
    ind_children = rng.permutation(num_available)[:num_children]
    children = []
    for ind in ind_children:
        children.append(available_nodes[ind])
    for j in children:
        parents[j] = i
        nodes_to_process.insert(0, j)
        available_nodes.remove(j)

print("Randomly picked tree:", parents)

# Design number of states for each node.
num_states = rng.randint(1, max_num_states + 1, size=num_nodes)
print("States per node:", num_states)

# generate random potentials
var_log_potentials = [rng.randn(n) for n in num_states]
edge_log_potentials = [rng.randn(num_states[parents[i]] * num_states[i])
                       for i in range(1, num_nodes)]

# 1) Build a factor graph using DENSE factors.
pairwise_fg = PFactorGraph()
multi_variables = []
for i in range(num_nodes):
    var = pairwise_fg.create_multi_variable(num_states[i])
    var.set_log_potentials(var_log_potentials[i])
    multi_variables.append(var)

description = ''
num_factors = 0

for i in range(1, num_nodes):
    p = parents[i]
    edge_variables = [multi_variables[p], multi_variables[i]]
    pairwise_fg.create_factor_dense(edge_variables,
                                    edge_log_potentials[i - 1])
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
            description += ' ' + str(edge_log_potentials[i - 1][t])
            t += 1
    description += '\n'

if upper_bound >= 0 or lower_bound >= 0:
    binary_vars = [var.get_state(0) for var in multi_variables]
    # Budget factor for upper bound.
    pairwise_fg.create_factor_budget(binary_vars, budget=upper_bound)
    num_factors += 1

    # Print factor to string.
    description += 'BUDGET ' + str(num_nodes)
    for i in range(num_nodes):
        description += ' ' + str(1 + multi_variables[i].get_state(0).get_id())
    description += ' ' + str(upper_bound)
    description += '\n'

    # Budget factor for lower bound.
    pairwise_fg.create_factor_budget(binary_vars,
                                     budget=num_nodes - lower_bound,
                                     negated=[True for _ in binary_vars])
    num_factors += 1

    # Print factor to string.
    description += 'BUDGET ' + str(num_nodes)
    for i in range(num_nodes):
        description += ' ' + str(
            -(1 + binary_vars[i].get_id()))
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
    value, posteriors, _, _ = pairwise_fg.solve(branch_and_bound=True)
    # Print solution.
    best_states = var_len_argmax(posteriors, num_states)
    print("Solution using DENSE and BUDGET factors:", best_states)

# 2) Build a factor graph using a GENERAL_TREE factor.
factor_graph = PFactorGraph()

flat_var_log_potentials = np.concatenate(var_log_potentials)

additional_log_potentials = []
num_current_states = num_states[0]
for i in range(1, num_nodes):
    p = parents[i]
    num_previous_states = num_states[p]
    num_current_states = num_states[i]
    count = 0
    for k in range(num_previous_states):
        for j in range(num_current_states):
            value = edge_log_potentials[i - 1][count]
            count += 1
            additional_log_potentials.append(value)

if upper_bound >= 0 or lower_bound >= 0:
    bounds = np.zeros(num_nodes + 1)
    ix = np.arange(num_nodes + 1)
    bounds[ix < lower_bound] = -1000
    bounds[ix > upper_bound] = -1000
    flat_var_log_potentials = np.concatenate([flat_var_log_potentials, bounds])

binary_variables = []
factors = []
for i in range(len(flat_var_log_potentials)):
    binary_variable = factor_graph.create_binary_variable()
    binary_variable.set_log_potential(flat_var_log_potentials[i])
    binary_variables.append(binary_variable)

if upper_bound >= 0 or lower_bound >= 0:
    factor = PFactorGeneralTreeCounts()

    f = open('example_general_tree_counts.fg', 'w')
    f.write(str(len(binary_variables)) + '\n')
    f.write(str(1) + '\n')
    for i in range(len(binary_variables)):
        f.write(str(flat_var_log_potentials[i]) + '\n')
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
    factor = PFactorGeneralTree()
variables = binary_variables
factor_graph.declare_factor(factor, variables, True)
factor.initialize(parents, num_states)
factor.set_additional_log_potentials(additional_log_potentials)
factors.append(factor)

# Run AD3.
value, posteriors, additionals, status = factor_graph.solve()

# Print solution.
best_states = var_len_argmax(posteriors, num_states)
print("Solution using GENERAL_TREE_COUNTS factor:", best_states)
