from __future__ import print_function
import numpy as np

import ad3.factor_graph as fg
from ad3 import solve

rng = np.random.RandomState(1)

num_nodes = 10
lower_bound = 3  # Minimum number of zeros.
upper_bound = num_nodes  # Maximum number of zeros.
max_num_bins = lower_bound + 2
counting_state = 1

# Decide whether each position counts for budget.
counts_for_budget = rng.uniform(size=num_nodes) < 0.2
print(counts_for_budget)

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
    children = [available_nodes[j] for j in ind_children]
    for j in children:
        parents[j] = i
        nodes_to_process.insert(0, j)
        available_nodes.remove(j)

print(parents)

# generate random potentials
var_log_potentials = rng.randn(num_nodes)
edge_log_potentials = rng.randn(num_nodes - 1, 4)

# 1) Build a factor graph using DENSE factors.
pairwise_fg = fg.PFactorGraph()
multi_variables = []
for i in range(num_nodes):
    multi_variable = pairwise_fg.create_multi_variable(2)
    multi_variable[0] = 0
    multi_variable[1] = var_log_potentials[i]
    multi_variables.append(multi_variable)


# random edge potentials
for i in range(1, num_nodes):
    var = multi_variables[i]
    parent = multi_variables[parents[i]]
    pairwise_fg.create_factor_dense([parent, var], edge_log_potentials[i - 1])

# If there are upper/lower bounds, add budget factors.
if upper_bound >= 0 or lower_bound >= 0:
    variables = [var.get_state(counting_state)
                 for var, flag in zip(multi_variables, counts_for_budget)
                 if flag]
    negated = [False for _ in variables]
    pairwise_fg.create_factor_budget(variables, negated, upper_bound)
    negated = [True for _ in variables]
    pairwise_fg.create_factor_budget(variables, negated,
                                     len(variables) - lower_bound)

# Run AD3.
value, posteriors, additionals, status = solve(pairwise_fg,
                                               verbose=0,
                                               branch_and_bound=True)

best_states = np.array(posteriors).reshape(-1, 2).argmax(axis=1)
print("Solution using DENSE and BUDGET factors:", best_states)

# 2) Build a factor graph using a BINARY_TREE factor.
tree_fg = fg.PFactorGraph()

variables = []
for i in range(num_nodes):
    var = tree_fg.create_binary_variable()
    var.set_log_potential(var_log_potentials[i])
    variables.append(var)

if upper_bound >= 0 or lower_bound >= 0:
    additionals = np.zeros(num_nodes + 1)
    ix = np.arange(num_nodes + 1)
    additionals[ix < lower_bound] = -1000
    additionals[ix > upper_bound] = -1000
    tree = fg.PFactorBinaryTreeCounts()
    tree_fg.declare_factor(tree, variables, True)
    has_count_scores = [False for _ in parents]
    has_count_scores[0] = True
    tree.initialize(parents, counts_for_budget, has_count_scores, max_num_bins)
    additionals = np.concatenate([edge_log_potentials.ravel(), additionals])
    tree.set_additional_log_potentials(additionals)
else:
    tree = fg.PFactorBinaryTree()
    tree_fg.declare_factor(tree, variables, True)
    tree.initialize(parents)
    tree.set_additional_log_potentials(edge_log_potentials.ravel())

# Run AD3.
value, posteriors, additionals, status = solve(tree_fg,
                                               verbose=0,
                                               branch_and_bound=False)

# for consistent printing with other approach
posteriors = np.array(posteriors).astype(np.int)
print("Solution using BINARY_TREE_COUNTS factor:", posteriors)
