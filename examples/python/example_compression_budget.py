from __future__ import print_function
import numpy as np

import ad3.factor_graph as fg

length = 30
budget = 10

rng = np.random.RandomState(1)

# Decide bigram_positions.
bigram_positions = []
for i in range(-1, length):
    value = rng.uniform()
    if value < 0.4:
        bigram_positions.append(i)

# Decide whether each position counts for budget.
counts_for_budget = rng.uniform(size=length) < 0.1

var_log_potentials = rng.randn(length)

# 1) Build a factor graph using a SEQUENCE and a BUDGET factor.
factor_graph = fg.PFactorGraph()
multi_variables = []
for i in range(length):
    multi_variable = factor_graph.create_multi_variable(2)
    multi_variable[0] = 0
    multi_variable[1] = var_log_potentials[i]
    multi_variables.append(multi_variable)

# generate sequence log potentials
initials = np.zeros(2)
finals = np.zeros(2)
transitions = np.zeros((length - 1, 2, 2))
transitions[:, 1, 1] = rng.randn(length - 1)
edge_log_potentials = np.concatenate([initials,
                                      transitions.ravel(),
                                      finals])

# Create a sequential factor.
factors = []

variables = []
num_states = []
for i in range(length):
    for state in range(2):
        variables.append(multi_variables[i].get_state(state))
    num_states.append(2)

factor = fg.PFactorSequence()
factor_graph.declare_factor(factor, variables)

factor.initialize(num_states)
factor.set_additional_log_potentials(edge_log_potentials)

# Create a budget factor.
variables = []
for i in range(length):
    if counts_for_budget[i]:
        variables.append(multi_variables[i].get_state(1))

factor_graph.create_factor_budget(variables, budget)

# Run AD3.
_, posteriors, _, _ = factor_graph.solve()

# Print solution.
best_states = np.array(posteriors).reshape(-1, 2).argmax(axis=1)

print("Solution using SEQUENCE + BUDGET factors:", best_states)

# 2) Build a factor graph using a COMPRESSION_BUDGET factor.
compression_factor_graph = fg.PFactorGraph()

variable_log_potentials = list(var_log_potentials)

additional_log_potentials = []
index = 0
for i in range(length + 1):
    if i == 0:
        num_previous_states = 1
    else:
        num_previous_states = 2
    if i == length:
        num_current_states = 1
    else:
        num_current_states = 2
    for k in range(num_previous_states):
        for l in range(num_current_states):
            value = edge_log_potentials[index]
            index += 1
            if (k == num_previous_states - 1 and
                    l == num_current_states - 1 and
                    i - 1 in bigram_positions):
                variable_log_potentials.append(value)
            else:
                additional_log_potentials.append(value)

binary_variables = []
factors = []
for potential in variable_log_potentials:
    binary_variable = compression_factor_graph.create_binary_variable()
    binary_variable.set_log_potential(potential)
    binary_variables.append(binary_variable)

factor = fg.PFactorCompressionBudget()

variables = binary_variables
compression_factor_graph.declare_factor(factor, variables)

factor.initialize(length, budget, counts_for_budget, bigram_positions)
factor.set_additional_log_potentials(additional_log_potentials)
factors.append(factor)

# Run AD3.
print("Bigrams at", bigram_positions)

_, posteriors, _, _ = compression_factor_graph.solve()

# Print solution.
best_states = np.array(posteriors[:length]) > 0.5

print("Solution using COMPRESSION_BUDGET factor:", best_states.astype(np.int))
