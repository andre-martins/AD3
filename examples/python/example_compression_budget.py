from __future__ import print_function
import numpy as np

import ad3.factor_graph as fg
from ad3 import solve

length = 30
budget = 10

# Decide bigram_positions.
bigram_positions = []
for i in range(-1, length):
    value = np.random.uniform()
    if value < 0.4:
        bigram_positions.append(i)

# Decide whether each position counts for budget.
counts_for_budget = []
for i in range(length):
    value = np.random.uniform()
    if value < 0.1:
        counts_for_budget.append(False)
    else:
        counts_for_budget.append(True)

# 1) Build a factor graph using a SEQUENCE and a BUDGET factor.
factor_graph = fg.PFactorGraph()
multi_variables = []
for i in range(length):
    multi_variable = factor_graph.create_multi_variable(2)
    multi_variable.set_log_potential(0, 0.0)
    value = np.random.normal()
    multi_variable.set_log_potential(1, value)
    multi_variables.append(multi_variable)

edge_log_potentials = []
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
            if k == 1 and l == 1:
                value = np.random.normal()
                edge_log_potentials.append(value)
            else:
                edge_log_potentials.append(0.0)

# Create a sequential factor.
factors = []

variables = []
num_states = []
for i in range(length):
    for state in range(2):
        variables.append(multi_variables[i].get_state(state))
    num_states.append(2)

num_factors = 0
factor = fg.PFactorSequence()
# Set True below to let the factor graph own the factor so that we
# don't need to delete it.
factor_graph.declare_factor(factor, variables, False)

factor.initialize(num_states)
factor.set_additional_log_potentials(edge_log_potentials)
factors.append(factor)
num_factors += 1

# Create a budget factor.
variables = []
for i in range(length):
    if counts_for_budget[i]:
        variables.append(multi_variables[i].get_state(1))

negated = [False] * len(variables)
factor_graph.create_factor_budget(variables, negated, budget)
num_factors += 1

# Run AD3.
_, posteriors, _, _ = solve(factor_graph)

# Print solution.
t = 0
best_states = []
for i in range(length):
    local_posteriors = posteriors[t:(t + 2)]
    j = np.argmax(local_posteriors)
    best_states.append(j)
    t += num_states[i]

print("Solution using SEQUENCE + BUDGET factors:", best_states)

# 2) Build a factor graph using a COMPRESSION_BUDGET factor.
compression_factor_graph = fg.PFactorGraph()

variable_log_potentials = []
for i in range(length):
    value = multi_variables[i].get_log_potential(1)
    variable_log_potentials.append(value)

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
compression_factor_graph.declare_factor(factor, variables, True)

factor.initialize(length, budget, counts_for_budget, bigram_positions)
factor.set_additional_log_potentials(additional_log_potentials)
factors.append(factor)

# Run AD3.
compression_factor_graph.set_eta_ad3(.1)
compression_factor_graph.adapt_eta_ad3(True)
compression_factor_graph.set_max_iterations_ad3(1000)

print("Bigrams at", bigram_positions)

_, posteriors, _, _ = solve(compression_factor_graph)

# Print solution.
t = 0
best_states = []
for i in range(length):
    if posteriors[i] > 0.5:
        j = 1
    else:
        j = 0
    best_states.append(j)

print("Solution using COMPRESSION_BUDGET factor:", best_states)
