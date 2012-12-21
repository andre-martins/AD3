import itertools
import numpy as np
import matplotlib.pyplot as plt
import pdb

import ad3

length = 30
budget = 10

# 1) Build a factor graph using a SEQUENCE and a BUDGET factor.
factor_graph = ad3.PFactorGraph()
multi_variables = []
for i in xrange(length):
    multi_variable = factor_graph.create_multi_variable(2)
    for state in xrange(2):
        value = np.random.normal()
        multi_variable.set_log_potential(state, value)
    multi_variables.append(multi_variable)

edge_log_potentials = []
for i in xrange(length+1):
    if i == 0:
        num_previous_states = 1
    else:
        num_previous_states = 2
    if i == length:
        num_current_states = 1
    else:
        num_current_states = 2
    for k in xrange(2):
        for l in xrange(2):
            if i != 0 and i != length:
                if k == 1 and l == 1:
                    value = np.random.normal()
                    edge_log_potentials.append(value)
                else:
                    edge_log_potentials.append(0.0)

# Create a sequential factor.
factors = []

variables = []
num_states = []
for i in xrange(length):
    for state in xrange(2):
        variables.append(multi_variables[i].get_state(state))
    num_states.append(2)

num_factors = 0
factor = ad3.PFactorSequence()
# Set True below to let the factor graph own the factor so that we
# don't need to delete it.
factor_graph.declare_factor(factor, variables, False)
factor.initialize(num_states)
factor.set_additional_log_potentials(edge_log_potentials)
factors.append(factor)
num_factors += 1
 
# Create a budget factor.
variables = []
for i in xrange(length):
    variables.append(multi_variables[i].get_state(1))
    
negated = [False] * length
factor_graph.create_factor_budget(variables, negated, budget)
num_factors += 1

# Run AD3.        
factor_graph.set_eta_ad3(.1)
factor_graph.adapt_eta_ad3(True)
factor_graph.set_max_iterations_ad3(1000)
value, posteriors, additional_posteriors = factor_graph.solve_lp_map_ad3()
  
# Print solution.
t = 0
best_states = []
for i in xrange(length):
    local_posteriors = posteriors[t:(t+2)]
    j = np.argmax(local_posteriors)
    best_states.append(j)
    t += num_states[i]
print best_states



# 2) Build a factor graph using a COMPRESSION_BUDGET factor.
factor_graph = ad3.PFactorGraph()

variable_log_potentials = []
additional_log_potentials = edge_log_potentials

for i in xrange(length):
    value = multi_variables[i].get_log_potential(1)
    variable_log_potentials.append(value)
            
binary_variables = []
factors = []
for i in xrange(len(variable_log_potentials)):
    binary_variable = factor_graph.create_binary_variable()
    binary_variable.set_log_potential(variable_log_potentials[i])
    binary_variables.append(binary_variable)

factor = ad3.PFactorCompressionBudget()
    
variables = binary_variables
factor_graph.declare_factor(factor, variables, True)
factor.initialize(length)
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
for i in xrange(length):
    if posteriors[i] > 0.5:
      j = 1
    else:
      j = 0
    best_states.append(j)
print best_states




