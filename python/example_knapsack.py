import numpy as np
import pdb
import ad3

def test_random_instance(n):
  costs = np.random.rand(n)
  budget = np.sum(costs) * np.random.rand()
  scores = np.random.randn(n)
  x_gurobi = solve_lp_knapsack_gurobi(scores, costs, budget)
  x = solve_lp_knapsack_ad3(scores, costs, budget)
  res = x - x_gurobi
  print x
  print x_gurobi
  if res.dot(res) > 1e-6:
    pdb.set_trace()

def solve_lp_knapsack_ad3(scores, costs, budget):
  factor_graph = ad3.PFactorGraph()
  binary_variables = []
  for i in xrange(len(scores)):
      binary_variable = factor_graph.create_binary_variable()
      binary_variable.set_log_potential(scores[i])
      binary_variables.append(binary_variable) 

  negated = [False] * len(binary_variables)
  factor_graph.create_factor_knapsack(binary_variables, negated, costs, budget)

  #pdb.set_trace()
  # Run AD3.        
  factor_graph.set_eta_ad3(.1)
  factor_graph.adapt_eta_ad3(True)
  factor_graph.set_max_iterations_ad3(1000)
  value, posteriors, additional_posteriors = factor_graph.solve_lp_map_ad3()

  return posteriors  


def solve_lp_knapsack_gurobi(scores, costs, budget):
  from gurobipy import *
  
  n = len(scores)
  
  # Create a new model.
  m = Model("lp_knapsack")
  
  # Create variables.
  for i in xrange(n):
    m.addVar(lb=0.0, ub=1.0)
  m.update()
  vars = m.getVars()
  
  # Set objective.
  obj = LinExpr()
  for i in xrange(n):
    obj += scores[i]*vars[i]  
  m.setObjective(obj, GRB.MAXIMIZE)
  
  # Add constraint.
  expr = LinExpr()
  for i in xrange(n):
    expr += costs[i]*vars[i]      
  m.addConstr(expr, GRB.LESS_EQUAL, budget)
  #pdb.set_trace()
  
  # Optimize.
  m.optimize()
  assert m.status == GRB.OPTIMAL
  x = np.zeros(n)
  for i in xrange(n):
    x[i] = vars[i].x
    
  return x        



if __name__ == "__main__":
  n = 10
  test_random_instance(n)

