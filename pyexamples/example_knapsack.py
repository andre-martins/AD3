import numpy as np
import pdb
import ad3.factor_graph as fg
import time

def test_random_instance(n):
  costs = np.random.rand(n)
  budget = np.sum(costs) * np.random.rand()
  scores = np.random.randn(n)
  
  tic = time.clock()
  x_gold = solve_lp_knapsack_lpsolve(scores, costs, budget)
  toc = time.clock()
  print 'lpsolve:', toc - tic
  
  tic = time.clock()
  x = solve_lp_knapsack_ad3(scores, costs, budget)
  toc = time.clock()
  print 'ad3:', toc - tic

  res = x - x_gold
  #print x
  #print x_gold
  if res.dot(res) > 1e-6:
    pdb.set_trace()

def solve_lp_knapsack_ad3(scores, costs, budget):
  factor_graph = fg.PFactorGraph()
  binary_variables = []
  for i in xrange(len(scores)):
      binary_variable = factor_graph.create_binary_variable()
      binary_variable.set_log_potential(scores[i])
      binary_variables.append(binary_variable) 

  negated = [False] * len(binary_variables)
  factor_graph.create_factor_knapsack(binary_variables, negated, costs, budget)

  #pdb.set_trace()
  # Run AD3.        
  factor_graph.set_verbosity(1)
  factor_graph.set_eta_ad3(.1)
  factor_graph.adapt_eta_ad3(True)
  factor_graph.set_max_iterations_ad3(1000)
  value, posteriors, additional_posteriors, status = factor_graph.solve_lp_map_ad3()

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

def solve_lp_knapsack_lpsolve(scores, costs, budget):
  import lpsolve55 as lps
  
  relax = True
  n = len(scores)
  
  lp = lps.lpsolve('make_lp', 0, n)        
  # Set verbosity level. 3 = only warnings and errors.
  lps.lpsolve('set_verbose', lp, 3)        
  lps.lpsolve('set_obj_fn', lp, -scores)
  
  lps.lpsolve('add_constraint', lp, costs, lps.LE, budget)
  
  lps.lpsolve('set_lowbo', lp, np.zeros(n))
  lps.lpsolve('set_upbo', lp, np.ones(n))

  if not relax:
      lps.lpsolve('set_int', lp, [True] * n)
  else:
      lps.lpsolve('set_int', lp, [False] * n)
               
  # Solve the ILP, and call the debugger if something went wrong.
  ret = lps.lpsolve('solve', lp)
  assert ret == 0, pdb.set_trace()

  # Retrieve solution and return
  [x, _] = lps.lpsolve('get_variables', lp)
  x = np.array(x)
    
  return x        


if __name__ == "__main__":
  n_tests = 100
  n = 100
  for i in xrange(n_tests):
    test_random_instance(n)

