import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pdb
import ad3.factor_graph as fg
from gurobipy import *

def generate_potts_grid(grid_size, num_states, edge_coupling, toroidal=True):
    n = grid_size
    node_potentials = [2.0*np.random.rand(num_states) - 1.0 \
                       for i, j in itertools.product(xrange(n), xrange(n))]

    node_indices = -np.ones((n,n), dtype=int)
    t = 0
    for i in xrange(n):
        for j in xrange(n):
            node_indices[i,j] = t
            t += 1

    edges = []
    edge_potentials = []
    for i in xrange(n):
        if i > 0 or not toroidal:
            prev_i = i-1
        else:
            prev_i = n-1
        for j in xrange(n):
            if j > 0 or not toroidal:
                prev_j = j-1
            else:
                prev_j = n-1

            t = len(edges)
            if prev_i >= 0:
                edges.append((node_indices[prev_i,j], node_indices[i,j]))
                edge_potentials.append(np.zeros((num_states, num_states)))
                for k in xrange(num_states):
                    edge_potentials[t][k,k] = \
                        edge_coupling*(2.0*np.random.rand(1) - 1.0)

            t = len(edges)
            if prev_j >= 0:
                edges.append((node_indices[i,prev_j], node_indices[i,j]))
                edge_potentials.append(np.zeros((num_states, num_states)))
                for k in xrange(num_states):
                    edge_potentials[t][k,k] = \
                        edge_coupling*(2.0*np.random.rand(1) - 1.0)

    return node_indices, edges, node_potentials, edge_potentials


def build_node_to_edges_map(num_nodes, edges):
    num_edges = len(edges)
    node_to_edges = [([], []) for i in xrange(num_nodes)]
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        node_to_edges[n1].append((j, 0))
        node_to_edges[n2].append((j, 1))

    return node_to_edges


def solve_max_marginals(scores, additional_scores):
    num_states1 = len(scores[0])
    num_states2 = len(scores[1])
    p = additional_scores.copy()
    for k in xrange(num_states1):
        for l in xrange(num_states2):
            p[k,l] += scores[k] + scores[l]

    max_marginals = [np.zeros(k), np.zeros(l)]

    # Compute max marginals for the first variable.
    for k in xrange(num_states1):
        max_marginals[0][k] = np.max(p[k,:])

    # Compute max marginals for the second variable.  
    for l in xrange(num_states2):
        max_marginals[1][l] = np.max(p[:,l])

    return max_marginals


def solve_map(scores, additional_scores):
    num_states1 = len(scores[0])
    num_states2 = len(scores[1])
    p = additional_scores.copy()
    for k in xrange(num_states1):
        for l in xrange(num_states2):
            p[k,l] += scores[k] + scores[l]
    k, l = np.unravel_index(p.argmax(), p.shape)
    value = p[k,l]
    posteriors = [np.zeros(k), np.zeros(l)]
    posteriors[0][k] = 1.0
    posteriors[1][l] = 1.0
    additional_posteriors[k,l] = 1.0

    return posteriors, additional_posteriors, value


def run_gurobi(edges, node_potentials, edge_potentials, relax=True):
    num_nodes = len(node_potentials)
    num_edges = len(edge_potentials)

    # Create a new model.
    m = Model("potts_grid")
  
    # Create variables.
    t = 0;
    node_variable_indices = []
    edge_variable_indices = []
    for i in xrange(num_nodes):
        node_variable_indices.append([])
        num_states = len(node_potentials[i])
        for k in xrange(num_states):
            if relax:
                m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
            else:
                m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
            node_variable_indices[i].append(t)
            t += 1
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        num_states1, num_states2 = np.shape(edge_potentials[j])
        edge_variable_indices.append(np.zeros((num_states1,
                                               num_states2),
                                              dtype=int))
        for k in xrange(num_states1):
            for l in xrange(num_states2):
                if relax:
                    m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
                else:
                    m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                edge_variable_indices[j][k,l] = t
                t += 1
    num_variables = t
    m.update()
    vars = m.getVars()
  
    # Set objective.
    obj = LinExpr()
    for i in xrange(num_nodes):
        num_states = len(node_potentials[i])
        for k in xrange(num_states):
            t = node_variable_indices[i][k]
            obj += node_potentials[i][k]*vars[t]
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        num_states1, num_states2 = np.shape(edge_potentials[j])
        for k in xrange(num_states1):
            for l in xrange(num_states2):
                t = edge_variable_indices[j][k,l]
                obj += edge_potentials[j][k,l]*vars[t]
    m.setObjective(obj, GRB.MAXIMIZE)
    
    # Add constraints.
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        num_states1, num_states2 = np.shape(edge_potentials[j])
        expr = LinExpr()
        for k in xrange(num_states1):
            for l in xrange(num_states2):
                t = edge_variable_indices[j][k,l]
                expr += vars[t]
        m.addConstr(expr, GRB.EQUAL, 1.0)
        for k in xrange(num_states1):
            expr = LinExpr()
            t = node_variable_indices[n1][k]
            expr -= vars[t]
            for l in xrange(num_states2):
                t = edge_variable_indices[j][k,l]
                expr += vars[t]
            m.addConstr(expr, GRB.EQUAL, 0.0)
        for l in xrange(num_states2):
            expr = LinExpr()
            t = node_variable_indices[n2][l]
            expr -= vars[t]
            for k in xrange(num_states1):
                t = edge_variable_indices[j][k,l]
                expr += vars[t]
            m.addConstr(expr, GRB.EQUAL, 0.0)

    # Optimize.
    m.optimize()
    assert m.status == GRB.OPTIMAL
    value = 0.0
    for i in xrange(num_nodes):
        num_states = len(node_potentials[i])
        for k in xrange(num_states):
            t = node_variable_indices[i][k]
            value += node_potentials[i][k]*vars[t].x
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        num_states1, num_states2 = np.shape(edge_potentials[j])
        for k in xrange(num_states1):
            for l in xrange(num_states2):
                t = edge_variable_indices[j][k,l]
                value += edge_potentials[j][k,l]*vars[t].x

    return value


def run_mplp(edges, node_potentials, edge_potentials, num_iterations=1000):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    posteriors = .5 * np.ones(num_nodes)
    edge_posteriors = np.zeros(num_edges)

    gammas = []
    deltas = []    
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        num_states1, num_states2 = np.shape(edge_potentials[j])
        gammas.append([])
        gammas[j].append(np.zeros(num_states1))
        gammas[j].append(np.zeros(num_states2))
        deltas.append([])
        deltas[j].append(np.zeros(num_states1))
        deltas[j].append(np.zeros(num_states2))

    node_to_edges  = build_node_to_edges_map(num_nodes, edges)

    dual_obj_seq = np.zeros(num_iterations)
    primal_obj_seq = np.zeros(num_iterations)
    for t in xrange(num_iterations):
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(node_to_edges[n1])
            d2 = len(node_to_edges[n2])

            # Update deltas.
            num_states = len(node_potentials[n1])
            for k in xrange(num_states):
                gamma_tot = sum([gammas[e,m][k] for e, m in node_to_edges[n1]])
                for e, m in node_to_edges[n1]:
                    deltas[e,m][k] = node_potentials[n1] + gamma_tot - gammas[e,m][k]
            num_states = len(node_potentials[n2])
            for k in xrange(num_states):
                gamma_tot = sum([gammas[e,m][k] for e, m in node_to_edges[n2]])
                for e, m in node_to_edges[n2]:
                    deltas[e,m][k] = node_potentials[n2] + gamma_tot - gammas[e,m][k]

            # Compute max-marginals and update gammas.
            scores = [deltas[j,0], deltas[j,1]]
            additional_scores = edge_potentials[j]
            max_marginals = solve_max_marginals(scores, additional_scores)

            factor_degree = 2
            gammas[j,0] = max_marginals[0] / float(factor_degree) - deltas[j,0]
            gammas[j,1] = max_marginals[1] / float(factor_degree) - deltas[j,1]

        # Compute dual objective.
        # First, get the contribution of the node variables to the dual objective.
        dual_obj = 0.0;        
        for i in xrange(num_nodes):
            num_states = len(node_potentials[i])
            vals = np.zeros(num_states)
            for k in xrange(num_states):
                gamma_tot = sum([gammas[e,m][k] for e, m in node_to_edges[i]])
                vals[k] = gamma_tot + node_potentials[i][k]
            k = np.argmax(vals)
            posteriors[i] = np.zeros(num_states)
            posteriors[i][k] = 1.0
            dual_obj += vals[k]
            
        # Now, get the contribution of the factors to the dual objective.
        for j in xrange(num_edges):
            scores = [-gammas[j,0], -gammas[j,1]]
            additional_scores = edge_potentials[j]
            local_posteriors, local_additional_posteriors, value = \
                solve_map(scores, additional_scores)
            dual_obj += value

        # Compute primal objective.
        primal_obj = 0.0;
        p_int = []
        for i in xrange(num_nodes):
            k = np.argmax(posteriors[i])
            p_int.append(k)
            primal_obj += node_potentials[i][k]
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            k = p_int[n1]
            l = p_int[n2]
            primal_obj += edge_potentials[j][k,l]

        if t % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal rel obj:', primal_rel_obj, 'Primal obj:', primal_obj

        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    return dual_obj_seq, primal_obj_seq


def run_ad3(edges, node_potentials, edge_potentials, num_iterations=1000, eta=1.0):
    factor_graph = fg.PFactorGraph()
    multi_variables = []
    num_nodes = len(node_potentials)
    num_edges = len(edges)

    for i in xrange(num_nodes):
        num_states = len(node_potentials[i])
        multi_variable = factor_graph.create_multi_variable(num_states)
        for state in xrange(num_states):
            multi_variable.set_log_potential(state, node_potentials[i][state])
        multi_variables.append(multi_variable)

    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        edge_variables = [multi_variables[n1], multi_variables[n2]]
        # pdb.set_trace()
        factor_graph.create_factor_dense(edge_variables,
                                         edge_potentials[j].ravel().tolist())

    factor_graph.store_primal_dual_sequences(True)
    factor_graph.set_eta_ad3(eta)
    factor_graph.adapt_eta_ad3(False)
#    factor_graph.adapt_eta_ad3(True)
    factor_graph.set_max_iterations_ad3(num_iterations)
    value, marginals, edge_marginals, solver_status = \
        factor_graph.solve_lp_map_ad3()
    primal_obj_seq, dual_obj_seq = factor_graph.get_primal_dual_sequences()

    return dual_obj_seq, primal_obj_seq


def find_stepwise_best(dual_obj_seq, primal_obj_seq):
    best_dual_obj_seq = np.zeros(len(dual_obj_seq))
    best_dual_obj = np.inf
    for i in xrange(len(dual_obj_seq)):
        if dual_obj_seq[i] < best_dual_obj:
            best_dual_obj = dual_obj_seq[i]
        best_dual_obj_seq[i] = best_dual_obj

    best_primal_obj_seq = np.zeros(len(primal_obj_seq))
    best_primal_obj = -np.inf
    for i in xrange(len(primal_obj_seq)):
        if primal_obj_seq[i] > best_primal_obj:
            best_primal_obj = primal_obj_seq[i]
        best_primal_obj_seq[i] = best_primal_obj

    return best_dual_obj_seq, best_primal_obj_seq


def trim_primal_dual_sequences(dual_obj_seq, primal_obj_seq, dual_value, primal_value, err_thres=1e-6):
    rel_err_dual = [abs(v-dual_value)/dual_value for v in dual_obj_seq]
    rel_err_primal = [abs(v-primal_value)/primal_value for v in primal_obj_seq]
    ind = [i for i in xrange(len(rel_err_dual)) if rel_err_dual[i] < err_thres]
    if len(ind) > 0:
        dual_obj_seq = dual_obj_seq[:(ind[0]+1)]
    ind = [i for i in xrange(len(rel_err_primal)) if rel_err_primal[i] < err_thres]
    if len(ind) > 0:
        primal_obj_seq = primal_obj_seq[:(ind[0]+1)]
    return dual_obj_seq, primal_obj_seq
        
    
if __name__ == "__main__": 
    if len(sys.argv) == 1:
        generate_grid = True
        grid_size = 20
        num_states = 8
        edge_coupling = 10.0
    else:
        generate_grid = bool(int(sys.argv[1]))
        grid_size = int(sys.argv[2])
        num_states = int(sys.argv[3])
        edge_coupling = float(sys.argv[4])

    filename = 'potts_gridsize-%d_coupling-%f.uai' % (grid_size, edge_coupling)
    if generate_grid:
        node_indices, edges, node_potentials, edge_potentials = \
            generate_potts_grid(grid_size, num_states, edge_coupling, toroidal=True)
        #save_potts(filename, edges, node_potentials, edge_potentials)
    else:
        pass
        #edges, node_potentials, edge_potentials = load_potts(filename)
    
    num_iterations = 1000
    
    use_mplp = True
    use_np = False
    use_accdd = False
    use_sdd = False
    use_psdd = False
    use_ad3 = True
    use_gurobi = True
    
    if use_gurobi:
        dual_value = run_gurobi(edges, node_potentials, edge_potentials, relax=True)
        print 'Optimal dual:', dual_value
        primal_value = np.inf #run_gurobi(edges, node_potentials, edge_potentials, relax=False)
        print 'Optimal primal:', primal_value
    else:
        dual_value = -np.inf
        primal_value = np.inf
    
    if use_mplp:
        print 'Running MPLP...'
        dual_obj_seq, primal_obj_seq = \
            run_mplp(edges, node_potentials, edge_potentials, num_iterations)
        dual_obj_seq_mplp, primal_obj_seq_mplp = find_stepwise_best(dual_obj_seq, primal_obj_seq)
        print 'Best primal:', primal_obj_seq_mplp[-1]
        dual_obj_seq_mplp, primal_obj_seq_mplp = \
            trim_primal_dual_sequences(dual_obj_seq_mplp, primal_obj_seq_mplp, dual_value, primal_value)

    
    if use_np:
        dual_obj_seq_np = [np.inf]
        for temperature in [0.001]: #[0.01, 0.1]:
            print 'Running NP with T =', temperature, '.'
            dual_obj_seq, primal_obj_seq = \
                run_norm_product(edges, node_potentials, edge_potentials, num_iterations, temperature)
            dual_obj_seq, primal_obj_seq = find_stepwise_best(dual_obj_seq, primal_obj_seq)
            if dual_obj_seq[-1] < dual_obj_seq_np[-1]:
                dual_obj_seq_np = dual_obj_seq
                primal_obj_seq_np = primal_obj_seq
                temperature_np = temperature
        print 'Best temperature np:', temperature_np
        dual_obj_seq_np, primal_obj_seq_np = \
            trim_primal_dual_sequences(dual_obj_seq_np, primal_obj_seq_np, dual_value, primal_value)
    
    if use_accdd:
        dual_obj_seq_accdd = [np.inf]
        for epsilon in [10.0]:
        #for epsilon in [1.0, 10.0, 100.0]:
            print 'Running ACCDD with epsilon =', epsilon, '.'
            dual_obj_seq, primal_obj_seq = \
                run_accdd(edges, node_potentials, edge_potentials, num_iterations, epsilon)
            dual_obj_seq, primal_obj_seq = find_stepwise_best(dual_obj_seq, primal_obj_seq)
            if dual_obj_seq[-1] < dual_obj_seq_accdd[-1]:
                dual_obj_seq_accdd = dual_obj_seq
                primal_obj_seq_accdd = primal_obj_seq
                epsilon_accdd = epsilon
        print 'Best epsilon ACCDD:', epsilon_accdd
        dual_obj_seq_accdd, primal_obj_seq_accdd = \
            trim_primal_dual_sequences(dual_obj_seq_accdd, primal_obj_seq_accdd, dual_value, primal_value)
    
    if use_sdd:
        dual_obj_seq_sdd = [np.inf]
    #    for eta in [0.001, 0.01, 0.1, 1, 10]:
        for eta, temperature in itertools.product([0.01, 0.1, 1], [0.01, 0.1]):
            print 'Running SDD with eta =', eta, 'and T =', temperature, '.'
            dual_obj_seq, primal_obj_seq = \
                run_sdd(edges, node_potentials, edge_potentials, num_iterations, eta, temperature)
            dual_obj_seq, primal_obj_seq = find_stepwise_best(dual_obj_seq, primal_obj_seq)
            if dual_obj_seq[-1] < dual_obj_seq_sdd[-1]:
                dual_obj_seq_sdd = dual_obj_seq
                primal_obj_seq_sdd = primal_obj_seq
                eta_sdd = eta
                temperature_sdd = temperature
        print 'Best eta SDD:', eta_sdd
        print 'Best temperature SDD:', temperature_sdd
        dual_obj_seq_sdd, primal_obj_seq_sdd = \
            trim_primal_dual_sequences(dual_obj_seq_sdd, primal_obj_seq_sdd, dual_value, primal_value)
    
    if use_psdd:
        dual_obj_seq_psdd = [np.inf]
        for eta in [0.001, 0.01, 0.1, 1, 10]:
            print 'Running PSDD with eta =', eta, '.'
            dual_obj_seq, primal_obj_seq = \
                run_psdd(edges, node_potentials, edge_potentials, num_iterations, eta=eta)
            dual_obj_seq, primal_obj_seq = find_stepwise_best(dual_obj_seq, primal_obj_seq)
            if dual_obj_seq[-1] < dual_obj_seq_psdd[-1]:
                dual_obj_seq_psdd = dual_obj_seq
                primal_obj_seq_psdd = primal_obj_seq
                eta_psdd = eta
        print 'Best eta PSDD:', eta_psdd
        dual_obj_seq_psdd, primal_obj_seq_psdd = \
            trim_primal_dual_sequences(dual_obj_seq_psdd, primal_obj_seq_psdd, dual_value, primal_value)
    
    if use_ad3:
        dual_obj_seq_ad3  = [np.inf]
        #for eta in [0.001, 0.01, 0.1, 1, 5.0]:
        for eta in [0.1]: #[0.1]:
            print 'Running AD3 with eta =', eta, '.'
            dual_obj_seq, primal_obj_seq = \
                run_ad3(edges, node_potentials, edge_potentials, num_iterations, eta=eta)
            dual_obj_seq, primal_obj_seq = find_stepwise_best(dual_obj_seq, primal_obj_seq)
            print 'Best primal:', primal_obj_seq[-1]
            if dual_obj_seq[-1] < dual_obj_seq_ad3[-1]:
                dual_obj_seq_ad3 = dual_obj_seq
                primal_obj_seq_ad3 = primal_obj_seq
                eta_ad3  = eta
        print 'Best eta AD3:', eta_ad3
        dual_obj_seq_ad3, primal_obj_seq_ad3 = \
            trim_primal_dual_sequences(dual_obj_seq_ad3, primal_obj_seq_ad3, dual_value, primal_value)

    fig = plt.figure()    
    if use_mplp:
        plt.plot(np.arange(len(dual_obj_seq_mplp)), dual_obj_seq_mplp, 'c-', label='MPLP dual')
        plt.hold(True)
    if use_np:
        plt.plot(np.arange(len(dual_obj_seq_np)), dual_obj_seq_np, 'y-', label='NP dual')
        plt.hold(True)
    if use_psdd:
        plt.plot(np.arange(len(dual_obj_seq_psdd)), dual_obj_seq_psdd, 'r-', label='PSDD dual')
        plt.hold(True)
    if use_sdd:
        plt.plot(np.arange(len(dual_obj_seq_sdd)), dual_obj_seq_sdd, 'b-', label='SDD dual')
        plt.hold(True)
    if use_accdd:
        plt.plot(np.arange(len(dual_obj_seq_accdd)), dual_obj_seq_accdd, 'm-', label='ACCDD dual')
        plt.hold(True)
    if use_ad3:
        plt.plot(np.arange(len(dual_obj_seq_ad3)), dual_obj_seq_ad3, 'g-', label='AD3 dual')
        plt.hold(True)
#    if use_gurobi:
#        plt.plot(np.arange(num_iterations), np.tile(dual_value, num_iterations), 'k-', label='Optimal dual')
#        plt.hold(True)
    if use_mplp:
        plt.plot(np.arange(len(primal_obj_seq_mplp)), primal_obj_seq_mplp, 'c:', label='MPLP primal')
        plt.hold(True)
    if use_np:
        plt.plot(np.arange(len(primal_obj_seq_np)), primal_obj_seq_np, 'y:', label='NP primal')
        plt.hold(True)
    if use_psdd:
        plt.plot(np.arange(len(primal_obj_seq_psdd)), primal_obj_seq_psdd, 'r:', label='PSDD primal')
        plt.hold(True)
    if use_sdd:
        plt.plot(np.arange(len(primal_obj_seq_sdd)), primal_obj_seq_sdd, 'b:', label='SDD primal')
        plt.hold(True)
    if use_accdd:
        plt.plot(np.arange(len(primal_obj_seq_accdd)), primal_obj_seq_accdd, 'm:', label='ACCDD primal')
        plt.hold(True)
    if use_ad3:
        plt.plot(np.arange(len(primal_obj_seq_ad3)), primal_obj_seq_ad3, 'g:', label='AD3 primal')
        plt.hold(True)
#    if use_gurobi:
#        plt.plot(np.arange(num_iterations), np.tile(primal_value, num_iterations), 'k:', label='Optimal primal')
#        plt.hold(True)
    
    
    plt.legend(loc=4) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Objective value')
    plt.xlabel('Number of iterations')
    
    ymin = np.max(primal_obj_seq_ad3) - 10.0
    ymax = np.min(dual_obj_seq_ad3) + 10.0
    
    #plt.ylim((ymin, ymax))
    plt.suptitle('Edge coupling: ' + str(edge_coupling))
    
    #pdb.set_trace()
    
    #plt.xticks(paramValues)
    #plt.grid(True)
    
    filename = 'potts_gridsize-%d_states-%d_coupling-%f.png' % \
        (grid_size, num_states, edge_coupling)
    fig.savefig(filename)

    plt.show()
    pdb.set_trace()
