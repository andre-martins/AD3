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


def save_potts(filename, edges, node_potentials, edge_potentials):
    f = open(filename, 'w')
    f.write('MARKOV\n')
    num_variables = len(node_potentials)
    num_factors = len(edge_potentials)
    f.write(str(num_variables) + '\n')
    f.write(' '.join([str(len(node_potentials[i])) \
                          for i in xrange(num_variables)]) + '\n')
    f.write(str(num_variables+num_factors) + '\n')
    for i in xrange(num_variables):
        f.write('1\t%d\n' % i)
    for j in xrange(num_factors):
        f.write('2\t%d\t%d\n' % (edges[j][0], edges[j][1]))

    for i in xrange(num_variables):
        f.write('\n')
        f.write('%d\n' % len(node_potentials[i]))
        f.write(' '.join([str(np.exp(val)) for val in node_potentials[i]]) + '\n')

    for j in xrange(num_factors):
        f.write('\n')
        num_states1, num_states2 = np.shape(edge_potentials[j])
        f.write('%d\n' % (num_states1*num_states2))
        for k in xrange(num_states1):
            f.write(' '.join([str(np.exp(val)) for val in edge_potentials[j][k,:]]) + '\n')
    f.close()


def load_potts(filename):
    f = open(filename)
    line = f.readline().rstrip('\n')
    assert line == 'MARKOV', pdb.set_trace()

    line = f.readline().rstrip('\n')
    num_variables = int(line)
    
    line = f.readline().rstrip('\n')
    num_variable_states = [int(field) for field in line.split(' ')]
    assert len(num_variable_states) == num_variables, pdb.set_trace()
    node_potentials = [np.ones(num_variable_states[i]) for i in xrange(num_variables)]

    line = f.readline().rstrip('\n')
    num_factors = int(line) - num_variables

    variable_indices = []
    edges = []
    edge_potentials = []
    for t in xrange(num_variables + num_factors):
        line = f.readline().rstrip('\n')
        fields = line.split('\t')
        degree = int(fields[0])
        assert len(fields[1:]) == degree, pdb.set_trace()
        if degree == 1:
            variable_indices.append(int(fields[1]))
        else:
            variable_indices.append(-1)
            assert degree == 2, pdb.set_trace()
            edges.append((int(fields[1]), int(fields[2])))
            num_states1 = num_variable_states[int(fields[1])]
            num_states2 = num_variable_states[int(fields[2])]
            edge_potentials.append(np.ones((num_states1, num_states2)))

    j = 0
    for t in xrange(num_variables + num_factors):
        line = f.readline().rstrip('\n')
        i = variable_indices[t]
        if i >= 0:
            line = f.readline().rstrip('\n')
            assert(int(line) == num_variable_states[i]), pdb.set_trace()
            line = f.readline().rstrip('\n')
            node_potentials[i][:] = \
                np.array([np.log(float(field)) for field in line.split(' ')])
        else:
            n1 = edges[j][0]
            n2 = edges[j][1]
            num_states1 = num_variable_states[n1]
            num_states2 = num_variable_states[n2]            
            line = f.readline().rstrip('\n')
            assert(int(line) == (num_states1*num_states2)), pdb.set_trace()
            for k in xrange(num_states1):
                line = f.readline().rstrip('\n')
                edge_potentials[j][k,:] = \
                    np.array([np.log(float(field)) for field in line.split(' ')])
            j += 1

    f.close()
    return edges, node_potentials, edge_potentials


def build_node_to_edges_map(num_nodes, edges):
    num_edges = len(edges)
    node_to_edges = [[] for i in xrange(num_nodes)]
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        node_to_edges[n1].append((j, 0))
        node_to_edges[n2].append((j, 1))

    return node_to_edges


def log_norm(v, p):
    # Computes u = log(||exp(v)||_p) = (1/p) log (\sum_i exp(vi*p)) = v_max + (1/p) log (\sum_i exp((vi-v_max)*p))
    m = np.max(v)
    if p == np.inf:
        return m
    else:
        return m + (1/p)*np.log(np.sum(np.exp((v-m)*p)))


def solve_max_marginals(scores, additional_scores):
    num_states1 = len(scores[0])
    num_states2 = len(scores[1])
    p = additional_scores.copy()
    for k in xrange(num_states1):
        for l in xrange(num_states2):
            p[k,l] += scores[0][k] + scores[1][l]

    max_marginals = [np.zeros(num_states1),
                     np.zeros(num_states2)]

    # Compute max marginals for the first variable.
    for k in xrange(num_states1):
        max_marginals[0][k] = np.max(p[k,:])

    # Compute max marginals for the second variable.  
    for l in xrange(num_states2):
        max_marginals[1][l] = np.max(p[:,l])

    return max_marginals


def solve_marginals(scores, additional_scores, temperature=1.0):
    num_states1 = len(scores[0])
    num_states2 = len(scores[1])
    p = additional_scores.copy()
    for k in xrange(num_states1):
        for l in xrange(num_states2):
            p[k,l] += scores[0][k] + scores[1][l]
    
    p /= temperature
    k, l = np.unravel_index(p.argmax(), p.shape)
    value = p[k,l]
    logZ = value + np.log(np.sum(np.exp(p - value)))

    marginals = [np.zeros(num_states1),
                     np.zeros(num_states2)]

    # Compute marginals for the first variable.
    for k in xrange(num_states1):
        marginals[0][k] = np.sum(np.exp(p[k,:] - logZ))

    # Compute marginals for the second variable.
    for l in xrange(num_states2):
        marginals[1][l] = np.sum(np.exp(p[:,l] - logZ))

    value = logZ * temperature
    return marginals, value


def solve_map(scores, additional_scores):
    num_states1 = len(scores[0])
    num_states2 = len(scores[1])
    p = additional_scores.copy()
    for k in xrange(num_states1):
        for l in xrange(num_states2):
            p[k,l] += scores[0][k] + scores[1][l]
    k, l = np.unravel_index(p.argmax(), p.shape)
    value = p[k,l]
    posteriors = [np.zeros(num_states1),
                  np.zeros(num_states2)]
    additional_posteriors = np.zeros((num_states1,
                                      num_states2))
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


def run_norm_product(edges, node_potentials, edge_potentials, num_iterations=1000, temperature=0.001):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    c_alphas = np.ones(num_edges)
    c_i = np.zeros(num_nodes)
    c_i_alphas = np.zeros((num_edges,2))
    posteriors = [[] for i in xrange(num_nodes)]
    edge_posteriors = [[] for j in xrange(num_edges)]

    m_messages = []
    n_messages = []    
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        num_states1, num_states2 = np.shape(edge_potentials[j])
        m_messages.append([])
        m_messages[j].append(np.zeros(num_states1))
        m_messages[j].append(np.zeros(num_states2))
        n_messages.append([])
        n_messages[j].append(np.zeros(num_states1))
        n_messages[j].append(np.zeros(num_states2))

    node_to_edges  = build_node_to_edges_map(num_nodes, edges)

    c_i_hat = c_i.copy()
    for i in xrange(num_nodes):
        c_i_hat[i] += sum([c_alphas[e] for e, m in node_to_edges[i]])

    dual_obj_seq = np.zeros(num_iterations)
    primal_obj_seq = np.zeros(num_iterations)
    for t in xrange(num_iterations):
        for i in xrange(num_nodes):
            # Update m-messages.
            num_states = len(node_potentials[i])
            for e, m in node_to_edges[i]:
                scores = [n_messages[e][0], n_messages[e][1]]
                additional_scores = edge_potentials[e]
                for k in xrange(num_states):
                    if m == 0:
                        v = scores[1] + additional_scores[k, :]
                    else:
                        v = scores[0] + additional_scores[:, k]
                    m_messages[e][m][k] = \
                        log_norm(v, 1.0/(temperature*c_alphas[e]))

            # Update n-messages.
            score = node_potentials[i].copy()
            for e, m in node_to_edges[i]:
                score += m_messages[e][m]
            for e, m in node_to_edges[i]:
                n_messages[e][m] = (score * (c_alphas[e]/c_i_hat[i])) - m_messages[e][m]
                value = np.mean(n_messages[e][m])
                n_messages[e][m] -= value

        # Compute beliefs.
        for i in xrange(num_nodes):
            beliefs = node_potentials[i].copy()
            for e, m in node_to_edges[i]:
                beliefs += m_messages[e][m]
            beliefs *= (1.0/(temperature*c_i_hat[i]))
            value = np.max(beliefs)
            value += np.log(np.sum(np.exp(beliefs - value)))
            beliefs = np.exp(beliefs - value)
            posteriors[i] = beliefs

        # Compute dual objective.
        dual_obj = 0.0
        dual_obj2 = 0.0
        for e in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            num_states1, num_states2 = np.shape(edge_potentials[j])
            v = edge_potentials[e].copy()
            for k in xrange(num_states1):
                for l in xrange(num_states2):
                    v[k,l] += n_messages[e][0][k] + n_messages[e][1][l]
            dual_obj += log_norm(v, 1.0/(temperature * c_alphas[e]))
            dual_obj2 += log_norm(v, np.inf)
        for i in xrange(num_nodes):
            v = node_potentials[i].copy()
            for e, m in node_to_edges[i]:
                v -= n_messages[e][m]
            if c_i[i] == 0:
                dual_obj += log_norm(v, np.inf)
            else:
                dual_obj += log_norm(v, 1.0/(temperature * c_i[i]))
            dual_obj2 += log_norm(v, np.inf)

        dual_obj = dual_obj2

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

        #pdb.set_trace()
        if (t+1) % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal obj:', primal_obj

        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    return dual_obj_seq, primal_obj_seq


def run_mplp(edges, node_potentials, edge_potentials, num_iterations=1000):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    posteriors = [[] for i in xrange(num_nodes)]
    edge_posteriors = [[] for j in xrange(num_edges)]

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
                gamma_tot = sum([gammas[e][m][k] for e, m in node_to_edges[n1]])
                for e, m in node_to_edges[n1]:
                    deltas[e][m][k] = node_potentials[n1][k] + gamma_tot - gammas[e][m][k]
            num_states = len(node_potentials[n2])
            for k in xrange(num_states):
                gamma_tot = sum([gammas[e][m][k] for e, m in node_to_edges[n2]])
                for e, m in node_to_edges[n2]:
                    deltas[e][m][k] = node_potentials[n2][k] + gamma_tot - gammas[e][m][k]

            # Compute max-marginals and update gammas.
            scores = [deltas[j][0], deltas[j][1]]
            additional_scores = edge_potentials[j]
            max_marginals = solve_max_marginals(scores, additional_scores)

            factor_degree = 2
            gammas[j][0] = max_marginals[0] / float(factor_degree) - deltas[j][0]
            gammas[j][1] = max_marginals[1] / float(factor_degree) - deltas[j][1]

        # Compute dual objective.
        # First, get the contribution of the node variables to the dual objective.
        dual_obj = 0.0;        
        for i in xrange(num_nodes):
            num_states = len(node_potentials[i])
            vals = np.zeros(num_states)
            for k in xrange(num_states):
                gamma_tot = sum([gammas[e][m][k] for e, m in node_to_edges[i]])
                vals[k] = gamma_tot + node_potentials[i][k]
            k = np.argmax(vals)
            posteriors[i] = np.zeros(num_states)
            posteriors[i][k] = 1.0
            dual_obj += vals[k]
            
        # Now, get the contribution of the factors to the dual objective.
        for j in xrange(num_edges):
            scores = [-gammas[j][0], -gammas[j][1]]
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

        if (t+1) % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal obj:', primal_obj

        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    return dual_obj_seq, primal_obj_seq


def run_accdd(edges, node_potentials, edge_potentials, num_iterations=1000, epsilon=1.0):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    posteriors = [[] for i in xrange(num_nodes)]
    edge_posteriors = [[] for j in xrange(num_edges)]

    p = [np.zeros(len(node_potentials[i])) for i in xrange(num_nodes)]
    q = []
    lambdas = []    
    zetas = []    
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        num_states1, num_states2 = np.shape(edge_potentials[j])
        q.append([])
        q[j].append(np.zeros(num_states1))
        q[j].append(np.zeros(num_states2))
        lambdas.append([])
        lambdas[j].append(np.zeros(num_states1))
        lambdas[j].append(np.zeros(num_states2))
        zetas.append([])
        zetas[j].append(np.zeros(num_states1))
        zetas[j].append(np.zeros(num_states2))

    sumlog_assig = num_edges*2.0*np.log(2.0)
    T = epsilon/(2.0*sumlog_assig);
    L = 2.0*sumlog_assig/epsilon;
    theta = 1.0

    node_to_edges  = build_node_to_edges_map(num_nodes, edges)

    dual_obj_prev = np.inf
    num_times_increment = 0;

    dual_smooth_obj_seq = np.zeros(num_iterations)
    dual_obj_seq = np.zeros(num_iterations)
    primal_obj_seq = np.zeros(num_iterations)
    for t in xrange(num_iterations):
        # Make updates and compute dual objective.
        dual_obj = 0.0;
        dual_smooth_obj = 0.0;
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(node_to_edges[n1])
            d2 = len(node_to_edges[n2])
            scores = [node_potentials[n1] / float(d1) + lambdas[j][0],
                      node_potentials[n2] / float(d2) + lambdas[j][1]]
            additional_scores = edge_potentials[j]
            posteriors, value = solve_marginals(scores, additional_scores, T)
            num_states1, num_states2 = np.shape(edge_potentials[j])
            value -= T*(np.log(num_states1) + np.log(num_states2))
            dual_smooth_obj += value

            q[j][0] = posteriors[0]
            q[j][1] = posteriors[1]

        # Check if dual improved so that num_times_increment 
        # can be incremented.
        #if dual_obj < dual_obj_prev:
        #    num_times_increment += 1;
        #dual_obj_prev = dual_obj;

        # Project (to update zetas).
        gammas = [np.zeros(len(node_potentials[i])) for i in xrange(num_nodes)]
        zetas_new = [[zetas[j][0].copy(), zetas[j][1].copy()] \
                         for j in xrange(num_edges)]
        for i in xrange(num_nodes):
            num_states = len(node_potentials[i])
            for k in xrange(num_states):
                gammas[i][k] = np.mean([theta*L*zetas[e][m][k] - q[e][m][k] \
                                            for e,m in node_to_edges[i]])
                p[i][k] = np.mean([q[e][m][k] for e,m in node_to_edges[i]])
            for e, m in node_to_edges[i]:
                zetas_new[e][m] = zetas[e][m] - (1/(theta*L)) * (q[e][m] + gammas[i])
        zetas = zetas_new

        # Update lambdas.
        for j in xrange(num_edges):
            lambdas[j][0] = (1-theta)*lambdas[j][0] + theta*zetas[j][0]
            lambdas[j][1] = (1-theta)*lambdas[j][1] + theta*zetas[j][1]

        theta = (np.sqrt(theta**4 + 4*(theta**2)) - theta**2)/2.0

        # Compute dual objective.
        dual_obj = 0.0;
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(node_to_edges[n1])
            d2 = len(node_to_edges[n2])
            scores = [node_potentials[n1] / float(d1) + lambdas[j][0],
                      node_potentials[n2] / float(d2) + lambdas[j][1]]
            additional_scores = edge_potentials[j]
            posteriors, additional_posteriors, value = solve_map(scores, additional_scores)
            dual_obj += value

            q[j][0] = posteriors[0]
            q[j][1] = posteriors[1]
            edge_posteriors[j] = additional_posteriors

        for i in xrange(num_nodes):
            num_states = len(node_potentials[i])
            for k in xrange(num_states):
                p[i][k] = np.mean([q[e][m][k] for e,m in node_to_edges[i]])

        # Compute relaxed primal objective.
        primal_rel_obj = 0.0;
        for i in xrange(num_nodes):
            primal_rel_obj += np.sum(p[i] * node_potentials[i])
        for j in xrange(num_edges):
            primal_rel_obj += np.sum(edge_posteriors[j] * edge_potentials[j])

        # Compute primal objective.
        primal_obj = 0.0;
        p_int = []
        for i in xrange(num_nodes):
            k = np.argmax(p[i])
            p_int.append(k)
            primal_obj += node_potentials[i][k]
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            k = p_int[n1]
            l = p_int[n2]
            primal_obj += edge_potentials[j][k,l]

        if (t+1) % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal rel obj:', primal_rel_obj, 'Primal obj:', primal_obj, 'Dual smooth obj:', dual_smooth_obj

        dual_smooth_obj_seq[t] = dual_smooth_obj
        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    return dual_obj_seq, primal_obj_seq


def run_psdd(edges, node_potentials, edge_potentials, num_iterations=1000, eta=0.1):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    posteriors = [[] for i in xrange(num_nodes)]
    edge_posteriors = [[] for j in xrange(num_edges)]

    p = [np.zeros(len(node_potentials[i])) for i in xrange(num_nodes)]
    q = []
    lambdas = []    
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        num_states1, num_states2 = np.shape(edge_potentials[j])
        q.append([])
        q[j].append(np.zeros(num_states1))
        q[j].append(np.zeros(num_states2))
        lambdas.append([])
        lambdas[j].append(np.zeros(num_states1))
        lambdas[j].append(np.zeros(num_states2))

    node_to_edges  = build_node_to_edges_map(num_nodes, edges)

    dual_obj_prev = np.inf
    num_times_increment = 0;

    dual_obj_seq = np.zeros(num_iterations)
    primal_obj_seq = np.zeros(num_iterations)
    for t in xrange(num_iterations):
        # Make updates and compute dual objective.
        dual_obj = 0.0;
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(node_to_edges[n1])
            d2 = len(node_to_edges[n2])
            scores = [node_potentials[n1] / float(d1) + lambdas[j][0],
                      node_potentials[n2] / float(d2) + lambdas[j][1]]
            additional_scores = edge_potentials[j]
            posteriors, additional_posteriors, value = solve_map(scores, additional_scores)
            dual_obj += value

            q[j][0] = posteriors[0]
            q[j][1] = posteriors[1]
            edge_posteriors[j] = additional_posteriors

        # Check if dual improved so that num_times_increment 
        # can be incremented.
        if dual_obj < dual_obj_prev:
            num_times_increment += 1;
        dual_obj_prev = dual_obj;

        for i in xrange(num_nodes):
            num_states = len(node_potentials[i])
            for k in xrange(num_states):
                p[i][k] = np.mean([q[e][m][k] for e,m in node_to_edges[i]])

        eta_t = eta / np.sqrt(num_times_increment);
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            lambdas[j][0] -= eta_t * (q[j][0] - p[n1])
            lambdas[j][1] -= eta_t * (q[j][1] - p[n2])

        # Compute relaxed primal objective.
        primal_rel_obj = 0.0;
        for i in xrange(num_nodes):
            primal_rel_obj += np.sum(p[i] * node_potentials[i])
        for j in xrange(num_edges):
            primal_rel_obj += np.sum(edge_posteriors[j] * edge_potentials[j])

        # Compute primal objective.
        primal_obj = 0.0;
        p_int = []
        for i in xrange(num_nodes):
            k = np.argmax(p[i])
            p_int.append(k)
            primal_obj += node_potentials[i][k]
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            k = p_int[n1]
            l = p_int[n2]
            primal_obj += edge_potentials[j][k,l]

        if (t+1) % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal rel obj:', primal_rel_obj, 'Primal obj:', primal_obj

        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    return dual_obj_seq, primal_obj_seq


def run_ad3(edges, node_potentials, edge_potentials, num_iterations=1000, eta=1.0,
            convert_to_binary=False):
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

    if convert_to_binary:
        factor_graph = factor_graph.convert_to_binary_factor_graph()
    factor_graph.store_primal_dual_sequences(True)
    factor_graph.set_eta_ad3(eta)
    factor_graph.adapt_eta_ad3(False)
#    factor_graph.adapt_eta_ad3(True)
    factor_graph.enable_caching_ad3(False)
    factor_graph.set_max_iterations_ad3(num_iterations)
    factor_graph.set_verbosity(0)
    value, marginals, edge_marginals, solver_status = \
        factor_graph.solve_lp_map_ad3()
    primal_obj_seq, dual_obj_seq = factor_graph.get_primal_dual_sequences()
    num_oracle_calls_seq = factor_graph.get_num_oracle_calls_sequence()
    num_oracle_calls_seq = [float(val)/num_edges for val in num_oracle_calls_seq]
    if convert_to_binary:
        for t in xrange(len(primal_obj_seq)):
            primal_obj_seq[t] = 0.0

    return dual_obj_seq, primal_obj_seq, num_oracle_calls_seq


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
        generate_grid = False #True
        grid_size = 20
        num_states = 8
        edge_coupling = 10.0
    else:
        generate_grid = bool(int(sys.argv[1]))
        grid_size = int(sys.argv[2])
        num_states = int(sys.argv[3])
        edge_coupling = float(sys.argv[4])

    filename = 'potts_gridsize-%d_states-%d_coupling-%f.uai' % \
        (grid_size, num_states, edge_coupling)
    if generate_grid:
        node_indices, edges, node_potentials, edge_potentials = \
            generate_potts_grid(grid_size, num_states, edge_coupling, toroidal=True)
        save_potts(filename, edges, node_potentials, edge_potentials)
    else:
        edges, node_potentials, edge_potentials = load_potts(filename)
    
    num_iterations = 1000
    
    use_mplp = True
    use_np = True
    use_accdd = True
    use_sdd = False
    use_psdd = True
    use_ad3 = True
    use_ad3_binary = True
    use_gurobi = True
    print_primal = False
    
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
        #for eta in [0.001, 0.01, 0.1, 1.0, 5.0]:
        for eta in [1.0]: #[0.1]:
            print 'Running AD3 with eta =', eta, '.'
            dual_obj_seq, primal_obj_seq, num_oracle_calls_seq = \
                run_ad3(edges, node_potentials, edge_potentials, num_iterations, eta=eta)
            dual_obj_seq, primal_obj_seq = find_stepwise_best(dual_obj_seq, primal_obj_seq)
            print 'Best dual:', dual_obj_seq[-1]
            print 'Best primal:', primal_obj_seq[-1]
            if dual_obj_seq[-1] < dual_obj_seq_ad3[-1]:
                dual_obj_seq_ad3 = dual_obj_seq
                primal_obj_seq_ad3 = primal_obj_seq
                num_oracle_calls_seq_ad3 = num_oracle_calls_seq
                eta_ad3  = eta
        print 'Best eta AD3:', eta_ad3
        dual_obj_seq_ad3, primal_obj_seq_ad3 = \
            trim_primal_dual_sequences(dual_obj_seq_ad3, primal_obj_seq_ad3, dual_value, primal_value)
        num_oracle_calls_seq_ad3 = num_oracle_calls_seq_ad3[:len(dual_obj_seq_ad3)]

    if use_ad3_binary:
        dual_obj_seq_ad3_binary  = [np.inf]
        for eta in [0.001, 0.01, 0.1, 1.0, 5.0]:
        #for eta in [0.1]: #[0.1]:
            print 'Running AD3 binary with eta =', eta, '.'
            dual_obj_seq, primal_obj_seq, _ = \
                run_ad3(edges, node_potentials, edge_potentials, num_iterations, eta=eta, convert_to_binary=True)
            dual_obj_seq, primal_obj_seq = find_stepwise_best(dual_obj_seq, primal_obj_seq)
            print 'Best dual:', dual_obj_seq[-1]
            print 'Best primal:', primal_obj_seq[-1]
            if dual_obj_seq[-1] < dual_obj_seq_ad3_binary[-1]:
                dual_obj_seq_ad3_binary = dual_obj_seq
                primal_obj_seq_ad3_binary = primal_obj_seq
                eta_ad3_binary  = eta
        print 'Best eta AD3 binary:', eta_ad3_binary
        dual_obj_seq_ad3_binary, primal_obj_seq_ad3_binary = \
            trim_primal_dual_sequences(dual_obj_seq_ad3_binary, primal_obj_seq_ad3_binary, dual_value, primal_value)

    fig = plt.figure()    
    if use_mplp:
        plt.plot(np.arange(len(dual_obj_seq_mplp)), dual_obj_seq_mplp, 'c-', label='MPLP dual', linewidth=2.0)
        plt.hold(True)
    if use_np:
        plt.plot(np.arange(len(dual_obj_seq_np)), dual_obj_seq_np, 'b-', label='Norm-Product dual', linewidth=2.0)
        plt.hold(True)
    if use_psdd:
        plt.plot(np.arange(len(dual_obj_seq_psdd)), dual_obj_seq_psdd, 'r-', label='PSDD dual', linewidth=2.0)
        plt.hold(True)
    if use_sdd:
        plt.plot(np.arange(len(dual_obj_seq_sdd)), dual_obj_seq_sdd, 'b-', label='SDD dual', linewidth=2.0)
        plt.hold(True)
    if use_accdd:
        plt.plot(np.arange(len(dual_obj_seq_accdd)), dual_obj_seq_accdd, 'm-', label='ACCDD dual', linewidth=2.0)
        plt.hold(True)
    if use_ad3:
        #plt.plot(np.arange(len(dual_obj_seq_ad3)), dual_obj_seq_ad3, 'g-', label='AD3 dual (with active set method)', linewidth=2.0)
        plt.plot(np.array(num_oracle_calls_seq_ad3)-1, dual_obj_seq_ad3, 'g-', label='AD3 dual (with active set method)', linewidth=2.0)
        plt.hold(True)
    if use_ad3_binary:
        plt.plot(np.arange(len(dual_obj_seq_ad3_binary)), dual_obj_seq_ad3_binary, 'y-', label='AD3 dual (with binarization)', linewidth=2.0)
        plt.hold(True)
    if use_gurobi:
        plt.plot(np.arange(num_iterations), np.tile(dual_value, num_iterations), 'k:', label='Optimal dual', linewidth=2.0)
        plt.hold(True)

    if print_primal:
        if use_mplp:
            plt.plot(np.arange(len(primal_obj_seq_mplp)), primal_obj_seq_mplp, 'c:', label='MPLP primal', linewidth=2.0)
            plt.hold(True)
        if use_np:
            plt.plot(np.arange(len(primal_obj_seq_np)), primal_obj_seq_np, 'b:', label='Norm-Product primal', linewidth=2.0)
            plt.hold(True)
        if use_psdd:
            plt.plot(np.arange(len(primal_obj_seq_psdd)), primal_obj_seq_psdd, 'r:', label='PSDD primal', linewidth=2.0)
            plt.hold(True)
        if use_sdd:
            plt.plot(np.arange(len(primal_obj_seq_sdd)), primal_obj_seq_sdd, 'b:', label='SDD primal', linewidth=2.0)
            plt.hold(True)
        if use_accdd:
            plt.plot(np.arange(len(primal_obj_seq_accdd)), primal_obj_seq_accdd, 'm:', label='ACCDD primal', linewidth=2.0)
            plt.hold(True)
        if use_ad3:
            #plt.plot(np.arange(len(primal_obj_seq_ad3)), primal_obj_seq_ad3, 'g:', label='AD3 primal (with active set method)', linewidth=2.0)
            plt.plot(np.array(num_oracle_calls_seq_ad3)-1, primal_obj_seq_ad3, 'g:', label='AD3 primal (with active set method)', linewidth=2.0)
            plt.hold(True)
        if use_ad3_binary:
            plt.plot(np.arange(len(primal_obj_seq_ad3_binary)), primal_obj_seq_ad3_binary, 'y:', label='AD3 primal (with binarization)', linewidth=2.0)
            plt.hold(True)
#    if use_gurobi:
#        plt.plot(np.arange(num_iterations), np.tile(primal_value, num_iterations), 'k:', label='Optimal primal')
#        plt.hold(True)
    
    
#    plt.legend(loc=4) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(loc=1) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Objective value')
    plt.xlabel('Number of iterations')
    
    if print_primal:
        ymin = np.max(primal_obj_seq_ad3) - 10.0
    else:
        ymin = np.min(dual_obj_seq_ad3) - 20.0
    ymax = np.min(dual_obj_seq_ad3) + 200.0
    
    plt.ylim((ymin, ymax))
    plt.xlim((0, num_iterations))
    plt.suptitle('Edge coupling: ' + str(edge_coupling))
    
    #pdb.set_trace()
    
    #plt.xticks(paramValues)
    #plt.grid(True)
    
    filename = 'potts_gridsize-%d_states-%d_coupling-%f.png' % \
        (grid_size, num_states, edge_coupling)
    fig.savefig(filename)

    plt.show()
    pdb.set_trace()
