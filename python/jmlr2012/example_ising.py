import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pdb

def generate_ising_grid(n, edge_coupling):
    node_potentials = 2.0*np.random.rand(n*n) - 1.0
    node_indices = -np.ones((n,n), dtype=int)
    t = 0
    for i in xrange(n):
        for j in xrange(n):
            node_indices[i,j] = t
            t += 1
    edges = []
    edge_potentials = np.zeros(2*n*n)
    for i in xrange(n):
        if i > 0:
            prev_i = i-1
        else:
            prev_i = n-1
        for j in xrange(n):
            if j > 0:
                prev_j = j-1
            else:
                prev_j = n-1
            t = len(edges)
            edges.append((node_indices[prev_i,j], node_indices[i,j]))
            edge_potentials[t] = 2.0*edge_coupling*(2.0*np.random.rand(1) - 1.0)
            node_potentials[node_indices[prev_i,j]] -= .5 * edge_potentials[t] 
            node_potentials[node_indices[i,j]] -= .5 * edge_potentials[t] 

            t = len(edges)
            edges.append((node_indices[i,prev_j], node_indices[i,j]))
            edge_potentials[t] = 2.0*edge_coupling*(2.0*np.random.rand(1) - 1.0)
            node_potentials[node_indices[i,prev_j]] -= .5 * edge_potentials[t] 
            node_potentials[node_indices[i,j]] -= .5 * edge_potentials[t] 

    return node_indices, edges, node_potentials, edge_potentials


def save_ising(filename, edges, node_potentials, edge_potentials):
    f = open(filename, 'w')
    num_variables = len(node_potentials)
    num_factors = len(edge_potentials)
    f.write(str(num_variables) + '\n')
    f.write(str(num_factors) + '\n')
    for i in xrange(num_variables):
        f.write(str(node_potentials[i]) + '\n')
#    pdb.set_trace()
    for j in xrange(num_factors):
        f.write('PAIR 2 %d %d %f\n' % (1+edges[j][0], 1+edges[j][1], edge_potentials[j]))
    f.close()


def load_ising(filename):
    f = open(filename)
    i = 0
    for line in f:
        line = line.rstrip('\n')
        if i == 0:
            num_variables = int(line)
            node_potentials = np.zeros(num_variables)
        elif i == 1:
            num_factors = int(line)
            edge_potentials = np.zeros(num_factors)
            edges = []
        elif i-2 < num_variables:
            node_potentials[i-2] = float(line)
        elif i-2-num_variables < num_factors:
            j = i-2-num_variables
            fields = line.split(' ')
            assert fields[0] == 'PAIR', pdb.set_trace()
            assert int(fields[1]) == 2, pdb.set_trace()
            edges.append((int(fields[2])-1, int(fields[3])-1))
            edge_potentials[j] = float(fields[4])
        else:
            assert False, pdb.set_trace()
        i += 1
            
    f.close()
    return edges, node_potentials, edge_potentials


def solve_qp(scores, additional_scores):
    # min 1/2 (u[0] - u0[0])^2 + (u[1] - u0[1])^2 + u0[2] * u[2], 
    # where u[2] is the edge marginal.
    # Remark: Assume inputs are NOT negated.
    x0 = [scores[0], scores[1], -additional_scores[0]]
    c = x0[2]
    if additional_scores[0] < 0:
        x0[0] -= c;
        x0[1] = 1 - x0[1];
        c = -c;

    if x0[0] > x0[1] - c:
        posteriors = [x0[0], x0[1] - c]
    elif x0[1] > x0[0] - c:
        posteriors = [x0[0] - c, x0[1]]
    else:
        posteriors = [0.5 * (x0[0] + x0[1] - c), 0.5 * (x0[0] + x0[1] - c)]

    # Project onto box.
    if posteriors[0] < 0.0:
        posteriors[0] = 0.0
    elif posteriors[0] > 1.0:
        posteriors[0] = 1.0
    if posteriors[1] < 0.0:
        posteriors[1] = 0.0
    elif posteriors[1] > 1.0:
        posteriors[1] = 1.0

    # u[2] = min(u[0], u[1]);
    additional_posteriors = [min(posteriors[0], posteriors[1])]

    if (additional_scores[0] < 0):
        # c > 0
        posteriors[1] = 1.0 - posteriors[1]
        additional_posteriors[0] = posteriors[0] - additional_posteriors[0] 

    return posteriors, additional_posteriors


def solve_max_marginals(scores, additional_scores):
    p = [0.0, # 00
         scores[1], # 01
         scores[0], # 10
         scores[0] + scores[1] + additional_scores[0]] # 11

    max_marginals_zeros = [0.0, 0.0]
    max_marginals_ones = [0.0, 0.0]

    #pdb.set_trace()
    # Compute max marginals for the first variable.  
    max_marginals_zeros[0] = np.max([p[0], p[1]])
    max_marginals_ones[0] = np.max([p[2], p[3]])

    # Compute max marginals for the second variable.  
    max_marginals_zeros[1] = np.max([p[0], p[2]]);
    max_marginals_ones[1] = np.max([p[1], p[3]]);

    return max_marginals_zeros, max_marginals_ones


def compute_sum_product_messages(scores, additional_scores, temperature=1.0):
    marginals, value = solve_marginals(scores, additional_scores, temperature)
    Z = np.exp(value/temperature)
    marginals_ones = [(m*Z)**temperature for m in marginals]
    marginals_zeros = [((1.0-m)*Z)**temperature for m in marginals]
    messages_zeros = [0.0, 0.0]
    messages_ones = [0.0, 0.0]
    for i in xrange(2):
        messages_ones[i] = marginals_ones[i] / np.exp(scores[i])
        messages_zeros[i] = marginals_zeros[i]
    return messages_zeros, messages_ones


def solve_marginals(scores, additional_scores, temperature=1.0):
    p = [0.0, # 00
         scores[1]/temperature, # 01
         scores[0]/temperature, # 10
         (scores[0] + scores[1] + additional_scores[0])/temperature] # 11

    marginals_zeros = [0.0, 0.0]
    marginals_ones = [0.0, 0.0]
    marginals = [0.0, 0.0]

    # Compute marginals for the first variable.  
    marginals_zeros[0] = np.logaddexp(p[0], p[1])
    marginals_ones[0] = np.logaddexp(p[2], p[3])

    # Compute marginals for the second variable.  
    marginals_zeros[1] = np.logaddexp(p[0], p[2]);
    marginals_ones[1] = np.logaddexp(p[1], p[3]);

    logZ = np.logaddexp(marginals_zeros[0], marginals_ones[0])
    marginals[0] = np.exp(marginals_ones[0] - logZ)
    marginals[1] = np.exp(marginals_ones[1] - logZ)
    value = logZ*temperature

    return marginals, value


def solve_map(scores, additional_scores):
    p = [0.0, # 00
         scores[1], # 01
         scores[0], # 10
         scores[0] + scores[1] + additional_scores[0]] # 11

    best = np.argmax(p)
    value = p[best]
    if best == 0:
        posteriors = [0.0, 0.0]
        additional_posteriors = [0.0]
    elif best == 1:
        posteriors = [0.0, 1.0]
        additional_posteriors = [0.0]
    elif best == 2:
        posteriors = [1.0, 0.0]
        additional_posteriors = [0.0]
    else: # best == 3
        posteriors = [1.0, 1.0]
        additional_posteriors = [1.0]

    return posteriors, additional_posteriors, value


def build_node_to_edges_map(num_nodes, edges):
    num_edges = len(edges)
    n1_to_edges = [[] for i in xrange(num_nodes)]
    n2_to_edges = [[] for i in xrange(num_nodes)]
    for j in xrange(num_edges):
        n1 = edges[j][0]
        n2 = edges[j][1]
        n1_to_edges[n1].append(j)
        n2_to_edges[n2].append(j)

    return n1_to_edges, n2_to_edges


def log_norm(v, p):
    # Computes u = log(||exp(v)||_p) = (1/p) log (\sum_i exp(vi*p)) = v_max + (1/p) log (\sum_i exp((vi-v_max)*p))
    m = np.max(v)
    if p == np.inf:
        return m
    else:
        return m + (1/p)*np.log(np.sum(np.exp((v-m)*p)))


def run_norm_product(edges, node_potentials, edge_potentials, num_iterations=500, temperature=0.01):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    c_alphas = np.ones(num_edges)
    c_i = np.zeros(num_nodes)
    c_i_alphas = np.zeros((num_edges,2))
    posteriors = .5 * np.ones(num_nodes)
    m_messages0 = np.zeros((num_edges,2))
    m_messages1 = np.zeros((num_edges,2))
    n_messages0 = np.zeros((num_edges,2))
    n_messages1 = np.zeros((num_edges,2))
    edge_posteriors = np.zeros(num_edges)

    n1_to_edges, n2_to_edges = build_node_to_edges_map(num_nodes, edges)

    c_i_hat = c_i.copy()
    for i in xrange(num_nodes):
        c_i_hat[i] += sum([c_alphas[e] for e in n1_to_edges[i]])
        c_i_hat[i] += sum([c_alphas[e] for e in n2_to_edges[i]])
    
    #dual_obj_prev = np.inf
    #num_times_increment = 0;

    dual_obj_seq = np.zeros(num_iterations)
    primal_obj_seq = np.zeros(num_iterations)
    for t in xrange(num_iterations):
        for i in xrange(num_nodes):
            # Update m-messages.
            for e in n1_to_edges[i]:
                scores = [n_messages1[e,0] - n_messages0[e,0],
                          n_messages1[e,1] - n_messages0[e,1]]
                additional_scores = [edge_potentials[e]]
                v = np.array([0.0, scores[1]])
                m_messages0[e,0] = log_norm(v, 1.0/(temperature*c_alphas[e]))
                v = np.array([0.0, scores[1] + additional_scores[0]])
                m_messages1[e,0] = log_norm(v, 1.0/(temperature*c_alphas[e]))           
            for e in n2_to_edges[i]:
                scores = [n_messages1[e,0] - n_messages0[e,0],
                          n_messages1[e,1] - n_messages0[e,1]]
                additional_scores = [edge_potentials[e]]
                v = np.array([scores[0], 0.0])
                m_messages0[e,1] = log_norm(v, 1.0/(temperature*c_alphas[e]))
                v = np.array([scores[0] + additional_scores[0], 0.0])
                m_messages1[e,1] = log_norm(v, 1.0/(temperature*c_alphas[e]))           

#        for i in xrange(num_nodes):
            # Update n-messages.
            value0 = 0.0
            value1 = node_potentials[i]
            for e in n1_to_edges[i]:
                value0 += m_messages0[e,0]
                value1 += m_messages1[e,0]
            for e in n2_to_edges[i]:
                value0 += m_messages0[e,1]
                value1 += m_messages1[e,1]
            for e in n1_to_edges[i]:
                n_messages0[e,0] = (value0 * (c_alphas[e]/c_i_hat[i])) - m_messages0[e,0]
                n_messages1[e,0] = (value1 * (c_alphas[e]/c_i_hat[i])) - m_messages1[e,0]
                value = (n_messages0[e,0] + n_messages1[e,0])/2.0
                n_messages0[e,0] -= value
                n_messages1[e,0] -= value
            for e in n2_to_edges[i]:
                n_messages0[e,1] = (value0 * (c_alphas[e]/c_i_hat[i])) - m_messages0[e,1]
                n_messages1[e,1] = (value1 * (c_alphas[e]/c_i_hat[i])) - m_messages1[e,1]
                value = (n_messages0[e,1] + n_messages1[e,1])/2.0
                n_messages0[e,1] -= value
                n_messages1[e,1] -= value


        # Compute beliefs and dual objective.
        entropy = 0.0
        dual_smooth_obj = 0.0
        for i in xrange(num_nodes):
            belief0 = 0.0
            belief1 = node_potentials[i]
            for e in n1_to_edges[i]:
                belief0 += m_messages0[e,0]
                belief1 += m_messages1[e,0]
            for e in n2_to_edges[i]:
                belief0 += m_messages0[e,1]
                belief1 += m_messages1[e,1]
            belief0 = belief0 * (1.0/(temperature*c_i_hat[i]))
            belief1 = belief1 * (1.0/(temperature*c_i_hat[i]))
            value = np.logaddexp(belief0, belief1)
            belief0 = np.exp(belief0 - value)
            belief1 = np.exp(belief1 - value)
            posteriors[i] = belief1
            entropy -= c_i[i] * (belief0 * np.log(belief0) + belief1 * np.log(belief1))
            dual_smooth_obj += node_potentials[i] * posteriors[i]

        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            belief00 = n_messages0[j,0] + n_messages0[j,1]
            belief01 = n_messages0[j,0] + n_messages1[j,1]
            belief10 = n_messages1[j,0] + n_messages0[j,1]
            belief11 = edge_potentials[j] + n_messages1[j,0] + n_messages1[j,1]
            belief00 = belief00 * (1.0/(temperature*c_alphas[j]))
            belief01 = belief01 * (1.0/(temperature*c_alphas[j]))
            belief10 = belief10 * (1.0/(temperature*c_alphas[j]))
            belief11 = belief11 * (1.0/(temperature*c_alphas[j]))
            value = np.logaddexp(np.logaddexp(belief00, belief01), np.logaddexp(belief10, belief11))
            belief00 = np.exp(belief00 - value)
            belief01 = np.exp(belief01 - value)
            belief10 = np.exp(belief10 - value)
            belief11 = np.exp(belief11 - value)
            edge_posteriors[j] = belief11
            entropy -= c_alphas[j] * (belief00 * np.log(belief00) + belief01 * np.log(belief01) + \
                                      belief10 * np.log(belief10) + belief11 * np.log(belief11))
            dual_smooth_obj += edge_potentials[j] * edge_posteriors[j]

        #pdb.set_trace()
        primal_rel_obj = dual_smooth_obj
        dual_smooth_obj += temperature * entropy


        dual_obj = 0.0
        dual_obj2 = 0.0
        for e in xrange(num_edges):
            v00 = n_messages0[e,0] + n_messages0[e,1]
            v01 = n_messages0[e,0] + n_messages1[e,1]
            v10 = n_messages1[e,0] + n_messages0[e,1]
            v11 = edge_potentials[e] + n_messages1[e,0] + n_messages1[e,1]
            v = np.array([v00, v01, v10, v11])
            #pdb.set_trace()
            dual_obj += log_norm(v, 1.0/(temperature * c_alphas[e]))
            dual_obj2 += log_norm(v, np.inf)
        #dual_obj1 = dual_obj
        for i in xrange(num_nodes):
            v0 = 0.0
            v1 = node_potentials[i]
            for e in n1_to_edges[i]:
#                v0 += max(-n_messages0[e,0], -n_messages1[e,0])
#                v1 += max(-n_messages0[e,0], -n_messages1[e,0])
                v0 -= n_messages0[e,0]
                v1 -= n_messages1[e,0]
            for e in n2_to_edges[i]:
#                v0 += max(-n_messages0[e,1], -n_messages1[e,1])
#                v1 += max(-n_messages0[e,1], -n_messages1[e,1])
                v0 -= n_messages0[e,1]
                v1 -= n_messages1[e,1]
            v = np.array([v0, v1])
#            print v[1]
            if c_i[i] == 0:
                dual_obj += log_norm(v, np.inf)
            else:
                dual_obj += log_norm(v, 1.0/(temperature * c_i[i]))
            dual_obj2 += log_norm(v, np.inf)
        #dual_obj2 = dual_obj-dual_obj1
        #print (dual_obj1,dual_obj2)

        dual_obj = dual_obj2 # Comment this line to use the smoothed dual objective.



        # Compute relaxed primal objective.
        #primal_rel_obj = 0.0;
        #for i in xrange(num_nodes):
        #    primal_rel_obj += p[i] * node_potentials[i]
        #for j in xrange(num_edges):
        #    primal_rel_obj += edge_posteriors[j] * edge_potentials[j]

        # Compute primal objective.
        primal_obj = 0.0;
        p = posteriors
        p_int = np.zeros(num_nodes)
        for i in xrange(num_nodes):
            p_int[i] = round(p[i])
            primal_obj += p_int[i] * node_potentials[i]
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            primal_obj += p_int[n1] * p_int[n2] * edge_potentials[j]

        if t % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal rel obj:', primal_rel_obj, 'Primal obj:', primal_obj

        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    return dual_obj_seq, primal_obj_seq, p_int



def run_mplp(edges, node_potentials, edge_potentials, num_iterations=500):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    posteriors = .5 * np.ones(num_nodes)
    gammas0 = np.zeros((num_edges,2))
    gammas1 = np.zeros((num_edges,2))
    deltas0 = np.zeros((num_edges,2))
    deltas1 = np.zeros((num_edges,2))
    edge_posteriors = np.zeros(num_edges)

    n1_to_edges, n2_to_edges = build_node_to_edges_map(num_nodes, edges)

    #dual_obj_prev = np.inf
    #num_times_increment = 0;

    dual_obj_seq = np.zeros(num_iterations)
    primal_obj_seq = np.zeros(num_iterations)
    for t in xrange(num_iterations):
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(n1_to_edges[n1]) + len(n2_to_edges[n1])
            d2 = len(n1_to_edges[n2]) + len(n2_to_edges[n2])

            # Update deltas.
            gamma0_tot = sum([gammas0[e,0] for e in n1_to_edges[n1]])
            gamma0_tot += sum([gammas0[e,1] for e in n2_to_edges[n1]])
            gamma1_tot = sum([gammas1[e,0] for e in n1_to_edges[n1]])
            gamma1_tot += sum([gammas1[e,1] for e in n2_to_edges[n1]])
            for e in n1_to_edges[n1]:
                deltas0[e,0] = gamma0_tot - gammas0[e,0]
                deltas1[e,0] = node_potentials[n1] + gamma1_tot - gammas1[e,0]
            for e in n2_to_edges[n1]:
                deltas0[e,1] = gamma0_tot - gammas0[e,1]
                deltas1[e,1] = node_potentials[n1] + gamma1_tot - gammas1[e,1]

            gamma0_tot = sum([gammas0[e,0] for e in n1_to_edges[n2]])
            gamma0_tot += sum([gammas0[e,1] for e in n2_to_edges[n2]])
            gamma1_tot = sum([gammas1[e,0] for e in n1_to_edges[n2]])
            gamma1_tot += sum([gammas1[e,1] for e in n2_to_edges[n2]])
            for e in n1_to_edges[n2]:
                deltas0[e,0] = gamma0_tot - gammas0[e,0]
                deltas1[e,0] = node_potentials[n2] + gamma1_tot - gammas1[e,0]
            for e in n2_to_edges[n2]:
                deltas0[e,1] = gamma0_tot - gammas0[e,1]
                deltas1[e,1] = node_potentials[n2] + gamma1_tot - gammas1[e,1]

            # Compute max-marginals and update gammas.
            constant = deltas0[j,0] + deltas0[j,1]
            scores = [deltas1[j,0] - deltas0[j,0], deltas1[j,1] - deltas0[j,1]]
            additional_scores = [edge_potentials[j]]
            max_marginals_zeros, max_marginals_ones = solve_max_marginals(scores, additional_scores)

            factor_degree = 2
            gammas0[j,0] = (max_marginals_zeros[0] + constant) / float(factor_degree) \
                - deltas0[j,0]
            gammas1[j,0] = (max_marginals_ones[0] + constant) / float(factor_degree) \
                - deltas1[j,0]
            gammas0[j,1] = (max_marginals_zeros[1] + constant) / float(factor_degree) \
                - deltas0[j,1]
            gammas1[j,1] = (max_marginals_ones[1] + constant) / float(factor_degree) \
                - deltas1[j,1]


        # Compute dual objective.
        # First, get the contribution of the node variables to the dual objective.
        dual_obj = 0.0;        
        for i in xrange(num_nodes):
            gamma0_tot = sum([gammas0[e,0] for e in n1_to_edges[i]])
            gamma0_tot += sum([gammas0[e,1] for e in n2_to_edges[i]])
            gamma1_tot = sum([gammas1[e,0] for e in n1_to_edges[i]])
            gamma1_tot += sum([gammas1[e,1] for e in n2_to_edges[i]])

            val = gamma1_tot - gamma0_tot + node_potentials[i]
            #posteriors[i] = val
            if val <= 0.0:
                dual_obj += gamma0_tot
                posteriors[i] = 0.0
            else:
                dual_obj += gamma1_tot + node_potentials[i]
                posteriors[i] = 1.0
            
        # Now, get the contribution of the factors to the dual objective.
        for j in xrange(num_edges):
            scores = [gammas0[j,0] - gammas1[j,0], gammas0[j,1] - gammas1[j,1]]
            additional_scores = [edge_potentials[j]]
            local_posteriors, local_additional_posteriors, value = \
                solve_map(scores, additional_scores)
            value -= (gammas0[j,0] + gammas0[j,1])

            n1 = edges[j][0]
            n2 = edges[j][1]
            #edge_posteriors[j] = posteriors[n1] * posteriors[n2]

            dual_obj += value

        # Compute relaxed primal objective.
        primal_rel_obj = 0.0;
        #for i in xrange(num_nodes):
        #    primal_rel_obj += p[i] * node_potentials[i]
        #for j in xrange(num_edges):
        #    primal_rel_obj += edge_posteriors[j] * edge_potentials[j]

        # Compute primal objective.
        primal_obj = 0.0;
        p = posteriors
        p_int = np.zeros(num_nodes)
        for i in xrange(num_nodes):
            p_int[i] = round(p[i])
            primal_obj += p_int[i] * node_potentials[i]
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            primal_obj += p_int[n1] * p_int[n2] * edge_potentials[j]

        if t % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal rel obj:', primal_rel_obj, 'Primal obj:', primal_obj

        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    return dual_obj_seq, primal_obj_seq, p_int



def run_accdd(edges, node_potentials, edge_potentials, num_iterations=500, epsilon=10.0):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    p = .5 * np.ones(num_nodes)
    q = .5 * np.ones((num_edges,2))
    lambdas = np.zeros((num_edges,2))
    zetas = np.zeros((num_edges, 2))
    edge_posteriors = np.zeros(num_edges)
    sumlog_assig = num_edges*2.0*np.log(2.0)
    T = epsilon/(2.0*sumlog_assig);
    L = 2.0*sumlog_assig/epsilon;
    theta = 1.0

    n1_to_edges, n2_to_edges = build_node_to_edges_map(num_nodes, edges)

    dual_obj_prev = np.inf
    num_times_increment = 0;

    dual_smooth_obj_seq = np.zeros(num_iterations)
    dual_obj_seq = np.zeros(num_iterations)
    primal_obj_seq = np.zeros(num_iterations)
    for t in xrange(num_iterations):
        # Make updates and compute "smoothed" dual objective.
        dual_smooth_obj = 0.0;
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(n1_to_edges[n1]) + len(n2_to_edges[n1])
            d2 = len(n1_to_edges[n2]) + len(n2_to_edges[n2])
            values = [node_potentials[n1] / float(d1) + 2.0 * lambdas[j,0],
                      node_potentials[n2] / float(d2) + 2.0 * lambdas[j,1]]
            delta = -(lambdas[j,0] + lambdas[j,1])
            scores = [values[0], values[1]]
            additional_scores = [edge_potentials[j]]
            posteriors, value = solve_marginals(scores, additional_scores, T)
            value -= T*2*np.log(2.0)
            dual_smooth_obj += value + delta

            q[j,0] = posteriors[0]
            q[j,1] = posteriors[1]

        # Check if dual improved so that num_times_increment 
        # can be incremented.
        #if dual_obj < dual_obj_prev:
        #    num_times_increment += 1;
        #dual_obj_prev = dual_obj;

        # Project (to update zetas).
        gammas = np.zeros(num_nodes);
        zetas_new = zetas.copy();
        for i in xrange(num_nodes):
            d = len(n1_to_edges[i]) + len(n2_to_edges[i])
            val = 0.0
            for j in n1_to_edges[i]:
                val += theta*L*zetas[j,0] - q[j,0]
            for j in n2_to_edges[i]:            
                val += theta*L*zetas[j,1] - q[j,1]
            val /= float(d)
            gammas[i] = val
            p[i] = (sum([q[j,0] for j in n1_to_edges[i]]) + sum([q[j,1] for j in n2_to_edges[i]])) / float(d)
            for j in n1_to_edges[i]:
                zetas_new[j,0] = zetas[j,0] - (1/(theta*L)) * (q[j,0] + gammas[i])
            for j in n2_to_edges[i]:
                zetas_new[j,1] = zetas[j,1] - (1/(theta*L)) * (q[j,1] + gammas[i])
        zetas = zetas_new

        # Update lambdas.
        for j in xrange(num_edges):
            lambdas[j,0] = (1-theta)*lambdas[j,0] + theta*zetas[j,0]
            lambdas[j,1] = (1-theta)*lambdas[j,1] + theta*zetas[j,1]
#        for i in xrange(num_nodes):
#            for j in n1_to_edges[i]:
#                lambdas[j,0] = (1-theta)*lambdas[j,0] + theta*zetas[j,0]
#            for j in n2_to_edges[i]:
#                lambdas[j,1] = (1-theta)*lambdas[j,1] + theta*zetas[j,1]

        theta = (np.sqrt(theta**4 + 4*(theta**2)) - theta**2)/2.0;

        # Compute dual objective.
        dual_obj = 0.0;
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(n1_to_edges[n1]) + len(n2_to_edges[n1])
            d2 = len(n1_to_edges[n2]) + len(n2_to_edges[n2])
            values = [node_potentials[n1] / float(d1) + 2.0 * lambdas[j,0],
                      node_potentials[n2] / float(d2) + 2.0 * lambdas[j,1]]
            #delta = -(lambdas[j,0] + lambdas[j,1])
            delta = lambdas[j,0] + lambdas[j,1] # note sign!!!
            scores = [values[0], values[1]]
            additional_scores = [edge_potentials[j]]
            posteriors, additional_posteriors, value = solve_map(scores, additional_scores)
            dual_obj += value + delta

            q[j,0] = posteriors[0]
            q[j,1] = posteriors[1]
            edge_posteriors[j] = additional_posteriors[0]


        for i in xrange(num_nodes):
            p[i] = (sum([q[j,0] for j in n1_to_edges[i]]) + sum([q[j,1] for j in n2_to_edges[i]])) / \
                     float(len(n1_to_edges[i]) + len(n2_to_edges[i]))



        # Compute relaxed primal objective.
        primal_rel_obj = 0.0;
        for i in xrange(num_nodes):
            primal_rel_obj += p[i] * node_potentials[i]
        for j in xrange(num_edges):
            primal_rel_obj += edge_posteriors[j] * edge_potentials[j]

        # Compute primal objective.
        primal_obj = 0.0;
        p_int = np.zeros(num_nodes)
        for i in xrange(num_nodes):
            p_int[i] = round(p[i])
            primal_obj += p_int[i] * node_potentials[i]
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            primal_obj += p_int[n1] * p_int[n2] * edge_potentials[j]

        if t % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal rel obj:', primal_rel_obj, 'Primal obj:', primal_obj, 'Dual smooth obj:', dual_smooth_obj

        dual_smooth_obj_seq[t] = dual_smooth_obj
        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    return dual_obj_seq, primal_obj_seq, p_int



def run_sdd(edges, node_potentials, edge_potentials, num_iterations=500, eta=5.0, temperature=0.01):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    p = .5 * np.ones(num_nodes)
    q = .5 * np.ones((num_edges,2))
    lambdas = np.zeros((num_edges,2))
    edge_posteriors = np.zeros(num_edges)

    n1_to_edges, n2_to_edges = build_node_to_edges_map(num_nodes, edges)

    dual_obj_prev = np.inf
    #num_times_increment = 0;

    dual_smooth_obj_seq = np.zeros(num_iterations)
    dual_obj_seq = np.zeros(num_iterations)
    primal_obj_seq = np.zeros(num_iterations)
    for t in xrange(num_iterations):
        # Make updates and compute dual smoothed objective.
        dual_smooth_obj = 0.0;
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(n1_to_edges[n1]) + len(n2_to_edges[n1])
            d2 = len(n1_to_edges[n2]) + len(n2_to_edges[n2])
            values = [node_potentials[n1] / float(d1) + 2.0 * lambdas[j,0],
                      node_potentials[n2] / float(d2) + 2.0 * lambdas[j,1]]
            delta = -(lambdas[j,0] + lambdas[j,1])
            scores = [values[0], values[1]]
            additional_scores = [edge_potentials[j]]
            posteriors, value = solve_marginals(scores, additional_scores, temperature)
            value -= temperature * 2.0*np.log(2.0)
            dual_smooth_obj += value + delta # NOTE: IS IT REALLY DELTA HERE?

            q[j,0] = posteriors[0]
            q[j,1] = posteriors[1]

        # Check if dual improved so that num_times_increment 
        # can be incremented.
        #if dual_obj < dual_obj_prev:
        #    num_times_increment += 1;
        #dual_obj_prev = dual_obj;

        for i in xrange(num_nodes):
            p[i] = (sum([q[j,0] for j in n1_to_edges[i]]) + sum([q[j,1] for j in n2_to_edges[i]])) / \
                     float(len(n1_to_edges[i]) + len(n2_to_edges[i]))

        #eta_t = eta / np.sqrt(num_times_increment);
        eta_t = eta / np.sqrt(t+1);
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            lambdas[j,0] -= eta_t * (q[j,0] - p[n1])
            lambdas[j,1] -= eta_t * (q[j,1] - p[n2])

        # Compute dual objective.
        dual_obj = 0.0;
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(n1_to_edges[n1]) + len(n2_to_edges[n1])
            d2 = len(n1_to_edges[n2]) + len(n2_to_edges[n2])
            values = [node_potentials[n1] / float(d1) + 2.0 * lambdas[j,0],
                      node_potentials[n2] / float(d2) + 2.0 * lambdas[j,1]]
            #delta = -(lambdas[j,0] + lambdas[j,1])
            delta = lambdas[j,0] + lambdas[j,1] # note sign!!!
            scores = [values[0], values[1]]
            additional_scores = [edge_potentials[j]]
            posteriors, additional_posteriors, value = solve_map(scores, additional_scores)
            dual_obj += value + delta

            q[j,0] = posteriors[0]
            q[j,1] = posteriors[1]
            edge_posteriors[j] = additional_posteriors[0]


        for i in xrange(num_nodes):
            p[i] = (sum([q[j,0] for j in n1_to_edges[i]]) + sum([q[j,1] for j in n2_to_edges[i]])) / \
                     float(len(n1_to_edges[i]) + len(n2_to_edges[i]))


        # Compute relaxed primal objective.
        primal_rel_obj = 0.0;
        for i in xrange(num_nodes):
            primal_rel_obj += p[i] * node_potentials[i]
        for j in xrange(num_edges):
            primal_rel_obj += edge_posteriors[j] * edge_potentials[j]

        # Compute primal objective.
        primal_obj = 0.0;
        p_int = np.zeros(num_nodes)
        for i in xrange(num_nodes):
            p_int[i] = round(p[i])
            primal_obj += p_int[i] * node_potentials[i]
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            primal_obj += p_int[n1] * p_int[n2] * edge_potentials[j]

        if t % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal rel obj:', primal_rel_obj, 'Primal obj:', primal_obj, 'Dual smooth obj:', dual_smooth_obj

        dual_smooth_obj_seq[t] = dual_smooth_obj
        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj


#    return dual_smooth_obj_seq, primal_obj_seq, p_int
    return dual_obj_seq, primal_obj_seq, p_int



def run_psdd(edges, node_potentials, edge_potentials, num_iterations=500, eta=5.0):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    p = .5 * np.ones(num_nodes)
    q = .5 * np.ones((num_edges,2))
    lambdas = np.zeros((num_edges,2))
    edge_posteriors = np.zeros(num_edges)

    n1_to_edges, n2_to_edges = build_node_to_edges_map(num_nodes, edges)

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
            d1 = len(n1_to_edges[n1]) + len(n2_to_edges[n1])
            d2 = len(n1_to_edges[n2]) + len(n2_to_edges[n2])
            values = [node_potentials[n1] / float(d1) + 2.0 * lambdas[j,0],
                      node_potentials[n2] / float(d2) + 2.0 * lambdas[j,1]]
            delta = -(lambdas[j,0] + lambdas[j,1])
            scores = [values[0], values[1]]
            additional_scores = [edge_potentials[j]]
            posteriors, additional_posteriors, value = solve_map(scores, additional_scores)
            dual_obj += value + delta

            q[j,0] = posteriors[0]
            q[j,1] = posteriors[1]
            edge_posteriors[j] = additional_posteriors[0]

        # Check if dual improved so that num_times_increment 
        # can be incremented.
        if dual_obj < dual_obj_prev:
            num_times_increment += 1;
        dual_obj_prev = dual_obj;

        for i in xrange(num_nodes):
            p[i] = (sum([q[j,0] for j in n1_to_edges[i]]) + sum([q[j,1] for j in n2_to_edges[i]])) / \
                     float(len(n1_to_edges[i]) + len(n2_to_edges[i]))

        eta_t = eta / np.sqrt(num_times_increment);
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            lambdas[j,0] -= eta_t * (q[j,0] - p[n1])
            lambdas[j,1] -= eta_t * (q[j,1] - p[n2])


        # Compute relaxed primal objective.
        primal_rel_obj = 0.0;
        for i in xrange(num_nodes):
            primal_rel_obj += p[i] * node_potentials[i]
        for j in xrange(num_edges):
            primal_rel_obj += edge_posteriors[j] * edge_potentials[j]

        # Compute primal objective.
        primal_obj = 0.0;
        p_int = np.zeros(num_nodes)
        for i in xrange(num_nodes):
            p_int[i] = round(p[i])
            primal_obj += p_int[i] * node_potentials[i]
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            primal_obj += p_int[n1] * p_int[n2] * edge_potentials[j]

        if t % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal rel obj:', primal_rel_obj, 'Primal obj:', primal_obj

        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    return dual_obj_seq, primal_obj_seq, p_int



def run_ad3(edges, node_potentials, edge_potentials, num_iterations=500, eta=5.0):
    num_nodes = len(node_potentials)
    num_edges = len(edges)
    p = .5 * np.ones(num_nodes)
    q = .5 * np.ones((num_edges,2))
    lambdas = np.zeros((num_edges,2))
    edge_posteriors = np.zeros(num_edges)

    n1_to_edges, n2_to_edges = build_node_to_edges_map(num_nodes, edges)

    dual_obj_seq = np.zeros(num_iterations)
    primal_obj_seq = np.zeros(num_iterations)
    for t in xrange(num_iterations):
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(n1_to_edges[n1]) + len(n2_to_edges[n1])
            d2 = len(n1_to_edges[n2]) + len(n2_to_edges[n2])
            values = [node_potentials[n1] / float(d1) + 2.0 * lambdas[j,0],
                      node_potentials[n2] / float(d2) + 2.0 * lambdas[j,1]]
            scores = [p[n1] + values[0] / (2.0 * eta),
                      p[n2] + values[1] / (2.0 * eta)]
            additional_scores = [edge_potentials[j] / (2.0 * eta)]

            posteriors, additional_posteriors = solve_qp(scores, additional_scores)
            q[j,0] = posteriors[0]
            q[j,1] = posteriors[1]
            edge_posteriors[j] = additional_posteriors[0]

        for i in xrange(num_nodes):
            p[i] = (sum([q[j,0] for j in n1_to_edges[i]]) + sum([q[j,1] for j in n2_to_edges[i]])) / \
                     float(len(n1_to_edges[i]) + len(n2_to_edges[i]))

        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            lambdas[j,0] -= eta * (q[j,0] - p[n1])
            lambdas[j,1] -= eta * (q[j,1] - p[n2])

        # Compute dual objective.
        dual_obj = 0.0;
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            d1 = len(n1_to_edges[n1]) + len(n2_to_edges[n1])
            d2 = len(n1_to_edges[n2]) + len(n2_to_edges[n2])
            values = [node_potentials[n1] / float(d1) + 2.0 * lambdas[j,0],
                      node_potentials[n2] / float(d2) + 2.0 * lambdas[j,1]]
            delta = -(lambdas[j,0] + lambdas[j,1])
            scores = [values[0], values[1]]
            additional_scores = [edge_potentials[j]]
            posteriors, additional_posteriors, value = solve_map(scores, additional_scores)
            dual_obj += value + delta

        # Compute relaxed primal objective.
        primal_rel_obj = 0.0;
        for i in xrange(num_nodes):
            primal_rel_obj += p[i] * node_potentials[i]
        for j in xrange(num_edges):
            primal_rel_obj += edge_posteriors[j] * edge_potentials[j]

        # Compute primal objective.
        primal_obj = 0.0;
        p_int = np.zeros(num_nodes)
        for i in xrange(num_nodes):
            p_int[i] = round(p[i])
            primal_obj += p_int[i] * node_potentials[i]
        for j in xrange(num_edges):
            n1 = edges[j][0]
            n2 = edges[j][1]
            primal_obj += p_int[n1] * p_int[n2] * edge_potentials[j]

        if t % 100 == 0:
            print 'Iteration:', t, 'Dual obj:', dual_obj, 'Primal rel obj:', primal_rel_obj, 'Primal obj:', primal_obj

        dual_obj_seq[t] = dual_obj
        primal_obj_seq[t] = primal_obj

    #return p, q, lambdas
    return dual_obj_seq, primal_obj_seq, p_int #p, q, lambdas


from gurobipy import *

def run_gurobi(edges, node_potentials, edge_potentials, relax=True):
    d1 = len(node_potentials) 
    d2 = len(edge_potentials)
    d = d1+d2

    # Create a new model.
    m = Model("grid_map")
  
    # Create variables.
    for i in xrange(d):
        if relax:
            m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
        else:
            m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
    m.update()
    vars = m.getVars()
  
    # Set objective.
    obj = LinExpr()
    for i in xrange(d1):
        obj += node_potentials[i]*vars[i]
    for j in xrange(d2):
        obj += edge_potentials[j]*vars[d1 + j]
    m.setObjective(obj, GRB.MAXIMIZE)
    
    # Add constraints.
    for j in xrange(d2):
        n1 = edges[j][0]
        n2 = edges[j][1]
        expr = LinExpr()
        expr += vars[d1+j] - vars[n1]
        m.addConstr(expr, GRB.LESS_EQUAL, 0.0)
        expr = LinExpr()
        expr += vars[d1+j] - vars[n2]
        m.addConstr(expr, GRB.LESS_EQUAL, 0.0)
        expr = LinExpr()
        expr += vars[n1] + vars[n2] - vars[d1+j]       
        m.addConstr(expr, GRB.LESS_EQUAL, 1.0)

    # Optimize.
    m.optimize()
    assert m.status == GRB.OPTIMAL
    posteriors = np.zeros(d1)
    edge_posteriors = np.zeros(d2)
    value = 0.0
    for i in xrange(d1):
        posteriors[i] = vars[i].x
        value += posteriors[i] * node_potentials[i]
    for j in xrange(d2):
        edge_posteriors[j] = vars[d1+j].x
        value += edge_posteriors[j] * edge_potentials[j]
    
    return value



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
    #ind = [i for i in xrange(len(rel_err_primal)) if rel_err_primal[i] < err_thres]
    #if len(ind) > 0:
        primal_obj_seq = primal_obj_seq[:(ind[0]+1)]
    return dual_obj_seq, primal_obj_seq
        
        
def compare_several_runs(generate_grid, edge_coupling, grid_size, num_runs):
    tol = 1e-6

    dual_obj_mplp_runs = []
    dual_obj_np_runs = []
    dual_obj_ad3_runs = []
    dual_obj_gurobi_runs = []
    primal_obj_mplp_runs = []
    primal_obj_np_runs = []
    primal_obj_ad3_runs = []
    primal_obj_gurobi_runs = []
    
    for ind_run in xrange(num_runs):
        filename = 'ising_gridsize-%d_coupling-%f_run-%d.fg' % (grid_size, edge_coupling, ind_run)
        if generate_grid:
            node_indices, edges, node_potentials, edge_potentials = generate_ising_grid(grid_size, edge_coupling)
            save_ising(filename, edges, node_potentials, edge_potentials)
        else:
            edges, node_potentials, edge_potentials = load_ising(filename)
        
        num_iterations = 500
        
        use_mplp = True
        use_np = True
        use_ad3 = True
        use_gurobi = True
        

        print '#######################'
        print 'Run', ind_run
        print '#######################'        

        dual_value = run_gurobi(edges, node_potentials, edge_potentials, relax=True)
        print 'Optimal dual:', dual_value
        primal_value = run_gurobi(edges, node_potentials, edge_potentials, relax=False)
        print 'Optimal primal:', primal_value
        
        dual_obj_gurobi_runs.append(dual_value)
        primal_obj_gurobi_runs.append(primal_value)
        
        if use_mplp:
            print 'Running MPLP...'
            dual_obj_seq, primal_obj_seq, p_int = \
                run_mplp(edges, node_potentials, edge_potentials, num_iterations)
            dual_obj_seq_mplp, primal_obj_seq_mplp = find_stepwise_best(dual_obj_seq, primal_obj_seq)
            print 'Best primal:', primal_obj_seq_mplp[-1]
            print 'Best dual:', dual_obj_seq_mplp[-1]
            dual_obj_mplp_runs.append(dual_obj_seq_mplp[-1])
            primal_obj_mplp_runs.append(primal_obj_seq_mplp[-1])
    
        
        if use_np:
            dual_obj_seq_np = [np.inf]
            for temperature in [0.001]: #[0.01, 0.1]:
                print 'Running NP with T =', temperature, '.'
                dual_obj_seq, primal_obj_seq, p_int = \
                    run_norm_product(edges, node_potentials, edge_potentials, num_iterations, temperature)
                dual_obj_seq, primal_obj_seq = find_stepwise_best(dual_obj_seq, primal_obj_seq)
                if dual_obj_seq[-1] < dual_obj_seq_np[-1]:
                    dual_obj_seq_np = dual_obj_seq
                    primal_obj_seq_np = primal_obj_seq
                    temperature_np = temperature
            print 'Best temperature np:', temperature_np
            print 'Best primal:', primal_obj_seq_np[-1]
            print 'Best dual:', dual_obj_seq_np[-1]
            dual_obj_np_runs.append(dual_obj_seq_np[-1])
            primal_obj_np_runs.append(primal_obj_seq_np[-1])
        
        
        if use_ad3:
            dual_obj_seq_ad3  = [np.inf]
            #for eta in [0.001, 0.01, 0.1, 1, 5.0]:
            for eta in [0.1]: #[0.1]:
                print 'Running AD3 with eta =', eta, '.'
                dual_obj_seq, primal_obj_seq, p_int = \
                    run_ad3(edges, node_potentials, edge_potentials, num_iterations, eta=eta)
                dual_obj_seq, primal_obj_seq = find_stepwise_best(dual_obj_seq, primal_obj_seq)
                if dual_obj_seq[-1] < dual_obj_seq_ad3[-1]:
                    dual_obj_seq_ad3 = dual_obj_seq
                    primal_obj_seq_ad3 = primal_obj_seq
                    eta_ad3  = eta
            print 'Best eta AD3:', eta_ad3
            print 'Best primal:', primal_obj_seq_ad3[-1]
            print 'Best dual:', dual_obj_seq_ad3[-1]
            dual_obj_ad3_runs.append(dual_obj_seq_ad3[-1])
            primal_obj_ad3_runs.append(primal_obj_seq_ad3[-1])
    
        dual_runs = np.zeros((ind_run+1, 4))
        dual_runs[:,0] = dual_obj_gurobi_runs
        dual_runs[:,1] = dual_obj_mplp_runs
        dual_runs[:,2] = dual_obj_np_runs
        dual_runs[:,3] = dual_obj_ad3_runs
        
        primal_runs = np.zeros((ind_run+1, 4))
        primal_runs[:,0] = primal_obj_gurobi_runs
        primal_runs[:,1] = primal_obj_mplp_runs
        primal_runs[:,2] = primal_obj_np_runs
        primal_runs[:,3] = primal_obj_ad3_runs

        print dual_runs
        print primal_runs
        
        systems = ['mplp', 'np', 'ad3']
        best_systems_dual = []
        best_systems_primal = []
        best_dual = np.min(dual_runs[ind_run, 1:])
        best_primal = np.max(primal_runs[ind_run, 1:])
        for j in xrange(len(systems)):
            gap = best_dual*tol
            if dual_runs[ind_run, 1+j]-gap < best_dual:
                best_systems_dual.append(systems[j])
            gap = best_primal*tol
            if primal_runs[ind_run, 1+j]+gap > best_primal:
                best_systems_primal.append(systems[j])
        print 'Dual winners:', best_systems_dual
        print 'Primal winners:', best_systems_primal
#        print dual_obj_mplp_runs, dual_obj_np_runs, dual_obj_ad3_runs, dual_obj_gurobi_runs
#        print primal_obj_mplp_runs, primal_obj_np_runs, primal_obj_ad3_runs, dual_obj_gurobi_runs
        
        

        
    
if __name__ == "__main__": 
    multiple_runs = False #True
    if len(sys.argv) == 1:
        generate_grid = False #True
        grid_size = 30
        edge_couplings = [0.1, 0.2, 0.5, 1.0]
    else:
        generate_grid = bool(int(sys.argv[1]))
        grid_size = int(sys.argv[2])
        edge_couplings = [float(val) for val in sys.argv[3:]]

    if multiple_runs:
        num_runs = 100
        for edge_coupling in edge_couplings:
            compare_several_runs(generate_grid, edge_coupling, grid_size, num_runs)
    else:
        
        #pdb.set_trace()
        plt.ion()
        fig = plt.figure()
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='sans-serif')
    
        for ind_run, edge_coupling in enumerate(edge_couplings):
            filename = 'ising_gridsize-%d_coupling-%f.fg' % (grid_size, edge_coupling)
            if generate_grid:
                node_indices, edges, node_potentials, edge_potentials = generate_ising_grid(grid_size, edge_coupling)
                save_ising(filename, edges, node_potentials, edge_potentials)
            else:
                edges, node_potentials, edge_potentials = load_ising(filename)
            
            num_iterations = 500
            
            use_mplp = True
            use_np = True
            use_accdd = True #True
            use_sdd = False #True
            use_psdd = True
            use_ad3 = True
            use_gurobi = True
            
            dual_value = run_gurobi(edges, node_potentials, edge_potentials, relax=True)
            print 'Optimal dual:', dual_value
            primal_value = run_gurobi(edges, node_potentials, edge_potentials, relax=False)
            print 'Optimal primal:', primal_value
            
            if use_mplp:
                print 'Running MPLP...'
                dual_obj_seq, primal_obj_seq, p_int = \
                    run_mplp(edges, node_potentials, edge_potentials, num_iterations)
                dual_obj_seq_mplp, primal_obj_seq_mplp = find_stepwise_best(dual_obj_seq, primal_obj_seq)
                print 'Best primal:', primal_obj_seq_mplp[-1]
                dual_obj_seq_mplp, primal_obj_seq_mplp = \
                    trim_primal_dual_sequences(dual_obj_seq_mplp, primal_obj_seq_mplp, dual_value, primal_value)
        
            
            if use_np:
                dual_obj_seq_np = [np.inf]
                for temperature in [0.001]: #[0.01, 0.1]:
                    print 'Running NP with T =', temperature, '.'
                    dual_obj_seq, primal_obj_seq, p_int = \
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
                    dual_obj_seq, primal_obj_seq, p_int = \
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
                    dual_obj_seq, primal_obj_seq, p_int = \
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
                    dual_obj_seq, primal_obj_seq, p_int = \
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
                    dual_obj_seq, primal_obj_seq, p_int = \
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
        
            plt.subplot(2, int(np.ceil(len(edge_couplings)/2.0)), ind_run+1)
     
            if use_mplp:
                plt.plot(np.arange(len(dual_obj_seq_mplp)), dual_obj_seq_mplp, 'co-', label='MPLP dual', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
            if use_np:
                plt.plot(np.arange(len(dual_obj_seq_np)), dual_obj_seq_np, 'bv-', label='Norm-Prod. dual', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
            if use_psdd:
                plt.plot(np.arange(len(dual_obj_seq_psdd)), dual_obj_seq_psdd, 'r^-', label='PSDD dual', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
            if use_sdd:
                plt.plot(np.arange(len(dual_obj_seq_sdd)), dual_obj_seq_sdd, 'y-', label='SDD dual', linewidth=2.0)
                plt.hold(True)
            if use_accdd:
                plt.plot(np.arange(len(dual_obj_seq_accdd)), dual_obj_seq_accdd, 'm*-', label='ACCDD dual', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
            if use_ad3:
                plt.plot(np.arange(len(dual_obj_seq_ad3)), dual_obj_seq_ad3, 'gs-', label='AD3 dual', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
        #    if use_gurobi:
        #        plt.plot(np.arange(num_iterations), np.tile(dual_value, num_iterations), 'k-', label='Optimal dual')
        #        plt.hold(True)
            if use_mplp:
                plt.plot(np.arange(len(primal_obj_seq_mplp)), primal_obj_seq_mplp, 'co--', label='MPLP primal', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
            if use_np:
                plt.plot(np.arange(len(primal_obj_seq_np)), primal_obj_seq_np, 'bv--', label='Norm-Prod primal', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
            if use_psdd:
                plt.plot(np.arange(len(primal_obj_seq_psdd)), primal_obj_seq_psdd, 'r^--', label='PSDD primal', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
            if use_sdd:
                plt.plot(np.arange(len(primal_obj_seq_sdd)), primal_obj_seq_sdd, 'y--', label='SDD primal', linewidth=2.0)
                plt.hold(True)
            if use_accdd:
                plt.plot(np.arange(len(primal_obj_seq_accdd)), primal_obj_seq_accdd, 'm*--', label='ACCDD primal', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
            if use_ad3:
                plt.plot(np.arange(len(primal_obj_seq_ad3)), primal_obj_seq_ad3, 'gs--', label='AD3 primal', linewidth=3.0, markevery=100, markeredgecolor='None', markersize=10)
                plt.hold(True)
        #    if use_gurobi:
        #        plt.plot(np.arange(num_iterations), np.tile(primal_value, num_iterations), 'k:', label='Optimal primal')
        #        plt.hold(True)
            
            if ind_run == 0:
                plt.legend(loc=4) #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                plt.ylabel('Objective value', fontsize=16)
                plt.xlabel('Number of iterations', fontsize=16)
                #plt.title(r'Edge coupling: $\rho=' + str(edge_coupling) + '$')
            else:
                pass
                #plt.title(r'$\rho=' + str(edge_coupling) + '$')
    
            plt.title('Edge coupling: ' + str(edge_coupling), fontsize=16)
            plt.setp(plt.gca().get_xticklabels(), fontsize=14)        
            plt.setp(plt.gca().get_yticklabels(), fontsize=14)        
    
            ymin = np.max(primal_obj_seq_ad3) - 10.0
            ymax = np.min(dual_obj_seq_ad3) + 10.0
            
            #plt.ylim((ymin, ymax))
            #plt.suptitle('Edge coupling: ' + str(edge_coupling))
            
            #pdb.set_trace()
            
            #plt.xticks(paramValues)
            #plt.grid(True)
            
        #filename = 'ising_gridsize-%d_coupling-%f.png' % (grid_size, edge_coupling)
        #fig.savefig(filename)
        
        plt.show()
        pdb.set_trace()
