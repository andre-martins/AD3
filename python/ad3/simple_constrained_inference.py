import itertools
import numpy as np
#from numpy.testing.utils import assert_array_almost_equal
import factor_graph as fg

def general_constrained_graph(unaries, edges, edge_weights, constraints, verbose=1, n_iterations=1000, eta=0.1, exact=False):
    """
    inference on a graph, taking into account logical constraints between unaries.
    
    The constraints must be a list of tuples like ( <operator>, <unaries>, <states>, <negated> )
    where:
    - operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
    - unaries is a list of the index of the unaries involved in this constraint
    - states is a list of unary states, 1 per involved unary. If the states are all the same, you can pass it directly as a scalar value.
    - negated is a list of boolean indicated if the unary must be negated. Again, if all values are the same, pass a single boolean value instead of a list 
    
    The graph is binarized as explained in Martins et al. ICML 2011 paper: "An Augmented Lagrangian Approach to Constrained MAP Inference".
    JL Meunier - 10/2016
    """
    if unaries.shape[1] != edge_weights.shape[1]:
        raise ValueError("incompatible shapes of unaries"
                         " and edge_weights.")
    if edge_weights.shape[1] != edge_weights.shape[2]:
        raise ValueError("Edge weights need to be of shape "
                         "(n_edges, n_states, n_states)!")
    if edge_weights.shape[0] != edges.shape[0]:
        raise ValueError("Number of edge weights different from number of"
                         "edges")

#     if exact:
#         raise Exception("Exact inference not supported in constrained inference for now at least.")
        
    factor_graph = fg.PFactorGraph()
    n_states = unaries.shape[-1]

    binary_variables = []   
    """
    define one binary variable Uik per possible state k of the node i.
    Uik = binary_variables[ i*n_states+k ]
    the Ith unaries is represented by [Uik for k in range(n_states)]
    """
    for u in unaries:
        lUi = []
        for state, cost in enumerate(u):
            Uik = factor_graph.create_binary_variable()
            Uik.set_log_potential(cost)
            lUi.append(Uik)
        #link these variable by a XOR factor
        factor_graph.create_factor_logic("XOR", lUi, [False]*len(lUi)) #False because they are not negated
        binary_variables.extend(lUi)

    """
    create the logical constraints
    """
    if constraints:
        for op, l_unary_i, l_state, l_negated in constraints:
            if not isinstance(l_state, list)  : l_state   = [l_state]   * len(l_unary_i) 
            if not isinstance(l_negated, list): l_negated = [l_negated] * len(l_unary_i) 
            if len(l_unary_i) != len(l_state):   raise ValueError("Number of states differs from unary index number.")
            if len(l_unary_i) != len(l_negated): raise ValueError("Number of negated differs from unary index number.")
            if max(l_state) >= n_states: raise ValueError("State should in [%d, %d]"%(0, n_states-1))
            lVar = [ binary_variables[i*n_states+k] for i,k in zip(l_unary_i, l_state) ]
            factor_graph.create_factor_logic(op, lVar, l_negated)
        
    """
    Define one Uijkl binary variable per edge i,j for each pair of state k,l 
    a) Link variable [Uijkl for all l] and not(Uik) for all k
    b) Link variable [Uijkl for all k] and not(Ujl) for all l
    """
    for ei, e in enumerate(edges):
        i,j = e
        lUij = [] # Uijkl = lUij[ k*n_states + l ]
        for k in range(n_states):
            Uik = binary_variables[ i*n_states+k ]
            lUijk = []
            for l in range(n_states):
                Uijkl = factor_graph.create_binary_variable()
                lUijk.append(Uijkl)
                cost = edge_weights[ei][k][l]
                Uijkl.set_log_potential(cost)
            
            #Let's do a)
            lUijkl_for_all_l = lUijk
            factor_graph.create_factor_logic("XOR", [Uik]+lUijkl_for_all_l, [True]+[False]*len(lUijkl_for_all_l)) 
            lUij.extend(lUijk)
            
        #now do b)
        for l in range(n_states):
            Ujl = binary_variables[ j*n_states+l ]
            Uijkl_for_all_k = [ lUij[ k*n_states + l ] for k in range(n_states)]
            factor_graph.create_factor_logic("XOR", [Ujl]+Uijkl_for_all_k, [True]+[False]*len(Uijkl_for_all_k)) 
        
        del lUij

    factor_graph.set_eta_ad3(eta)
    #factor_graph.adapt_eta_ad3(True)
    factor_graph.adapt_eta_ad3(False)
    factor_graph.set_max_iterations_ad3(n_iterations)
    factor_graph.set_verbosity(verbose)
    #factor_graph.fix_multi_variables_without_factors()  # NEEDED ???
    
    if exact:
        #NOT SUPPORTED - NO IDEA WHAT IT MEANS AND HOW DO DEAL WITH THIS
        value, marginals, edge_marginals, solver_status =\
            factor_graph.solve_exact_map_ad3()
    else:
        value, marginals, edge_marginals, solver_status =\
            factor_graph.solve_lp_map_ad3()
            
    marginals = np.array(marginals[:len(binary_variables)]).reshape(unaries.shape)

    #assert_array_almost_equal(np.sum(marginals, axis=-1), 1)
    #edge_marginals  is []  edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)
    solver_string = ["integral", "fractional", "infeasible", "unsolved"]

    return marginals, edge_marginals, value, solver_string[solver_status]
