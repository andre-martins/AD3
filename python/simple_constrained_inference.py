# -*- coding: utf-8 -*-

"""
    Inference on CRF graphs, possibly with hard-logic constraints, possibly with multiple node types

    Copyright Xerox(C) 2017 JL. Meunier

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union's Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np
import factor_graph as fg

def general_constrained_graph(unaries, edges, edge_weights, constraints, verbose=1, n_iterations=1000, eta=0.1, exact=False):
    """
    inference on a graph, taking into account logical constraints between unaries.
    
    The constraints structure differs for single- versus multi-type graphs. See in each function below.
    
    NOTE: I had to re-compile AD3 since v2.0.1 from Anaconda missed the create_binary_variable method
    
    JL Meunier - January 2017
    """
    if isinstance(unaries, list):
        #this must be a graph with multiple node types
        return general_constrained_graph_multitype(unaries, edges, edge_weights, constraints, verbose, n_iterations, eta, exact)
    else:
        return general_constrained_graph_singletype(unaries, edges, edge_weights, constraints, verbose, n_iterations, eta, exact)

#---------  SINGLE-TYPE -----------------------------------------------------------------------------------------------------------------

def general_constrained_graph_singletype(unaries, edges, edge_weights, constraints, verbose=1, n_iterations=1000, eta=0.1, exact=False):
    """
    inference on a graph, with one type of node, taking into account logical constraints between unaries.

    The constraints must be a list of tuples like ( <operator>, <unaries>, <states>, <negated> )
    The tuple is defined differently for single- and multi-type inference. See in each function below.
    
    where:
    - operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
    - unaries is a list of the index of the unaries involved in this constraint
    - states is a list of unary states, 1 per involved unary. If the states are all the same, you can pass it directly as a scalar value.
    - negated is a list of boolean indicated if the unary must be negated. Again, if all values are the same, pass a single boolean value instead of a list 
    
    The graph is binarized as explained in Martins et al. ICML 2011 paper: "An Augmented Lagrangian Approach to Constrained MAP Inference".
    
    NOTE: I had to re-compile AD3 since v2.0.1 from Anaconda missed the create_binary_variable method
    
    JL Meunier - October 2016
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
        for _, cost in enumerate(u):
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
        #this is a trick to force the graph binarization
        if constraints != True:
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
        edge_weights_ei = edge_weights[ei]
        for k in range(n_states):
            Uik = binary_variables[ i*n_states+k ]
            lUijk = []
            for l in range(n_states):
                Uijkl = factor_graph.create_binary_variable()
                lUijk.append(Uijkl)
                cost = edge_weights_ei[k,l]
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

    value, marginals, edge_marginals, solver_status = factor_graph.solve(
        eta=eta,
        adapt=True,
        max_iter=n_iterations,
        branch_and_bound=exact,
        verbose=verbose)

    edge_marginals = np.array(marginals[len(binary_variables):]).reshape(edge_weights.shape)
    marginals = np.array(marginals[:len(binary_variables)]).reshape(unaries.shape)

    #assert_array_almost_equal(np.sum(marginals, axis=-1), 1)
    #edge_marginals  is []  edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)

    return marginals, edge_marginals, value, solver_status


#---------  MULTY-TYPE -----------------------------------------------------------------------------------------------------------------

def general_constrained_graph_multitype(l_unaries, l_edges, l_edge_weights, constraints, verbose=1, n_iterations=1000, eta=0.1, exact=False):
    """
    inference on a graph with multiple node types, taking into account logical constraints between unaries.
    
    The constraints must be a list of tuples like ( <operator>, <l_unaries>, <l_states>, <l_negated> )
                                               or ( <operator>, <l_unaries>, <l_states>, <l_negated> , (type, unary, state, negated))
    
    
    where:
    - operator is one of 'XOR' 'XOROUT' 'ATMOSTONE' 'OR' 'OROUT' 'ANDOUT' 'IMPLY'
    
    
    - l_unaries is a list of unaries per type. Each item is a list of the index of the unaries of that type involved in this constraint
    - l_states is a list of states per type. Each item is a list of the state of the involved unaries. 
            If the states are all the same for a type, you can pass it directly as a scalar value.
    - l_negated is a list of "negated" per type. Each item is a list of booleans indicating if the unary must be negated. 
            Again, if all values are the same for a type, pass a single boolean value instead of a list 
    - the optional (type, unary, state, negated) allows to refer to a certain unary of a certain type in a certain state, possibly negated.
      This is the final node of the logic operator, which can be key for instance for XOROUT, OROUT, ANDOUT, IMPLY operators. 
          (Because the other terms of the operator are grouped and ordered by type) 
    
    The graph is binarized as explained in a publication to come.
    
    JL Meunier - January 2017
    """
    l_n_nodes, l_n_states = zip(*[unary.shape for unary in l_unaries])  #number of nodes and of states per type
#     n_types = len(l_unaries)                       #number of node types
#     n_nodes = sum(l_n_nodes)                        #number of nodes
#     n_states = sum(l_n_states)                      #total number of states across types
#     n_edges = sum( edges.shape[0] for edges in l_edges)
    
    #BASIC CHECKING
    assert len(l_unaries)**2 == len(l_edges)
    assert len(l_edges) == len(l_edge_weights)
    
    #when  making binary variable, index of 1st variable given a type
    a_binaryvariable_startindex_by_type = np.cumsum([0]+[_n_states * _n_nodes for _n_states, _n_nodes in zip(l_n_states, l_n_nodes)]) 
    
    factor_graph = fg.PFactorGraph()

    #table giving the index of first Uik for i
    #variables indicating the graph node states        
    unary_binary_variables = []   
    """
    define one binary variable Uik per possible state k of the node i of type T
    Uik = unary_binary_variables[ a_binaryvariable_startindex_by_type[T] + i_in_typ*typ_n_states + k ]

    the Ith unaries is represented by [Uik for k in range(n_states)]
    """
    for _n_nodes, _n_states, type_unaries in zip(l_n_nodes, l_n_states, l_unaries):
        assert type_unaries.shape == (_n_nodes, _n_states)
        for i in xrange(_n_nodes):
            lUi = list() #All binary nodes for that node of that type
            
            for state in xrange(_n_states):
                Uik = factor_graph.create_binary_variable()
                Uik.set_log_potential( type_unaries[i,state] )
                lUi.append(Uik)

            #link these variable by a XOR factor
            factor_graph.create_factor_logic("XOR", lUi, [False]*len(lUi)) #False because they are not negated
            
            unary_binary_variables.extend(lUi)

    """
    create the logical constraints
    """
    if constraints:
        for tup in constraints:
            try:
                op, l_l_unary_i, l_l_state, l_l_negated = tup
                last_type = None
            except ValueError:
                op, l_l_unary_i, l_l_state, l_l_negated, (last_type, last_unary, last_state, last_neg) = tup
                
            lVar = list()      #listing the implied unaries
            lNegated = list()  #we will flatten it from the per type information
            for typ, (_l_unary_i, _l_state, _l_negated) in enumerate(zip(l_l_unary_i, l_l_state, l_l_negated)):
                if not _l_unary_i: continue
                if not isinstance(_l_state  , list): _l_state   = [_l_state]   * len(_l_unary_i) 
                if not isinstance(_l_negated, list): _l_negated = [_l_negated] * len(_l_unary_i) 
                if len(_l_unary_i) != len(_l_state)  : raise ValueError("Type %d: Number of states differs from unary index number."%typ)
                if len(_l_unary_i) != len(_l_negated): raise ValueError("type %d: Number of negated differs from unary index number."%typ)
                typ_n_states = l_n_states[typ]
                if max(_l_state) >= typ_n_states    : raise ValueError("Type %d: State should in [%d, %d]"%(typ, 0, typ_n_states-1))
                start_type_index = a_binaryvariable_startindex_by_type[typ]
                
                lTypeVar = [ unary_binary_variables[start_type_index + i*typ_n_states + k] for i,k in zip(_l_unary_i, _l_state) ]
                lVar    .extend(lTypeVar)
                lNegated.extend(_l_negated)
            
            if last_type:
                typ_n_states = l_n_states[last_type]
                if last_state >= typ_n_states    : raise ValueError("(last) Type %d: State should in [%d, %d]"%(typ, 0, typ_n_states-1))
                start_type_index = a_binaryvariable_startindex_by_type[last_type]
                u = unary_binary_variables[start_type_index + last_unary*typ_n_states + last_state]
                lVar    .append(u)
                lNegated.append(last_neg)
                
            factor_graph.create_factor_logic(op, lVar, lNegated)
        
    """
    Define one Uijkl binary variable per edge i,j for each pair of state k,l 
    a) Link variable [Uijkl for all l] and not(Uik) for all k
    b) Link variable [Uijkl for all k] and not(Ujl) for all l
    """
    i_typ_typ = 0
    for typ_i, n_states_i in enumerate(l_n_states):
        for typ_j, n_states_j in enumerate(l_n_states):
            edges        = l_edges       [i_typ_typ] #pop would modify the list..
            edge_weights = l_edge_weights[i_typ_typ] #pop would modify the list..
            i_typ_typ += 1
            
            if len(edges) or len(edge_weights): assert edge_weights.shape[1:] == (n_states_i, n_states_j)
            for e, cost in zip(edges, edge_weights):
                i,j = e

                #NOTE: Uik = unary_binary_variables[ a_binaryvariable_startindex_by_type[T] + i_in_typ*typ_n_states + k ]
                index_Ui_base = a_binaryvariable_startindex_by_type[typ_i] + i*n_states_i
                index_Uj_base = a_binaryvariable_startindex_by_type[typ_j] + j*n_states_j
                
                lUij = [] # list all binary nodes reflecting the edge between node i and node j, for all possible pairs of states
                lUijl_by_l = [list() for _ in xrange(n_states_j)]  # Uij for all l
                for k in range(n_states_i):
                    cost_k = cost[k]
                    Uik = unary_binary_variables[ index_Ui_base + k ]
                    lUijk = []
                    for l in range(n_states_j):
                        Uijkl = factor_graph.create_binary_variable()
                        lUijk.append(Uijkl)
                        lUijl_by_l[l].append(Uijkl)     #lUijl_by_l[l] is Uijkl for all k
                        Uijkl.set_log_potential(cost_k[l])
        
                    #Let's do a)        lUijk is "Uijkl for all l"
                    factor_graph.create_factor_logic("XOR", [Uik]+lUijk, [True]+[False]*len(lUijk)) 
                    lUij.extend(lUijk)
        #             print "XOR not_", dUik_name[Uik], [dUik_name[o] for o in lUijk]
                    
                
                #now do b)
                for l in range(n_states_j):
                    Ujl = unary_binary_variables[ index_Uj_base + l ]
                    Uijkl_for_all_k = [ lUij[ k*n_states_j + l ] for k in range(n_states_i) ]
                    factor_graph.create_factor_logic("XOR", [Ujl]+Uijkl_for_all_k, [True]+[False]*len(Uijkl_for_all_k)) 
        #             print "XOR not_", dUik_name[Ujl], [dUik_name[o] for o in Uijkl_for_all_k]
                
                del lUij, lUijl_by_l             
            
    value, marginals, edge_marginals, solver_status = factor_graph.solve(
        eta=eta,
        adapt=True,
        max_iter=n_iterations,
        branch_and_bound=exact,
        verbose=verbose)
            
    #put back the values of the binary variables into the marginals
    aux_marginals = np.asarray(marginals)

    #THE NODE MARGINALS
    ret_node_marginals = list()
    k = 0
    for (_n_nodes, _n_states) in zip(l_n_nodes, l_n_states):  #iteration by type
        _n_binaries = _n_nodes * _n_states
        ret_node_marginals.append( aux_marginals[ k:k+_n_binaries ].reshape(_n_nodes, _n_states) )
        k += _n_binaries
    #assert k == len(unary_binary_variables)
    
    #NOW THE EDGE MARGINALS
    aux_marginals = aux_marginals[len(unary_binary_variables):]

    ret_edge_marginals = list()

    i_typ_typ = 0
    i_marg_start = 0
    for typ_i, n_states_i in enumerate(l_n_states):
        for typ_j, n_states_j in enumerate(l_n_states):
            edges = l_edges[i_typ_typ] #pop would modify the list..
            i_typ_typ += 1
            
            _n_edges = len(edges)
            _n_edge_states = n_states_i * n_states_j
            _n_marg = _n_edges * _n_edge_states
            
            if _n_edges:
                ret_edge_marginals.append( aux_marginals[ i_marg_start : i_marg_start + _n_marg ].reshape((_n_edges, _n_edge_states)) )
            else:
                ret_edge_marginals.append( np.zeros( (0, _n_edge_states) ) )
            i_marg_start  += _n_marg
    #assert i_marg_start == len(aux_marginals)
    
    return ret_node_marginals, ret_edge_marginals, value, solver_status

