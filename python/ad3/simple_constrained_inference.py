# -*- coding: utf-8 -*-

"""
    Inference on CRF graphs, possibly with hard-logic constraints, possibly with multiple node types

    Copyright Xerox(C) 2017 JL. Meunier

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    
    Developed  for the EU project READ. The READ project has received funding 
    from the European Union�s Horizon 2020 research and innovation programme 
    under grant agreement No 674943.
    
"""
import numpy as np
import factor_graph as fg

def general_constrained_graph(unaries, edges, edge_weights, constraints, verbose=1, n_iterations=1000, eta=0.1, exact=False, nodetype=None):
    """
    inference on a graph, taking into account logical constraints between unaries.
    
    The constraints structure differs for single- versus multi-type graphs. See in each function below.
    
    NOTE: I had to re-compile AD3 since v2.0.1 from Anaconda missed the create_binary_variable method
    
    JL Meunier - January 2017
    """
    if nodetype:
        #this must be a graph with multiple node types
        return general_constrained_graph_multitype(unaries, edges, edge_weights, constraints, verbose, n_iterations, eta, exact, nodetype)
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


    dUik_name = {}
    
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
            
            name = "var_%d_state_%d_cost_%.3f"%(len(dUik_name), state, cost)
            print name
            dUik_name[Uik] = name
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
                
                name = "edge_%d_%d->%d_cost=%.3f" %(len(dUik_name),k,l, cost)
                print name
                dUik_name[Uijkl] = name
                
            #Let's do a)
            lUijkl_for_all_l = lUijk
            factor_graph.create_factor_logic("XOR", [Uik]+lUijkl_for_all_l, [True]+[False]*len(lUijkl_for_all_l)) 
            print "XOR not_", dUik_name[Uik], [dUik_name[o] for o in lUijkl_for_all_l]
            lUij.extend(lUijk)
            
        #now do b)
        for l in range(n_states):
            Ujl = binary_variables[ j*n_states+l ]
            Uijkl_for_all_k = [ lUij[ k*n_states + l ] for k in range(n_states)]
            factor_graph.create_factor_logic("XOR", [Ujl]+Uijkl_for_all_k, [True]+[False]*len(Uijkl_for_all_k)) 
            print "XOR not_", dUik_name[Ujl], [dUik_name[o] for o in Uijkl_for_all_k]
        
        del lUij

    factor_graph.set_eta_ad3(eta)
    factor_graph.adapt_eta_ad3(True)
    #factor_graph.adapt_eta_ad3(False)   #seem to lead to 'unsolved' solver status
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
#     print " exact=", exact, "   edge_marginals =", edge_marginals
#     print "SOLVER ONE TYPE OUPUT ", "value=",value, "solver_status=",solver_status
    #print "SOLVER OUTPUT ", show((value, marginals, edge_marginals, solver_status))

    edge_marginals = np.array(marginals[len(binary_variables):]).reshape(edge_weights.shape)
    marginals = np.array(marginals[:len(binary_variables)]).reshape(unaries.shape)

    #assert_array_almost_equal(np.sum(marginals, axis=-1), 1)
    #edge_marginals  is []  edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)
    solver_string = ["integral", "fractional", "infeasible", "unsolved"]

    return marginals, edge_marginals, value, solver_string[solver_status]


#---------  MULTY-TYPE -----------------------------------------------------------------------------------------------------------------

def general_constrained_graph_multitype(unaries, edges, edge_weights, constraints, verbose=1, n_iterations=1000, eta=0.1, exact=False, nodetype=None):
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
    assert nodetype, "Internal error"
    
    l_n_nodes, l_n_states = nodetype                #number of nodes per node type, number of states per node type
    n_nodes = sum(l_n_nodes)                        #number of nodes
    n_types = len(l_n_states)                       #number of node types
    n_states = sum(l_n_states)                      #total number of states across types
    
    #CHECKING
    assert unaries.shape == (n_nodes, n_states)
    assert len(edges) == len(edge_weights)
    assert edge_weights.shape == (len(edges), n_states, n_states)
    
    #internal stuff
    #when putting weights in a single sequence, index of 1st weight for type i
    l_state_startindex_by_type = [ sum(l_n_states[:i]) for i in range(n_types+1)]   #CAUTION, we go beyond the last type
    l_node_startindex_by_type  = [ sum(l_n_nodes [:i]) for i in range(n_types+1)]   #CAUTION, we go beyond the last type

    #when  making binary variable, index of 1st variable given a type
    a_binaryvariable_startindex_by_type = np.cumsum([0]+[_n_states * _n_nodes for _n_states, _n_nodes in zip(l_n_states, l_n_nodes)]) 
    
    #an index giving the index of Ui for 1st state of node i in the list of unary_binary_variables
    a_Ui_index_by_i = np.zeros( (n_nodes,), np.uint32)
    
    factor_graph = fg.PFactorGraph()

    #make a table giving the node type
    a_binary_variable_type = np.empty((n_nodes, ), dtype=np.uint32)
    for i in range(n_types):
        a_binary_variable_type[ l_node_startindex_by_type[i]:l_node_startindex_by_type[i+1] ] = i

#     dUik_name = {}

    #table giving the index of first Uik for i
    #variables indicating the graph node states        
    unary_binary_variables = []   
    """
    define one binary variable Uik per possible state k of the node i of type T
    Uik = unary_binary_variables[ a_binaryvariable_startindex_by_type[T] + i_in_typ*typ_n_states + k ]
    faster access:
        Uik = unary_binary_variables[ a_Ui_index_by_i[ global_node_index ] + k ]

    the Ith unaries is represented by [Uik for k in range(n_states)]
    """
    i_node_type_start = 0
    for typ, (_n_nodes, _n_states) in enumerate(zip(l_n_nodes, l_n_states)):
        
        i_state_type_start = l_state_startindex_by_type[typ]
        
        #NOTE: unaries.shape == (n_nodes, n_states)
        type_unaries = unaries[ i_node_type_start:i_node_type_start+_n_nodes, i_state_type_start:i_state_type_start+_n_states ]
        
        assert len(unary_binary_variables) == a_binaryvariable_startindex_by_type[typ]
        
        for i in xrange(_n_nodes):
            lUi = list() #All binary nodes for that node of that type
            
            for state in xrange(_n_states):
                Uik = factor_graph.create_binary_variable()
                Uik.set_log_potential( type_unaries[i,state] )
                lUi.append(Uik)

#                 name = "var_%d_state_%d_cost_%.3f"%(len(dUik_name), state, type_unaries[i,state])
#                 print name
#                 dUik_name[Uik] = name                
                
            #link these variable by a XOR factor
            factor_graph.create_factor_logic("XOR", lUi, [False]*len(lUi)) #False because they are not negated
            
            a_Ui_index_by_i[i_node_type_start+i] = len(unary_binary_variables)
            
            unary_binary_variables.extend(lUi)
        i_node_type_start += _n_nodes
        
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
                if not isinstance(_l_state  , list): _l_state   = [_l_state]   * len(_l_unary_i) 
                if not isinstance(_l_negated, list): _l_negated = [_l_negated] * len(_l_unary_i) 
                if len(_l_unary_i) != len(_l_state)  : raise ValueError("Type %d: Number of states differs from unary index number."%typ)
                if len(_l_unary_i) != len(_l_negated): raise ValueError("type %d: Number of negated differs from unary index number."%typ)
                if not _l_unary_i: continue
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
    for ei, e in enumerate(edges):
        i,j = e
        typ_i, typ_j = a_binary_variable_type[i], a_binary_variable_type[j]
        start_typ_i, start_typ_j = l_state_startindex_by_type[typ_i], l_state_startindex_by_type[typ_j]
        n_states_i, n_states_j = l_n_states[typ_i], l_n_states[typ_j]
        #NOTE: edge_weights.shape == (len(edges), n_states, n_states)
        edge_weights_ei = edge_weights[ei]
        
        lUij = [] # list all binary nodes reflecting the edge between node i and node j, for all possible pairs of states
        lUijl_by_l = [list() for _ in xrange(n_states_j)]  # Uij for all l
        #Uijkl = lUij[ k*n_states + l ]
        for k in range(n_states_i):
            Uik = unary_binary_variables[ a_Ui_index_by_i[i] + k ]
            lUijk = []
            for l in range(n_states_j):
                Uijkl = factor_graph.create_binary_variable()
                lUijk.append(Uijkl)
                lUijl_by_l[l].append(Uijkl)     #lUijl_by_l[l] is Uijkl for all k
                cost = edge_weights_ei[start_typ_i+k, start_typ_j+l]
                Uijkl.set_log_potential(cost)

#                 name = "edge_%d_%d->%d_cost=%.3f" %(len(dUik_name),k,l, cost)
#                 print name
#                 dUik_name[Uijkl] = name            

            #Let's do a)        lUijk is "Uijkl for all l"
            factor_graph.create_factor_logic("XOR", [Uik]+lUijk, [True]+[False]*len(lUijk)) 
            lUij.extend(lUijk)
#             print "XOR not_", dUik_name[Uik], [dUik_name[o] for o in lUijk]
            
        
        #now do b)
        for l in range(n_states_j):
            #Ujl = unary_binary_variables[ a_binaryvariable_startindex_by_type[typ_j] + j*n_states_j + l ]
            Ujl = unary_binary_variables[ a_Ui_index_by_i[j] + l ]
            Uijkl_for_all_k = [ lUij[ k*n_states_j + l ] for k in range(n_states_i) ]
            factor_graph.create_factor_logic("XOR", [Ujl]+Uijkl_for_all_k, [True]+[False]*len(Uijkl_for_all_k)) 
#             print "XOR not_", dUik_name[Ujl], [dUik_name[o] for o in Uijkl_for_all_k]
        
        del lUij, lUijl_by_l 

    factor_graph.set_eta_ad3(eta)
    factor_graph.adapt_eta_ad3(True)
    factor_graph.set_max_iterations_ad3(n_iterations)
    factor_graph.set_verbosity(verbose)
    #factor_graph.fix_multi_variables_without_factors()  # NEEDED ???
    
    if exact:
        #NOT SUPPORTED - NO IDEA WHAT IT MEANS AND HOW DO DEAL WITH THIS
        assert False
        value, marginals, edge_marginals, solver_status =\
            factor_graph.solve_exact_map_ad3()
    else:
        value, marginals, edge_marginals, solver_status =\
            factor_graph.solve_lp_map_ad3()
#     print " exact=", exact, "   edge_marginals =", edge_marginals
#     print "SOLVER MULTITYPE OUPUT ", "value=",value, "solver_status=",solver_status
    
#     print "SOLVER OUTPUT ", show((value, marginals, edge_marginals, solver_status))
    
    #put back the values of the binary variables into the marginals
    ret_marginals = np.zeros(unaries.shape, dtype=unaries.dtype)    #this will be the returned node marginals
    aux_marginals = np.asarray(marginals)
    
    #first the node marginals
    #NOTE: unaries.shape == (n_nodes, n_states)
    i,j,k = 0,0,0
    for (_n_nodes, _n_states) in zip(l_n_nodes, l_n_states):  #iteration by type
        _n_binaries = _n_nodes * _n_states
        ret_marginals[ i:i+_n_nodes , j:j+_n_states ] = aux_marginals[ k:k+_n_binaries ].reshape(_n_nodes, _n_states)
        i += _n_nodes
        j += _n_states
        k += _n_binaries

    aux_marginals = marginals[len(unary_binary_variables):]
    
    ret_edge_marginals = np.zeros((edge_weights.shape[0], n_states**2), dtype=unaries.dtype)   #returned edge marginals

    if False:
        #when putting edge states in a single sequence, index of 1st state of an edge of type (typ1, typ2)
        l_edgetype_start_index = [] 
        i_start = 0
        for typ1_n_states in l_n_states:
            for typ2_n_states in l_n_states:
                l_edgetype_start_index.append(i_start)
                i_start += typ1_n_states*typ2_n_states 
        l_edgetype_start_index.append(i_start)
        assert i_start == n_states**2
          
      
        for ei, e in enumerate(edges):  #iteration over edges, as when creating the bianry variables reflecting each edge
            i,j = e
            typ_i, typ_j = a_binary_variable_type[i], a_binary_variable_type[j]
            n_states_i, n_states_j = l_n_states[typ_i], l_n_states[typ_j]
            i_start = l_edgetype_start_index[typ_i*n_states+typ_j]
             
            ret_edge_marginals_ei = ret_edge_marginals[ei]
            for k in range(n_states_i):
                for l in range(n_states_j):
                    ret_edge_marginals_ei[i_start+k*n_states_j+l] = aux_marginals.pop(0)
    else:
        #when putting edge states in a single sequence, index of 1st state of an edge of type (typ1, typ2)
        a_startindex_by_typ_typ = np.zeros((n_types, n_types), dtype=np.int32)
        i_state_start = 0
        for typ1, typ1_n_states in enumerate(l_n_states):
            for typ2, typ2_n_states in enumerate(l_n_states):
                a_startindex_by_typ_typ[typ1,typ2] = i_state_start
                i_state_start += typ1_n_states*typ2_n_states 
                    
        i_start = 0
        for ei, e in enumerate(edges):  #iteration over edges, as when creating the bianry variables reflecting each edge
            i,j = e
            typ_i, typ_j = a_binary_variable_type[i], a_binary_variable_type[j]
            n = l_n_states[typ_i] * l_n_states[typ_j]
            i_stop = i_start + n
            i_state_start = a_startindex_by_typ_typ[typ_i,typ_j]
            ret_edge_marginals[ei, i_state_start:i_state_start+n] = aux_marginals[i_start:i_stop]
            i_start = i_stop
     
    #assert_array_almost_equal(np.sum(marginals, axis=-1), 1)
    #edge_marginals  is []  edge_marginals = np.array(edge_marginals).reshape(-1, n_states ** 2)
    solver_string = ["integral", "fractional", "infeasible", "unsolved"]

    return ret_marginals, ret_edge_marginals, value, solver_string[solver_status]

# def getY_from_typedmarginals(unary_marginals, nodetypedata):
#     """
#     we use argmax, but need to use the class index by type of node, not the global one
#     """
#     l_n_nodes, l_n_states = nodetypedata                #number of nodes per node type, number of states per node type
# 
#     Y_pred = np.argmax(unary_marginals, axis=-1)
#     cum_n_states,k = 0,0
#     for (n_nodes, n_states) in zip(l_n_nodes, l_n_states):
#         Y_pred[ k:k+n_nodes ] -= cum_n_states
#         cum_n_states += n_states
#         k += n_nodes
#     return Y_pred
