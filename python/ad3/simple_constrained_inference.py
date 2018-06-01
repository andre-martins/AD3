# -*- coding: utf-8 -*-

"""
Inference on constrained, multi-type factor graphs.

January 2017 JL. Meunier

Developed  for the EU project READ. The READ project has received funding
from the European Union's Horizon 2020 research and innovation programme
under grant agreement No 674943.
"""

import numpy as np
from . import factor_graph as fg


def general_constrained_graph(unaries, edges, edge_weights, constraints,
                              verbose=1, n_iterations=1000, eta=0.1,
                              exact=False):
    """Inference on a general graph, with logical constraints.

    This graph can take one of two forms:
    - simple form, where all nodes share the same set of L labels.
    - typed form, where each node is typed and the node type t defines the set
    of Lt labels that the node can take.

    The input parameters differ according to the form of graph.

    Parameters
    ----------
    unaries : array or list of arrays
        `unaries` gives the unary potentials of the nodes of the graph.
        In the simple form, for N nodes, this is a N x L array.
        In the typed form, for T types, this is a list of Nt x Lt arrays, where
        Nt is the number of nodes of type t, and Lt is the number of labels of
        type t.
    edges : 2-column array or list of 2-columns arrays
        `edges` defines the edges of the graph.
        In the simple form, for E edges, this is a E x 2 array, giving the
        source node index and target node index of each edge of the graph.
        In the typed form, this is a list of length T^2 of 2-column array, each
        such array defining all the Et1t2 edges from a source node of type t1
        to a target node of type t2, for t1 in [1..T] and t2 in [1..T].
    edge_weights : array of list of arrays
        `edge_weights` gives the pairwise potentials of the pairs of nodes
        linked by an edge in the graph.
        In the simple form, for E edges, it is a E x L x L array.
        In the typed form, it is a list of Et1t2 x Lt1 x Lt2 array, for t1 in
        [1..T] and t2 in [1..T].
    constraints : list of tuples
        In the simple form, `constraints` is a list of tuples like:
                ( <operator>, <nodes>, <states>, <negated> )
        where:
        - `operator` is one of the strings: XOR XOROUT ATMOSTONE OR OROUT ANDOUT
            IMPLY
        - `nodes` is a list of the index of the nodes involved in this
            constraint
        - states is a list of states (int), 1 per involved node.
            If the states are all the same for all the involved node, you can
            pass it directly as a scalar value.
        - negated is a list of boolean indicated if the node state is be
            negated. Again, if all values are the same, pass a single boolean
            value instead of a list
        In the typed form, `constraints` is a list of tuples like:
               ( <operator>, <l_nodes>, <l_states>, <l_negated> )
            or ( <operator>, <l_nodes>, <l_states>, <l_negated> ,
                 (type, node, state, negated))
        where:
        - operator is one of the strings XOR XOROUT ATMOSTONE OR OROUT ANDOUT
            IMPLY
        - l_nodes is a list of nodes per type. Each item is a list of the index
            of the nodes of that type involved in this constraint
        - l_states is a list of states per type. Each item is a list of the state
            of the involved nodes. If the states are all the same for a type,
            you can pass it directly as a scalar value.
        - l_negated is a list of "negated" per type. Each item is a list of
            booleans indicating if the unary must be negated.
            Again, if all values are the same for a type, pass a single boolean
            value instead of a list
        - the optional (type, node, state, negated) allows to refer to a certain
            node of a certain type in a certain state, possibly negated.
            This is the final node of the logic operator, which can be key for
            instance for XOROUT, OROUT, ANDOUT, IMPLY operators.
            (Because the other terms of the operator are grouped and ordered by
            type)
        `constraints` differs for simple form and typed form of graphs.
    verbose : AD3 verbosity level
    n_iterations : AD3 number of iterations
    eta : AD3 eta
    exact : AD3 type of inference

    Returns
    -------
    marginals : array
        Marginals for all nodes of the graph, in same order as the `unaries`,
        after flattening.
    edge_marginals : array
        Marginals for all edges of the graph, in same order as the
        `edge_weights`, after flattening.
    value : float
        Graph energy.
    solver_status : str
        status of the solver.
    """
    if isinstance(unaries, list):
        # this must be a graph with multiple node types
        return general_constrained_graph_multitype(unaries, edges,
                                                   edge_weights, constraints,
                                                   verbose, n_iterations, eta,
                                                   exact)
    else:
        return general_constrained_graph_singletype(unaries, edges,
                                                    edge_weights, constraints,
                                                    verbose, n_iterations, eta,
                                                    exact)


def general_constrained_graph_singletype(unaries, edges, edge_weights,
                                         constraints, verbose=1,
                                         n_iterations=1000, eta=0.1,
                                         exact=False):
    """Inference on a general constrained graph with a single node type.

    The graph is binarized as explained in Martins et al. ICML 2011
    paper: "An Augmented Lagrangian Approach to Constrained MAP Inference".
    See also Meunier, CAp 2017, "Joint Structured Learning and Predictions
    under Logical Constraints in Conditional Random Fields"

    This graph can take one of two forms:
    - simple form, where all nodes share the same set of L labels.
    - typed form, where each node is typed and the node type t defines the set
    of Lt labels that the node can take.

    The input parameters differ according to the form of graph.

    Parameters
    ----------
    unaries : array or list of arrays
        `unaries` gives the unary potentials of the nodes of the graph.
        In the simple form, for N nodes, this is a N x L array.
        In the typed form, for T types, this is a list of Nt x Lt arrays, where
        Nt is the number of nodes of type t, and Lt is the number of labels of
        type t.
    edges : 2-column array or list of 2-columns arrays
        `edges` defines the edges of the graph.
        In the simple form, for E edges, this is a E x 2 array, giving the
        source node index and target node index of each edge of the graph.
        In the typed form, this is a list of length T^2 of 2-column array, each
        such array defining all the Et1t2 edges from a source node of type t1
        to a target node of type t2, for t1 in [1..T] and t2 in [1..T].
    edge_weights : array of list of arrays
        `edge_weights` gives the pairwise potentials of the pairs of nodes
        linked by an edge in the graph.
        In the simple form, for E edges, it is a E x L x L array.
        In the typed form, it is a list of Et1t2 x Lt1 x Lt2 array, for t1 in
        [1..T] and t2 in [1..T].
    constraints : list of tuples
        `constraints` is a list of tuples like:
                ( <operator>, <nodes>, <states>, <negated> )
        where:
        - `operator` is one of the strings: XOR XOROUT ATMOSTONE OR OROUT ANDOUT
            IMPLY
        - `nodes` is a list of the index of the nodes involved in this
            constraint
        - states is a list of states (int), 1 per involved node.
            If the states are all the same for all the involved node, you can
            pass it directly as a scalar value.
        - negated is a list of boolean indicated if the node state is be
            negated. Again, if all values are the same, pass a single boolean
            value instead of a list
    verbose : AD3 verbosity level
    n_iterations : AD3 number of iterations
    eta : AD3 eta
    exact : AD3 type of inference

    Returns
    -------
    marginals : array
        Marginals for all nodes of the graph, in same order as the `unaries`,
        after flattening.
    edge_marginals : array
        Marginals for all edges of the graph, in same order as the
        `edge_weights`, after flattening.
    value : float
        Graph energy.
    solver_status : str
        status of the solver.
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

    factor_graph = fg.PFactorGraph()
    n_states = unaries.shape[-1]

    binary_variables = []

    #  define one binary variable Uik per possible state k of the node i.
    #  Uik = binary_variables[ i*n_states+k ]
    #  the Ith unaries is represented by [Uik for k in range(n_states)]

    for u in unaries:
        lUi = []
        for _, cost in enumerate(u):
            Uik = factor_graph.create_binary_variable()
            Uik.set_log_potential(cost)
            lUi.append(Uik)

        # link these variable by a XOR factor
        # (False because they are not negated)
        factor_graph.create_factor_logic("XOR", lUi, [False]*len(lUi))
        binary_variables.extend(lUi)

    #  create the logical constraints
    if constraints:
        # this is a trick to force the graph binarization
        if constraints is not True:
            for op, l_unary_i, l_state, l_negated in constraints:
                if not isinstance(l_state, list):
                    l_state = [l_state] * len(l_unary_i)
                if not isinstance(l_negated, list):
                    l_negated = [l_negated] * len(l_unary_i)
                if len(l_unary_i) != len(l_state):
                    raise ValueError("Number of states differs from unary"
                                     " index number.")
                if len(l_unary_i) != len(l_negated):
                    raise ValueError("Number of negated differs from unary"
                                     " index number.")
                if max(l_state) >= n_states:
                    raise ValueError("State should in [%d, %d]"
                                     % (0, n_states-1))
                lVar = [binary_variables[i*n_states+k]
                        for i, k in zip(l_unary_i, l_state)]
                factor_graph.create_factor_logic(op, lVar, l_negated)

    #  Define one Uijkl binary variable per edge i,j for each pair of state k,l
    #  a) Link variable [Uijkl for all l] and not(Uik) for all k
    #  b) Link variable [Uijkl for all k] and not(Ujl) for all l
    for ei, e in enumerate(edges):
        i, j = e
        lUij = []  # Uijkl = lUij[ k*n_states + l ]
        edge_weights_ei = edge_weights[ei]
        for k in range(n_states):
            Uik = binary_variables[i*n_states+k]
            lUijk = []
            for l in range(n_states):
                Uijkl = factor_graph.create_binary_variable()
                lUijk.append(Uijkl)
                cost = edge_weights_ei[k, l]
                Uijkl.set_log_potential(cost)

            # Let's do a)
            lUijkl_for_all_l = lUijk
            factor_graph.create_factor_logic("XOR",
                                             [Uik] + lUijkl_for_all_l,
                                             [True] +
                                             [False] * len(lUijkl_for_all_l)
                                             )
            lUij.extend(lUijk)

        # now do b)
        for l in range(n_states):
            Ujl = binary_variables[j*n_states+l]
            Uijkl_for_all_k = [lUij[k*n_states + l] for k in range(n_states)]
            factor_graph.create_factor_logic("XOR",
                                             [Ujl] + Uijkl_for_all_k,
                                             [True] +
                                             [False] * len(Uijkl_for_all_k)
                                             )

        del lUij

    value, marginals, edge_marginals, solver_status = factor_graph.solve(
        eta=eta,
        adapt=True,
        max_iter=n_iterations,
        branch_and_bound=exact,
        verbose=verbose)

    edge_marginals = np.array(marginals[len(binary_variables):])
    edge_marginals = edge_marginals.reshape(edge_weights.shape)
    marginals = np.array(marginals[:len(binary_variables)])
    marginals = marginals.reshape(unaries.shape)

    # assert_array_almost_equal(np.sum(marginals, axis=-1), 1)
    # edge_marginals  is []  edge_marginals =
    #     np.array(edge_marginals).reshape(-1, n_states ** 2)

    return marginals, edge_marginals, value, solver_status


def general_constrained_graph_multitype(l_unaries, l_edges, l_edge_weights,
                                        constraints, verbose=1,
                                        n_iterations=1000, eta=0.1,
                                        exact=False):
    """Inference on a general constrained graph with multiple node types.

    The graph is binarized as explained in Martins et al. ICML 2011
    paper: "An Augmented Lagrangian Approach to Constrained MAP Inference".
    See also Meunier, CAp 2017, "Joint Structured Learning and Predictions
    under Logical Constraints in Conditional Random Fields"

    This graph can take one of two forms:
    - simple form, where all nodes share the same set of L labels.
    - typed form, where each node is typed and the node type t defines the set
    of Lt labels that the node can take.

    The input parameters differ according to the form of graph.

    Parameters
    ----------
    unaries : array or list of arrays
        `unaries` gives the unary potentials of the nodes of the graph.
        In the simple form, for N nodes, this is a N x L array.
        In the typed form, for T types, this is a list of Nt x Lt arrays, where
        Nt is the number of nodes of type t, and Lt is the number of labels of
        type t.
    edges : 2-column array or list of 2-columns arrays
        `edges` defines the edges of the graph.
        In the simple form, for E edges, this is a E x 2 array, giving the
        source node index and target node index of each edge of the graph.
        In the typed form, this is a list of length T^2 of 2-column array, each
        such array defining all the Et1t2 edges from a source node of type t1
        to a target node of type t2, for t1 in [1..T] and t2 in [1..T].
    edge_weights : array of list of arrays
        `edge_weights` gives the pairwise potentials of the pairs of nodes
        linked by an edge in the graph.
        In the simple form, for E edges, it is a E x L x L array.
        In the typed form, it is a list of Et1t2 x Lt1 x Lt2 array, for t1 in
        [1..T] and t2 in [1..T].
    constraints : list of tuples
        The constraints is a list of tuples like:
           ( <operator>, <l_nodes>, <l_states>, <l_negated> )
        or ( <operator>, <l_nodes>, <l_states>, <l_negated> ,
             (type, node, state, negated))
        where:
        - operator is one of the strings XOR XOROUT ATMOSTONE OR OROUT ANDOUT
            IMPLY
        - l_nodes is a list of nodes per type. Each item is a list of the index
            of the nodes of that type involved in this constraint
        - l_states is a list of states per type. Each item is a list of the state
            of the involved nodes. If the states are all the same for a type,
            you can pass it directly as a scalar value.
        - l_negated is a list of "negated" per type. Each item is a list of
            booleans indicating if the unary must be negated.
            Again, if all values are the same for a type, pass a single boolean
            value instead of a list
        - the optional (type, node, state, negated) allows to refer to a certain
            node of a certain type in a certain state, possibly negated.
            This is the final node of the logic operator, which can be key for
            instance for XOROUT, OROUT, ANDOUT, IMPLY operators.
            (Because the other terms of the operator are grouped and ordered by
            type)
    verbose : AD3 verbosity level
    n_iterations : AD3 number of iterations
    eta : AD3 eta
    exact : AD3 type of inference

    Returns
    -------
    marginals : array
        Marginals for all nodes of the graph, in same order as the `unaries`,
        after flattening.
    edge_marginals : array
        Marginals for all edges of the graph, in same order as the
        `edge_weights`, after flattening.
    value : float
        Graph energy.
    solver_status : str
        status of the solver.
    """

    # number of nodes and of states per type
    l_n_nodes, l_n_states = zip(*[unary.shape for unary in l_unaries])

    #     n_types = len(l_unaries)   #number of node types
    #     n_nodes = sum(l_n_nodes)   #number of nodes
    #     n_states = sum(l_n_states) #total number of states across types
    #     n_edges = sum( edges.shape[0] for edges in l_edges)

    # BASIC CHECKING
    assert len(l_unaries)**2 == len(l_edges)
    assert len(l_edges) == len(l_edge_weights)

    # when  making binary variable, index of 1st variable given a type
    #Before PEP8 its name was: a_binaryvariable_startindex_by_type
    a_by_type = np.cumsum([0] +[_n_states * _n_nodes
                               for _n_states, _n_nodes
                               in zip(l_n_states, l_n_nodes)])

    factor_graph = fg.PFactorGraph()

    # table giving the index of first Uik for i
    # variables indicating the graph node states
    unary_binary_variables = []
    #  define one binary variable Uik per possible state k of the node i of type T
    #  Uik = unary_binary_variables[ a_by_type[T] + i_in_typ*typ_n_states + k ]

    #  the Ith unaries is represented by [Uik for k in range(n_states)]
    for _n_nodes, _n_states, type_unaries in zip(l_n_nodes, l_n_states,
                                                 l_unaries):
        assert type_unaries.shape == (_n_nodes, _n_states)
        for i in range(_n_nodes):
            lUi = list()  # All binary nodes for that node of that type

            for state in range(_n_states):
                Uik = factor_graph.create_binary_variable()
                Uik.set_log_potential(type_unaries[i, state])
                lUi.append(Uik)

            # link these variable by a XOR factor
            # (False because they are not negated)
            factor_graph.create_factor_logic("XOR", lUi, [False]*len(lUi))

            unary_binary_variables.extend(lUi)

    #  create the logical constraints
    if constraints:
        for tup in constraints:
            try:
                op, l_l_unary_i, l_l_state, l_l_negated = tup
                last_type = None
            except ValueError:
                (op, l_l_unary_i, l_l_state, l_l_negated,
                 (last_type, last_unary, last_state, last_neg)) = tup

            lVar = list()      # listing the implied unaries
            lNegated = list()  # we flatten it from the per type information
            for typ, (_l_unary_i,
                      _l_state,
                      _l_negated) in enumerate(zip(l_l_unary_i,
                                                   l_l_state,
                                                   l_l_negated)):
                if not _l_unary_i:
                    continue
                if not isinstance(_l_state, list):
                    _l_state = [_l_state] * len(_l_unary_i)
                if not isinstance(_l_negated, list):
                    _l_negated = [_l_negated] * len(_l_unary_i)
                if len(_l_unary_i) != len(_l_state):
                    raise ValueError("Type %d: Number of states differs"
                                     " from unary index number." % typ)
                if len(_l_unary_i) != len(_l_negated):
                    raise ValueError("type %d: Number of negated differs"
                                     " from unary index number." % typ)
                typ_n_states = l_n_states[typ]
                if max(_l_state) >= typ_n_states:
                    raise ValueError("Type %d: State should in [%d, %d]"
                                     % (typ, 0, typ_n_states-1))
                start_type_index = a_by_type[typ]

                lTypeVar = [unary_binary_variables[start_type_index +
                                                   i*typ_n_states + k]
                            for i, k in zip(_l_unary_i, _l_state)]
                lVar.extend(lTypeVar)
                lNegated.extend(_l_negated)

            if last_type:
                typ_n_states = l_n_states[last_type]
                if last_state >= typ_n_states:
                    raise ValueError("(last) Type %d: State should in [%d, %d]"
                                     % (typ, 0, typ_n_states-1))
                start_type_index = a_by_type[last_type]
                u = unary_binary_variables[start_type_index +
                                           last_unary*typ_n_states +
                                           last_state]
                lVar.append(u)
                lNegated.append(last_neg)

            factor_graph.create_factor_logic(op, lVar, lNegated)

    #  Define one Uijkl binary variable per edge i,j for each pair of state k,l
    #  a) Link variable [Uijkl for all l] and not(Uik) for all k
    #  b) Link variable [Uijkl for all k] and not(Ujl) for all l
    i_typ_typ = 0
    for typ_i, n_states_i in enumerate(l_n_states):
        for typ_j, n_states_j in enumerate(l_n_states):
            edges = l_edges[i_typ_typ]
            edge_weights = l_edge_weights[i_typ_typ]
            i_typ_typ += 1

            if len(edges) or len(edge_weights):
                assert edge_weights.shape[1:] == (n_states_i, n_states_j)
            for e, cost in zip(edges, edge_weights):
                i, j = e

                # NOTE: Uik = unary_binary_variables[
                #    a_by_type[T] +
                #                                   i_in_typ*typ_n_states + k ]
                index_Ui_base = a_by_type[typ_i] + i*n_states_i
                index_Uj_base = a_by_type[typ_j] + j*n_states_j

                # lUij : list all binary nodes reflecting the edge between
                #          node i and node j, for all possible pairs of states
                lUij = []
                # Uij for all l
                lUijl_by_l = [list() for _ in range(n_states_j)]
                for k in range(n_states_i):
                    cost_k = cost[k]
                    Uik = unary_binary_variables[index_Ui_base + k]
                    lUijk = []
                    for l in range(n_states_j):
                        Uijkl = factor_graph.create_binary_variable()
                        lUijk.append(Uijkl)
                        # lUijl_by_l[l] is Uijkl for all k
                        lUijl_by_l[l].append(Uijkl)
                        Uijkl.set_log_potential(cost_k[l])

                    # Let's do a)        lUijk is "Uijkl for all l"
                    factor_graph.create_factor_logic("XOR",
                                                     [Uik] + lUijk,
                                                     [True] +
                                                     [False]*len(lUijk))
                    lUij.extend(lUijk)

                # now do b)
                for l in range(n_states_j):
                    Ujl = unary_binary_variables[index_Uj_base + l]
                    Uijkl_for_all_k = [lUij[k*n_states_j + l]
                                       for k in range(n_states_i)]
                    factor_graph.create_factor_logic("XOR",
                                                     [Ujl] + Uijkl_for_all_k,
                                                     [True] +
                                                     [False] *
                                                     len(Uijkl_for_all_k))
                del lUij, lUijl_by_l

    value, marginals, edge_marginals, solver_status = factor_graph.solve(
        eta=eta,
        adapt=True,
        max_iter=n_iterations,
        branch_and_bound=exact,
        verbose=verbose)

    # put back the values of the binary variables into the marginals
    aux_marginals = np.asarray(marginals)

    # THE NODE MARGINALS
    ret_node_marginals = list()
    k = 0
    # iteration by type
    for (_n_nodes, _n_states) in zip(l_n_nodes, l_n_states):
        _n_binaries = _n_nodes * _n_states
        ret_node_marginals.append(aux_marginals[k:k+_n_binaries
                                                ].reshape(_n_nodes, _n_states))
        k += _n_binaries
    # assert k == len(unary_binary_variables)

    # NOW THE EDGE MARGINALS
    aux_marginals = aux_marginals[len(unary_binary_variables):]

    ret_edge_marginals = list()

    i_typ_typ = 0
    i_marg_start = 0
    for typ_i, n_states_i in enumerate(l_n_states):
        for typ_j, n_states_j in enumerate(l_n_states):
            edges = l_edges[i_typ_typ]  # pop would modify the list..
            i_typ_typ += 1

            _n_edges = len(edges)
            _n_edge_states = n_states_i * n_states_j
            _n_marg = _n_edges * _n_edge_states

            if _n_edges:
                ret_edge_marginals.append(
                    aux_marginals[i_marg_start:i_marg_start +
                                  _n_marg].reshape((_n_edges, _n_edge_states))
                                          )
            else:
                ret_edge_marginals.append(np.zeros((0, _n_edge_states)))
            i_marg_start += _n_marg
    # assert i_marg_start == len(aux_marginals)

    return ret_node_marginals, ret_edge_marginals, value, solver_status
