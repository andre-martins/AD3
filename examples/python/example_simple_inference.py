# -*- coding: utf-8 -*-
"""
An example of use of the simple_inference module.

JL Meunier
Dec 2017
"""

from __future__ import print_function
import numpy as np
from ad3.simple_inference import general_graph


def get_grid_label(nRow, nCol, nodes):
    assert nRow * nCol == nodes.shape[0]
    labels = nodes.argmax(axis=1).reshape(nRow, nCol).astype(np.int)
    return labels


def prepare_single_type():
    """
    return the prepared data

    We define a 2x3 grid.
    N--N--N
    |  |  |
    N--N--N
    Each node takes one of 6 labels.
    All node weights are increasing per label. ("higher" labels preferred)

    All of the edge weights rewards an edge going from label I to I+1.
    """
    L = 6  # labels

    # we look at nodes from left-right, top, bottom
    N = 2 * 3  # nodes
    unaries_uniform = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float)
    unaries = np.array([unaries_uniform] * N, dtype=np.float)
    assert unaries.shape == (N, L)

    E = 7  # edges
    edges = np.array([[0, 1], [1, 2],
                      [0, 3], [1, 4], [2, 5],
                      [3, 4], [4, 5]], dtype=np.int)
    assert edges.shape == (E, 2)

    # the edge weights just favour well behaved neighbors, i.e. I and I+1
    edge_weights_uniform = np.array([[0, 1, 0, 0, 0, 0],
                                     [1, 0, 1, 0, 0, 0],
                                     [0, 1, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 1, 0],
                                     [0, 0, 0, 1, 0, 1],
                                     [0, 0, 0, 0, 1, 0],
                                     ], dtype=np.float)
    # same weight for all edges
    edge_weights = np.array([edge_weights_uniform] * E, dtype=np.float)
    assert edge_weights.shape == (E, L, L)

    # simple case
    unaries[0, 3] = 1.0  # pushing for node 0 being label 3
    unaries[1, 4] = 1.0  # pushing for node 1 being label 4
    print(get_grid_label(2, 3, unaries))

    return unaries, edges, edge_weights


def prepare_multi_type():
    """
    Now we add another sort of node M that takes one of 3 labels

    We create this grid:
    N--N--N
    |  |  |
    M--M--M

    we keep the same spirit for node and edge weights
    """

    N_N = 3
    L_N = 6
    unaries_N = np.array([np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])] * N_N)
    assert unaries_N.shape == (N_N, L_N)

    N_M = 3
    L_M = 3
    unaries_M = np.array([np.array([0.1, 0.2, 0.3])] * N_M)
    assert unaries_M.shape == (N_M, L_M)

    # edges are grouped by type: N-N, N-M, M-N, M-M
    edges_NN = np.array([[0, 1], [1, 2]])
    edges_NM = np.array([[0, 0], [1, 1], [2, 2]])
    edges_MN = np.array([])
    edges_MM = np.array([[0, 1], [1, 2]])

    # the edge weights just favour well behaved neighbors, i.e. I and I+1
    edge_weights_uniform_NN = np.array([[0, 1, 0, 0, 0, 0],
                                        [1, 0, 1, 0, 0, 0],
                                        [0, 1, 0, 1, 0, 0],
                                        [0, 0, 1, 0, 1, 0],
                                        [0, 0, 0, 1, 0, 1],
                                        [0, 0, 0, 0, 1, 0],
                                        ], dtype=np.float)
    edge_weights_uniform_NM = np.array([[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0],
                                        [0, 0, 1],
                                        [0, 0, 0],
                                        [0, 0, 0],
                                        ], dtype=np.float)
    edge_weights_uniform_MN = edge_weights_uniform_NM.transpose()
    edge_weights_uniform_MM = np.array([[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0],
                                        ], dtype=np.float)
    # same weights for all edges
    edge_weights_NN = np.array([edge_weights_uniform_NN] * len(edges_NN))
    edge_weights_NM = np.array([edge_weights_uniform_NM] * len(edges_NM))
    edge_weights_MN = np.array([edge_weights_uniform_MN] * len(edges_MN))
    edge_weights_MM = np.array([edge_weights_uniform_MM] * len(edges_MM))

    unaries_N[0, 3] = 1.0  # pushing for node N 0 being label 3
    unaries_N[1, 4] = 1.0  # pushing for node N 1 being label 4
    print(get_grid_label(1, 3, unaries_N))
    print(get_grid_label(1, 3, unaries_M))

    return (
        [unaries_N, unaries_M],
        [edges_NN, edges_NM, edges_MN, edges_MM],
        [edge_weights_NN, edge_weights_NM, edge_weights_MN, edge_weights_MM]
        )


if __name__ == "__main__":

    print("--- SINGLE TYPE ---")

    unaries, edges, edge_weights = prepare_single_type()
    marginals, edge_marginals, value, status = general_graph(unaries,
                                                             edges,
                                                             edge_weights,
                                                             exact=True
                                                             )
    labels = get_grid_label(2, 3, marginals)
    print("--->")
    print(labels)

    assert labels.tolist() == [[3, 4, 5],
                               [4, 5, 4]]

    print("--- MULTI TYPE ---")

    lUnary, lEdges, lEdgeWeights = prepare_multi_type()
    marginals, edge_marginals, value, status = general_graph(lUnary,
                                                             lEdges,
                                                             lEdgeWeights,
                                                             exact=True
                                                             )
    print("--->")
    labels_N = get_grid_label(1, 3, marginals[0])
    print(labels_N)
    labels_M = get_grid_label(1, 3, marginals[1])
    print(labels_M)
    # it is better to have N1 as 2 to get the +1 bonus from the vertical edge
    # it is better to have N2 as 3 to get the +1 bonus from the vertical edge
    assert labels_N.tolist() == [[3, 2, 3]]
    assert labels_M.tolist() == [[2, 1, 2]]

    # ========   TEST  ========
    unaries_N, unaries_M = lUnary
    unaries_N[1, 4] = 2.0  # pushing HARDER for node N 1 being label 4

    # inference will prefer gaining 2.0 to losing .2 for gaining 1 (the edge)
    marginals, edge_marginals, value, status = general_graph(lUnary,
                                                             lEdges,
                                                             lEdgeWeights,
                                                             exact=True
                                                             )
    print("--->")
    labels_N = get_grid_label(1, 3, marginals[0])
    print(labels_N)
    labels_M = get_grid_label(1, 3, marginals[1])
    print(labels_M)

    assert labels_N.tolist() == [[3, 4, 3]]
    assert labels_M.tolist() == [[2, 1, 2]]

