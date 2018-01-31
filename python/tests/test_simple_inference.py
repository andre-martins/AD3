'''
Created on 30 Jan 2017

@author: meunier
'''
import future
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_almost_equal, assert_equal)
from ..simple_inference import general_graph

def test_general_graph():
    unaries         = np.array([[ 10, 11,       0 ],
                                [ 1000, 1100, 1200]], dtype=np.float64)
    edges           = np.array([[0        ,1]])
    edge_weights    = np.array([
                                [[.00, .01, .02],
                                 [.10, .11, .12],
                                 [0,     0,   0]]
                                ], dtype=np.float64)
    ret = general_graph(unaries, edges, edge_weights, verbose=1, exact=False)
    print(ret)
    marginals, edge_marginals, value, solver_status = ret
    assert (marginals == np.array([[ 0.,  1.,  0.],
                                  [ 0.,  0.,  1.]])).all()     
    assert (edge_marginals == np.array([[ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.]])).all()
    assert solver_status == 'integral'

def test_general_graph_multitype():
    unaries         = [ np.array([[ 10, 11]])
                       ,np.array([[ 1000, 1100, 1200]])]
    edges           = [ np.zeros((0,0)), np.array([[0,0]]), np.zeros((0,0)), np.zeros((0,0)) ]
    edge_weights    = [ np.zeros((0,0))
                       , np.array([[[.00, .01, .02],
                                    [.10, .11, .12]]])
                       , np.zeros((0,0))
                       , np.zeros((0,0))
                       ]
    print([o.shape for o in edge_weights])
    ret = general_graph(unaries, edges, edge_weights, verbose=1)
    
    marginals, edge_marginals, value, solver_status = ret
    print(ret)
    assert_array_almost_equal( marginals[0], np.array([[ 0,  1]]) )
    assert_array_almost_equal( marginals[1], np.array([[0, 0, 1]]) )
    
    assert_array_almost_equal( edge_marginals[0], np.zeros( (0,4) ) )
    assert_array_almost_equal( edge_marginals[1].reshape(2,3), np.array( [[0, 0, 0],
                                                                          [0, 0, 1]] ) , 5)
    assert_array_almost_equal( edge_marginals[2], np.zeros( (0,6) ) )
    assert_array_almost_equal( edge_marginals[3], np.zeros( (0,9) ) )

    
if __name__ == "__main__":
    test_general_graph()
    test_general_graph_multitype()
    print("OK")