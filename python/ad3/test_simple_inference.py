'''
Created on 30 Jan 2017

@author: meunier
'''
import numpy as np

from simple_inference import general_graph

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
    print ret
    marginals, edge_marginals, value, solver_status = ret
    assert (marginals == np.array([[ 0.,  1.,  0.],
                                  [ 0.,  0.,  1.]])).all()     
    assert (edge_marginals == np.array([[ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.]])).all()
    assert solver_status == 'integral'
    
if __name__ == "__main__":
    test_general_graph()
    print "OK"