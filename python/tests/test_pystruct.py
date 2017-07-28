"""Test pystruct integration"""
# Author: Vlad Niculae <vlad@vene.ro>

try:
    from pystruct.inference import inference_ad3
    missing_pystruct = False
except ImportError:
    missing_pystruct = True
    pass

import pytest
import numpy as np
from numpy.testing import assert_array_equal


@pytest.mark.skipif(missing_pystruct,
                    reason="pystruct is not available")
def test_pystruct():

    unaries = np.zeros((3, 5))
    unaries[1, 2] = 2
    pairwise = np.eye(5)
    edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.intp)

    # no parameters
    labels = inference_ad3(unaries, pairwise, edges)
    assert_array_equal(labels, [2, 2, 2])

    # exact decoding
    labels_exact = inference_ad3(unaries, pairwise, edges,
                                 branch_and_bound=True)
    assert_array_equal(labels_exact, [2, 2, 2])

    # request energy
    labels, energy = inference_ad3(unaries, pairwise, edges,
                                   return_energy=True)
    assert_array_equal(energy, -5)

    # exact decoding and request energy
    labels, energy = inference_ad3(unaries, pairwise, edges,
                                   branch_and_bound=True, return_energy=True)
    assert_array_equal(energy, -5)
