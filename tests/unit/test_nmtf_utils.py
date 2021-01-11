import pytest
import numpy as np

from nmtf.modules.nmtf_utils import sparse_opt, shift

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.mark.parametrize(
    "b, alpha, two_sided, expected",
    [
        (arr, 0.5, True, np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.62158264,
        3.09527575,  5.56896887,  8.04266198, 10.5163551 , 12.99004821])),
        (arr, 0.5, False, np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.62158264,
        3.09527575,  5.56896887,  8.04266198, 10.5163551 , 12.99004821])),
    ],
)

def test_sparse_opt(b, alpha, two_sided, expected):
    np.testing.assert_equal(np.round(expected, decimals=6), np.round(sparse_opt(b, alpha, two_sided), decimals=6))

@pytest.mark.parametrize(
    "b, num, fill_value, expected",
    [
        (arr, 2, 0, np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])),
        (arr, -2, 0, np.array([ 3,  4,  5,  6,  7,  8,  9, 10,  0,  0])),
    ],
)

def test_shift(b, num, fill_value, expected):
    np.testing.assert_equal(expected, shift(b, num, fill_value))