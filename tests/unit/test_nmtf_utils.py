import pytest
import numpy as np

from adnmtf.nmtf_utils import sparse_opt

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.mark.parametrize(
    "b, alpha, two_sided, expected",
    [
        (
            arr,
            0.5,
            True,
            np.array([0.0, 0.0, 0.0, 0.0, 0.62158264, 3.09527575, 5.56896887, 8.04266198, 10.5163551, 12.99004821]),
        ),
        (
            arr,
            0.5,
            False,
            np.array([0.0, 0.0, 0.0, 0.0, 0.62158264, 3.09527575, 5.56896887, 8.04266198, 10.5163551, 12.99004821]),
        ),
    ],
)
def test_sparse_opt(b, alpha, two_sided, expected):
    np.testing.assert_equal(np.round(expected, decimals=6), np.round(sparse_opt(b, alpha, two_sided), decimals=6))
