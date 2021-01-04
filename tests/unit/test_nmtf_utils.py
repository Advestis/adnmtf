import pytest
import numpy as np

from nmtf.modules.nmtf_utils import sparse_opt

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.mark.parametrize(
    "b, alpha, two_sided, expected",
    [
        (arr, 0.1, True, np.array([0, 0, 0, 0, 0, 0, 0, 0, np.nan])),
        (arr, 0.1, False, np.array([0, 0, 0, 0, 0, 0, 0, 0, np.nan])),
    ],
)
def test_sparse_opt(b, alpha, two_sided, expected):
    np.testing.assert_equal(expected, sparse_opt(b, alpha, two_sided))
