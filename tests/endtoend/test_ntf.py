import pytest
import numpy as np

from .generate_ntf import estimator_1, expected_estimator_1, estimator_2, expected_estimator_2


@pytest.mark.parametrize(
    "estimator, expected_estimator", [(estimator_1, expected_estimator_1), (estimator_2, expected_estimator_2)]
)
def test_ntf(estimator, expected_estimator):
    for param in estimator.keys():
        print("")
        print(f"Testing {param}...")

        param_exp = param
        if param not in expected_estimator:
            param_exp = param.upper()

        assert param_exp in expected_estimator
        assert isinstance(estimator[param], type(expected_estimator[param_exp]))
        if isinstance(estimator[param], np.ndarray):
            np.testing.assert_array_almost_equal(estimator[param], expected_estimator[param_exp])
        elif isinstance(estimator[param], float):
            assert pytest.approx(estimator[param], rel=1e-10) == expected_estimator[param_exp]
        else:
            assert estimator[param] == expected_estimator[param_exp]
