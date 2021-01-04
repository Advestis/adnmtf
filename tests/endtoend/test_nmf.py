import pytest
import numpy as np

from .generate_nmf import estimator, expected_estimator


@pytest.mark.parametrize(
    "param", list(estimator.keys())
)
def test_nmf(param):
    print("")
    print(f"Testing {param}...")
    if param.lower() == "wb" or param.lower() == "hb":
        print(f"Ignoring {param}...")
        return
    
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
