import pytest
import numpy as np

from .generate_nmf import inputs
from pathlib import Path
DATA_PATH = Path(__file__).parent.parent / "data"

@pytest.mark.parametrize(
    "inputs_", inputs
)
def test_nmf(inputs_):
    print("")
    estimator = inputs_[0]
    expected_estimator = inputs_[1]
    test_name = inputs_[2]
    for param in estimator:
        print(f"Testing {param}...")
        if param.lower() == "wb" or param.lower() == "hb":
            print(f"Ignoring {param}...")
        else:
            if test_name == "test_nmf3":
                np.savetxt(DATA_PATH / "test_nmf3_W.csv", estimator['W'])
                np.savetxt(DATA_PATH / "test_nmf3_H.csv", estimator['H'])
                print(estimator['diff'])

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
