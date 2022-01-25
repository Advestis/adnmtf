import pytest
import numpy as np

from .generate_nmf import compute_test_ntf
# from .generate_json import make_json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize(
    "path, iloc, n_blocks, n_components, sparsity, n_bootstrap, expected",
    (
        ("data_sntf.csv", 1, 6, 6, 0, 0, "expected_result_sntf.json"),
        (
            "data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms_miss.csv",
            1,
            None,
            1,
            0,
            None,
            "expected_data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms_miss.json",
        ),
        ("data_ntf.csv", 0, 5, 5, 0, None, "expected_result_ntf.json"),
        ("data_ntf.csv", 0, 5, 5, 0.8, 10, "expected_result_ntf_with_sparsity_bootstrap.json"),
        (
            (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]),
                np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]),
            ),
            0,
            None,
            2,
            0.8,
            10,
            "expected_result_nmf.json",
        ),
        (
            "data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms.csv",
            1,
            None,
            1,
            0,
            0,
            "expected_data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms.json",
        ),
        ("data_nmf_brunet.csv", 2, None, 4, 0, 10, "expected_data_nmf_brunet.json"),
    ),
)
def test_nmf(path, iloc, n_blocks, n_components, sparsity, n_bootstrap, expected):
    # make_json(path, iloc, n_blocks, n_components, sparsity, n_bootstrap, expected)
    inputs = compute_test_ntf(path, iloc, n_blocks, n_components, sparsity, n_bootstrap, expected)
    print("")
    estimator = inputs[0]
    expected_estimator = inputs[1]
    for param in estimator:
        print(f"Testing {param}...")
        if param.lower() == "wb" or param.lower() == "hb":
            print(f"Ignoring {param}...")
        else:

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
