from nmtf import NMF

import pytest
import numpy as np
import json
import sys
from pathlib import Path
from ..utils.json_encoder import JSONEncoder

DATA_PATH = Path(__file__).parent.parent / "data"


def test():
    expected_estimator = {}
    with open(DATA_PATH / "expected_result_nmf.json", "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator[key] = (
                np.asarray(decoded_array[key]) if isinstance(decoded_array[key], list) else decoded_array[key]
            )

    w = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10],
                  [11, 12]])

    h = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                  [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    m0 = w.dot(h)
    my_nt_fmodel = NMF(n_components=2)
    estimator = my_nt_fmodel.fit_transform(m0, sparsity=0.8, n_bootstrap=10)
    estimator = my_nt_fmodel.predict(estimator)

    # Uncomment to save the estimator in a file
    # with open(DATA_PATH / "expected_result_nmf.json", "w") as ofile:
    #     ofile.write(json.dumps(estimator, cls=JSONEncoder))
    # exit(0)

    failed = False
    for key in estimator:
        print("")
        print(f"Testing {key}...")
        if key.lower() == "wb" or key.lower() == "hb":
            print(f"Ignoring {key}...")
            continue
        key_exp = key
        if key not in expected_estimator:
            key_exp = key.upper()

        if key_exp not in expected_estimator:
            print(f"{key} not found in expected elements")
            failed = True
            continue

        if not isinstance(estimator[key], type(expected_estimator[key_exp])):
            print("")
            print(f"Type of {key} is {type(estimator[key])} while expected type if {type(expected_estimator[key_exp])}")
            failed = True
            continue

        try:
            if isinstance(estimator[key], np.ndarray):
                np.testing.assert_array_almost_equal(estimator[key], expected_estimator[key_exp])
            elif isinstance(estimator[key], float):
                assert pytest.approx(estimator[key], rel=1e-10) == expected_estimator[key_exp]
            else:
                assert estimator[key] == expected_estimator[key_exp]
            print("...ok")
        except AssertionError:
            np.set_printoptions(threshold=sys.maxsize)
            print("")
            print(f"Estimator[{key}]:{estimator[key]}")
            print(f"Expected:{expected_estimator[key_exp]}")
            print("Differences:", estimator[key] - expected_estimator[key_exp])
            failed = True

    if failed:
        raise AssertionError("Some tests failed")
