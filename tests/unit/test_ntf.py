from nmtf import NTF

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def test():
    df = pd.read_csv(DATA_PATH / "data_ntf.csv")
    expected_estimator = {}
    with open(DATA_PATH / "expected_result_ntf.json", "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator[key] = np.asarray(decoded_array[key])
    m0 = df.values
    n_blocks = 5
    my_nt_fmodel = NTF(n_components=5)
    estimator = my_nt_fmodel.fit_transform(m0, n_blocks, sparsity=.8, n_bootstrap=10)
    estimator = my_nt_fmodel.predict(estimator)
    failed = False
    for key in estimator:
        print("")
        print(f"Testing {key}...")
        key_exp = key
        if key not in expected_estimator:
            key_exp = key.upper()

        if key not in expected_estimator:
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
