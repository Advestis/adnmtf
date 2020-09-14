from nmtf import NTF

import pytest
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import date, datetime

DATA_PATH = Path(__file__).parent.parent / "data"


class JSONEncoder(json.JSONEncoder):
    """
    Extending the JSON encoder so it knows how to serialise a dataframe
    """

    def default(self, obj):
        if hasattr(obj, "to_json"):
            if callable(obj.to_json):
                try:
                    return obj.to_json(orient="records")
                except TypeError:
                    return obj.to_json()
            else:
                return obj.to_json
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return json.JSONEncoder.default(self, obj)


def test():
    df = pd.read_csv(DATA_PATH / "data_ntf.csv")
    expected_estimator = {}
    with open(DATA_PATH / "expected_result_ntf.json", "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator[key] = (
                np.asarray(decoded_array[key]) if isinstance(decoded_array[key], list) else decoded_array[key]
            )
    m0 = df.values
    n_blocks = 5
    my_nt_fmodel = NTF(n_components=5)
    estimator = my_nt_fmodel.fit_transform(m0, n_blocks, sparsity=0.8, n_bootstrap=10)
    estimator = my_nt_fmodel.predict(estimator)

    # Uncomment to save the estimator in a file
    # with open(DATA_PATH / "expected_result_ntf_new.json", "w") as ofile:
    #     ofile.write(json.dumps(estimator, cls=JSONEncoder))

    failed = False
    for key in estimator:
        print("")
        print(f"Testing {key}...")
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
