from nmtf import NTF

import pandas as pd
import numpy as np
import json
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
    for key in estimator:
        key_exp = key
        if key not in expected_estimator:
            key_exp = key.upper()
        print("")
        print(f"Estimator[{key}]:{estimator[key]}")
        print(f"Expected:{expected_estimator[key_exp]}")
        assert (estimator[key] == expected_estimator[key_exp]).all()
