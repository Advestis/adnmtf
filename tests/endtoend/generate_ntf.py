from nmtf import NTF

import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def compute(input_file, json_file, sparsity=0, n_bootstrap=0):
    df = pd.read_csv(input_file)
    expected_estimator_ = {}
    with open(json_file, "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator_[key] = (
                np.asarray(decoded_array[key]) if isinstance(decoded_array[key], list) else decoded_array[key]
            )
    m0 = df.values
    n_blocks = 5
    my_nt_fmodel = NTF(n_components=5, random_state=123)
    estimator_ = my_nt_fmodel.fit_transform(m0, n_blocks, sparsity=sparsity, n_bootstrap=n_bootstrap)
    estimator_ = my_nt_fmodel.predict(estimator_)

    return estimator_, expected_estimator_


estimator_1, expected_estimator_1 = compute(input_file=DATA_PATH / "data_ntf.csv",
                                        json_file=DATA_PATH / "expected_result_ntf.json")
estimator_2, expected_estimator_2 = compute(input_file=DATA_PATH / "data_ntf.csv",
                                        json_file=DATA_PATH / "expected_result_ntf_with_sparsity_bootstrap.json",
                                          sparsity=.8, n_bootstrap=10)