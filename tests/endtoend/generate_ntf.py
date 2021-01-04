from nmtf import NTF

import pandas as pd
import numpy as np
import json
from pathlib import Path
from ..utils.json_encoder import JSONEncoder

DATA_PATH = Path(__file__).parent.parent / "data"


def compute():
    df = pd.read_csv(DATA_PATH / "data_ntf.csv")
    expected_estimator_ = {}
    with open(DATA_PATH / "expected_result_ntf.json", "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator_[key] = (
                np.asarray(decoded_array[key]) if isinstance(decoded_array[key], list) else decoded_array[key]
            )
    m0 = df.values
    n_blocks = 5
    my_nt_fmodel = NTF(n_components=5)
    estimator_ = my_nt_fmodel.fit_transform(m0, n_blocks, sparsity=0.8, n_bootstrap=10)
    estimator_ = my_nt_fmodel.predict(estimator_)

    return estimator_, expected_estimator_


estimator, expected_estimator = compute()