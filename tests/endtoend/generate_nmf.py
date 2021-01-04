from nmtf import NMF

import numpy as np
import json
from pathlib import Path
from ..utils.json_encoder import JSONEncoder

DATA_PATH = Path(__file__).parent.parent / "data"


def compute():
    expected_estimator_ = {}
    with open(DATA_PATH / "expected_result_nmf.json", "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator_[key] = (
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
    estimator_ = my_nt_fmodel.fit_transform(m0, sparsity=0.8, n_bootstrap=10)
    estimator_ = my_nt_fmodel.predict(estimator_)
    return estimator_, expected_estimator_



estimator, expected_estimator = compute()
