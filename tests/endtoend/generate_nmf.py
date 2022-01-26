from nmtf import NMF
from nmtf import NTF

import numpy as np
import pandas as pd
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def compute_test_ntf(
    path, row_header, n_blocks, n_components, sparsity, n_bootstrap, json_file
):
    expected_estimator_ = {}
    with open(DATA_PATH / json_file, "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator_[key] = (
                np.asarray(decoded_array[key]) if isinstance(decoded_array[key], list) else decoded_array[key]
            )

    if isinstance(path, str):
        df = pd.read_csv(DATA_PATH / path)
        if row_header > 0:
            df = df.iloc[:, row_header:]
        m0 = df.values
    else:
        m0 = path[0].dot(path[1])

    if n_blocks is not None:
        my_ntfmodel = NTF(n_components=n_components, random_state=123)
        estimator_ = my_ntfmodel.fit_transform(m=m0, n_blocks=n_blocks, sparsity=sparsity, n_bootstrap=n_bootstrap)
        estimator_ = my_ntfmodel.predict(estimator_)
    else:
        my_nmfmodel = NMF(n_components=n_components, random_state=123)
        estimator_ = my_nmfmodel.fit_transform(m=m0, sparsity=sparsity, n_bootstrap=n_bootstrap)

    return estimator_, expected_estimator_
