from adnmtf import NMF
from adnmtf import NTF

import numpy as np
import pandas as pd
import json
from pathlib import Path
from adnmtf.estimator import Estimator

DATA_PATH = Path(__file__).parent.parent / "data"


def compute_test_ntf(path, row_header, n_blocks, n_components, sparsity, n_bootstrap, json_file):
    expected_estimator = Estimator(w=None, h=None, volume=None, diff=None, leverage=None, verbose=None)
    with open(DATA_PATH / json_file, "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            setattr(
                expected_estimator,
                key,
                np.asarray(decoded_array[key]) if isinstance(decoded_array[key], list) else decoded_array[key],
            )

    if isinstance(path, str):
        df = pd.read_csv(DATA_PATH / path)
        if row_header > 0:
            df = df.iloc[:, row_header:]
        m0 = df.values
    else:
        m0 = path[0].dot(path[1])

    m0 = m0.astype(float)

    if n_blocks is not None:
        my_ntfmodel = NTF(n_components=n_components, random_state=123)
        estimator = my_ntfmodel.fit_transform(m=m0, n_blocks=n_blocks, sparsity=sparsity, n_bootstrap=n_bootstrap)
        my_ntfmodel.predict(estimator)
    else:
        my_nmfmodel = NMF(n_components=n_components, random_state=123)
        estimator = my_nmfmodel.fit_transform(m=m0, sparsity=sparsity, n_bootstrap=n_bootstrap)
        my_nmfmodel.predict(estimator)

    return estimator, expected_estimator
