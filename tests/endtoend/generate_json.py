from nmtf import NTF
from nmtf import NMF
import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def make_json(path, row_header, n_blocks, n_components, sparsity, n_bootstrap, expected):
    if isinstance(path, str):
        df = pd.read_csv(DATA_PATH / path)
        if row_header > 0:
            df = df.iloc[:, row_header:]
        m0 = df.values
    else:
        m0 = path[0].dot(path[1])
    if n_blocks is not None:
        my_model = NTF(n_components=n_components, random_state=123)
        estimator_ = my_model.fit_transform(m0, n_blocks, sparsity=sparsity, n_bootstrap=n_bootstrap)
    else:
        my_model = NMF(n_components=n_components, random_state=123)
        estimator_ = my_model.fit_transform(m0, sparsity=sparsity, n_bootstrap=n_bootstrap)
    estimator_ = my_model.predict(estimator_)
    for item in estimator_:
        if isinstance(estimator_[item], np.ndarray):
            estimator_[item] = estimator_[item].tolist()
    with open(DATA_PATH / expected, "w") as ifile:
        json.dump(estimator_, ifile)

    return estimator_
