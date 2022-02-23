from adnmtf import NTF
from adnmtf import NMF
import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"
from . import estimator_attributes


def make_json(path, row_header, n_blocks, n_components, sparsity, n_bootstrap, expected):
    if isinstance(path, str):
        df = pd.read_csv(DATA_PATH / path)
        if row_header > 0:
            df = df.iloc[:, row_header:]
        m0 = df.values
    else:
        m0 = path[0].dot(path[1])

    m0 = m0.astype(float)
    if n_blocks is not None:
        my_model = NTF(n_components=n_components, random_state=123)
        estimator = my_model.fit_transform(m0, n_blocks=n_blocks, sparsity=sparsity, n_bootstrap=n_bootstrap)
    else:
        my_model = NMF(n_components=n_components, random_state=123)
        estimator = my_model.fit_transform(m0, sparsity=sparsity, n_bootstrap=n_bootstrap)
    my_model.predict(estimator)
    estimator_ = {}
    for item in estimator_attributes:
        if isinstance(getattr(estimator, item), np.ndarray):
            estimator_[item] = getattr(estimator, item).tolist()
        else:
            estimator_[item] = getattr(estimator, item)
    with open(DATA_PATH / expected, "w") as ifile:
        json.dump(estimator_, ifile)
