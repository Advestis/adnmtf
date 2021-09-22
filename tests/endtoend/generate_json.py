from nmtf import NTF

import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def compute():
    df = pd.read_csv(DATA_PATH / "data_ntf.csv")
    m0 = df.values
    n_blocks = 5
    my_nt_fmodel = NTF(n_components=5, random_state=123)
    estimator_ = my_nt_fmodel.fit_transform(m0, n_blocks, sparsity=.8, n_bootstrap=10)
    estimator_ = my_nt_fmodel.predict(estimator_)
    for item in estimator_:
        if isinstance(estimator_[item], np.ndarray):
            estimator_[item] = estimator_[item].tolist()
    with open(DATA_PATH / "expected_result_ntf_with_sparsity_bootstrap.json", "w") as ifile:
        json.dump(estimator_, ifile)

    return estimator_


estimator = compute()