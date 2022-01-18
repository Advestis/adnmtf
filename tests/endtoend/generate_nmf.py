from nmtf import NMF
from nmtf import NTF

import numpy as np
import pandas as pd
import json
from pathlib import Path
from ..utils.json_encoder import JSONEncoder

DATA_PATH = Path(__file__).parent.parent / "data"

def compute_test_ntf(test_name, json_file, input_file, row_header=0, n_components=1, n_blocks = 2, sparsity=0, n_bootstrap=0):
    df = pd.read_csv(input_file)
    expected_estimator_ = {}
    with open(json_file, "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator_[key] = (
                np.asarray(decoded_array[key]) if isinstance(decoded_array[key], list) else decoded_array[key]
            )

    if row_header > 0:
        m0 = df.iloc[:, row_header:].values
    else:
        m0 = df.values

    n_blocks = n_blocks
    my_ntfmodel = NTF(n_components=n_components, random_state=123)
    estimator_ = my_ntfmodel.fit_transform(m0, n_blocks, sparsity=sparsity, n_bootstrap=n_bootstrap)
    estimator_ = my_ntfmodel.predict(estimator_)

    return estimator_, expected_estimator_, test_name


def compute_test_nmf(test_name, json_file, input_file=None, row_header=0, w=None, h=None, n_components=1, sparsity=0, n_bootstrap=0):
    expected_estimator_ = {}
    with open(DATA_PATH / json_file, "r") as ifile:
        decoded_array = json.load(ifile)
        for key in decoded_array:
            expected_estimator_[key] = (
                np.asarray(decoded_array[key]) if isinstance(decoded_array[key], list) else decoded_array[key]
            )

    if input_file is None:
        m0 = w.dot(h)
    else:
        df = pd.read_csv(input_file)
        if row_header > 0:
            m0 = df.iloc[:, row_header:].values
        else:
            m0 = df.values

    my_nmfmodel = NMF(n_components=n_components, random_state=123)
    estimator_ = my_nmfmodel.fit_transform(m0, sparsity=sparsity, n_bootstrap=n_bootstrap)
    return estimator_, expected_estimator_, test_name

inputs = (
    compute_test_nmf(test_name="test_nmf1",
                     json_file=DATA_PATH / "expected_result_nmf.json",
                     w=np.array(
                        [[1, 2],
                        [3, 4],
                        [5, 6],
                        [7, 8],
                        [9, 10],
                        [11, 12]]),
                     h=np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]]),
                     n_components=2,
                     sparsity=0.8,
                     n_bootstrap=10
                     ),
    compute_test_nmf(test_name="test_nmf2",
                     json_file=DATA_PATH / "expected_data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms.json",
                     input_file=DATA_PATH / "data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms.csv",
                     row_header=1,
                     n_components=1
                     ),
    compute_test_nmf(test_name="test_nmf3",
                     json_file=DATA_PATH / "expected_data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms_miss.json",
                     input_file=DATA_PATH / "data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms_miss.csv",
                     row_header=1,
                     n_components=1
                     ),
    compute_test_nmf(test_name="test_nmf4",
                     json_file=DATA_PATH / "expected_data_nmf_brunet.json",
                     input_file=DATA_PATH / "data_nmf_brunet.csv",
                     row_header=2,
                     n_components=4,
                     n_bootstrap=10
                     ),
    compute_test_ntf(test_name="test_ntf",
                     json_file=DATA_PATH / "expected_result_ntf.json",
                     input_file=DATA_PATH / "data_ntf.csv",
                     n_components=5,
                     n_blocks=5
                     ),
    compute_test_ntf(test_name="test_ntf_with_sparsity_bootstrap",
                     json_file=DATA_PATH / "expected_result_ntf_with_sparsity_bootstrap.json",
                     input_file=DATA_PATH / "data_ntf.csv",
                     n_components=5,
                     n_blocks=5,
                     sparsity=.8,
                     n_bootstrap=10
                     ),
    compute_test_ntf(test_name="test_sntf",
                     json_file=DATA_PATH / "expected_result_sntf.json",
                     input_file=DATA_PATH / "data_sntf.csv",
                     row_header=1,
                     n_components=6,
                     n_blocks=6,
                     n_bootstrap=10
                     )
    )

#np.savetxt(Path(__file__).parent / 'H.csv',estimator['H'], delimiter=',')
#np.savetxt(Path(__file__).parent /'H_expected.csv',expected_estimator['H'], delimiter=',')
