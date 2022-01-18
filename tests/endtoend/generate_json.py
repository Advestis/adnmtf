from nmtf import NTF
from nmtf import NMF
import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def compute_test_ntf1():
    df = pd.read_csv(DATA_PATH / "data_ntf.csv")
    m0 = df.values
    n_blocks = 5
    my_ntfmodel = NTF(n_components=5, random_state=123)
    estimator_ = my_ntfmodel.fit_transform(m0, n_blocks)
    estimator_ = my_ntfmodel.predict(estimator_)
    for item in estimator_:
        if isinstance(estimator_[item], np.ndarray):
            estimator_[item] = estimator_[item].tolist()
    with open(DATA_PATH / "expected_result_ntf.json", "w") as ifile:
        json.dump(estimator_, ifile)

    return estimator_

def compute_test_ntf2():
    df = pd.read_csv(DATA_PATH / "data_ntf.csv")
    m0 = df.values
    n_blocks = 5
    my_ntfmodel = NTF(n_components=5, random_state=123)
    estimator_ = my_ntfmodel.fit_transform(m0, n_blocks, sparsity=.8, n_bootstrap=10)
    estimator_ = my_ntfmodel.predict(estimator_)
    for item in estimator_:
        if isinstance(estimator_[item], np.ndarray):
            estimator_[item] = estimator_[item].tolist()
    with open(DATA_PATH / "expected_result_ntf_with_sparsity_bootstrap.json", "w") as ifile:
        json.dump(estimator_, ifile)

    return estimator_

def compute_test_ntf3():
    df = pd.read_csv(DATA_PATH / "data_sntf.csv")
    m0 = df.iloc[:, 1:].values
    n_blocks = 6
    my_ntfmodel = NTF(n_components=6, random_state=123, init_type=1)
    estimator_ = my_ntfmodel.fit_transform(m0, n_blocks, n_bootstrap=0)
    estimator_ = my_ntfmodel.predict(estimator_)
    for item in estimator_:
        if isinstance(estimator_[item], np.ndarray):
            estimator_[item] = estimator_[item].tolist()
    with open(DATA_PATH / "expected_result_sntf.json", "w") as ifile:
        json.dump(estimator_, ifile)

    return estimator_

def compute_test_nmf1():
    w = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10],
                  [11, 12]])

    h = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                  [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    m0 = w.dot(h)
    my_nmfmodel = NMF(n_components=2, random_state=123)
    estimator_ = my_nmfmodel.fit_transform(m0, sparsity=0.8, n_bootstrap=10)
    estimator_ = my_nmfmodel.predict(estimator_)
    for item in estimator_:
        if isinstance(estimator_[item], np.ndarray):
            estimator_[item] = estimator_[item].tolist()
    with open(DATA_PATH / "expected_result_nmf.json", "w") as ifile:
        json.dump(estimator_, ifile)

    return estimator_

def compute_test_nmf2():
    df = pd.read_csv(DATA_PATH / "data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms.csv", header=0)
    m0 = df.iloc[:, 1:].values
    my_nmfmodel = NMF(n_components=1, random_state=123)
    estimator_ = my_nmfmodel.fit_transform(m0, n_bootstrap=0)
    for item in estimator_:
        if isinstance(estimator_[item], np.ndarray):
            estimator_[item] = estimator_[item].tolist()
    with open(DATA_PATH / "expected_data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms.json", "w") as ifile:
        json.dump(estimator_, ifile)

def compute_test_nmf3():
    df = pd.read_csv(DATA_PATH / "data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms_miss.csv", header=0)
    m0 = df.iloc[:, 1:].values
    my_nmfmodel = NMF(n_components=1, random_state=123)
    estimator_ = my_nmfmodel.fit_transform(m0, n_bootstrap=0)
    for item in estimator_:
        if isinstance(estimator_[item], np.ndarray):
            estimator_[item] = estimator_[item].tolist()
    with open(DATA_PATH / "expected_data_nmf_nc1_corrmin0.9_corrmax0.1_noise0_RandNorms_miss.json", "w") as ifile:
        json.dump(estimator_, ifile)

def compute_test_nmf4():
    df = pd.read_csv(DATA_PATH / "data_nmf_brunet.csv", header=0)
    m0 = df.iloc[:, 2:].values
    my_nmfmodel = NMF(n_components=4, random_state=123)
    estimator_ = my_nmfmodel.fit_transform(m0, n_bootstrap=10)
    for item in estimator_:
        if isinstance(estimator_[item], np.ndarray):
            estimator_[item] = estimator_[item].tolist()
    with open(DATA_PATH / "expected_data_nmf_brunet.json", "w") as ifile:
        json.dump(estimator_, ifile)

estimator = compute_test_ntf1()
estimator = compute_test_ntf2()
estimator = compute_test_ntf3()
estimator = compute_test_nmf1()
estimator = compute_test_nmf2()
estimator = compute_test_nmf3()
estimator = compute_test_nmf4()