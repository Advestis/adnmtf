"""Non-negative matrix and tensor factorization basic functions
"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20
# Initialize progressbar

from typing import List, Union, Tuple
import numpy as np
from scipy.sparse.linalg import svds
import logging

from .nmtf_core import ntf_stack, ntf_solve
from .nmtf_utils import calc_leverage

logger = logging.getLogger(__name__)
EPSILON = np.finfo(np.float32).eps


# TODO (pcotte): Typing
# TODO (pcotte): group similar methods


def init(m, mmis, nc):
    n, p = m.shape
    mmis = mmis.astype(np.int)
    n_mmis = mmis.shape[0]
    if n_mmis == 0:
        missing_values_indexes = np.where(np.isnan(m) == 1)
        n_mmis = missing_values_indexes[0].size
        if n_mmis > 0:
            mmis = np.isnan(m) == 0
            mmis = mmis.astype(np.int)
            m[mmis == 0] = 0
    else:
        m[mmis == 0] = 0

    nc = int(nc)
    return n, p, nc, n_mmis


def nmf_init(m, mmis, mt0, mw0, nc) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize NMF components using NNSVD

    Parameters
    ----------
    m: Input matrix
    mmis: Define missing values (0 = missing cell, 1 = real cell)
    mt0: Initial left hand matrix (may be empty)
    mw0: Initial right hand matrix (may be empty)
    nc: NMF rank

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]: Left hand matrix and right hand matrix

    Reference
    ---------
    C. Boutsidis, E. Gallopoulos (2008) SVD based initialization: A head start for nonnegative matrix factorization
    Pattern Recognition Pattern Recognition Volume 41, Issue 4, April 2008, Pages 1350-1362
    """

    n, p, nc, n_mmis = init(m, mmis, nc)

    mt = np.copy(mt0)
    mw = np.copy(mw0)
    if (mt.shape[0] == 0) or (mw.shape[0] == 0):
        # Note that if there are missing values, SVD is performed on matrix imputed with 0's
        np.random.seed(3)
        if nc >= min(n, p):
            # arpack does not accept to factorize at full rank -> need to duplicate in both dimensions to force it work
            # noinspection PyTypeChecker
            t, d, w = svds(
                np.concatenate((np.concatenate((m, m), axis=1), np.concatenate((m, m), axis=1)), axis=0),
                k=nc,
                v0=np.random.uniform(size=2 * min(n, p)),
            )
            d /= 2
            # svd causes mem allocation problem with large matrices
            # t, d, w = np.linalg.svd(m)
            # mt = t
            # mw = w.T
        else:
            t, d, w = svds(m, k=nc, v0=np.random.uniform(size=min(n, p)))
            # t, d, w = np.linalg.svd(m)

        mt = t[:n, :nc]
        mw = w[:nc, :p].T
        # svds returns singular vectors in reverse order
        mt = mt[:, ::-1]
        mw = mw[:, ::-1]

    for k in range(0, nc):
        u1 = mt[:, k]
        u2 = -mt[:, k]
        u1[u1 < 0] = 0
        u2[u2 < 0] = 0
        v1 = mw[:, k]
        v2 = -mw[:, k]
        v1[v1 < 0] = 0
        v2[v2 < 0] = 0
        u1 = np.reshape(u1, (n, 1))
        v1 = np.reshape(v1, (1, p))
        u2 = np.reshape(u2, (n, 1))
        v2 = np.reshape(v2, (1, p))
        if np.linalg.norm(u1 @ v1) > np.linalg.norm(u2 @ v2):
            mt[:, k] = np.reshape(u1, n)
            mw[:, k] = np.reshape(v1, p)
        else:
            mt[:, k] = np.reshape(u2, n)
            mw[:, k] = np.reshape(v2, p)

    # Warm up using multiplicative rules
    precision = EPSILON
    mt += precision
    mw += precision
    for _ in range(0, 100):
        if n_mmis > 0:
            mw = mw * ((mt.T @ (m * mmis)) / (mt.T @ ((mt @ mw.T) * mmis) + precision)).T
            mt = mt * ((m * mmis) @ mw / (((mt @ mw.T) * mmis) @ mw + precision))
        else:
            mw = mw * ((mt.T @ m) / ((mt.T @ mt) @ mw.T + precision)).T
            mt = mt * (m @ mw / (mt @ (mw.T @ mw) + precision))

    # np.savetxt("C:/Users/paul_/PycharmProjects/nmtf_private/tests/data/datatest_W.csv", mt)
    # np.savetxt("C:/Users/paul_/PycharmProjects/nmtf_private/tests/data/datatest_H.csv", mw)

    return mt, mw


def init_ntf_type_1(m, mmis, n_blocks, nc, mt_nmf, mw_nmf, tolerance, log_iter, status0, my_status_box, n, p):
    # Init legacy
    mstacked, mmis_stacked = ntf_stack(m=m, mmis=mmis, n_blocks=n_blocks)
    nc2 = min(nc, n_blocks)  # factorization rank can't be > number of blocks
    if (mt_nmf.shape[0] == 0) or (mw_nmf.shape[0] == 0):
        mt_nmf, mw_nmf = nmf_init(m=mstacked, mmis=mmis_stacked, mt0=np.array([]), mw0=np.array([]), nc=nc2)
    else:
        mt_nmf, mw_nmf = nmf_init(m=mstacked, mmis=mmis_stacked, mt0=mt_nmf, mw0=mw_nmf, nc=nc2)

    # Quick NMF
    _, mt_nmf, mw_nmf, mb, diff, cancel_pressed = ntf_solve(
        m=mstacked,
        mmis=mmis_stacked,
        mt0=mt_nmf,
        mw0=mw_nmf,
        mb0=np.array([]),
        nc=nc2,
        tolerance=tolerance,
        log_iter=log_iter,
        status0=status0,
        max_iterations=10,
        nmf_fix_user_lhe=0,
        nmf_fix_user_rhe=0,
        nmf_fix_user_bhe=1,
        nmf_sparse_level=0,
        ntf_unimodal=0,
        ntf_smooth=0,
        ntf_left_components=0,
        ntf_right_components=0,
        ntf_block_components=0,
        n_blocks=1,
        nmf_priors=np.array([]),
        my_status_box=my_status_box,
    )

    # Factorize Left vectors and distribute multiple factors if nc2 < nc
    mt = np.zeros((n, nc))
    mw = np.zeros((int(p / n_blocks), nc))
    mb = np.zeros((n_blocks, nc))
    n_fact = int(np.ceil(nc / n_blocks))
    for k in range(0, nc2):
        my_status_box.update_status(status="Start SVD...")
        u, d, v = svds(np.reshape(mt_nmf[:, k], (int(p / n_blocks), n)).T, k=n_fact)
        v = v.T
        # svds returns singular vectors in reverse order
        u = u[:, ::-1]
        v = v[:, ::-1]
        d = d[::-1]

        my_status_box.update_status(status="SVD completed")
        for i_fact in range(0, n_fact):
            ind = i_fact * n_blocks + k
            if ind < nc:
                u1 = u[:, i_fact]
                u2 = -u[:, i_fact]
                u1[u1 < 0] = 0
                u2[u2 < 0] = 0
                v1 = v[:, i_fact]
                v2 = -v[:, i_fact]
                v1[v1 < 0] = 0
                v2[v2 < 0] = 0
                u1 = np.reshape(u1, (n, 1))
                v1 = np.reshape(v1, (1, int(p / n_blocks)))
                u2 = np.reshape(u2, (n, 1))
                v2 = np.reshape(v2, (1, int(p / n_blocks)))
                if np.linalg.norm(u1 @ v1) > np.linalg.norm(u2 @ v2):
                    mt[:, ind] = np.reshape(u1, n)
                    mw[:, ind] = d[i_fact] * np.reshape(v1, int(p / n_blocks))
                else:
                    mt[:, ind] = np.reshape(u2, n)
                    mw[:, ind] = d[i_fact] * np.reshape(v2, int(p / n_blocks))

                mb[:, ind] = mw_nmf[:, k]
    return mt, mw, mb, nc2 - 1, cancel_pressed


def init_ntf_type_2(
    m,
    mmis,
    n_blocks,
    nc,
    mt_nmf,
    mw_nmf,
    ntf_unimodal,
    ntf_left_components,
    tolerance,
    log_iter,
    status0,
    my_status_box,
    n,
    p,
):
    # Init default
    if (mt_nmf.shape[0] == 0) or (mw_nmf.shape[0] == 0):
        mt_nmf, mw_nmf = nmf_init(m=m, mmis=mmis, mt0=np.array([]), mw0=np.array([]), nc=nc)
    else:
        mt_nmf, mw_nmf = nmf_init(m=m, mmis=mmis, mt0=mt_nmf, mw0=mw_nmf, nc=nc)

    # Quick NMF
    _, mt_nmf, mw_nmf, mb, diff, cancel_pressed = ntf_solve(
        m=m,
        mmis=mmis,
        mt0=mt_nmf,
        mw0=mw_nmf,
        mb0=np.array([]),
        nc=nc,
        tolerance=tolerance,
        log_iter=log_iter,
        status0=status0,
        max_iterations=10,
        nmf_fix_user_lhe=0,
        nmf_fix_user_rhe=0,
        nmf_fix_user_bhe=1,
        nmf_sparse_level=0,
        ntf_unimodal=0,
        ntf_smooth=0,
        ntf_left_components=0,
        ntf_right_components=0,
        ntf_block_components=0,
        n_blocks=1,
        nmf_priors=np.array([]),
        my_status_box=my_status_box,
    )

    # Factorize Left vectors
    mt = np.zeros((n, nc))
    mw = np.zeros((int(p / n_blocks), nc))
    mb = np.zeros((n_blocks, nc))

    for k in range(0, nc):
        my_status_box.update_status(status="Start SVD...")
        # noinspection PyTypeChecker
        u, d, v = svds(np.reshape(mw_nmf[:, k], (int(p / n_blocks), n_blocks)), k=1)
        v = v.T
        u = np.abs(u)
        v = np.abs(v)
        my_status_box.update_status(status="SVD completed")
        mt[:, k] = mt_nmf[:, k]
        mw[:, k] = d[0] * np.reshape(u, int(p / n_blocks))
        mb[:, k] = np.reshape(v, n_blocks)

    for k in range(0, nc):
        if (ntf_unimodal > 0) & (ntf_left_components > 0):
            #                 Enforce unimodal distribution
            tmax = np.argmax(mt[:, k])
            for i in range(tmax + 1, n):
                mt[i, k] = min(mt[i - 1, k], mt[i, k])

            for i in range(tmax - 1, -1, -1):
                mt[i, k] = min(mt[i + 1, k], mt[i, k])

    return mt, mw, mb, nc - 1, cancel_pressed


def ntf_init(
    m,
    mmis,
    mt_nmf,
    mw_nmf,
    nc,
    tolerance,
    log_iter,
    ntf_unimodal,
    ntf_left_components,
    ntf_right_components,
    ntf_block_components,
    n_blocks,
    init_type,
    my_status_box,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str, int]:
    """Initialize NTF components for HALS

    Parameters
    ----------
    m: Input tensor
    mmis: Define missing values (0 = missing cell, 1 = real cell)
    mt_nmf: initialization of LHM in NMF(unstacked tensor), may be empty
    mw_nmf: initialization of RHM of NMF(unstacked tensor), may be empty
    nc: NTF rank
    tolerance: Convergence threshold
    log_iter: Log results through iterations
    ntf_unimodal: Apply Unimodal constraint on factoring vectors
    ntf_left_components: Apply Unimodal/Smooth constraint on left hand matrix
    ntf_right_components: Apply Unimodal/Smooth constraint on right hand matrix
    ntf_block_components: Apply Unimodal/Smooth constraint on block hand matrix
    n_blocks: Number of NTF blocks
    init_type : integer, default 0\n
        * init_type = 0 : NMF initialization applied on the reshaped matrix [1st dim x vectorized (2nd & 3rd dim)]\n
        * init_type = 1 : NMF initialization applied on the reshaped matrix [vectorized (1st & 2nd dim) x 3rd dim]\n
    my_status_box

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str, int]\n
      * mt: Left hand matrix\n
      * mw: Right hand matrix\n
      * mb: Block hand matrix\n
    """

    n, p, nc, n_mmis = init(m, mmis, nc)

    n_blocks = int(n_blocks)
    init_type = int(init_type)

    status0 = f"Step 1 - Quick NMF Ncomp={nc}: "

    if init_type == 1:
        mt, mw, mb, k, cancel_pressed = init_ntf_type_1(
            m=m,
            mmis=mmis,
            n_blocks=n_blocks,
            nc=nc,
            mt_nmf=mt_nmf,
            mw_nmf=mw_nmf,
            tolerance=tolerance,
            log_iter=log_iter,
            status0=status0,
            my_status_box=my_status_box,
            n=n,
            p=p,
        )
    else:
        mt, mw, mb, k, cancel_pressed = init_ntf_type_2(
            m=m,
            mmis=mmis,
            n_blocks=n_blocks,
            nc=nc,
            mt_nmf=mt_nmf,
            mw_nmf=mw_nmf,
            ntf_unimodal=ntf_unimodal,
            ntf_left_components=ntf_left_components,
            tolerance=tolerance,
            log_iter=log_iter,
            status0=status0,
            my_status_box=my_status_box,
            n=n,
            p=p,
        )

    if ntf_unimodal > 0 and ntf_right_components > 0:
        # Enforce unimodal distribution
        wmax = np.argmax(mw[:, k])
        for j in range(wmax + 1, int(p / n_blocks)):
            mw[j, k] = min(mw[j - 1, k], mw[j, k])

        for j in range(wmax - 1, -1, -1):
            mw[j, k] = min(mw[j + 1, k], mw[j, k])

    if ntf_unimodal > 0 and ntf_block_components > 0:
        # Enforce unimodal distribution
        bmax = np.argmax(mb[:, k])
        for i_block in range(bmax + 1, n_blocks):
            mb[i_block, k] = min(mb[i_block - 1, k], mb[i_block, k])

        for i_block in range(bmax - 1, -1, -1):
            mb[i_block, k] = min(mb[i_block + 1, k], mb[i_block, k])

    return mt, mw, mb, [], "", cancel_pressed


def r_ntf_solve(
    m,
    mmis,
    mt0,
    mw0,
    mb0,
    nc,
    tolerance,
    log_iter,
    max_iterations,
    nmf_fix_user_lhe,
    nmf_fix_user_rhe,
    nmf_fix_user_bhe,
    nmf_algo,
    nmf_robust_n_runs,
    nmf_calculate_leverage,
    nmf_use_robust_leverage,
    nmf_sparse_level,
    ntf_unimodal,
    ntf_smooth,
    ntf_left_components,
    ntf_right_components,
    ntf_block_components,
    n_blocks,
    nmf_priors,
    my_status_box,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Union[np.ndarray, float],
    Union[np.ndarray, float],
    Union[np.ndarray, float],
    List[str],
    str,
    int,
]:
    """Estimate NTF matrices (robust version)

    Parameters
    ----------
    m: Input matrix
    mmis: Define missing values (0 = missing cell, 1 = real cell)
    mt0: Initial left hand matrix
    mw0: Initial right hand matrix
    mb0: Initial block hand matrix
    nc: NTF rank
    tolerance: Convergence threshold
    log_iter: Log results through iterations
    max_iterations: Max iterations
    nmf_fix_user_lhe: fix left hand matrix columns: = 1, else = 0
    nmf_fix_user_rhe: fix  right hand matrix columns: = 1, else = 0
    nmf_fix_user_bhe: fix  block hand matrix columns: = 1, else = 0
    nmf_algo: "non-robust" version or "robust" version
    nmf_robust_n_runs: Number of bootstrap runs
    nmf_calculate_leverage: Calculate leverages
    nmf_use_robust_leverage: Calculate leverages based on robust max across factoring columns
    nmf_sparse_level : sparsity level (as defined by Hoyer); +/- = make RHE/LHe sparse
    ntf_unimodal: Apply Unimodal constraint on factoring vectors
    ntf_smooth: Apply Smooth constraint on factoring vectors
    ntf_left_components: Apply Unimodal/Smooth constraint on left hand matrix
    ntf_right_components: Apply Unimodal/Smooth constraint on right hand matrix
    ntf_block_components: Apply Unimodal/Smooth constraint on block hand matrix
    n_blocks: Number of NTF blocks
    nmf_priors: Elements in mw that should be updated (others remain 0)
    my_status_box


    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[np.ndarray, float], Union[np.ndarray, float],
          Union[np.ndarray, float], List[str], str, int]\n
      * mt_conv: np.ndarray\n
          Convolutional Left hand matrix\n
      * mt: np.ndarray\n
          Left hand matrix\n
      * mw: np.ndarray\n
          Right hand matrix\n
      * mb: np.ndarray\n
          Block hand matrix\n
      * mt_pct: Union[np.ndarray, float]\n
          Percent robust clustered rows\n
      * mw_pct: Union[np.ndarray, float]\n
          Percent robust clustered columns\n
      * diff : float\n
          Objective minimum achieved\n
      * add_message: List[str]\n
      * err_message: str\n
      * cancel_pressed: int
    """

    add_message = []
    err_message = ""
    cancel_pressed = 0
    n, p0 = m.shape
    nc = int(nc)
    n_blocks = int(n_blocks)
    p = int(p0 / n_blocks)
    if nmf_fix_user_lhe * nmf_fix_user_rhe * nmf_fix_user_bhe == 1:
        return (
            np.zeros((n, nc)),
            mt0,
            mw0,
            mb0,
            np.zeros((n, p0)),
            np.ones((n, nc)),
            np.ones((p, nc)),
            add_message,
            err_message,
            cancel_pressed,
        )

    mmis = mmis.astype(np.int)
    n_mmis = mmis.shape[0]
    if n_mmis == 0:
        missing_values_indexes = np.where(np.isnan(m) == 1)
        n_mmis = missing_values_indexes[0].size
        if n_mmis > 0:
            mmis = np.isnan(m) == 0
            mmis = mmis.astype(np.int)
            m[mmis == 0] = 0
    else:
        m[mmis == 0] = 0

    nmf_robust_n_runs = int(nmf_robust_n_runs)
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    mb = np.copy(mb0)

    # Check parameter consistency (and correct if needed)
    if nc == 1 or nmf_algo == "non-robust":
        nmf_robust_n_runs = 0

    if nmf_robust_n_runs == 0:
        mt_pct = np.nan
        mw_pct = np.nan

    # Step 1: NTF
    status0 = f"Step 1 - NTF Ncomp={nc}:"
    mt_conv, mt, mw, mb, diff, cancel_pressed = ntf_solve(
        m=m,
        mmis=mmis,
        mt0=mt,
        mw0=mw,
        mb0=mb,
        nc=nc,
        tolerance=tolerance,
        log_iter=log_iter,
        status0=status0,
        max_iterations=max_iterations,
        nmf_fix_user_lhe=nmf_fix_user_lhe,
        nmf_fix_user_rhe=nmf_fix_user_rhe,
        nmf_fix_user_bhe=nmf_fix_user_bhe,
        nmf_sparse_level=nmf_sparse_level,
        ntf_unimodal=ntf_unimodal,
        ntf_smooth=ntf_smooth,
        ntf_left_components=ntf_left_components,
        ntf_right_components=ntf_right_components,
        ntf_block_components=ntf_block_components,
        n_blocks=n_blocks,
        nmf_priors=nmf_priors,
        my_status_box=my_status_box,
    )

    mtsup = np.copy(mt)
    mwsup = np.copy(mw)
    mbsup = np.copy(mb)
    diff_sup = diff
    # Bootstrap to assess robust clustering
    if nmf_robust_n_runs > 1:
        #     Update mwsup
        mw_pct = np.zeros((p, nc))
        mw_blk = np.zeros((p, nmf_robust_n_runs * nc))
        for i_bootstrap in range(0, nmf_robust_n_runs):
            boot = np.random.randint(n, size=n)
            status0 = f"Step 2 - boot {i_bootstrap + 1}/{nmf_robust_n_runs} NTF Ncomp={nc}"
            if n_mmis > 0:
                mt_conv, mt, mw, mb, diff, cancel_pressed = ntf_solve(
                    m=m[boot, :],
                    mmis=mmis[boot, :],
                    mt0=mtsup[boot, :],
                    mw0=mwsup,
                    mb0=mb,
                    nc=nc,
                    tolerance=1.0e-3,
                    log_iter=log_iter,
                    status0=status0,
                    max_iterations=max_iterations,
                    nmf_fix_user_lhe=1,
                    nmf_fix_user_rhe=0,
                    nmf_fix_user_bhe=nmf_fix_user_bhe,
                    nmf_sparse_level=nmf_sparse_level,
                    ntf_unimodal=ntf_unimodal,
                    ntf_smooth=ntf_smooth,
                    ntf_left_components=ntf_left_components,
                    ntf_right_components=ntf_right_components,
                    ntf_block_components=ntf_block_components,
                    n_blocks=n_blocks,
                    nmf_priors=nmf_priors,
                    my_status_box=my_status_box,
                )
            else:
                mt_conv, mt, mw, mb, diff, cancel_pressed = ntf_solve(
                    m=m[boot, :],
                    mmis=np.array([]),
                    mt0=mtsup[boot, :],
                    mw0=mwsup,
                    mb0=mb,
                    nc=nc,
                    tolerance=1.0e-3,
                    log_iter=log_iter,
                    status0=status0,
                    max_iterations=max_iterations,
                    nmf_fix_user_lhe=1,
                    nmf_fix_user_rhe=0,
                    nmf_fix_user_bhe=nmf_fix_user_bhe,
                    nmf_sparse_level=nmf_sparse_level,
                    ntf_unimodal=ntf_unimodal,
                    ntf_smooth=ntf_smooth,
                    ntf_left_components=ntf_left_components,
                    ntf_right_components=ntf_right_components,
                    ntf_block_components=ntf_block_components,
                    n_blocks=n_blocks,
                    nmf_priors=nmf_priors,
                    my_status_box=my_status_box,
                )

            for k in range(0, nc):
                mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] = mw[:, k]

            mwn = np.zeros((p, nc))
            for k in range(0, nc):
                scale_mw = np.linalg.norm(mw_blk[:, k * nmf_robust_n_runs + i_bootstrap])
                if scale_mw > 0:
                    mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] = (
                        mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] / scale_mw
                    )

                mwn[:, k] = mw_blk[:, k * nmf_robust_n_runs + i_bootstrap]

            col_clust = np.zeros(p, dtype=int)
            if nmf_calculate_leverage > 0:
                mwn, add_message, err_message, cancel_pressed = calc_leverage(
                    mwn, nmf_use_robust_leverage, add_message, my_status_box
                )

            for j in range(0, p):
                col_clust[j] = np.argmax(np.array(mwn[j, :]))
                mw_pct[j, col_clust[j]] = mw_pct[j, col_clust[j]] + 1

        mw_pct = mw_pct / nmf_robust_n_runs

        #     Update Mtsup
        mt_pct = np.zeros((n, nc))
        for i_bootstrap in range(0, nmf_robust_n_runs):
            status0 = f"Step 3 - boot {i_bootstrap + 1}/{nmf_robust_n_runs} NTF Ncomp={nc}"
            mw = np.zeros((p, nc))
            for k in range(0, nc):
                mw[:, k] = mw_blk[:, k * nmf_robust_n_runs + i_bootstrap]

            mt_conv, mt, mw, mb, diff, cancel_pressed = ntf_solve(
                m=m,
                mmis=mmis,
                mt0=mtsup,
                mw0=mw,
                mb0=mb,
                nc=nc,
                tolerance=1.0e-3,
                log_iter=log_iter,
                status0=status0,
                max_iterations=max_iterations,
                nmf_fix_user_lhe=0,
                nmf_fix_user_rhe=1,
                nmf_fix_user_bhe=nmf_fix_user_bhe,
                nmf_sparse_level=nmf_sparse_level,
                ntf_unimodal=ntf_unimodal,
                ntf_smooth=ntf_smooth,
                ntf_left_components=ntf_left_components,
                ntf_right_components=ntf_right_components,
                ntf_block_components=ntf_block_components,
                n_blocks=n_blocks,
                nmf_priors=nmf_priors,
                my_status_box=my_status_box,
            )

            row_clust = np.zeros(n, dtype=int)
            if nmf_calculate_leverage > 0:
                mtn, add_message, err_message, cancel_pressed = calc_leverage(
                    v=mt,
                    nmf_use_robust_leverage=nmf_use_robust_leverage,
                    add_message=add_message,
                    my_status_box=my_status_box,
                )
            else:
                mtn = mt

            for i in range(0, n):
                row_clust[i] = np.argmax(mtn[i, :])
                mt_pct[i, row_clust[i]] = mt_pct[i, row_clust[i]] + 1

        mt_pct = mt_pct / nmf_robust_n_runs

    mt = mtsup
    mw = mwsup
    mb = mbsup
    diff = diff_sup

    # TODO (pcotte) : mt_pct and mw_pct can be not yet referenced : fix that
    return mt_conv, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed


def init_factorization(m, n_components):
    n, p = m.shape
    # Identify missing values
    mmis = np.array([])
    mmis = mmis.astype(np.int)
    missing_values_indexes = np.where(np.isnan(m) == 1)
    n_mmis = missing_values_indexes[0].size
    if n_mmis > 0:
        mmis = np.isnan(m) == 0
        mmis = mmis.astype(np.int)
        m[mmis == 0] = 0
    else:
        m[mmis == 0] = 0

    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components
    return m, n, p, mmis, nc
