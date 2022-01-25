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
from .nmtf_utils import calc_leverage, StatusBoxTqdm, nmf_det, build_clusters, global_sign

logger = logging.getLogger(__name__)
EPSILON = np.finfo(np.float32).eps


# TODO (pcotte): Typing
# TODO (pcotte): group similar methods


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
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    if (mt.shape[0] == 0) or (mw.shape[0] == 0):
        # Note that if there are missing values, SVD is performed on matrix imputed with 0's
        if nc >= min(n, p):
            # arpack does not accept to factorize at full rank -> need to duplicate in both dimensions to force it work
            # noinspection PyTypeChecker
            t, d, w = svds(
                np.concatenate((np.concatenate((m, m), axis=1), np.concatenate((m, m), axis=1)), axis=0), k=nc
            )
            t *= np.sqrt(2)
            w *= np.sqrt(2)
            d /= 2
            # svd causes mem allocation problem with large matrices
            # t, d, w = np.linalg.svd(M)
            # mt = t
            # mw = w.T
        else:
            t, d, w = svds(m, k=nc)

        mt = t[:n, :]
        mw = w[:, :p].T
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

    return mt, mw


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
    init_type : integer, default 0
        * init_type = 0 : NMF initialization applied on the reshaped matrix [1st dim x vectorized (2nd & 3rd dim)]
        * init_type = 1 : NMF initialization applied on the reshaped matrix [vectorized (1st & 2nd dim) x 3rd dim]
    my_status_box

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], str, int]
        mt: Left hand matrix
        mw: Right hand matrix
        mb: Block hand matrix
    """
    add_message = []

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
    n_blocks = int(n_blocks)
    init_type = int(init_type)

    status0 = f"Step 1 - Quick NMF Ncomp={nc}: "

    if init_type == 1:
        # Init legacy
        mstacked, mmis_stacked = ntf_stack(m=m, mmis=mmis, n_blocks=n_blocks)
        nc2 = min(nc, n_blocks)  # factorization rank can't be > number of blocks
        if (mt_nmf.shape[0] == 0) or (mw_nmf.shape[0] == 0):
            mt_nmf, mw_nmf = nmf_init(m=mstacked, mmis=mmis_stacked, mt0=np.array([]), mw0=np.array([]), nc=nc2)
        else:
            mt_nmf, mw_nmf = nmf_init(m=mstacked, mmis=mmis_stacked, mt0=mt_nmf, mw0=mw_nmf, nc=nc2)

        # Quick NMF
        dummy, mt_nmf, mw_nmf, mb, diff, cancel_pressed = ntf_solve(
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
        err_message = ""

        # Factorize Left vectors and distribute multiple factors if nc2 < nc
        mt = np.zeros((n, nc))
        mw = np.zeros((int(p / n_blocks), nc))
        mb = np.zeros((n_blocks, nc))
        n_fact = int(np.ceil(nc / n_blocks))
        for k in range(0, nc2):
            my_status_box.update_status(delay=1, status="Start SVD...")
            u, d, v = svds(np.reshape(mt_nmf[:, k], (int(p / n_blocks), n)).T, k=n_fact)
            v = v.T
            # svds returns singular vectors in reverse order
            u = u[:, ::-1]
            v = v[:, ::-1]
            d = d[::-1]

            my_status_box.update_status(delay=1, status="SVD completed")
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
    else:
        # Init default
        if (mt_nmf.shape[0] == 0) or (mw_nmf.shape[0] == 0):
            mt_nmf, mw_nmf = nmf_init(m=m, mmis=mmis, mt0=np.array([]), mw0=np.array([]), nc=nc)
        else:
            mt_nmf, mw_nmf = nmf_init(m=m, mmis=mmis, mt0=mt_nmf, mw0=mw_nmf, nc=nc)

        # Quick NMF
        dummy, mt_nmf, mw_nmf, mb, diff, cancel_pressed = ntf_solve(
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
        err_message = ""

        # Factorize Left vectors
        mt = np.zeros((n, nc))
        mw = np.zeros((int(p / n_blocks), nc))
        mb = np.zeros((n_blocks, nc))

        for k in range(0, nc):
            my_status_box.update_status(delay=1, status="Start SVD...")
            # noinspection PyTypeChecker
            u, d, v = svds(np.reshape(mw_nmf[:, k], (int(p / n_blocks), n_blocks)), k=1)
            v = v.T
            u = np.abs(u)
            v = np.abs(v)
            my_status_box.update_status(delay=1, status="SVD completed")
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

    if (ntf_unimodal > 0) & (ntf_right_components > 0):
        #                 Enforce unimodal distribution
        # TODO (pcotte) : k seems to be defined as a for loop iterator. VERY bad practice !
        wmax = np.argmax(mw[:, k])
        for j in range(wmax + 1, int(p / n_blocks)):
            mw[j, k] = min(mw[j - 1, k], mw[j, k])

        for j in range(wmax - 1, -1, -1):
            mw[j, k] = min(mw[j + 1, k], mw[j, k])

    if (ntf_unimodal > 0) & (ntf_block_components > 0):
        #                 Enforce unimodal distribution
        bmax = np.argmax(mb[:, k])
        for i_block in range(bmax + 1, n_blocks):
            mb[i_block, k] = min(mb[i_block - 1, k], mb[i_block, k])

        for i_block in range(bmax - 1, -1, -1):
            mb[i_block, k] = min(mb[i_block + 1, k], mb[i_block, k])

    return mt, mw, mb, add_message, err_message, cancel_pressed


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
    nmf_algo: =5: Non-robust version, =6: Robust version
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
          Union[np.ndarray, float], List[str], str, int]
        mt_conv: np.ndarray
            Convolutional Left hand matrix
        mt: np.ndarray
            Left hand matrix
        mw: np.ndarray
            Right hand matrix
        mb: np.ndarray
            Block hand matrix
        mt_pct: Union[np.ndarray, float]
            Percent robust clustered rows
        mw_pct: Union[np.ndarray, float]
            Percent robust clustered columns
        diff : float
            Objective minimum achieved
        add_message: List[str]
        err_message: str
        cancel_pressed: int
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
    if (nc == 1) | (nmf_algo == 5):
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


def non_negative_factorization(
    x,
    w=None,
    h=None,
    n_components=None,
    update_w=True,
    update_h=True,
    n_bootstrap=None,
    tol=1e-6,
    max_iter=150,
    regularization=None,
    sparsity=0,
    leverage="standard",
    random_state=None,
    verbose=0,
) -> dict:
    """Compute Non-negative Matrix Factorization (NMF)

    Find two non-negative matrices (W, H) such as x = W @ H.T + Error.
    This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is minimized with an alternating minimization of W
    and H.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features)
        Constant matrix.
    w : array-like, shape (n_samples, n_components)
        prior W
        If n_update_W == 0 , it is used as a constant, to solve for H only.
    h : array-like, shape (n_features, n_components)
        prior H
        If n_update_H = 0 , it is used as a constant, to solve for W only.
    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)
    update_w : boolean, default: True
        Update or keep W fixed
    update_h : boolean, default: True
        Update or keep H fixed
    n_bootstrap : integer, default: 0
        Number of bootstrap runs.
    tol : float, default: 1e-6
        Tolerance of the stopping condition.
    max_iter : integer, default: 200
        Maximum number of iterations.
    regularization :  None | 'components' | 'transformation'
        Select whether the regularization affects the components (H), the
        transformation (W) or none of them.
    sparsity : float, default: 0
        Sparsity target with 0 <= sparsity < 1 representing either:
        - the % rows in W or H set to 0 (when use_hals = False)
        - the mean % rows per column in W or H set to 0 (when use_hals = True)
        sparsity == 1: adaptive sparsity through hard thresholding and hhi
    leverage :  None | 'standard' | 'robust', default 'standard'
        Calculate leverage of W and H rows on each component.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------
    dict: Estimator with following entries
        W : array-like, shape (n_samples, n_components)
            Solution to the non-negative least squares problem.
        H : array-like, shape (n_features, n_components)
            Solution to the non-negative least squares problem.
        volume : scalar, volume occupied by W and H
        WB : array-like, shape (n_samples, n_components)
            Percent consistently clustered rows for each component.
            only if n_bootstrap > 0.
        HB : array-like, shape (n_features, n_components)
            Percent consistently clustered columns for each component.
            only if n_bootstrap > 0.
        B : array-like, shape (n_observations, n_components) or (n_features, n_components)
            only if active convex variant, H = B.T @ X or W = X @ B
        diff : Objective minimum achieved
    """

    m = x
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

    nmf_algo = 2
    log_iter = verbose
    my_status_box = StatusBoxTqdm(verbose=log_iter)
    tolerance = tol
    if (w is None) & (h is None):
        mt, mw = nmf_init(m, mmis, np.array([]), np.array([]), nc)
    else:
        if h is None:
            mw = np.ones((p, nc))
        elif w is None:
            mt = np.ones((n, nc))

        for k in range(0, nc):
            # TODO (pcotte) : mt and mw can be not yet referenced : fix that
            mt[:, k] = mt[:, k] / np.linalg.norm(mt[:, k])
            mw[:, k] = mw[:, k] / np.linalg.norm(mw[:, k])

    if n_bootstrap is None:
        nmf_robust_n_runs = 0
    else:
        nmf_robust_n_runs = n_bootstrap

    if nmf_robust_n_runs > 1:
        nmf_algo += 2

    if update_w is True:
        nmf_fix_user_lhe = 0
    else:
        nmf_fix_user_lhe = 1

    if update_h is True:
        nmf_fix_user_rhe = 0
    else:
        nmf_fix_user_rhe = 1

    max_iterations = max_iter
    if regularization is None:
        nmf_sparse_level = 0
    else:
        if regularization == "components":
            nmf_sparse_level = sparsity
        elif regularization == "transformation":
            nmf_sparse_level = -sparsity
        else:
            nmf_sparse_level = 0

    if leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif leverage == "robust":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 1
    else:
        nmf_calculate_leverage = 0
        nmf_use_robust_leverage = 0

    if random_state is not None:
        random_seed = random_state
        np.random.seed(random_seed)

    if nmf_algo <= 2:
        ntf_algo = 5
    else:
        ntf_algo = 6

    dummy, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed = r_ntf_solve(
        m=m,
        mmis=mmis,
        mt0=mt,
        mw0=mw,
        mb0=np.array([]),
        nc=nc,
        tolerance=tolerance,
        log_iter=log_iter,
        max_iterations=max_iterations,
        nmf_fix_user_lhe=nmf_fix_user_lhe,
        nmf_fix_user_rhe=nmf_fix_user_rhe,
        nmf_fix_user_bhe=1,
        nmf_algo=ntf_algo,
        nmf_robust_n_runs=nmf_robust_n_runs,
        nmf_calculate_leverage=nmf_calculate_leverage,
        nmf_use_robust_leverage=nmf_use_robust_leverage,
        nmf_sparse_level=nmf_sparse_level,
        ntf_unimodal=0,
        ntf_smooth=0,
        ntf_left_components=0,
        ntf_right_components=0,
        ntf_block_components=0,
        n_blocks=1,
        nmf_priors=np.array([]),
        my_status_box=my_status_box,
    )
    mev = np.ones(nc)
    if (nmf_fix_user_lhe == 0) & (nmf_fix_user_rhe == 0):
        # Scale
        for k in range(0, nc):
            scale_mt = np.linalg.norm(mt[:, k])
            scale_mw = np.linalg.norm(mw[:, k])
            mev[k] = scale_mt * scale_mw
            if mev[k] > 0:
                mt[:, k] = mt[:, k] / scale_mt
                mw[:, k] = mw[:, k] / scale_mw

    volume = nmf_det(mt, mw, 1)

    for message in add_message:
        logger.info(message)

    my_status_box.close()

    # Order by decreasing scale
    r_mev = np.argsort(-mev)
    mev = mev[r_mev]
    mt = mt[:, r_mev]
    mw = mw[:, r_mev]
    if isinstance(mt_pct, np.ndarray):
        mt_pct = mt_pct[:, r_mev]
        mw_pct = mw_pct[:, r_mev]

    # Scale by max com p
    for k in range(0, nc):
        max_col = np.max(mt[:, k])
        if max_col > 0:
            mt[:, k] /= max_col
            mw[:, k] *= mev[k] * max_col
            mev[k] = 1
        else:
            mev[k] = 0

    estimator = {}
    if nmf_robust_n_runs <= 1:
        estimator.update([("W", mt), ("H", mw), ("volume", volume), ("diff", diff)])
    else:
        estimator.update([("W", mt), ("H", mw), ("volume", volume), ("WB", mt_pct), ("HB", mw_pct), ("diff", diff)])

    return estimator


def nmf_predict(
    estimator, leverage="robust", blocks=None, cluster_by_stability=False, custom_order=False, verbose=0
) -> dict:
    """Derives ordered sample and feature indexes for future use in ordered heatmaps

    Parameters
    ----------
    estimator: dict
        As returned by non_negative_factorization
    leverage :  None | 'standard' | 'robust', default 'robust'
        Calculate leverage of W and H rows on each component.
    blocks : array-like, shape(n_blocks), default None
        Size of each block (if any) in ordered heatmap.
    cluster_by_stability : boolean, default False
         Use stability instead of leverage to assign samples/features to clusters
    custom_order :  boolean, default False
         if False samples/features with highest leverage or stability appear on top of each cluster
         if True within cluster ordering is modified to suggest a continuum  between adjacent clusters
    verbose : integer, default: 0
        The verbosity level (0/1).

    Returns
    -------
    dict: Completed estimator with following entries:
        WL : array-like, shape (n_samples, n_components)
             Sample leverage on each component
        HL : array-like, shape (n_features, n_components)
             Feature leverage on each component
        QL : array-like, shape (n_blocks, n_components)
             Block leverage on each component (NTF only)
        WR : vector-like, shape (n_samples)
             Ranked sample indexes (by cluster and leverage or stability)
             Used to produce ordered heatmaps
        HR : vector-like, shape (n_features)
             Ranked feature indexes (by cluster and leverage or stability)
             Used to produce ordered heatmaps
        WN : vector-like, shape (n_components)
             Sample cluster bounds in ordered heatmap
        HN : vector-like, shape (n_components)
             Feature cluster bounds in ordered heatmap
        WC : vector-like, shape (n_samples)
             Sample assigned cluster
        HC : vector-like, shape (n_features)
             Feature assigned cluster
        QC : vector-like, shape (size(blocks))
             Block assigned cluster (NTF only)
    """
    mt = estimator["W"]
    mw = estimator["H"]
    if "Q" in estimator:
        # X is a 3D tensor, in unfolded form of a 2D array
        # horizontal concatenation of blocks of equal size.
        mb = estimator["Q"]
        nmf_algo = 5
        n_blocks = mb.shape[0]
        blk_size = mw.shape[0] * np.ones(n_blocks)
    else:
        mb = np.array([])
        nmf_algo = 0
        if blocks is None:
            n_blocks = 1
            blk_size = np.array([mw.shape[0]])
        else:
            n_blocks = blocks.shape[0]
            blk_size = blocks

    if "WB" in estimator:
        mt_pct = estimator["WB"]
    else:
        mt_pct = None

    if "HB" in estimator:
        mw_pct = estimator["HB"]
    else:
        mw_pct = None

    if leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif leverage == "robust":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 1
    else:
        nmf_calculate_leverage = 0
        nmf_use_robust_leverage = 0

    if cluster_by_stability is True:
        nmf_robust_cluster_by_stability = 1
    else:
        nmf_robust_cluster_by_stability = 0

    if custom_order is True:
        cell_plot_ordered_clusters = 1
    else:
        cell_plot_ordered_clusters = 0

    add_message = []
    my_status_box = StatusBoxTqdm(verbose=verbose)

    (
        mtn,
        mwn,
        mbn,
        r_ct,
        r_cw,
        n_ct,
        n_cw,
        row_clust,
        col_clust,
        block_clust,
        add_message,
        err_message,
        cancel_pressed,
    ) = build_clusters(
        mt=mt,
        mw=mw,
        mb=mb,
        mt_pct=mt_pct,
        mw_pct=mw_pct,
        n_blocks=n_blocks,
        blk_size=blk_size,
        nmf_calculate_leverage=nmf_calculate_leverage,
        nmf_use_robust_leverage=nmf_use_robust_leverage,
        nmf_algo=nmf_algo,
        nmf_robust_cluster_by_stability=nmf_robust_cluster_by_stability,
        cell_plot_ordered_clusters=cell_plot_ordered_clusters,
        add_message=add_message,
        my_status_box=my_status_box,
    )
    for message in add_message:
        logger.info(message)

    my_status_box.close()
    if "Q" in estimator:
        estimator.update(
            [
                ("WL", mtn),
                ("HL", mwn),
                ("WR", r_ct),
                ("HR", r_cw),
                ("WN", n_ct),
                ("HN", n_cw),
                ("WC", row_clust),
                ("HC", col_clust),
                ("QL", mbn),
                ("QC", block_clust),
            ]
        )
    else:
        estimator.update(
            [
                ("WL", mtn),
                ("HL", mwn),
                ("WR", r_ct),
                ("HR", r_cw),
                ("WN", n_ct),
                ("HN", n_cw),
                ("WC", row_clust),
                ("HC", col_clust),
                ("QL", None),
                ("QC", None),
            ]
        )
    return estimator


# TODO (pcotte): this function is not called by any pytest. Make a pytest calling it.
def nmf_permutation_test_score(estimator, y, n_permutations=100, verbose=0) -> dict:
    """Do a permutation test to assess association between ordered samples and some covariate

    Parameters
    ----------
    estimator : tuplet as returned by non_negative_factorization and nmf_predict
    y :  array-like, group to be predicted
    n_permutations :  integer, default: 100
    verbose : integer, default: 0
        The verbosity level (0/1).

    Returns
    -------
    dict: Completed estimator with following entries:
        score : float
             The true score without permuting targets.
        pvalue : float
             The p-value, which approximates the probability that the score would be obtained by chance.
        CS : array-like, shape(n_components)
             The size of each cluster
        CP : array-like, shape(n_components)
             The pvalue of the most significant group within each cluster
        CG : array-like, shape(n_components)
             The index of the most significant group within each cluster
        CN : array-like, shape(n_components, n_groups)
             The size of each group within each cluster
    """
    mt = estimator["W"]
    r_ct = estimator["WR"]
    n_ct = estimator["WN"]
    row_groups = y
    uniques, index = np.unique([row for row in row_groups], return_index=True)
    list_groups = row_groups[index]
    nb_groups = list_groups.shape[0]
    ngroup = np.zeros(nb_groups)
    for group in range(0, nb_groups):
        ngroup[group] = np.where(row_groups == list_groups[group])[0].shape[0]

    nrun = n_permutations
    my_status_box = StatusBoxTqdm(verbose=verbose)
    cluster_size, pglob, prun, cluster_prob, cluster_group, cluster_ngroup, cancel_pressed = global_sign(
        nrun, nb_groups, mt, r_ct, n_ct, row_groups, list_groups, ngroup, my_status_box
    )

    estimator.update(
        [
            ("score", prun),
            ("pvalue", pglob),
            ("CS", cluster_size),
            ("CP", cluster_prob),
            ("CG", cluster_group),
            ("CN", cluster_ngroup),
        ]
    )
    return estimator


def non_negative_tensor_factorization(
    x,
    n_blocks,
    w=None,
    h=None,
    q=None,
    n_components=None,
    update_w=True,
    update_h=True,
    update_q=True,
    regularization=None,
    sparsity=0,
    unimodal=False,
    smooth=False,
    apply_left=False,
    apply_right=False,
    apply_block=False,
    n_bootstrap=None,
    tol=1e-6,
    max_iter=150,
    leverage="standard",
    random_state=None,
    init_type=0,
    verbose=0,
) -> dict:
    """Compute Non-negative Tensor Factorization (NTF)

    Find three non-negative matrices (W, H, F) such as x = W @@ H @@ F + Error (@@ = tensor product).
    This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is minimized with an alternating minimization of W
    and H.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features x n_blocks)
        Constant matrix.
        X is a tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd dimensions.
    n_blocks : integer
    w : array-like, shape (n_samples, n_components)
        prior W
    h : array-like, shape (n_features, n_components)
        prior H
    q : array-like, shape (n_blocks, n_components)
        prior Q
    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)
    update_w : boolean, default: True
        Update or keep W fixed
    update_h : boolean, default: True
        Update or keep H fixed
    update_q : boolean, default: True
        Update or keep Q fixed
    regularization :  None | 'components' | 'transformation'
        Select whether the regularization affects the components (H), the
        transformation (W) or none of them.
    sparsity : float, default: 0
        Sparsity target with 0 <= sparsity <= 1 representing the mean % rows per column in W or H set to 0
    unimodal : Boolean, default: False
    smooth : Boolean, default: False
    apply_left : Boolean, default: False
    apply_right : Boolean, default: False
    apply_block : Boolean, default: False
    n_bootstrap : integer, default: 0
        Number of bootstrap runs.
    tol : float, default: 1e-6
        Tolerance of the stopping condition.
    max_iter : integer, default: 200
        Maximum number of iterations.
    leverage :  None | 'standard' | 'robust', default 'standard'
        Calculate leverage of W and H rows on each component.
    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    init_type : integer, default 0
        init_type = 0 : NMF initialization applied on the reshaped matrix [1st dim x vectorized (2nd & 3rd dim)]
        init_type = 1 : NMF initialization applied on the reshaped matrix [vectorized (1st & 2nd dim) x 3rd dim]
    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------
    dict: Estimator with following entries
        W : array-like, shape (n_samples, n_components)
            Solution to the non-negative least squares problem.
        H : array-like, shape (n_features, n_components)
            Solution to the non-negative least squares problem.
        Q : array-like, shape (n_blocks, n_components)
            Solution to the non-negative least squares problem.
        volume : scalar, volume occupied by W and H
        WB : array-like, shape (n_samples, n_components)
            Percent consistently clustered rows for each component.
            only if n_bootstrap > 0.
        HB : array-like, shape (n_features, n_components)
            Percent consistently clustered columns for each component.
            only if n_bootstrap > 0.

    Reference
    ---------
    A. Cichocki, P.H.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

    """

    m = x
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

    n_blocks = n_blocks
    p_block = int(p / n_blocks)
    tolerance = tol
    log_iter = verbose
    if regularization is None:
        nmf_sparse_level = 0
    else:
        if regularization == "components":
            nmf_sparse_level = sparsity
        elif regularization == "transformation":
            nmf_sparse_level = -sparsity
        else:
            nmf_sparse_level = 0
    ntf_unimodal = unimodal
    ntf_smooth = smooth
    ntf_left_components = apply_left
    ntf_right_components = apply_right
    ntf_block_components = apply_block
    if random_state is not None:
        random_seed = random_state
        np.random.seed(random_seed)

    my_status_box = StatusBoxTqdm(verbose=log_iter)

    if (w is None) & (h is None) & (q is None):
        mt0, mw0, mb0, add_message, err_message, cancel_pressed = ntf_init(
            m=m,
            mmis=mmis,
            mt_nmf=np.array([]),
            mw_nmf=np.array([]),
            nc=nc,
            tolerance=tolerance,
            log_iter=log_iter,
            ntf_unimodal=ntf_unimodal,
            ntf_left_components=ntf_left_components,
            ntf_right_components=ntf_right_components,
            ntf_block_components=ntf_block_components,
            n_blocks=n_blocks,
            init_type=init_type,
            my_status_box=my_status_box,
        )
    else:
        if w is None:
            mt0 = np.ones((n, nc))
        else:
            mt0 = np.copy(w)

        if h is None:
            mw0 = np.ones((p_block, nc))
        else:
            mw0 = np.copy(h)

        if q is None:
            mb0 = np.ones((n_blocks, nc))
        else:
            mb0 = np.copy(q)

        mfit = np.zeros((n, p))
        for k in range(0, nc):
            for i_block in range(0, n_blocks):
                mfit[:, i_block * p_block : (i_block + 1) * p_block] += (
                    mb0[i_block, k] * np.reshape(mt0[:, k], (n, 1)) @ np.reshape(mw0[:, k], (1, p_block))
                )

        scale_ratio = (np.linalg.norm(mfit) / np.linalg.norm(m)) ** (1 / 3)
        for k in range(0, nc):
            mt0[:, k] /= scale_ratio
            mw0[:, k] /= scale_ratio
            mb0[:, k] /= scale_ratio

        mfit = np.zeros((n, p))
        for k in range(0, nc):
            for i_block in range(0, n_blocks):
                mfit[:, i_block * p_block : (i_block + 1) * p_block] += (
                    mb0[i_block, k] * np.reshape(mt0[:, k], (n, 1)) @ np.reshape(mw0[:, k], (1, p_block))
                )

    max_iterations = max_iter

    if n_bootstrap is None:
        nmf_robust_n_runs = 0
    else:
        nmf_robust_n_runs = n_bootstrap

    if nmf_robust_n_runs <= 1:
        nmf_algo = 5
    else:
        nmf_algo = 6

    if leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif leverage == "robust":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 1
    else:
        nmf_calculate_leverage = 0
        nmf_use_robust_leverage = 0

    if random_state is not None:
        random_seed = random_state
        np.random.seed(random_seed)

    if update_w:
        nmf_fix_user_lhe = 0
    else:
        nmf_fix_user_lhe = 1

    if update_h:
        nmf_fix_user_rhe = 0
    else:
        nmf_fix_user_rhe = 1

    if update_q:
        nmf_fix_user_bhe = 0
    else:
        nmf_fix_user_bhe = 1

    mt_conv, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed = r_ntf_solve(
        m=m,
        mmis=mmis,
        mt0=mt0,
        mw0=mw0,
        mb0=mb0,
        nc=nc,
        tolerance=tolerance,
        log_iter=log_iter,
        max_iterations=max_iterations,
        nmf_fix_user_lhe=nmf_fix_user_lhe,
        nmf_fix_user_rhe=nmf_fix_user_rhe,
        nmf_fix_user_bhe=nmf_fix_user_bhe,
        nmf_algo=nmf_algo,
        nmf_robust_n_runs=nmf_robust_n_runs,
        nmf_calculate_leverage=nmf_calculate_leverage,
        nmf_use_robust_leverage=nmf_use_robust_leverage,
        nmf_sparse_level=nmf_sparse_level,
        ntf_unimodal=ntf_unimodal,
        ntf_smooth=ntf_smooth,
        ntf_left_components=ntf_left_components,
        ntf_right_components=ntf_right_components,
        ntf_block_components=ntf_block_components,
        n_blocks=n_blocks,
        nmf_priors=np.array([]),
        my_status_box=my_status_box,
    )

    volume = nmf_det(mt, mw, 1)

    for message in add_message:
        logger.info(message)

    my_status_box.close()

    estimator = {}
    if nmf_robust_n_runs <= 1:
        estimator.update([("W", mt), ("H", mw), ("Q", mb), ("volume", volume), ("diff", diff)])
    else:
        estimator.update(
            [("W", mt), ("H", mw), ("Q", mb), ("volume", volume), ("WB", mt_pct), ("HB", mw_pct), ("diff", diff)]
        )

    return estimator
