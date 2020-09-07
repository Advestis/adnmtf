"""Non-negative matrix and tensor factorization basic functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20
# Initialize progressbar
import math
import numpy as np
from sklearn.utils.extmath import randomized_svd

from .nmtf_core import nmf_solve, ntf_solve, ntf_solve_fast, ntf_stack
from .nmtf_utils import StatusBoxTqdm, global_sign, nmf_det, build_clusters
from .small_function import set_nmf_attributes, set_uv, update_m_mmis, update_col_clust, update_row_clust, update_m, \
    fast_code_and_svd_algo, do_svd_algo, do_mfit

import sys

if not hasattr(sys, "argv"):
    sys.argv = [""]

EPSILON = np.finfo(np.float32).eps
compatibility_flag = False


def nmf_init(m, mmis, mt0, mw0, nc, tolerance, log_iter, my_status_box):
    """Initialize NMF components using NNSVD

    Input:
        m: Input matrix
        mmis: Define missing values (0 = missing cell, 1 = real cell)
        mt0: Initial left hand matrix (may be empty)
        mw0: Initial right hand matrix (may be empty)
        nc: NMF rank
    Output:
        m: Left hand matrix
        mw: Right hand matrix

    Reference
    ---------

    x. Boutsidis, E. Gallopoulos (2008) SVD based initialization: a head start for nonnegative matrix factorization
    Pattern Recognition Pattern Recognition Volume 41, Issue 4, April 2008, Pages 1350-1362

    """

    n, p, n_mmis, m, mmis, nc = set_nmf_attributes(m, mmis, nc)
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    if (mt.shape[0] == 0) or (mw.shape[0] == 0):
        if n_mmis == 0:
            t, d, w = randomized_svd(m, n_components=nc, n_iter="auto", random_state=None)
            mt = t
            mw = w.T
        else:
            mt, d, mw, mmis, mmsr, mmsr2, add_message, err_message, cancel_pressed = r_svd_solve(
                m, mmis, nc, tolerance, log_iter, 0, "", 200, 1, 1, 1, my_status_box
            )

    for k in range(0, nc):
        u1, u2, v1, v2 = set_uv(mt, mw, k, n)
        v1 = np.reshape(v1, (1, p))
        u2 = np.reshape(u2, (n, 1))
        v2 = np.reshape(v2, (1, p))
        if np.linalg.norm(u1 @ v1) > np.linalg.norm(u2 @ v2):
            mt[:, k] = np.reshape(u1, n)
            mw[:, k] = np.reshape(v1, p)
        else:
            mt[:, k] = np.reshape(u2, n)
            mw[:, k] = np.reshape(v2, p)

    return [mt, mw]


def r_nmf_solve(
    m,
    mmis,
    mt0,
    mw0,
    nc,
    tolerance,
    precision,
    log_iter,
    max_iterations,
    nmf_algo,
    nmf_fix_user_lhe,
    nmf_fix_user_rhe,
    nmf_max_interm,
    nmf_sparse_level,
    nmf_robust_resample_columns,
    nmf_robust_n_runs,
    nmf_calculate_leverage,
    nmf_use_robust_leverage,
    nmf_find_parts,
    nmf_find_centroids,
    nmf_kernel,
    nmf_reweigh_columns,
    nmf_priors,
    my_status_box,
):
    """Estimate left and right hand matrices (robust version)

    Input:
         m: Input matrix
         mmis: Define missing values (0 = missing cell, 1 = real cell)
         mt0: Initial left hand matrix
         mw0: Initial right hand matrix
         nc: NMF rank
         tolerance: Convergence threshold
         precision: Replace 0-values in multiplication rules
         log_iter: Log results through iterations
          max_iterations: Max iterations
         nmf_algo: =1,3: Divergence; =2,4: Least squares;
         nmf_fix_user_lhe: = 1 => fixed left hand matrix columns
         nmf_fix_user_rhe: = 1 => fixed  right hand matrix columns
         nmf_max_interm: Max iterations for warmup multiplication rules
         nmf_sparse_level: Requested sparsity in terms of relative number of rows with 0 values in right hand matrix
         nmf_robust_resample_columns: Resample columns during bootstrap
         nmf_robust_n_runs: Number of bootstrap runs
         nmf_calculate_leverage: Calculate leverages
         nmf_use_robust_leverage: Calculate leverages based on robust max across factoring columns
         nmf_find_parts: Enforce convexity on left hand matrix
         nmf_find_centroids: Enforce convexity on right hand matrix
         nmf_kernel: Type of kernel used; 1: linear; 2: quadraitc; 3: radial
         nmf_reweigh_columns: Reweigh columns in 2nd step of parts-based NMF
         nmf_priors: Priors on right hand matrix
    Output:
         m: Left hand matrix
         mw: Right hand matrix
         mt_pct: Percent robust clustered rows
         mw_pct: Percent robust clustered columns
         diff: Objective minimum achieved
         mh: Convexity matrix
         flag_nonconvex: Updated non-convexity flag on left hand matrix

    """

    # Check parameter consistency (and correct if needed)
    add_message = []
    err_message = ""
    cancel_pressed = 0
    nc = int(nc)
    if nmf_fix_user_lhe * nmf_fix_user_rhe == 1:
        return mt0, mw0, np.array([]), np.array([]), 0, np.array([]), 0, add_message, err_message, cancel_pressed

    if (nc == 1) & (nmf_algo > 2):
        nmf_algo -= 2

    if nmf_algo <= 2:
        nmf_robust_n_runs = 0

    m, mmis, n_mmis = update_m_mmis(m, mmis)

    if nmf_robust_resample_columns > 0:
        m = np.copy(m).T
        if n_mmis > 0:
            mmis = np.copy(mmis).T

        mtemp = np.copy(mw0)
        mw0 = np.copy(mt0)
        mt0 = mtemp
        nmf_fix_user_lh_etemp = nmf_fix_user_lhe
        nmf_fix_user_lhe = nmf_fix_user_rhe
        nmf_fix_user_rhe = nmf_fix_user_lh_etemp

    n, p = m.shape
    # noinspection PyBroadException
    try:
        n_nmf_priors, nc = nmf_priors.shape
    except BaseException:
        n_nmf_priors = 0

    nmf_robust_n_runs = int(nmf_robust_n_runs)
    mt_pct = np.nan
    mw_pct = np.nan
    flag_nonconvex = 0

    # Step 1: NMF
    status = "Step 1 - NMF Ncomp=" + str(nc) + ": "
    mt, mw, diffsup, mhsup, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
        m,
        mmis,
        mt0,
        mw0,
        nc,
        tolerance,
        precision,
        log_iter,
        status,
        max_iterations,
        nmf_algo,
        nmf_fix_user_lhe,
        nmf_fix_user_rhe,
        nmf_max_interm,
        100,
        nmf_sparse_level,
        nmf_find_parts,
        nmf_find_centroids,
        nmf_kernel,
        nmf_reweigh_columns,
        nmf_priors,
        flag_nonconvex,
        add_message,
        my_status_box,
    )
    mtsup = np.copy(mt)
    mwsup = np.copy(mw)
    if (n_nmf_priors > 0) & (nmf_reweigh_columns > 0):
        #     Run again with fixed LHE & no priors
        status = "Step 1bis - NMF (fixed LHE) Ncomp=" + str(nc) + ": "
        mw = np.ones((p, nc)) / math.sqrt(p)
        mt, mw, diffsup, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
            m,
            mmis,
            mtsup,
            mw,
            nc,
            tolerance,
            precision,
            log_iter,
            status,
            max_iterations,
            nmf_algo,
            nc,
            0,
            nmf_max_interm,
            100,
            nmf_sparse_level,
            nmf_find_parts,
            nmf_find_centroids,
            nmf_kernel,
            0,
            nmf_priors,
            flag_nonconvex,
            add_message,
            my_status_box,
        )
        mtsup = np.copy(mt)
        mwsup = np.copy(mw)

    # Bootstrap to assess robust clustering
    if nmf_robust_n_runs > 1:
        #     Update mwsup
        mw_pct = np.zeros((p, nc))
        mw_blk = np.zeros((p, nmf_robust_n_runs * nc))
        for i_bootstrap in range(0, nmf_robust_n_runs):
            boot = np.random.randint(n, size=n)
            status = (
                "Step 2 - "
                + "boot "
                + str(i_bootstrap + 1)
                + "/"
                + str(nmf_robust_n_runs)
                + " NMF Ncomp="
                + str(nc)
                + ": "
            )
            if n_mmis > 0:
                mt, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
                    m[boot, :],
                    mmis[boot, :],
                    mtsup[boot, :],
                    mwsup,
                    nc,
                    1.0e-3,
                    precision,
                    log_iter,
                    status,
                    max_iterations,
                    nmf_algo,
                    nc,
                    0,
                    nmf_max_interm,
                    20,
                    nmf_sparse_level,
                    nmf_find_parts,
                    nmf_find_centroids,
                    nmf_kernel,
                    nmf_reweigh_columns,
                    nmf_priors,
                    flag_nonconvex,
                    add_message,
                    my_status_box,
                )
            else:
                mt, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
                    m[boot, :],
                    mmis,
                    mtsup[boot, :],
                    mwsup,
                    nc,
                    1.0e-3,
                    precision,
                    log_iter,
                    status,
                    max_iterations,
                    nmf_algo,
                    nc,
                    0,
                    nmf_max_interm,
                    20,
                    nmf_sparse_level,
                    nmf_find_parts,
                    nmf_find_centroids,
                    nmf_kernel,
                    nmf_reweigh_columns,
                    nmf_priors,
                    flag_nonconvex,
                    add_message,
                    my_status_box,
                )

            for k in range(0, nc):
                mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] = mw[:, k]

            mwn = np.zeros((p, nc))
            for k in range(0, nc):
                if (nmf_algo == 2) | (nmf_algo == 4):
                    scale_mw = np.linalg.norm(mw_blk[:, k * nmf_robust_n_runs + i_bootstrap])
                else:
                    scale_mw = np.sum(mw_blk[:, k * nmf_robust_n_runs + i_bootstrap])

                if scale_mw > 0:
                    mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] = (
                        mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] / scale_mw
                    )

                mwn[:, k] = mw_blk[:, k * nmf_robust_n_runs + i_bootstrap]

            col_clust, mwn, mw_pct, add_message, tmp_err_message, tmp_cancel_pressed = update_col_clust(
                p, nmf_calculate_leverage, mwn, mw_pct, nmf_use_robust_leverage, add_message, my_status_box
            )
            err_message = tmp_err_message if tmp_err_message is not None else err_message
            cancel_pressed = tmp_cancel_pressed if tmp_cancel_pressed is not None else cancel_pressed

        mw_pct = mw_pct / nmf_robust_n_runs

        #     Update mtsup
        mt_pct = np.zeros((n, nc))
        for i_bootstrap in range(0, nmf_robust_n_runs):
            status = (
                "Step 3 - "
                + "boot "
                + str(i_bootstrap + 1)
                + "/"
                + str(nmf_robust_n_runs)
                + " NMF Ncomp="
                + str(nc)
                + ": "
            )
            mw = np.zeros((p, nc))
            for k in range(0, nc):
                mw[:, k] = mw_blk[:, k * nmf_robust_n_runs + i_bootstrap]

            mt, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
                m,
                mmis,
                mtsup,
                mw,
                nc,
                1.0e-3,
                precision,
                log_iter,
                status,
                max_iterations,
                nmf_algo,
                0,
                nc,
                nmf_max_interm,
                20,
                nmf_sparse_level,
                nmf_find_parts,
                nmf_find_centroids,
                nmf_kernel,
                nmf_reweigh_columns,
                nmf_priors,
                flag_nonconvex,
                add_message,
                my_status_box,
            )
            row_clust, mtn, mt_pct, add_message, tmp_err_message, tmp_cancel_pressed = update_row_clust(
                n, nmf_calculate_leverage, mt, mt_pct, nmf_use_robust_leverage, add_message, my_status_box
            )
            err_message = tmp_err_message if tmp_err_message is not None else err_message
            cancel_pressed = tmp_cancel_pressed if tmp_cancel_pressed is not None else cancel_pressed

        mt_pct = mt_pct / nmf_robust_n_runs

    mt = mtsup
    mw = mwsup
    mh = mhsup
    diff = diffsup

    if nmf_robust_resample_columns > 0:
        mtemp = np.copy(mt)
        mt = np.copy(mw)
        mw = mtemp
        mtemp = np.copy(mt_pct)
        mt_pct = np.copy(mw_pct)
        mw_pct = mtemp

    return mt, mw, mt_pct, mw_pct, diff, mh, flag_nonconvex, add_message, err_message, cancel_pressed


def ntf_init(
    m,
    mmis,
    mtx_mw,
    mb2,
    nc,
    tolerance,
    precision,
    log_iter,
    ntf_unimodal,
    ntf_left_components,
    ntf_right_components,
    ntf_block_components,
    n_blocks,
    my_status_box,
):
    """Initialize NTF components for HALS

     Input:
         m: Input tensor
         mmis: Define missing values (0 = missing cell, 1 = real cell)
         mtx_mw: initialization of LHM in NMF(unstacked tensor), may be empty
         mb2: initialization of RHM of NMF(unstacked tensor), may be empty
         n_blocks: Number of NTF blocks
         nc: NTF rank
         tolerance: Convergence threshold
         precision: Replace 0-values in multiplication rules
         log_iter: Log results through iterations
     Output:
         m: Left hand matrix
         mw: Right hand matrix
         mb: Block hand matrix
     """
    add_message = []
    n, p, n_mmis, m, mmis, nc = set_nmf_attributes(m, mmis, nc)

    n_blocks = int(n_blocks)
    status0 = "Step 1 - Quick NMF Ncomp=" + str(nc) + ": "
    mstacked, mmis_stacked = ntf_stack(m, mmis, n_blocks)
    nc2 = min(nc, n_blocks)  # factorization rank can't be > number of blocks
    if (mtx_mw.shape[0] == 0) or (mb2.shape[0] == 0):
        mtx_mw, mb2 = nmf_init(
            mstacked, mmis_stacked, np.array([]), np.array([]), nc2, tolerance, log_iter, my_status_box
        )
    # NOTE: nmf_init (NNSVD) should always be called to prevent initializing NMF with signed components.
    # Creates minor differences in AD clustering, correction non implemented
    # in Galderma version
    if not compatibility_flag:
        mtx_mw, mb2 = nmf_init(mstacked, mmis_stacked, mtx_mw, mb2, nc2, tolerance, log_iter, my_status_box)
    else:
        print("In ntf_init, nmf_init was not called for the sake of compatibility with previous versions")

    # Quick NMF
    mtx_mw, mb2, diff, mh, dummy1, dummy2, add_message, err_message, cancel_pressed = nmf_solve(
        mstacked,
        mmis_stacked,
        mtx_mw,
        mb2,
        nc2,
        tolerance,
        precision,
        log_iter,
        status0,
        10,
        2,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        np.array([]),
        0,
        add_message,
        my_status_box,
    )

    # Factorize Left vectors and distribute multiple factors if nc2 < nc
    mt = np.zeros((n, nc))
    mw = np.zeros((int(p / n_blocks), nc))
    mb = np.zeros((n_blocks, nc))
    n_fact = int(np.ceil(nc / n_blocks))
    for k in range(0, nc2):
        my_status_box.update_status(delay=1, status="Start SVD...")
        u, d, v = randomized_svd(
            np.reshape(mtx_mw[:, k], (int(p / n_blocks), n)).T, n_components=n_fact, n_iter="auto", random_state=None
        )
        v = v.T
        my_status_box.update_status(delay=1, status="SVD completed")
        for i_fact in range(0, n_fact):
            ind = i_fact * n_blocks + k
            if ind < nc:
                u1, u2, v1, v2 = set_uv(u, v, i_fact, n)
                v1 = np.reshape(v1, (1, int(p / n_blocks)))
                u2 = np.reshape(u2, (n, 1))
                v2 = np.reshape(v2, (1, int(p / n_blocks)))
                if np.linalg.norm(u1 @ v1) > np.linalg.norm(u2 @ v2):
                    mt[:, ind] = np.reshape(u1, n)
                    mw[:, ind] = d[i_fact] * np.reshape(v1, int(p / n_blocks))
                else:
                    mt[:, ind] = np.reshape(u2, n)
                    mw[:, ind] = d[i_fact] * np.reshape(v2, int(p / n_blocks))

                mb[:, ind] = mb2[:, k]

    for k in range(0, nc):
        mt = update_m(ntf_unimodal, ntf_left_components, mt, k, n)

        if (ntf_unimodal > 0) & (ntf_right_components > 0):
            #                 Enforce unimodal distribution
            wmax = np.argmax(mw[:, k])
            for j in range(wmax + 1, int(p / n_blocks)):
                mw[j, k] = min(mw[j - 1, k], mw[j, k])

            for j in range(wmax - 1, -1, -1):
                mw[j, k] = min(mw[j + 1, k], mw[j, k])

        mb = update_m(ntf_unimodal, ntf_block_components, mb, k, n_blocks)

    return [mt, mw, mb, add_message, err_message, cancel_pressed]


def r_ntf_solve(
    m,
    mmis,
    mt0,
    mw0,
    mb0,
    nc,
    tolerance,
    precision,
    log_iter,
    max_iterations,
    nmf_fix_user_lhe,
    nmf_fix_user_rhe,
    nmf_fix_user_bhe,
    nmf_algo,
    nmf_robust_n_runs,
    nmf_calculate_leverage,
    nmf_use_robust_leverage,
    ntf_fast_hals,
    ntfn_iterations,
    nmf_sparse_level,
    ntf_unimodal,
    ntf_smooth,
    ntf_left_components,
    ntf_right_components,
    ntf_block_components,
    n_blocks,
    ntfn_conv,
    nmf_priors,
    my_status_box,
):
    """Estimate NTF matrices (robust version)

     Input:
         m: Input matrix
         mmis: Define missing values (0 = missing cell, 1 = real cell)
         mt0: Initial left hand matrix
         mw0: Initial right hand matrix
         mb0: Initial block hand matrix
         nc: NTF rank
         tolerance: Convergence threshold
         precision: Replace 0-values in multiplication rules
         log_iter: Log results through iterations
         max_iterations: Max iterations
         nmf_fix_user_lhe: fix left hand matrix columns: = 1, else = 0
         nmf_fix_user_rhe: fix  right hand matrix columns: = 1, else = 0
         nmf_fix_user_bhe: fix  block hand matrix columns: = 1, else = 0
         nmf_algo: =5: Non-robust version, =6: Robust version
         nmf_robust_n_runs: Number of bootstrap runs
         nmf_calculate_leverage: Calculate leverages
         nmf_use_robust_leverage: Calculate leverages based on robust max across factoring columns
         ntf_fast_hals: Use Fast HALS (does not accept handle missing values and convolution)
         ntfn_iterations: Warmup iterations for fast HALS
         nmf_sparse_level : sparsity level (as defined by Hoyer); +/- = make RHE/LHe sparse
         ntf_unimodal: Apply Unimodal constraint on factoring vectors
         ntf_smooth: Apply Smooth constraint on factoring vectors
         compo: Apply Unimodal/Smooth constraint on left hand matrix
         ntf_right_components: Apply Unimodal/Smooth constraint on right hand matrix
         compo1: Apply Unimodal/Smooth constraint on block hand matrix
         n_blocks: Number of NTF blocks
         ntfn_conv: Half-Size of the convolution window on 3rd-dimension of the tensor
         nmf_priors: Elements in mw that should be updated (others remain 0)


     Output:
         mt_conv: Convolutional Left hand matrix
         m: Left hand matrix
         mw: Right hand matrix
         mb: Block hand matrix
         mt_pct: Percent robust clustered rows
         mw_pct: Percent robust clustered columns
         diff : Objective minimum achieved
     """
    mt_pct = mw_pct = None
    add_message = []
    err_message = ""
    cancel_pressed = 0
    n, p0 = m.shape
    nc = int(nc)
    n_blocks = int(n_blocks)
    p = int(p0 / n_blocks)
    ntfn_conv = int(ntfn_conv)
    if nmf_fix_user_lhe * nmf_fix_user_rhe * nmf_fix_user_bhe == 1:
        return (
            np.zeros((n, nc * (2 * ntfn_conv + 1))),
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

    m, mmis, n_mmis = update_m_mmis(m, mmis)

    ntfn_iterations = int(ntfn_iterations)
    nmf_robust_n_runs = int(nmf_robust_n_runs)
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    mb = np.copy(mb0)
    mt_conv = np.array([])

    # Check parameter consistency (and correct if needed)
    if (nc == 1) | (nmf_algo == 5):
        nmf_robust_n_runs = 0

    if nmf_robust_n_runs == 0:
        mt_pct = np.nan
        mw_pct = np.nan

    if (n_mmis > 0 or ntfn_conv > 0 or nmf_sparse_level != 0) and ntf_fast_hals > 0:
        ntf_fast_hals = 0
        reverse2_hals = 1
    else:
        reverse2_hals = 0

    # Step 1: NTF
    status0 = "Step 1 - NTF Ncomp=" + str(nc) + ": "
    if ntf_fast_hals > 0:
        if ntfn_iterations > 0:
            mt_conv, mt, mw, mb, diff, cancel_pressed = ntf_solve(
                m,
                mmis,
                mt,
                mw,
                mb,
                nc,
                tolerance,
                log_iter,
                status0,
                ntfn_iterations,
                nmf_fix_user_lhe,
                nmf_fix_user_rhe,
                nmf_fix_user_bhe,
                0,
                ntf_unimodal,
                ntf_smooth,
                ntf_left_components,
                ntf_right_components,
                ntf_block_components,
                n_blocks,
                ntfn_conv,
                nmf_priors,
                my_status_box,
            )

        mt, mw, mb, diff, cancel_pressed = ntf_solve_fast(
            m,
            mmis,
            mt,
            mw,
            mb,
            nc,
            tolerance,
            precision,
            log_iter,
            status0,
            max_iterations,
            nmf_fix_user_lhe,
            nmf_fix_user_rhe,
            nmf_fix_user_bhe,
            ntf_unimodal,
            ntf_smooth,
            ntf_left_components,
            ntf_right_components,
            ntf_block_components,
            n_blocks,
            my_status_box,
        )
    else:
        mt_conv, mt, mw, mb, diff, cancel_pressed = ntf_solve(
            m,
            mmis,
            mt,
            mw,
            mb,
            nc,
            tolerance,
            log_iter,
            status0,
            max_iterations,
            nmf_fix_user_lhe,
            nmf_fix_user_rhe,
            nmf_fix_user_bhe,
            nmf_sparse_level,
            ntf_unimodal,
            ntf_smooth,
            ntf_left_components,
            ntf_right_components,
            ntf_block_components,
            n_blocks,
            ntfn_conv,
            nmf_priors,
            my_status_box,
        )

    mtsup = np.copy(mt)
    mwsup = np.copy(mw)
    # Bootstrap to assess robust clustering
    if nmf_robust_n_runs > 1:
        #     Update mwsup
        mw_pct = np.zeros((p, nc))
        mw_blk = np.zeros((p, nmf_robust_n_runs * nc))
        for i_bootstrap in range(0, nmf_robust_n_runs):
            boot = np.random.randint(n, size=n)
            status0 = (
                "Step 2 - "
                + "boot "
                + str(i_bootstrap + 1)
                + "/"
                + str(nmf_robust_n_runs)
                + " NTF Ncomp="
                + str(nc)
                + ": "
            )
            if ntf_fast_hals > 0:
                if n_mmis > 0:
                    mt, mw, mb, diff, cancel_pressed = ntf_solve_fast(
                        m[boot, :],
                        mmis[boot, :],
                        mtsup[boot, :],
                        mwsup,
                        mb,
                        nc,
                        1.0e-3,
                        precision,
                        log_iter,
                        status0,
                        max_iterations,
                        1,
                        0,
                        nmf_fix_user_bhe,
                        ntf_unimodal,
                        ntf_smooth,
                        ntf_left_components,
                        ntf_right_components,
                        ntf_block_components,
                        n_blocks,
                        my_status_box,
                    )
                else:
                    mt, mw, mb, diff, cancel_pressed = ntf_solve_fast(
                        m[boot, :],
                        np.array([]),
                        mtsup[boot, :],
                        mwsup,
                        mb,
                        nc,
                        1.0e-3,
                        precision,
                        log_iter,
                        status0,
                        max_iterations,
                        1,
                        0,
                        nmf_fix_user_bhe,
                        ntf_unimodal,
                        ntf_smooth,
                        ntf_left_components,
                        ntf_right_components,
                        ntf_block_components,
                        n_blocks,
                        my_status_box,
                    )
            else:
                if n_mmis > 0:
                    mt_conv, mt, mw, mb, diff, cancel_pressed = ntf_solve(
                        m[boot, :],
                        mmis[boot, :],
                        mtsup[boot, :],
                        mwsup,
                        mb,
                        nc,
                        1.0e-3,
                        log_iter,
                        status0,
                        max_iterations,
                        1,
                        0,
                        nmf_fix_user_bhe,
                        nmf_sparse_level,
                        ntf_unimodal,
                        ntf_smooth,
                        ntf_left_components,
                        ntf_right_components,
                        ntf_block_components,
                        n_blocks,
                        ntfn_conv,
                        nmf_priors,
                        my_status_box,
                    )
                else:
                    mt_conv, mt, mw, mb, diff, cancel_pressed = ntf_solve(
                        m[boot, :],
                        np.array([]),
                        mtsup[boot, :],
                        mwsup,
                        mb,
                        nc,
                        1.0e-3,
                        log_iter,
                        status0,
                        max_iterations,
                        1,
                        0,
                        nmf_fix_user_bhe,
                        nmf_sparse_level,
                        ntf_unimodal,
                        ntf_smooth,
                        ntf_left_components,
                        ntf_right_components,
                        ntf_block_components,
                        n_blocks,
                        ntfn_conv,
                        nmf_priors,
                        my_status_box,
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

            col_clust, mwn, mw_pct, add_message, tmp_err_message, tmp_cancel_pressed = update_col_clust(
                p, nmf_calculate_leverage, mwn, mw_pct, nmf_use_robust_leverage, add_message, my_status_box
            )
            err_message = tmp_err_message if tmp_err_message is not None else err_message
            cancel_pressed = tmp_cancel_pressed if tmp_cancel_pressed is not None else cancel_pressed

        mw_pct = mw_pct / nmf_robust_n_runs

        #     Update mtsup
        mt_pct = np.zeros((n, nc))
        for i_bootstrap in range(0, nmf_robust_n_runs):
            status0 = (
                "Step 3 - "
                + "boot "
                + str(i_bootstrap + 1)
                + "/"
                + str(nmf_robust_n_runs)
                + " NTF Ncomp="
                + str(nc)
                + ": "
            )
            mw = np.zeros((p, nc))
            for k in range(0, nc):
                mw[:, k] = mw_blk[:, k * nmf_robust_n_runs + i_bootstrap]

            if ntf_fast_hals > 0:
                mt, mw, mb, diff, cancel_pressed = ntf_solve_fast(
                    m,
                    mmis,
                    mtsup,
                    mw,
                    mb,
                    nc,
                    1.0e-3,
                    precision,
                    log_iter,
                    status0,
                    max_iterations,
                    0,
                    1,
                    nmf_fix_user_bhe,
                    ntf_unimodal,
                    ntf_smooth,
                    ntf_left_components,
                    ntf_right_components,
                    ntf_block_components,
                    n_blocks,
                    my_status_box,
                )
            else:
                mt_conv, mt, mw, mb, diff, cancel_pressed = ntf_solve(
                    m,
                    mmis,
                    mtsup,
                    mw,
                    mb,
                    nc,
                    1.0e-3,
                    log_iter,
                    status0,
                    max_iterations,
                    0,
                    1,
                    nmf_fix_user_bhe,
                    nmf_sparse_level,
                    ntf_unimodal,
                    ntf_smooth,
                    ntf_left_components,
                    ntf_right_components,
                    ntf_block_components,
                    n_blocks,
                    ntfn_conv,
                    nmf_priors,
                    my_status_box,
                )

            row_clust, mtn, mt_pct, add_message, tmp_err_message, tmp_cancel_pressed = update_row_clust(
                n, nmf_calculate_leverage, mt, mt_pct, nmf_use_robust_leverage, add_message, my_status_box
            )
            err_message = tmp_err_message if tmp_err_message is not None else err_message
            cancel_pressed = tmp_cancel_pressed if tmp_cancel_pressed is not None else cancel_pressed

        mt_pct = mt_pct / nmf_robust_n_runs

    mt = mtsup
    mw = mwsup
    if reverse2_hals > 0:
        add_message.insert(
            len(add_message),
            "Currently, Fast HALS cannot be applied with missing data or convolution window and was reversed to "
            "Simple HALS.",
        )

    return mt_conv, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed


def r_svd_solve(
    m,
    mmis,
    nc,
    tolerance,
    log_iter,
    log_trials,
    status0,
    max_iterations,
    svd_algo,
    svd_coverage,
    svdn_trials,
    my_status_box,
):
    """Estimate SVD matrices (robust version)

     Input:
         m: Input matrix
         mmis: Define missing values (0 = missing cell, 1 = real cell)
         nc: SVD rank
         tolerance: Convergence threshold
         log_iter: Log results through iterations
         log_trials: Log results through trials
         status0: Initial displayed status to be updated during iterations
         max_iterations: Max iterations
         svd_algo: =1: Non-robust version, =2: Robust version
         svd_coverage: Coverage non-outliers (robust version)
         svdn_trials: Number of trials (robust version)

     Output:
         m: Left hand matrix
         mev: Scaling factors
         mw: Right hand matrix
         mmis: Matrix of missing/flagged outliers
         mmsr: Vector of Residual SSQ
         mmsr2: Vector of Reidual variance

     Reference
     ---------

     L. Liu et al (2003) Robust singular value decomposition analysis of microarray data
     PNAS November 11, 2003 vol. 100 no. 23 13167–13172

    """

    best_trial = diff_trial = diff0 = diff = None
    add_message = []
    err_message = ""
    cancel_pressed = 0

    # m0 is the running matrix (to be factorized, initialized from m)
    m0 = np.copy(m)
    n, p = m0.shape
    mmis = mmis.astype(np.bool_)
    n_mmis = mmis.shape[0]

    if n_mmis > 0:
        m0[mmis is False] = np.nan
    else:
        mmis = np.isnan(m0) is False
        mmis = mmis.astype(np.bool_)
        n_mmis = mmis.shape[0]

    trace0 = np.sum(m0[mmis] ** 2)
    nc = int(nc)
    svdn_trials = int(svdn_trials)
    nxp = n * p
    nxpcov = int(round(nxp * svd_coverage, 0))
    mmsr = np.zeros(nc)
    mmsr2 = np.zeros(nc)
    mev = np.zeros(nc)
    if svd_algo == 2:
        max_trial = svdn_trials
    else:
        max_trial = 1

    mw = np.zeros((p, nc))
    mt = np.zeros((n, nc))
    mdiff = np.zeros((n, p))
    w = np.zeros(p)
    t = np.zeros(n)
    w_trial = np.zeros(p)
    t_trial = np.zeros(n)
    mmis_trial = np.zeros((n, p), dtype=np.bool)
    # Outer-reference m becomes local reference m, which is the running matrix
    # within ALS/LTS loop.
    m = np.zeros((n, p))
    wnorm = np.zeros((p, n))
    tnorm = np.zeros((n, p))
    denomw = np.zeros(n)
    denomt = np.zeros(p)
    step_iter = math.ceil(max_iterations / 100)
    pbar_step = 100 * step_iter / max_iterations
    if (n_mmis == 0) & (svd_algo == 1):
        fast_code = 1
    else:
        fast_code = 0

    if (fast_code == 0) and (svd_algo == 1):
        denomw[np.count_nonzero(mmis, axis=1) < 2] = np.nan
        denomt[np.count_nonzero(mmis, axis=0) < 2] = np.nan

    for k in range(0, nc):
        for i_trial in range(0, max_trial):
            my_status_box.init_bar(delay=1)
            # Copy values of m0 into m
            m[:, :] = m0
            status1 = status0 + "Ncomp " + str(k + 1) + " Trial " + str(i_trial + 1) + ": "
            if svd_algo == 2:
                #         Select a random subset
                m = np.reshape(m, (nxp, 1))
                m[np.argsort(np.random.rand(nxp))[nxpcov:nxp]] = np.nan
                m = np.reshape(m, (n, p))

            mmis[:, :] = np.isnan(m) is False

            #         Initialize w
            for j in range(0, p):
                w[j] = np.median(m[mmis[:, j], j])

            if np.where(w > 0)[0].size == 0:
                w[:] = 1

            w /= np.linalg.norm(w)
            # Replace missing values by 0's before regression
            m[mmis is False] = 0

            # nitialize t (LTS  =stochastic)
            wnorm, denomw, t, mdiff, m, mmis = fast_code_and_svd_algo(fast_code, wnorm, w, n, mmis, denomw, svd_algo,
                                                                      t, m, mdiff, m0, nxp, nxpcov, p)

            i_iter = 0
            cont = 1
            while (cont > 0) & (i_iter < max_iterations):
                #                 build w
                if fast_code == 0:
                    tnorm[:, :] = np.repeat(t[:, np.newaxis] ** 2, p, axis=1) * mmis
                    denomt[:] = np.sum(tnorm, axis=0)
                    # Request at least 2 non-missing values to perform column
                    # regression
                    if svd_algo == 2:
                        denomt[np.count_nonzero(mmis, axis=0) < 2] = np.nan

                    w[:] = m.T @ t / denomt
                else:
                    w[:] = m.T @ t / np.linalg.norm(t) ** 2

                w[np.isnan(w)] = np.median(w[np.isnan(w) is False])
                # normalize w
                w /= np.linalg.norm(w)
                mdiff, m, mmis = do_svd_algo(w, n, mmis, svd_algo, t, m, mdiff, m0, nxp, nxpcov, p)
                # build t

                wnorm, denomw, t, mdiff, m, mmis = fast_code_and_svd_algo(fast_code, wnorm, w, n, mmis, denomw,
                                                                          svd_algo, t, m, mdiff, m0, nxp, nxpcov, p)

                if i_iter % step_iter == 0:
                    if svd_algo == 1:
                        mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))

                    status = status1 + "Iteration: %s" % int(i_iter)
                    my_status_box.update_status(delay=1, status=status)
                    my_status_box.update_bar(delay=1, step=pbar_step)
                    if my_status_box.cancel_pressed:
                        cancel_pressed = 1
                        return [mt, mev, mw, mmis, mmsr, mmsr2, add_message, err_message, cancel_pressed]

                    diff = np.linalg.norm(mdiff[mmis]) ** 2 / np.where(mmis)[0].size
                    if log_iter == 1:
                        if svd_algo == 2:
                            my_status_box.my_print(
                                "Ncomp: "
                                + str(k)
                                + " Trial: "
                                + str(i_trial)
                                + " Iter: "
                                + str(i_iter)
                                + " MSR: "
                                + str(diff)
                            )
                        else:
                            my_status_box.my_print("Ncomp: " + str(k) + " Iter: " + str(i_iter) + " MSR: " + str(diff))

                    if i_iter > 0 and abs(diff - diff0) / diff0 < tolerance:
                        cont = 0

                    diff0 = diff

                i_iter += 1

            #         save trial
            # Does not make sense : both parts of if do the same thing, and diff_trial does not even exit yet anyway
            # if i_trial == 0:
            #     best_trial = i_trial
            #     diff_trial = diff
            #     t_trial[:] = t
            #     w_trial[:] = w
            #     mmis_trial[:, :] = mmis
            # elif diff < diff_trial:
            #     best_trial = i_trial
            #     diff_trial = diff
            #     t_trial[:] = t
            #     w_trial[:] = w
            #     mmis_trial[:, :] = mmis

            if log_trials == 1:
                my_status_box.my_print("Ncomp: " + str(k) + " Trial: " + str(i_trial) + " MSR: " + str(diff))

        if log_trials:
            my_status_box.my_print("Ncomp: " + str(k) + " Best trial: " + str(best_trial) + " MSR: " + str(diff_trial))

        t[:] = t_trial
        w[:] = w_trial
        mw[:, k] = w
        #         compute eigen value
        if svd_algo == 2:
            #             Robust regression of m on tw`
            mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
            r_mdiff = np.argsort(np.reshape(mdiff, nxp))
            t /= np.linalg.norm(t)  # Normalize t
            mt[:, k] = t
            mmis = np.reshape(mmis, nxp)
            mmis[r_mdiff[nxpcov:nxp]] = False
            ycells = np.reshape(m0, (nxp, 1))[mmis]
            xcells = np.reshape(np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)), (nxp, 1))[mmis]
            mev[k] = ycells.T @ xcells / np.linalg.norm(xcells) ** 2
            mmis = np.reshape(mmis, (n, p))
        else:
            mev[k] = np.linalg.norm(t)
            mt[:, k] = t / mev[k]  # normalize t

        if k == 0:
            mmsr[k] = mev[k] ** 2
        else:
            mmsr[k] = mmsr[k - 1] + mev[k] ** 2
            mmsr2[k] = mmsr[k] - mev[0] ** 2

        # m0 is deflated before calculating next component
        m0 = m0 - mev[k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k].T, (1, p))

    trace02 = trace0 - mev[0] ** 2
    mmsr = 1 - mmsr / trace0
    mmsr[mmsr > 1] = 1
    mmsr[mmsr < 0] = 0
    mmsr2 = 1 - mmsr2 / trace02
    mmsr2[mmsr2 > 1] = 1
    mmsr2[mmsr2 < 0] = 0
    if nc > 1:
        r_mev = np.argsort(-mev)
        mev = mev[r_mev]
        mw0 = mw
        mt0 = mt
        for k in range(0, nc):
            mw[:, k] = mw0[:, r_mev[k]]
            mt[:, k] = mt0[:, r_mev[k]]

    mmis[:, :] = True
    mmis[mmis_trial is False] = False
    # mmis.astype(dtype=int)

    return [mt, mev, mw, mmis, mmsr, mmsr2, add_message, err_message, cancel_pressed]


def non_negative_factorization(
    x,
    w=None,
    h=None,
    n_components=None,
    update_w=True,
    update_h=True,
    beta_loss="frobenius",
    use_hals=False,
    n_bootstrap=None,
    tol=1e-6,
    max_iter=150,
    max_iter_mult=20,
    regularization=None,
    sparsity=0,
    leverage="standard",
    convex=None,
    kernel="linear",
    skewness=False,
    null_priors=False,
    random_state=None,
    verbose=0,
):
    """Compute Non-negative Matrix Factorization (NMF)

    Find two non-negative matrices (w, h) such as x = w @ h.T + Error.
    This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is minimized with an alternating minimization of w
    and h.

    Parameters
    ----------

    x : array-like, shape (n_samples, n_features)
        Constant matrix.

    w : array-like, shape (n_samples, n_components)
        prior w
        If n_update_W == 0 , it is used as a constant, to solve for h only.

    h : array-like, shape (n_features, n_components)
        prior h
        If n_update_H = 0 , it is used as a constant, to solve for w only.

    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    update_w : boolean, default: True
        Update or keep w fixed

    update_h : boolean, default: True
        Update or keep h fixed

    beta_loss : string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler'}.
        Beta divergence to be minimized, measuring the distance between x
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss == 'kullback-leibler', the input
        matrix x cannot contain zeros.

    use_hals : boolean
        True -> HALS algorithm (note that convex and kullback-leibler loss opions are not supported)
        False-> Projected gradiant

    n_bootstrap : integer, default: 0
        Number of bootstrap runs.

    tol : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations.

    max_iter_mult : integer, default: 20
        Maximum number of iterations in multiplicative warm-up to projected gradient (beta_loss = 'frobenius' only).

    regularization :  None | 'components' | 'transformation'
        Select whether the regularization affects the components (h), the
        transformation (w) or none of them.

    sparsity : integer, default: 0
        Sparsity target with 0 <= sparsity <= 1 representing the % rows in w or h set to 0.

    leverage :  None | 'standard' | 'robust', default 'standard'
        Calculate the_leverage of w and h rows on each component.

    convex :  None | 'components' | 'transformation', default None
        Apply convex constraint on w or h.

    kernel :  'linear', 'quadratic', 'radial', default 'linear'
        Can be set if convex = 'transformation'.

    null_priors : boolean, default False
        Cells of h with prior cells = 0 will not be updated.
        Can be set only if prior h has been defined.

    skewness : boolean, default False
        When solving mixture problems, columns of x at the extremities of the convex hull will be given largest weights.
        The column weight is a function of the skewness and its sign.
        The expected sign of the skewness is based on the skewness of w components, as returned by the first pass
        of a 2-steps convex NMF. Thus, during the first pass, skewness must be set to False.
        Can be set only if convex = 'transformation' and prior w and h have been defined.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------

    Estimator (dictionary) with following entries

    w : array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    h : array-like, shape (n_features, n_components)
        Solution to the non-negative least squares problem.

    volume : scalar, volume occupied by w and h

    WB : array-like, shape (n_samples, n_components)
        Percent consistently clustered rows for each component.
        only if n_bootstrap > 0.

    HB : array-like, shape (n_features, n_components)
        Percent consistently clustered columns for each component.
        only if n_bootstrap > 0.

    b : array-like, shape (n_observations, n_components) or (n_features, n_components)
        only if active convex variant, h = b.T @ x or w = x @ b

    diff : Objective minimum achieved

        """
    nmf_find_centroids = nmf_kernel = None
    mt = mw = nmf_find_parts = mh = None
    if use_hals:
        # convex and kullback-leibler loss options are not supported
        beta_loss = "frobenius"
        convex = None

    m = x
    n, p = m.shape
    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components

    if beta_loss == "frobenius":
        nmf_algo = 2
    else:
        nmf_algo = 1

    log_iter = verbose
    my_status_box = StatusBoxTqdm(verbose=log_iter)
    tolerance = tol
    precision = EPSILON
    if (w is None) & (h is None):
        mt, mw = nmf_init(m, np.array([]), np.array([]), np.array([]), nc, tolerance, log_iter, my_status_box)
        init = "nndsvd"
    else:
        if h is None:
            mw = np.ones((p, nc))
            init = "custom_W"
        elif w is None:
            mt = np.ones((n, nc))
            init = "custom_H"
        else:
            init = "custom"

        for k in range(0, nc):
            if nmf_algo == 2:
                mt[:, k] = mt[:, k] / np.linalg.norm(mt[:, k])
                mw[:, k] = mw[:, k] / np.linalg.norm(mw[:, k])
            else:
                mt[:, k] = mt[:, k] / np.sum(mt[:, k])
                mw[:, k] = mw[:, k] / np.sum(mw[:, k])

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
    nmf_max_interm = max_iter_mult
    if regularization is None:
        nmf_sparse_level = 0
    else:
        if regularization == "components":
            nmf_sparse_level = sparsity
        elif regularization == "transformation":
            nmf_sparse_level = -sparsity
        else:
            nmf_sparse_level = 0

    nmf_robust_resample_columns = 0

    if leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif leverage == "robust":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 1
    else:
        nmf_calculate_leverage = 0
        nmf_use_robust_leverage = 0

    if convex is None:
        nmf_find_parts = 0
        nmf_find_centroids = 0
        nmf_kernel = 1
    elif convex == "transformation":
        nmf_find_parts = 1
        nmf_find_centroids = 0
        nmf_kernel = 1
    elif convex == "components":
        nmf_find_parts = 0
        nmf_find_centroids = 1
        if kernel == "linear":
            nmf_kernel = 1
        elif kernel == "quadratic":
            nmf_kernel = 2
        elif kernel == "radial":
            nmf_kernel = 3
        else:
            nmf_kernel = 1

    if (null_priors is True) & ((init == "custom") | (init == "custom_H")):
        nmf_priors = h
    else:
        nmf_priors = np.array([])

    if convex is None:
        nmf_reweigh_columns = 0
    else:
        if (convex == "transformation") & (init == "custom"):
            if skewness is True:
                nmf_reweigh_columns = 1
            else:
                nmf_reweigh_columns = 0

        else:
            nmf_reweigh_columns = 0

    if random_state is not None:
        random_seed = random_state
        np.random.seed(random_seed)

    if use_hals:
        if nmf_algo <= 2:
            ntf_algo = 5
        else:
            ntf_algo = 6

        mt_conv, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed = r_ntf_solve(
            m,
            np.array([]),
            mt,
            mw,
            np.array([]),
            nc,
            tolerance,
            precision,
            log_iter,
            max_iterations,
            nmf_fix_user_lhe,
            nmf_fix_user_rhe,
            1,
            ntf_algo,
            nmf_robust_n_runs,
            nmf_calculate_leverage,
            nmf_use_robust_leverage,
            0,
            0,
            nmf_sparse_level,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            np.array([]),
            my_status_box,
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

    else:
        mt, mw, mt_pct, mw_pct, diff, mh, flag_nonconvex, add_message, err_message, cancel_pressed = r_nmf_solve(
            m,
            np.array([]),
            mt,
            mw,
            nc,
            tolerance,
            precision,
            log_iter,
            max_iterations,
            nmf_algo,
            nmf_fix_user_lhe,
            nmf_fix_user_rhe,
            nmf_max_interm,
            nmf_sparse_level,
            nmf_robust_resample_columns,
            nmf_robust_n_runs,
            nmf_calculate_leverage,
            nmf_use_robust_leverage,
            nmf_find_parts,
            nmf_find_centroids,
            nmf_kernel,
            nmf_reweigh_columns,
            nmf_priors,
            my_status_box,
        )

        mev = np.ones(nc)
        if (nmf_find_parts == 0) & (nmf_find_centroids == 0) & (nmf_fix_user_lhe == 0) & (nmf_fix_user_rhe == 0):
            # Scale
            for k in range(0, nc):
                if (nmf_algo == 2) | (nmf_algo == 4):
                    scale_mt = np.linalg.norm(mt[:, k])
                    scale_mw = np.linalg.norm(mw[:, k])
                else:
                    scale_mt = np.sum(mt[:, k])
                    scale_mw = np.sum(mw[:, k])

                mev[k] = scale_mt * scale_mw
                if mev[k] > 0:
                    mt[:, k] = mt[:, k] / scale_mt
                    mw[:, k] = mw[:, k] / scale_mw

    volume = nmf_det(mt, mw, 1)

    for message in add_message:
        print(message)

    my_status_box.close()

    # Order by decreasing scale
    r_mev = np.argsort(-mev)
    mev = mev[r_mev]
    mt = mt[:, r_mev]
    mw = mw[:, r_mev]
    if isinstance(mt_pct, np.ndarray):
        mt_pct = mt_pct[:, r_mev]
        mw_pct = mw_pct[:, r_mev]

    if (nmf_find_parts == 0) & (nmf_find_centroids == 0):
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
        if (nmf_find_parts == 0) & (nmf_find_centroids == 0):
            estimator.update([("w", mt), ("h", mw), ("volume", volume), ("diff", diff)])
        else:
            estimator.update([("w", mt), ("h", mw), ("volume", volume), ("b", mh), ("diff", diff)])

    else:
        if (nmf_find_parts == 0) & (nmf_find_centroids == 0):
            estimator.update([("w", mt), ("h", mw), ("volume", volume), ("WB", mt_pct), ("HB", mw_pct), ("diff", diff)])
        else:
            estimator.update(
                [("w", mt), ("h", mw), ("volume", volume), ("b", mh), ("WB", mt_pct), ("HB", mw_pct), ("diff", diff)]
            )

    return estimator


def nmf_predict(estimator, leverage="robust", blocks=None, cluster_by_stability=False, custom_order=False, verbose=0):
    """Derives ordered sample and feature indexes for future use in ordered heatmaps

    Parameters
    ----------

    estimator : tuplet as returned by non_negative_factorization

    leverage :  None | 'standard' | 'robust', default 'robust'
        Calculate the_leverage of w and h rows on each component.

    blocks : array-like, shape(n_blocks), default None
        Size of each block (if any) in ordered heatmap.

    cluster_by_stability : boolean, default False
         Use stability instead of the_leverage to assign samples/features to clusters

    custom_order :  boolean, default False
         if False samples/features with highest the_leverage or stability appear on top of each cluster
         if True within cluster ordering is modified to suggest a continuum  between adjacent clusters

    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------

    Completed estimator with following entries:
    WL : array-like, shape (n_samples, n_components)
         Sample the_leverage on each component

    HL : array-like, shape (n_features, n_components)
         Feature the_leverage on each component

    QL : array-like, shape (n_blocks, n_components)
         Block the_leverage on each component (NTF only)

    WR : vector-like, shape (n_samples)
         Ranked sample indexes (by cluster and the_leverage or stability)
         Used to produce ordered heatmaps

    HR : vector-like, shape (n_features)
         Ranked feature indexes (by cluster and the_leverage or stability)
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

    mt = estimator["w"]
    mw = estimator["h"]
    if "q" in estimator:
        # x is a 3D tensor, in unfolded form of a 2D array
        # horizontal concatenation of blocks of equal size.
        mb = estimator["q"]
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
        mt,
        mw,
        mb,
        mt_pct,
        mw_pct,
        n_blocks,
        blk_size,
        nmf_calculate_leverage,
        nmf_use_robust_leverage,
        nmf_algo,
        nmf_robust_cluster_by_stability,
        cell_plot_ordered_clusters,
        add_message,
        my_status_box,
    )
    for message in add_message:
        print(message)

    my_status_box.close()
    if "q" in estimator:
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


def nmf_permutation_test_score(estimator, y, n_permutations=100, verbose=0):
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

    Completed estimator with following entries:

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
    mt = estimator["w"]
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
    fast_hals=True,
    n_iter_hals=2,
    n_shift=0,
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
    verbose=0,
):
    """Compute Non-negative Tensor Factorization (NTF)

    Find three non-negative matrices (w, h, f) such as x = w @@ h @@ f + Error (@@ = tensor product).
    This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is minimized with an alternating minimization of w
    and h.

    Parameters
    ----------

    x : array-like, shape (n_samples, n_features x n_blocks)
        Constant matrix.
        x is a tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd dimensions.

    n_blocks : integer

    w : array-like, shape (n_samples, n_components)
        prior w

    h : array-like, shape (n_features, n_components)
        prior h

    q : array-like, shape (n_blocks, n_components)
        prior q

    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    update_w : boolean, default: True
        Update or keep w fixed

    update_h : boolean, default: True
        Update or keep h fixed

    update_q : boolean, default: True
        Update or keep q fixed

    fast_hals : boolean, default: True
        Use fast implementation of HALS

    n_iter_hals : integer, default: 2
        Number of HALS iterations prior to fast HALS

    n_shift : integer, default: 0
        max shifting in convolutional NTF

    sparsity : integer, default: 0
        sparsity level (as defined by Hoyer); +/- = make w/h sparse

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
        Calculate the_leverage of w and h rows on each component.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------

        Estimator (dictionary) with following entries

        w : array-like, shape (n_samples, n_components)
            Solution to the non-negative least squares problem.

        h : array-like, shape (n_features, n_components)
            Solution to the non-negative least squares problem.

        q : array-like, shape (n_blocks, n_components)
            Solution to the non-negative least squares problem.

        volume : scalar, volume occupied by w and h

        WB : array-like, shape (n_samples, n_components)
            Percent consistently clustered rows for each component.
            only if n_bootstrap > 0.

        HB : array-like, shape (n_features, n_components)
            Percent consistently clustered columns for each component.
            only if n_bootstrap > 0.

    Reference
    ---------

    a. Cichocki, P.h.a.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

    """

    m = x
    n, p = m.shape
    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components

    # n_blocks = n_blocks  # The fuck ?
    p_block = int(p / n_blocks)
    tolerance = tol
    precision = EPSILON
    log_iter = verbose
    nmf_sparse_level = sparsity
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
            m,
            np.array([]),
            np.array([]),
            np.array([]),
            nc,
            tolerance,
            precision,
            log_iter,
            ntf_unimodal,
            ntf_left_components,
            ntf_right_components,
            ntf_block_components,
            n_blocks,
            my_status_box,
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

        mfit = do_mfit(n, p, nc, n_blocks, p_block, mb0, mt0, mw0)

        scale_ratio = (np.linalg.norm(mfit) / np.linalg.norm(m)) ** (1 / 3)
        for k in range(0, nc):
            mt0[:, k] /= scale_ratio
            mw0[:, k] /= scale_ratio
            mb0[:, k] /= scale_ratio

    ntf_fast_hals = fast_hals
    ntfn_iterations = n_iter_hals
    max_iterations = max_iter
    ntfn_conv = n_shift
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
        m,
        np.array([]),
        mt0,
        mw0,
        mb0,
        nc,
        tolerance,
        precision,
        log_iter,
        max_iterations,
        nmf_fix_user_lhe,
        nmf_fix_user_rhe,
        nmf_fix_user_bhe,
        nmf_algo,
        nmf_robust_n_runs,
        nmf_calculate_leverage,
        nmf_use_robust_leverage,
        ntf_fast_hals,
        ntfn_iterations,
        nmf_sparse_level,
        ntf_unimodal,
        ntf_smooth,
        ntf_left_components,
        ntf_right_components,
        ntf_block_components,
        n_blocks,
        ntfn_conv,
        np.array([]),
        my_status_box,
    )

    volume = nmf_det(mt, mw, 1)

    for message in add_message:
        print(message)

    my_status_box.close()

    estimator = {}
    if nmf_robust_n_runs <= 1:
        estimator.update([("w", mt), ("h", mw), ("q", mb), ("volume", volume), ("diff", diff)])
    else:
        estimator.update(
            [("w", mt), ("h", mw), ("q", mb), ("volume", volume), ("WB", mt_pct), ("HB", mw_pct), ("diff", diff)]
        )

    return estimator
