"""Non-negative matrix and tensor factorization core functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20
from typing import Tuple

import numpy as np
from .nmtf_utils import EPSILON, sparse_opt
import logging

logger = logging.getLogger(__name__)

# TODO (pcotte): typing
# TODO (pcotte): docstrings (with parameters and returns)


def ntf_stack(m, mmis, n_blocks):
    """Unfold tensor M
    for future use with NMF
    """
    n, p = m.shape
    mmis = mmis.astype(np.int)
    n_mmis = mmis.shape[0]
    n_blocks = int(n_blocks)

    mstacked = np.zeros((int(n * p / n_blocks), n_blocks))
    if n_mmis > 0:
        mmis_stacked = np.zeros((int(n * p / n_blocks), n_blocks))
    else:
        mmis_stacked = np.array([])

    for i_block in range(0, n_blocks):
        for j in range(0, int(p / n_blocks)):
            i1 = j * n
            i2 = i1 + n
            mstacked[i1:i2, i_block] = m[:, int(i_block * p / n_blocks + j)]
            if n_mmis > 0:
                mmis_stacked[i1:i2, i_block] = mmis[:, int(i_block * p / n_blocks + j)]

    return mstacked, mmis_stacked


def ntf_solve(
    m,
    mmis,
    mt0,
    mw0,
    mb0,
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
    nmf_priors,
    my_status_box,
):
    """Interface to:
    - NTFSolve_simple
    """

    if len(nmf_priors) > 0:
        n_nmf_priors, nc = nmf_priors.shape
    else:
        n_nmf_priors = 0

    if n_nmf_priors > 0:
        nmf_priors[nmf_priors > 0] = 1

    return ntf_solve_simple(
        m=m,
        mmis=mmis,
        mt0=mt0,
        mw0=mw0,
        mb0=mb0,
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


def ntf_solve_simple(
    m,
    mmis,
    mt0,
    mw0,
    mb0,
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
    nmf_priors,
    my_status_box,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Estimate NTF matrices (HALS)

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
     status0: Initial displayed status to be updated during iterations
     max_iterations: Max iterations
     nmf_fix_user_lhe: = 1 => fixed left hand matrix columns
     nmf_fix_user_rhe: = 1 => fixed  right hand matrix columns
     nmf_fix_user_bhe: = 1 => fixed  block hand matrix columns
     nmf_sparse_level: sparsity level (as defined by Hoyer); +/- = make RHE/LHe sparse
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
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]\n
       * mt: Left hand matrix\n
       * mw: Right hand matrix\n
       * mb: Block hand matrix\n
       * diff: objective cost\n
       * cancel_pressed\n

    Reference
    ---------
    a. Cichocki, P.H.a.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
    IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.
    """

    cancel_pressed = 0

    n, p0 = m.shape
    n_mmis = mmis.shape[0]
    nc = int(nc)
    n_blocks = int(n_blocks)
    p = int(p0 / n_blocks)
    nxp = int(n * p)
    nxp0 = int(n * p0)
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    mb = np.copy(mb0)
    #     step_iter = math.ceil(MaxIterations/10)
    step_iter = 1
    pbar_step = 100 * step_iter / max_iterations

    id_blockp = np.arange(0, (n_blocks - 1) * p + 1, p)
    a = np.zeros(n)
    b = np.zeros(p)
    c = np.zeros(n_blocks)
    alpha = np.zeros(nc)

    # Compute Residual tensor
    mfit = np.zeros((n, p0))
    for k in range(0, nc):
        if n_blocks > 1:
            for i_block in range(0, n_blocks):
                mfit[:, id_blockp[i_block]: id_blockp[i_block] + p] += (
                    mb[i_block, k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
                )
        else:
            mfit[:, id_blockp[0]: id_blockp[0] + p] += np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))

    denomt = np.zeros(n)
    denomw = np.zeros(p)
    denom_block = np.zeros((n_blocks, nc))
    mt2 = np.zeros(n)
    mw2 = np.zeros(p)
    mt_mw = np.zeros(nxp)
    denom_cutoff = 0.1

    if n_mmis > 0:
        mres = (m - mfit) * mmis
    else:
        mres = m - mfit

    my_status_box.init_bar()

    # Loop
    cont = 1
    i_iter = 0
    diff0 = 1.0e99
    mpart = np.zeros((n, p0))
    if abs(nmf_sparse_level) < 1:
        alpha[0] = nmf_sparse_level * 0.8
    else:
        alpha[0] = nmf_sparse_level

    percent_zeros = 0
    iter_sparse = 0

    while (cont > 0) & (i_iter < max_iterations):
        for k in range(0, nc):
            (
                n_blocks,
                mpart,
                id_blockp,
                p,
                mb,
                k,
                mt,
                n,
                mw,
                n_mmis,
                mmis,
                mres,
                nmf_fix_user_lhe,
                denomt,
                mw2,
                denom_cutoff,
                alpha,
                ntf_unimodal,
                ntf_left_components,
                ntf_smooth,
                a,
                nmf_fix_user_rhe,
                denomw,
                mt2,
                ntf_right_components,
                b,
                nmf_fix_user_bhe,
                mt_mw,
                nxp,
                denom_block,
                ntf_block_components,
                c,
                mfit,
                nmf_priors,
            ) = ntf_update(
                n_blocks=n_blocks,
                mpart=mpart,
                id_blockp=id_blockp,
                p=p,
                mb=mb,
                k=k,
                mt=mt,
                n=n,
                mw=mw,
                n_mmis=n_mmis,
                mmis=mmis,
                mres=mres,
                nmf_fix_user_lhe=nmf_fix_user_lhe,
                denomt=denomt,
                mw2=mw2,
                denom_cutoff=denom_cutoff,
                alpha=alpha,
                ntf_unimodal=ntf_unimodal,
                ntf_left_components=ntf_left_components,
                ntf_smooth=ntf_smooth,
                a=a,
                nmf_fix_user_rhe=nmf_fix_user_rhe,
                denomw=denomw,
                mt2=mt2,
                ntf_right_components=ntf_right_components,
                b=b,
                nmf_fix_user_bhe=nmf_fix_user_bhe,
                mt_mw=mt_mw,
                nxp=nxp,
                denom_block=denom_block,
                ntf_block_components=ntf_block_components,
                c=c,
                mfit=mfit,
                nmf_priors=nmf_priors,
            )

        if i_iter % step_iter == 0:
            # Check convergence
            diff = np.linalg.norm(mres) ** 2 / nxp0
            if (diff0 - diff) / diff0 < tolerance:
                cont = 0
            else:
                if diff > diff0:
                    my_status_box.my_print(f"{status0} Iter: {i_iter} MSR does not improve")

                diff0 = diff

            Status = f"{status0} Iteration: {i_iter}"

            if nmf_sparse_level != 0:
                Status = f"{Status} ; Achieved sparsity: {round(percent_zeros, 2)}; alpha: {round(alpha[0], 2)}"
                if log_iter == 1:
                    my_status_box.my_print(Status)

            my_status_box.update_status(status=Status)
            my_status_box.update_bar(step=pbar_step)
            if my_status_box.cancel_pressed:
                cancel_pressed = 1
                return np.array([]), mt, mw, mb, mres, cancel_pressed

            if log_iter == 1:
                my_status_box.my_print(status0 + " Iter: " + str(i_iter) + " MSR: " + str(diff))

        i_iter += 1

        if cont == 0 or i_iter == max_iterations or (cont == 0 and abs(nmf_sparse_level) == 1):
            if 0 < nmf_sparse_level < 1:
                sparse_test = np.zeros((nc, 1))
                percent_zeros0 = percent_zeros
                for k in range(0, nc):
                    sparse_test[k] = np.where(mw[:, k] == 0)[0].size

                percent_zeros = np.mean(sparse_test) / p
                if percent_zeros < percent_zeros0:
                    iter_sparse += 1
                else:
                    iter_sparse = 0

                if (percent_zeros < 0.99 * nmf_sparse_level) & (iter_sparse < 50):
                    alpha[0] *= min(1.05 * nmf_sparse_level / percent_zeros, 1.1)
                    if alpha[0] < 1:
                        i_iter = 0
                        cont = 1

            elif 0 > nmf_sparse_level > -1:
                sparse_test = np.zeros((nc, 1))
                percent_zeros0 = percent_zeros
                for k in range(0, nc):
                    sparse_test[k] = np.where(mt[:, k] == 0)[0].size

                percent_zeros = np.mean(sparse_test) / n
                if percent_zeros < percent_zeros0:
                    iter_sparse += 1
                else:
                    iter_sparse = 0

                if (percent_zeros < 0.99 * abs(nmf_sparse_level)) & (iter_sparse < 50):
                    alpha[0] *= min(1.05 * abs(nmf_sparse_level) / percent_zeros, 1.1)
                    if abs(alpha[0]) < 1:
                        i_iter = 0
                        cont = 1

            elif abs(alpha[0]) == 1:
                if alpha[0] == -1:
                    for k in range(0, nc):
                        if np.max(mt[:, k]) > 0:
                            hhi = int(
                                np.round(
                                    (np.linalg.norm(mt[:, k], ord=1) / (np.linalg.norm(mt[:, k], ord=2) + EPSILON))
                                    ** 2,
                                    decimals=0,
                                )
                            )
                            alpha[k] = -1 - (n - hhi) / (n - 1)
                        else:
                            alpha[k] = 0
                else:
                    for k in range(0, nc):
                        if np.max(mw[:, k]) > 0:
                            hhi = int(
                                np.round(
                                    (np.linalg.norm(mw[:, k], ord=1) / (np.linalg.norm(mw[:, k], ord=2) + EPSILON))
                                    ** 2,
                                    decimals=0,
                                )
                            )
                            alpha[k] = 1 + (p - hhi) / (p - 1)
                        else:
                            alpha[k] = 0

                if alpha[0] <= -1:
                    alpha_real = -(alpha + 1)
                    # noinspection PyTypeChecker
                    alpha_min = min(alpha_real)
                    for k in range(0, nc):
                        # noinspection PyUnresolvedReferences
                        alpha[k] = min(alpha_real[k], 2 * alpha_min)
                        alpha[k] = -alpha[k] - 1
                else:
                    alpha_real = alpha - 1
                    alpha_min = min(alpha_real)
                    for k in range(0, nc):
                        alpha[k] = min(alpha_real[k], 2 * alpha_min)
                        alpha[k] = alpha[k] + 1

                i_iter = 0
                cont = 1
                diff0 = 1.0e99

    for k in range(0, nc):
        hhi = np.round((np.linalg.norm(mt[:, k], ord=1) / np.linalg.norm(mt[:, k], ord=2)) ** 2, decimals=0)
        logger.info(f"component: {k}, left hhi: {hhi}")
        hhi = np.round((np.linalg.norm(mw[:, k], ord=1) / np.linalg.norm(mw[:, k], ord=2)) ** 2, decimals=0)
        logger.info(f"component: {k} right hhi: {hhi}")

    if (n_mmis > 0) & (nmf_fix_user_bhe == 0):
        mb *= denom_block

    # TODO (pcotte): mt and mw can be not yet referenced: fix that
    return np.array([]), mt, mw, mb, diff, cancel_pressed


def ntf_update(
    n_blocks,
    mpart,
    id_blockp,
    p,
    mb,
    k,
    mt,
    n,
    mw,
    n_mmis,
    mmis,
    mres,
    nmf_fix_user_lhe,
    denomt,
    mw2,
    denom_cutoff,
    alpha,
    ntf_unimodal,
    ntf_left_components,
    ntf_smooth,
    a,
    nmf_fix_user_rhe,
    denomw,
    mt2,
    ntf_right_components,
    b,
    nmf_fix_user_bhe,
    mt_mw,
    nxp,
    denom_block,
    ntf_block_components,
    c,
    mfit,
    nmf_priors,
):
    """Core updating code called by NTFSolve_simple & NTF Solve_conv
    Input:
        All variables in the calling function used in the function
    Output:
        Same as Input
    """

    if len(nmf_priors) > 0:
        n_nmf_priors, nc = nmf_priors.shape
    else:
        n_nmf_priors = 0

    # Compute kth-part
    if n_blocks > 1:
        for i_block in range(0, n_blocks):
            mpart[:, id_blockp[i_block]: id_blockp[i_block] + p] = (
                mb[i_block, k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
            )
    else:
        mpart[:, id_blockp[0]: id_blockp[0] + p] = np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))

    if n_mmis > 0:
        mpart *= mmis

    mpart += mres

    if nmf_fix_user_bhe > 0:
        norm_bhe = True
        if nmf_fix_user_rhe == 0:
            norm_lhe = True
            norm_rhe = False
        else:
            norm_lhe = False
            norm_rhe = True
    else:
        norm_bhe = False
        norm_lhe = True
        norm_rhe = True

    if (nmf_fix_user_lhe > 0) & norm_lhe:
        norm = np.linalg.norm(mt[:, k])
        if norm > 0:
            mt[:, k] /= norm

    if (nmf_fix_user_rhe > 0) & norm_rhe:
        norm = np.linalg.norm(mw[:, k])
        if norm > 0:
            mw[:, k] /= norm

    if (nmf_fix_user_bhe > 0) & norm_bhe & (n_blocks > 1):
        norm = np.linalg.norm(mb[:, k])
        if norm > 0:
            mb[:, k] /= norm

    if nmf_fix_user_lhe == 0:
        # Update Mt
        mt[:, k] = 0
        if n_blocks > 1:
            for i_block in range(0, n_blocks):
                mt[:, k] += mb[i_block, k] * mpart[:, id_blockp[i_block]: id_blockp[i_block] + p] @ mw[:, k]
        else:
            mt[:, k] += mpart[:, id_blockp[0]: id_blockp[0] + p] @ mw[:, k]

        if n_mmis > 0:
            denomt[:] = 0
            mw2[:] = mw[:, k] ** 2
            if n_blocks > 1:
                for i_block in range(0, n_blocks):
                    # Broadcast missing cells into Mw to calculate Mw.T * Mw
                    denomt += mb[i_block, k] ** 2 * mmis[:, id_blockp[i_block]: id_blockp[i_block] + p] @ mw2
            else:
                denomt += mmis[:, id_blockp[0]: id_blockp[0] + p] @ mw2

            denomt /= np.max(denomt)
            denomt[denomt < denom_cutoff] = denom_cutoff
            mt[:, k] /= denomt

        mt[mt[:, k] < 0, k] = 0
        if alpha[0] < 0:
            if alpha[0] <= -1:
                if (alpha[0] == -1) & (np.max(mt[:, k]) > 0):
                    t_threshold = mt[:, k]
                    hhi = int(
                        np.round(
                            (np.linalg.norm(t_threshold, ord=1) / (np.linalg.norm(t_threshold, ord=2) + EPSILON)) ** 2,
                            decimals=0,
                        )
                    )
                    t_rank = np.argsort(t_threshold)
                    t_threshold[t_rank[0: n - hhi]] = 0
                else:
                    mt[:, k] = sparse_opt(mt[:, k], -alpha[k] - 1, False)
            else:
                mt[:, k] = sparse_opt(mt[:, k], -alpha[0], False)

        if (ntf_unimodal > 0) & (ntf_left_components > 0):
            #             Enforce unimodal distribution
            tmax = np.argmax(mt[:, k])
            for i in range(tmax + 1, n):
                mt[i, k] = min(mt[i - 1, k], mt[i, k])

            for i in range(tmax - 1, -1, -1):
                mt[i, k] = min(mt[i + 1, k], mt[i, k])

        if (ntf_smooth > 0) & (ntf_left_components > 0):
            #             Smooth distribution
            a[0] = 0.75 * mt[0, k] + 0.25 * mt[1, k]
            a[n - 1] = 0.25 * mt[n - 2, k] + 0.75 * mt[n - 1, k]
            for i in range(1, n - 1):
                a[i] = 0.25 * mt[i - 1, k] + 0.5 * mt[i, k] + 0.25 * mt[i + 1, k]

            mt[:, k] = a

        if norm_lhe:
            norm = np.linalg.norm(mt[:, k])
            if norm > 0:
                mt[:, k] /= norm

    if nmf_fix_user_rhe == 0:
        # Update Mw
        mw[:, k] = 0
        if n_blocks > 1:
            for i_block in range(0, n_blocks):
                mw[:, k] += mpart[:, id_blockp[i_block]: id_blockp[i_block] + p].T @ mt[:, k] * mb[i_block, k]
        else:
            mw[:, k] += mpart[:, id_blockp[0]: id_blockp[0] + p].T @ mt[:, k]

        if n_mmis > 0:
            denomw[:] = 0
            mt2[:] = mt[:, k] ** 2
            if n_blocks > 1:
                for i_block in range(0, n_blocks):
                    # Broadcast missing cells into Mw to calculate Mt.T * Mt
                    denomw += mb[i_block, k] ** 2 * mmis[:, id_blockp[i_block]: id_blockp[i_block] + p].T @ mt2
            else:
                denomw += mmis[:, id_blockp[0]: id_blockp[0] + p].T @ mt2

            denomw /= np.max(denomw)
            denomw[denomw < denom_cutoff] = denom_cutoff
            mw[:, k] /= denomw

        mw[mw[:, k] < 0, k] = 0

        if alpha[0] > 0:
            if alpha[0] >= 1:
                if (alpha[0] == 1) & (np.max(mw[:, k]) > 0):
                    w_threshold = mw[:, k]
                    hhi = int(
                        np.round(
                            (np.linalg.norm(w_threshold, ord=1) / (np.linalg.norm(w_threshold, ord=2) + EPSILON)) ** 2,
                            decimals=0,
                        )
                    )
                    w_rank = np.argsort(w_threshold)
                    w_threshold[w_rank[0: p - hhi]] = 0
                else:
                    mw[:, k] = sparse_opt(mw[:, k], alpha[k] - 1, False)
            else:
                mw[:, k] = sparse_opt(mw[:, k], alpha[0], False)

        if (ntf_unimodal > 0) & (ntf_right_components > 0):
            # Enforce unimodal distribution
            wmax = np.argmax(mw[:, k])
            for j in range(wmax + 1, p):
                mw[j, k] = min(mw[j - 1, k], mw[j, k])

            for j in range(wmax - 1, -1, -1):
                mw[j, k] = min(mw[j + 1, k], mw[j, k])

        if (ntf_smooth > 0) & (ntf_right_components > 0):
            #             Smooth distribution
            b[0] = 0.75 * mw[0, k] + 0.25 * mw[1, k]
            b[p - 1] = 0.25 * mw[p - 2, k] + 0.75 * mw[p - 1, k]
            for j in range(1, p - 1):
                b[j] = 0.25 * mw[j - 1, k] + 0.5 * mw[j, k] + 0.25 * mw[j + 1, k]

            mw[:, k] = b

        if n_nmf_priors > 0:
            mw[:, k] = mw[:, k] * nmf_priors[:, k]

        if norm_rhe:
            norm = np.linalg.norm(mw[:, k])
            if norm > 0:
                mw[:, k] /= norm

    if nmf_fix_user_bhe == 0:
        # Update Mb
        mb[:, k] = 0
        mt_mw[:] = np.reshape((np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))), nxp)

        for i_block in range(0, n_blocks):
            mb[i_block, k] = np.reshape(mpart[:, id_blockp[i_block]: id_blockp[i_block] + p], nxp).T @ mt_mw

        if n_mmis > 0:
            mt_mw[:] = mt_mw[:] ** 2
            for i_block in range(0, n_blocks):
                # Broadcast missing cells into Mb to calculate Mb.T * Mb
                denom_block[i_block, k] = (
                    np.reshape(mmis[:, id_blockp[i_block]: id_blockp[i_block] + p], (1, nxp)) @ mt_mw
                )

            maxdenom_block = np.max(denom_block[:, k])
            denom_block[denom_block[:, k] < denom_cutoff * maxdenom_block] = denom_cutoff * maxdenom_block
            mb[:, k] /= denom_block[:, k]

        mb[mb[:, k] < 0, k] = 0

        if (ntf_unimodal > 0) & (ntf_block_components > 0):
            #                 Enforce unimodal distribution
            bmax = np.argmax(mb[:, k])
            for i_block in range(bmax + 1, n_blocks):
                mb[i_block, k] = min(mb[i_block - 1, k], mb[i_block, k])

            for i_block in range(bmax - 1, -1, -1):
                mb[i_block, k] = min(mb[i_block + 1, k], mb[i_block, k])

        if (ntf_smooth > 0) & (ntf_block_components > 0):
            #             Smooth distribution
            c[0] = 0.75 * mb[0, k] + 0.25 * mb[1, k]
            c[n_blocks - 1] = 0.25 * mb[n_blocks - 2, k] + 0.75 * mb[n_blocks - 1, k]
            for i_block in range(1, n_blocks - 1):
                c[i_block] = 0.25 * mb[i_block - 1, k] + 0.5 * mb[i_block, k] + 0.25 * mb[i_block + 1, k]

            mb[:, k] = c

        if norm_bhe:
            norm = np.linalg.norm(mb[:, k])
            if norm > 0:
                mb[:, k] /= norm

    # Update residual tensor
    mfit[:, :] = 0
    if n_blocks > 1:
        for i_block in range(0, n_blocks):
            mfit[:, id_blockp[i_block]: id_blockp[i_block] + p] += (
                mb[i_block, k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
            )
    else:
        mfit[:, id_blockp[0]: id_blockp[0] + p] += np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))

    if n_mmis > 0:
        mres[:, :] = (mpart - mfit) * mmis
    else:
        mres[:, :] = mpart - mfit

    return (
        n_blocks,
        mpart,
        id_blockp,
        p,
        mb,
        k,
        mt,
        n,
        mw,
        n_mmis,
        mmis,
        mres,
        nmf_fix_user_lhe,
        denomt,
        mw2,
        denom_cutoff,
        alpha,
        ntf_unimodal,
        ntf_left_components,
        ntf_smooth,
        a,
        nmf_fix_user_rhe,
        denomw,
        mt2,
        ntf_right_components,
        b,
        nmf_fix_user_bhe,
        mt_mw,
        nxp,
        denom_block,
        ntf_block_components,
        c,
        mfit,
        nmf_priors,
    )
