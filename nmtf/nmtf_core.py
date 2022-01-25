"""Non-negative matrix and tensor factorization core functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20

import math
import numpy as np

# from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from scipy.stats import hypergeom
from scipy.optimize import nnls

from .nmtf_utils import *


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

    return [mstacked, mmis_stacked]


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

    try:
        n_NMFPriors, nc = nmf_priors.shape
    except:
        n_NMFPriors = 0

    if n_NMFPriors > 0:
        nmf_priors[nmf_priors > 0] = 1

    return NTFSolve_simple(
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
    )


def NTFSolve_simple(
    M,
    Mmis,
    Mt0,
    Mw0,
    Mb0,
    nc,
    tolerance,
    LogIter,
    Status0,
    MaxIterations,
    NMFFixUserLHE,
    NMFFixUserRHE,
    NMFFixUserBHE,
    NMFSparseLevel,
    NTFUnimodal,
    NTFSmooth,
    NTFLeftComponents,
    NTFRightComponents,
    NTFBlockComponents,
    NBlocks,
    NMFPriors,
    myStatusBox,
):
    """
    Estimate NTF matrices (HALS)
    Input:
         M: Input matrix
         Mmis: Define missing values (0 = missing cell, 1 = real cell)
         Mt0: Initial left hand matrix
         Mw0: Initial right hand matrix
         Mb0: Initial block hand matrix
         nc: NTF rank
         tolerance: Convergence threshold
         LogIter: Log results through iterations
         Status0: Initial displayed status to be updated during iterations
         MaxIterations: Max iterations
         NMFFixUserLHE: = 1 => fixed left hand matrix columns
         NMFFixUserRHE: = 1 => fixed  right hand matrix columns
         NMFFixUserBHE: = 1 => fixed  block hand matrix columns
         NMFSparseLevel : sparsity level (as defined by Hoyer); +/- = make RHE/LHe sparse
         NTFUnimodal: Apply Unimodal constraint on factoring vectors
         NTFSmooth: Apply Smooth constraint on factoring vectors
         NTFLeftComponents: Apply Unimodal/Smooth constraint on left hand matrix
         NTFRightComponents: Apply Unimodal/Smooth constraint on right hand matrix
         NTFBlockComponents: Apply Unimodal/Smooth constraint on block hand matrix
         NBlocks: Number of NTF blocks
         NMFPriors: Elements in Mw that should be updated (others remain 0)

    Output:
         Mt: Left hand matrix
         Mw: Right hand matrix
         Mb: Block hand matrix
         diff: objective cost

    Reference
    ---------

    A. Cichocki, P.H.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

    """

    cancel_pressed = 0

    n, p0 = M.shape
    n_Mmis = Mmis.shape[0]
    nc = int(nc)
    NBlocks = int(NBlocks)
    p = int(p0 / NBlocks)
    nxp = int(n * p)
    nxp0 = int(n * p0)
    Mt = np.copy(Mt0)
    Mw = np.copy(Mw0)
    Mb = np.copy(Mb0)
    #     StepIter = math.ceil(MaxIterations/10)
    StepIter = 1
    pbar_step = 100 * StepIter / MaxIterations

    IDBlockp = np.arange(0, (NBlocks - 1) * p + 1, p)
    A = np.zeros(n)
    B = np.zeros(p)
    C = np.zeros(NBlocks)
    alpha = np.zeros(nc)

    # Compute Residual tensor
    Mfit = np.zeros((n, p0))
    for k in range(0, nc):
        if NBlocks > 1:
            for iBlock in range(0, NBlocks):
                Mfit[:, IDBlockp[iBlock] : IDBlockp[iBlock] + p] += (
                    Mb[iBlock, k] * np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))
                )
        else:
            Mfit[:, IDBlockp[0] : IDBlockp[0] + p] += np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))

    denomt = np.zeros(n)
    denomw = np.zeros(p)
    denomBlock = np.zeros((NBlocks, nc))
    Mt2 = np.zeros(n)
    Mw2 = np.zeros(p)
    MtMw = np.zeros(nxp)
    denomCutoff = 0.1

    if n_Mmis > 0:
        Mres = (M - Mfit) * Mmis
    else:
        Mres = M - Mfit

    myStatusBox.init_bar(delay=1)

    # Loop
    cont = 1
    iIter = 0
    diff0 = 1.0e99
    Mpart = np.zeros((n, p0))
    if abs(NMFSparseLevel) < 1:
        alpha[0] = NMFSparseLevel * 0.8
    else:
        alpha[0] = NMFSparseLevel

    PercentZeros = 0
    iterSparse = 0

    while (cont > 0) & (iIter < MaxIterations):
        for k in range(0, nc):
            (
                NBlocks,
                Mpart,
                IDBlockp,
                p,
                Mb,
                k,
                Mt,
                n,
                Mw,
                n_Mmis,
                Mmis,
                Mres,
                NMFFixUserLHE,
                denomt,
                Mw2,
                denomCutoff,
                alpha,
                NTFUnimodal,
                NTFLeftComponents,
                NTFSmooth,
                A,
                NMFFixUserRHE,
                denomw,
                Mt2,
                NTFRightComponents,
                B,
                NMFFixUserBHE,
                MtMw,
                nxp,
                denomBlock,
                NTFBlockComponents,
                C,
                Mfit,
                NMFPriors,
            ) = NTFUpdate(
                NBlocks,
                Mpart,
                IDBlockp,
                p,
                Mb,
                k,
                Mt,
                n,
                Mw,
                n_Mmis,
                Mmis,
                Mres,
                NMFFixUserLHE,
                denomt,
                Mw2,
                denomCutoff,
                alpha,
                NTFUnimodal,
                NTFLeftComponents,
                NTFSmooth,
                A,
                NMFFixUserRHE,
                denomw,
                Mt2,
                NTFRightComponents,
                B,
                NMFFixUserBHE,
                MtMw,
                nxp,
                denomBlock,
                NTFBlockComponents,
                C,
                Mfit,
                NMFPriors,
            )

        if iIter % StepIter == 0:
            # Check convergence
            diff = np.linalg.norm(Mres) ** 2 / nxp0
            if (diff0 - diff) / diff0 < tolerance:
                cont = 0
            else:
                if diff > diff0:
                    myStatusBox.myPrint(Status0 + " Iter: " + str(iIter) + " MSR does not improve")

                diff0 = diff

            Status = Status0 + "Iteration: %s" % int(iIter)

            if NMFSparseLevel != 0:
                Status = (
                    Status
                    + "; Achieved sparsity: "
                    + str(round(PercentZeros, 2))
                    + "; alpha: "
                    + str(round(alpha[0], 2))
                )
                if LogIter == 1:
                    myStatusBox.myPrint(Status)

            myStatusBox.update_status(delay=1, status=Status)
            myStatusBox.update_bar(delay=1, step=pbar_step)
            if myStatusBox.cancel_pressed:
                cancel_pressed = 1
                return [np.array([]), Mt, Mw, Mb, Mres, cancel_pressed]

            if LogIter == 1:
                myStatusBox.myPrint(Status0 + " Iter: " + str(iIter) + " MSR: " + str(diff))

        iIter += 1

        if (cont == 0) | (iIter == MaxIterations) | ((cont == 0) & (abs(NMFSparseLevel) == 1)):
            if (NMFSparseLevel > 0) & (NMFSparseLevel < 1):
                SparseTest = np.zeros((nc, 1))
                PercentZeros0 = PercentZeros
                for k in range(0, nc):
                    SparseTest[k] = np.where(Mw[:, k] == 0)[0].size

                PercentZeros = np.mean(SparseTest) / p
                if PercentZeros < PercentZeros0:
                    iterSparse += 1
                else:
                    iterSparse = 0

                if (PercentZeros < 0.99 * NMFSparseLevel) & (iterSparse < 50):
                    alpha[0] *= min(1.05 * NMFSparseLevel / PercentZeros, 1.1)
                    if alpha[0] < 1:
                        iIter = 0
                        cont = 1

            elif (NMFSparseLevel < 0) & (NMFSparseLevel > -1):
                SparseTest = np.zeros((nc, 1))
                PercentZeros0 = PercentZeros
                for k in range(0, nc):
                    SparseTest[k] = np.where(Mt[:, k] == 0)[0].size

                PercentZeros = np.mean(SparseTest) / n
                if PercentZeros < PercentZeros0:
                    iterSparse += 1
                else:
                    iterSparse = 0

                if (PercentZeros < 0.99 * abs(NMFSparseLevel)) & (iterSparse < 50):
                    alpha[0] *= min(1.05 * abs(NMFSparseLevel) / PercentZeros, 1.1)
                    if abs(alpha[0]) < 1:
                        iIter = 0
                        cont = 1

            elif abs(alpha[0]) == 1:
                if alpha[0] == -1:
                    for k in range(0, nc):
                        if np.max(Mt[:, k]) > 0:
                            hhi = int(
                                np.round(
                                    (np.linalg.norm(Mt[:, k], ord=1) / (np.linalg.norm(Mt[:, k], ord=2) + EPSILON))
                                    ** 2,
                                    decimals=0,
                                )
                            )
                            alpha[k] = -1 - (n - hhi) / (n - 1)
                        else:
                            alpha[k] = 0
                else:
                    for k in range(0, nc):
                        if np.max(Mw[:, k]) > 0:
                            hhi = int(
                                np.round(
                                    (np.linalg.norm(Mw[:, k], ord=1) / (np.linalg.norm(Mw[:, k], ord=2) + EPSILON))
                                    ** 2,
                                    decimals=0,
                                )
                            )
                            alpha[k] = 1 + (p - hhi) / (p - 1)
                        else:
                            alpha[k] = 0

                if alpha[0] <= -1:
                    alpha_real = -(alpha + 1)
                    alpha_min = min(alpha_real)
                    for k in range(0, nc):
                        alpha[k] = min(alpha_real[k], 2 * alpha_min)
                        alpha[k] = -alpha[k] - 1
                else:
                    alpha_real = alpha - 1
                    alpha_min = min(alpha_real)
                    for k in range(0, nc):
                        alpha[k] = min(alpha_real[k], 2 * alpha_min)
                        alpha[k] = alpha[k] + 1

                iIter = 0
                cont = 1
                diff0 = 1.0e99

    for k in range(0, nc):
        hhi = np.round((np.linalg.norm(Mt[:, k], ord=1) / np.linalg.norm(Mt[:, k], ord=2)) ** 2, decimals=0)
        print("component: ", k, "; left hhi: ", hhi)
        hhi = np.round((np.linalg.norm(Mw[:, k], ord=1) / np.linalg.norm(Mw[:, k], ord=2)) ** 2, decimals=0)
        print("component: ", k, "; right hhi: ", hhi)

    if (n_Mmis > 0) & (NMFFixUserBHE == 0):
        Mb *= denomBlock

    return [np.array([]), Mt, Mw, Mb, diff, cancel_pressed]


def NTFUpdate(
    NBlocks,
    Mpart,
    IDBlockp,
    p,
    Mb,
    k,
    Mt,
    n,
    Mw,
    n_Mmis,
    Mmis,
    Mres,
    NMFFixUserLHE,
    denomt,
    Mw2,
    denomCutoff,
    alpha,
    NTFUnimodal,
    NTFLeftComponents,
    NTFSmooth,
    A,
    NMFFixUserRHE,
    denomw,
    Mt2,
    NTFRightComponents,
    B,
    NMFFixUserBHE,
    MtMw,
    nxp,
    denomBlock,
    NTFBlockComponents,
    C,
    Mfit,
    NMFPriors,
):
    """Core updating code called by NTFSolve_simple & NTF Solve_conv
    Input:
        All variables in the calling function used in the function
    Output:
        Same as Input
    """

    try:
        n_NMFPriors, nc = NMFPriors.shape
    except:
        n_NMFPriors = 0

    # Compute kth-part
    if NBlocks > 1:
        for iBlock in range(0, NBlocks):
            Mpart[:, IDBlockp[iBlock] : IDBlockp[iBlock] + p] = (
                Mb[iBlock, k] * np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))
            )
    else:
        Mpart[:, IDBlockp[0] : IDBlockp[0] + p] = np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))

    if n_Mmis > 0:
        Mpart *= Mmis

    Mpart += Mres

    if NMFFixUserBHE > 0:
        NormBHE = True
        if NMFFixUserRHE == 0:
            NormLHE = True
            NormRHE = False
        else:
            NormLHE = False
            NormRHE = True
    else:
        NormBHE = False
        NormLHE = True
        NormRHE = True

    if (NMFFixUserLHE > 0) & NormLHE:
        norm = np.linalg.norm(Mt[:, k])
        if norm > 0:
            Mt[:, k] /= norm

    if (NMFFixUserRHE > 0) & NormRHE:
        norm = np.linalg.norm(Mw[:, k])
        if norm > 0:
            Mw[:, k] /= norm

    if (NMFFixUserBHE > 0) & NormBHE & (NBlocks > 1):
        norm = np.linalg.norm(Mb[:, k])
        if norm > 0:
            Mb[:, k] /= norm

    if NMFFixUserLHE == 0:
        # Update Mt
        Mt[:, k] = 0
        if NBlocks > 1:
            for iBlock in range(0, NBlocks):
                Mt[:, k] += Mb[iBlock, k] * Mpart[:, IDBlockp[iBlock] : IDBlockp[iBlock] + p] @ Mw[:, k]
        else:
            Mt[:, k] += Mpart[:, IDBlockp[0] : IDBlockp[0] + p] @ Mw[:, k]

        if n_Mmis > 0:
            denomt[:] = 0
            Mw2[:] = Mw[:, k] ** 2
            if NBlocks > 1:
                for iBlock in range(0, NBlocks):
                    # Broadcast missing cells into Mw to calculate Mw.T * Mw
                    denomt += Mb[iBlock, k] ** 2 * Mmis[:, IDBlockp[iBlock] : IDBlockp[iBlock] + p] @ Mw2
            else:
                denomt += Mmis[:, IDBlockp[0] : IDBlockp[0] + p] @ Mw2

            denomt /= np.max(denomt)
            denomt[denomt < denomCutoff] = denomCutoff
            Mt[:, k] /= denomt

        Mt[Mt[:, k] < 0, k] = 0
        if alpha[0] < 0:
            if alpha[0] <= -1:
                if (alpha[0] == -1) & (np.max(Mt[:, k]) > 0):
                    t_threshold = Mt[:, k]
                    hhi = int(
                        np.round(
                            (np.linalg.norm(t_threshold, ord=1) / (np.linalg.norm(t_threshold, ord=2) + EPSILON)) ** 2,
                            decimals=0,
                        )
                    )
                    t_rank = np.argsort(t_threshold)
                    t_threshold[t_rank[0 : n - hhi]] = 0
                else:
                    Mt[:, k] = sparse_opt(Mt[:, k], -alpha[k] - 1, False)
            else:
                Mt[:, k] = sparse_opt(Mt[:, k], -alpha[0], False)

        if (NTFUnimodal > 0) & (NTFLeftComponents > 0):
            #             Enforce unimodal distribution
            tmax = np.argmax(Mt[:, k])
            for i in range(tmax + 1, n):
                Mt[i, k] = min(Mt[i - 1, k], Mt[i, k])

            for i in range(tmax - 1, -1, -1):
                Mt[i, k] = min(Mt[i + 1, k], Mt[i, k])

        if (NTFSmooth > 0) & (NTFLeftComponents > 0):
            #             Smooth distribution
            A[0] = 0.75 * Mt[0, k] + 0.25 * Mt[1, k]
            A[n - 1] = 0.25 * Mt[n - 2, k] + 0.75 * Mt[n - 1, k]
            for i in range(1, n - 1):
                A[i] = 0.25 * Mt[i - 1, k] + 0.5 * Mt[i, k] + 0.25 * Mt[i + 1, k]

            Mt[:, k] = A

        if NormLHE:
            norm = np.linalg.norm(Mt[:, k])
            if norm > 0:
                Mt[:, k] /= norm

    if NMFFixUserRHE == 0:
        # Update Mw
        Mw[:, k] = 0
        if NBlocks > 1:
            for iBlock in range(0, NBlocks):
                Mw[:, k] += Mpart[:, IDBlockp[iBlock] : IDBlockp[iBlock] + p].T @ Mt[:, k] * Mb[iBlock, k]
        else:
            Mw[:, k] += Mpart[:, IDBlockp[0] : IDBlockp[0] + p].T @ Mt[:, k]

        if n_Mmis > 0:
            denomw[:] = 0
            Mt2[:] = Mt[:, k] ** 2
            if NBlocks > 1:
                for iBlock in range(0, NBlocks):
                    # Broadcast missing cells into Mw to calculate Mt.T * Mt
                    denomw += Mb[iBlock, k] ** 2 * Mmis[:, IDBlockp[iBlock] : IDBlockp[iBlock] + p].T @ Mt2
            else:
                denomw += Mmis[:, IDBlockp[0] : IDBlockp[0] + p].T @ Mt2

            denomw /= np.max(denomw)
            denomw[denomw < denomCutoff] = denomCutoff
            Mw[:, k] /= denomw

        Mw[Mw[:, k] < 0, k] = 0

        if alpha[0] > 0:
            if alpha[0] >= 1:
                if (alpha[0] == 1) & (np.max(Mw[:, k]) > 0):
                    w_threshold = Mw[:, k]
                    hhi = int(
                        np.round(
                            (np.linalg.norm(w_threshold, ord=1) / (np.linalg.norm(w_threshold, ord=2) + EPSILON)) ** 2,
                            decimals=0,
                        )
                    )
                    w_rank = np.argsort(w_threshold)
                    w_threshold[w_rank[0 : p - hhi]] = 0
                else:
                    Mw[:, k] = sparse_opt(Mw[:, k], alpha[k] - 1, False)
            else:
                Mw[:, k] = sparse_opt(Mw[:, k], alpha[0], False)

        if (NTFUnimodal > 0) & (NTFRightComponents > 0):
            # Enforce unimodal distribution
            wmax = np.argmax(Mw[:, k])
            for j in range(wmax + 1, p):
                Mw[j, k] = min(Mw[j - 1, k], Mw[j, k])

            for j in range(wmax - 1, -1, -1):
                Mw[j, k] = min(Mw[j + 1, k], Mw[j, k])

        if (NTFSmooth > 0) & (NTFRightComponents > 0):
            #             Smooth distribution
            B[0] = 0.75 * Mw[0, k] + 0.25 * Mw[1, k]
            B[p - 1] = 0.25 * Mw[p - 2, k] + 0.75 * Mw[p - 1, k]
            for j in range(1, p - 1):
                B[j] = 0.25 * Mw[j - 1, k] + 0.5 * Mw[j, k] + 0.25 * Mw[j + 1, k]

            Mw[:, k] = B

        if n_NMFPriors > 0:
            Mw[:, k] = Mw[:, k] * NMFPriors[:, k]

        if NormRHE:
            norm = np.linalg.norm(Mw[:, k])
            if norm > 0:
                Mw[:, k] /= norm

    if NMFFixUserBHE == 0:
        # Update Mb
        Mb[:, k] = 0
        MtMw[:] = np.reshape((np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))), nxp)

        for iBlock in range(0, NBlocks):
            Mb[iBlock, k] = np.reshape(Mpart[:, IDBlockp[iBlock] : IDBlockp[iBlock] + p], nxp).T @ MtMw

        if n_Mmis > 0:
            MtMw[:] = MtMw[:] ** 2
            for iBlock in range(0, NBlocks):
                # Broadcast missing cells into Mb to calculate Mb.T * Mb
                denomBlock[iBlock, k] = np.reshape(Mmis[:, IDBlockp[iBlock] : IDBlockp[iBlock] + p], (1, nxp)) @ MtMw

            maxdenomBlock = np.max(denomBlock[:, k])
            denomBlock[denomBlock[:, k] < denomCutoff * maxdenomBlock] = denomCutoff * maxdenomBlock
            Mb[:, k] /= denomBlock[:, k]

        Mb[Mb[:, k] < 0, k] = 0

        if (NTFUnimodal > 0) & (NTFBlockComponents > 0):
            #                 Enforce unimodal distribution
            bmax = np.argmax(Mb[:, k])
            for iBlock in range(bmax + 1, NBlocks):
                Mb[iBlock, k] = min(Mb[iBlock - 1, k], Mb[iBlock, k])

            for iBlock in range(bmax - 1, -1, -1):
                Mb[iBlock, k] = min(Mb[iBlock + 1, k], Mb[iBlock, k])

        if (NTFSmooth > 0) & (NTFBlockComponents > 0):
            #             Smooth distribution
            C[0] = 0.75 * Mb[0, k] + 0.25 * Mb[1, k]
            C[NBlocks - 1] = 0.25 * Mb[NBlocks - 2, k] + 0.75 * Mb[NBlocks - 1, k]
            for iBlock in range(1, NBlocks - 1):
                C[iBlock] = 0.25 * Mb[iBlock - 1, k] + 0.5 * Mb[iBlock, k] + 0.25 * Mb[iBlock + 1, k]

            Mb[:, k] = C

        if NormBHE:
            norm = np.linalg.norm(Mb[:, k])
            if norm > 0:
                Mb[:, k] /= norm

    # Update residual tensor
    Mfit[:, :] = 0
    if NBlocks > 1:
        for iBlock in range(0, NBlocks):
            Mfit[:, IDBlockp[iBlock] : IDBlockp[iBlock] + p] += (
                Mb[iBlock, k] * np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))
            )
    else:
        Mfit[:, IDBlockp[0] : IDBlockp[0] + p] += np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))

    if n_Mmis > 0:
        Mres[:, :] = (Mpart - Mfit) * Mmis
    else:
        Mres[:, :] = Mpart - Mfit

    return (
        NBlocks,
        Mpart,
        IDBlockp,
        p,
        Mb,
        k,
        Mt,
        n,
        Mw,
        n_Mmis,
        Mmis,
        Mres,
        NMFFixUserLHE,
        denomt,
        Mw2,
        denomCutoff,
        alpha,
        NTFUnimodal,
        NTFLeftComponents,
        NTFSmooth,
        A,
        NMFFixUserRHE,
        denomw,
        Mt2,
        NTFRightComponents,
        B,
        NMFFixUserBHE,
        MtMw,
        nxp,
        denomBlock,
        NTFBlockComponents,
        C,
        Mfit,
        NMFPriors,
    )
