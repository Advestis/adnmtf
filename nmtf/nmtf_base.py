"""Non-negative matrix and tensor factorization basic functions
"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20
# Initialize progressbar

from typing import List
import numpy as np
from scipy.sparse.linalg import svds

from .nmtf_core import NTFStack, ntf_solve
from .nmtf_utils import Leverage, StatusBoxTqdm, NMFDet, BuildClusters, GlobalSign

EPSILON = np.finfo(np.float32).eps


def nmf_init(m, mmis, mt0, mw0, nc) -> List[np.ndarray]:
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
        List[np.ndarray]: Left hand matrix and: Right hand matrix

    Reference
    ---------
    C. Boutsidis, E. Gallopoulos (2008) SVD based initialization: A head start for nonnegative matrix factorization
    Pattern Recognition Pattern Recognition Volume 41, Issue 4, April 2008, Pages 1350-1362
    """

    n, p = m.shape
    mmis = mmis.astype(np.int)
    n_Mmis = mmis.shape[0]
    if n_Mmis == 0:
        ID = np.where(np.isnan(m) is True)
        n_Mmis = ID[0].size
        if n_Mmis > 0:
            mmis = np.isnan(m) is False
            mmis = mmis.astype(np.int)
            m[mmis == 0] = 0
    else:
        m[mmis == 0] = 0

    nc = int(nc)
    Mt = np.copy(mt0)
    Mw = np.copy(mw0)
    if (Mt.shape[0] == 0) or (Mw.shape[0] == 0):
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
            # Mt = t
            # Mw = w.T
        else:
            t, d, w = svds(m, k=nc)

        Mt = t[:n, :]
        Mw = w[:, :p].T
        # svds returns singular vectors in reverse order
        Mt = Mt[:, ::-1]
        Mw = Mw[:, ::-1]

    for k in range(0, nc):
        U1 = Mt[:, k]
        U2 = -Mt[:, k]
        U1[U1 < 0] = 0
        U2[U2 < 0] = 0
        V1 = Mw[:, k]
        V2 = -Mw[:, k]
        V1[V1 < 0] = 0
        V2[V2 < 0] = 0
        U1 = np.reshape(U1, (n, 1))
        V1 = np.reshape(V1, (1, p))
        U2 = np.reshape(U2, (n, 1))
        V2 = np.reshape(V2, (1, p))
        if np.linalg.norm(U1 @ V1) > np.linalg.norm(U2 @ V2):
            Mt[:, k] = np.reshape(U1, n)
            Mw[:, k] = np.reshape(V1, p)
        else:
            Mt[:, k] = np.reshape(U2, n)
            Mw[:, k] = np.reshape(V2, p)

    return [Mt, Mw]


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
) -> List[np.ndarray]:
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
    List[np.ndarray]
        Mt: Left hand matrix
        Mw: Right hand matrix
        Mb: Block hand matrix
    """
    AddMessage = []

    n, p = m.shape
    mmis = mmis.astype(np.int)
    n_Mmis = mmis.shape[0]
    if n_Mmis == 0:
        ID = np.where(np.isnan(m) is True)
        n_Mmis = ID[0].size
        if n_Mmis > 0:
            mmis = np.isnan(m) is False
            mmis = mmis.astype(np.int)
            m[mmis == 0] = 0
    else:
        m[mmis == 0] = 0

    nc = int(nc)
    n_blocks = int(n_blocks)
    init_type = int(init_type)

    Status0 = "Step 1 - Quick NMF Ncomp=" + str(nc) + ": "

    if init_type == 1:
        # Init legacy
        Mstacked, Mmis_stacked = NTFStack(m, mmis, n_blocks)
        nc2 = min(nc, n_blocks)  # factorization rank can't be > number of blocks
        if (mt_nmf.shape[0] == 0) or (mw_nmf.shape[0] == 0):
            mt_nmf, mw_nmf = nmf_init(
                Mstacked, Mmis_stacked, np.array([]), np.array([]), nc2
            )
        else:
            mt_nmf, mw_nmf = nmf_init(Mstacked, Mmis_stacked, mt_nmf, mw_nmf, nc2)

        # Quick NMF
        dummy, mt_nmf, mw_nmf, Mb, diff, cancel_pressed = ntf_solve(
            Mstacked,
            Mmis_stacked,
            mt_nmf,
            mw_nmf,
            np.array([]),
            nc2,
            tolerance,
            log_iter,
            Status0,
            10,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            np.array([]),
            my_status_box,
        )
        ErrMessage = ""

        # Factorize Left vectors and distribute multiple factors if nc2 < nc
        Mt = np.zeros((n, nc))
        Mw = np.zeros((int(p / n_blocks), nc))
        Mb = np.zeros((n_blocks, nc))
        NFact = int(np.ceil(nc / n_blocks))
        for k in range(0, nc2):
            my_status_box.update_status(delay=1, status="Start SVD...")
            U, d, V = svds(np.reshape(mt_nmf[:, k], (int(p / n_blocks), n)).T, k=NFact)
            V = V.T
            # svds returns singular vectors in reverse order
            U = U[:, ::-1]
            V = V[:, ::-1]
            d = d[::-1]

            my_status_box.update_status(delay=1, status="SVD completed")
            for iFact in range(0, NFact):
                ind = iFact * n_blocks + k
                if ind < nc:
                    U1 = U[:, iFact]
                    U2 = -U[:, iFact]
                    U1[U1 < 0] = 0
                    U2[U2 < 0] = 0
                    V1 = V[:, iFact]
                    V2 = -V[:, iFact]
                    V1[V1 < 0] = 0
                    V2[V2 < 0] = 0
                    U1 = np.reshape(U1, (n, 1))
                    V1 = np.reshape(V1, (1, int(p / n_blocks)))
                    U2 = np.reshape(U2, (n, 1))
                    V2 = np.reshape(V2, (1, int(p / n_blocks)))
                    if np.linalg.norm(U1 @ V1) > np.linalg.norm(U2 @ V2):
                        Mt[:, ind] = np.reshape(U1, n)
                        Mw[:, ind] = d[iFact] * np.reshape(V1, int(p / n_blocks))
                    else:
                        Mt[:, ind] = np.reshape(U2, n)
                        Mw[:, ind] = d[iFact] * np.reshape(V2, int(p / n_blocks))

                    Mb[:, ind] = mw_nmf[:, k]
    else:
        # Init default
        if (mt_nmf.shape[0] == 0) or (mw_nmf.shape[0] == 0):
            mt_nmf, mw_nmf = nmf_init(m, mmis, np.array([]), np.array([]), nc)
        else:
            mt_nmf, mw_nmf = nmf_init(m, mmis, mt_nmf, mw_nmf, nc)

        # Quick NMF
        dummy, mt_nmf, mw_nmf, Mb, diff, cancel_pressed = ntf_solve(
            m,
            mmis,
            mt_nmf,
            mw_nmf,
            np.array([]),
            nc,
            tolerance,
            log_iter,
            Status0,
            10,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            np.array([]),
            my_status_box,
        )
        ErrMessage = ""

        # Factorize Left vectors
        Mt = np.zeros((n, nc))
        Mw = np.zeros((int(p / n_blocks), nc))
        Mb = np.zeros((n_blocks, nc))

        for k in range(0, nc):
            my_status_box.update_status(delay=1, status="Start SVD...")
            # noinspection PyTypeChecker
            U, d, V = svds(np.reshape(mw_nmf[:, k], (int(p / n_blocks), n_blocks)), k=1)
            V = V.T
            U = np.abs(U)
            V = np.abs(V)
            my_status_box.update_status(delay=1, status="SVD completed")
            Mt[:, k] = mt_nmf[:, k]
            Mw[:, k] = d[0] * np.reshape(U, int(p / n_blocks))
            Mb[:, k] = np.reshape(V, n_blocks)

        for k in range(0, nc):
            if (ntf_unimodal > 0) & (ntf_left_components > 0):
                #                 Enforce unimodal distribution
                tmax = np.argmax(Mt[:, k])
                for i in range(tmax + 1, n):
                    Mt[i, k] = min(Mt[i - 1, k], Mt[i, k])

                for i in range(tmax - 1, -1, -1):
                    Mt[i, k] = min(Mt[i + 1, k], Mt[i, k])

    if (ntf_unimodal > 0) & (ntf_right_components > 0):
        #                 Enforce unimodal distribution
        # TODO (pcotte) : k seems to be defined as a for loop iterator. VERY bad practice !
        wmax = np.argmax(Mw[:, k])
        for j in range(wmax + 1, int(p / n_blocks)):
            Mw[j, k] = min(Mw[j - 1, k], Mw[j, k])

        for j in range(wmax - 1, -1, -1):
            Mw[j, k] = min(Mw[j + 1, k], Mw[j, k])

    if (ntf_unimodal > 0) & (ntf_block_components > 0):
        #                 Enforce unimodal distribution
        bmax = np.argmax(Mb[:, k])
        for iBlock in range(bmax + 1, n_blocks):
            Mb[iBlock, k] = min(Mb[iBlock - 1, k], Mb[iBlock, k])

        for iBlock in range(bmax - 1, -1, -1):
            Mb[iBlock, k] = min(Mb[iBlock + 1, k], Mb[iBlock, k])

    return [Mt, Mw, Mb, AddMessage, ErrMessage, cancel_pressed]


def r_ntf_solve(
        M,
        Mmis,
        Mt0,
        Mw0,
        Mb0,
        nc,
        tolerance,
        LogIter,
        MaxIterations,
        NMFFixUserLHE,
        NMFFixUserRHE,
        NMFFixUserBHE,
        NMFAlgo,
        NMFRobustNRuns,
        NMFCalculateLeverage,
        NMFUseRobustLeverage,
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
    """Estimate NTF matrices (robust version)

    Input:
        M: Input matrix
        Mmis: Define missing values (0 = missing cell, 1 = real cell)
        Mt0: Initial left hand matrix
        Mw0: Initial right hand matrix
        Mb0: Initial block hand matrix
        nc: NTF rank
        tolerance: Convergence threshold
        LogIter: Log results through iterations
        MaxIterations: Max iterations
        NMFFixUserLHE: fix left hand matrix columns: = 1, else = 0
        NMFFixUserRHE: fix  right hand matrix columns: = 1, else = 0
        NMFFixUserBHE: fix  block hand matrix columns: = 1, else = 0
        NMFAlgo: =5: Non-robust version, =6: Robust version
        NMFRobustNRuns: Number of bootstrap runs
        NMFCalculateLeverage: Calculate leverages
        NMFUseRobustLeverage: Calculate leverages based on robust max across factoring columns
        NMFSparseLevel : sparsity level (as defined by Hoyer); +/- = make RHE/LHe sparse
        NTFUnimodal: Apply Unimodal constraint on factoring vectors
        NTFSmooth: Apply Smooth constraint on factoring vectors
        NTFLeftComponents: Apply Unimodal/Smooth constraint on left hand matrix
        NTFRightComponents: Apply Unimodal/Smooth constraint on right hand matrix
        NTFBlockComponents: Apply Unimodal/Smooth constraint on block hand matrix
        NBlocks: Number of NTF blocks
        NMFPriors: Elements in Mw that should be updated (others remain 0)


    Output:
        Mt_conv: Convolutional Left hand matrix
        Mt: Left hand matrix
        Mw: Right hand matrix
        Mb: Block hand matrix
        MtPct: Percent robust clustered rows
        MwPct: Percent robust clustered columns
        diff : Objective minimum achieved
    """

    AddMessage = []
    ErrMessage = ""
    cancel_pressed = 0
    n, p0 = M.shape
    nc = int(nc)
    NBlocks = int(NBlocks)
    p = int(p0 / NBlocks)
    if NMFFixUserLHE * NMFFixUserRHE * NMFFixUserBHE == 1:
        return (
            np.zeros((n, nc)),
            Mt0,
            Mw0,
            Mb0,
            np.zeros((n, p0)),
            np.ones((n, nc)),
            np.ones((p, nc)),
            AddMessage,
            ErrMessage,
            cancel_pressed,
        )

    Mmis = Mmis.astype(np.int)
    n_Mmis = Mmis.shape[0]
    if n_Mmis == 0:
        ID = np.where(np.isnan(M) is True)
        n_Mmis = ID[0].size
        if n_Mmis > 0:
            Mmis = np.isnan(M) is False
            Mmis = Mmis.astype(np.int)
            M[Mmis == 0] = 0
    else:
        M[Mmis == 0] = 0

    NMFRobustNRuns = int(NMFRobustNRuns)
    Mt = np.copy(Mt0)
    Mw = np.copy(Mw0)
    Mb = np.copy(Mb0)

    # Check parameter consistency (and correct if needed)
    if (nc == 1) | (NMFAlgo == 5):
        NMFRobustNRuns = 0

    if NMFRobustNRuns == 0:
        MtPct = np.nan
        MwPct = np.nan

    # Step 1: NTF
    Status0 = "Step 1 - NTF Ncomp=" + str(nc) + ": "
    Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = ntf_solve(
        M,
        Mmis,
        Mt,
        Mw,
        Mb,
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
    )

    Mtsup = np.copy(Mt)
    Mwsup = np.copy(Mw)
    Mbsup = np.copy(Mb)
    diff_sup = diff
    # Bootstrap to assess robust clustering
    if NMFRobustNRuns > 1:
        #     Update Mwsup
        MwPct = np.zeros((p, nc))
        MwBlk = np.zeros((p, NMFRobustNRuns * nc))
        for iBootstrap in range(0, NMFRobustNRuns):
            Boot = np.random.randint(n, size=n)
            Status0 = f"Step 2 - Boot {iBootstrap + 1}/{NMFRobustNRuns} NTF Ncomp={nc}"
            if n_Mmis > 0:
                Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = ntf_solve(
                    M[Boot, :],
                    Mmis[Boot, :],
                    Mtsup[Boot, :],
                    Mwsup,
                    Mb,
                    nc,
                    1.0e-3,
                    LogIter,
                    Status0,
                    MaxIterations,
                    1,
                    0,
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
                )
            else:
                Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = ntf_solve(
                    M[Boot, :],
                    np.array([]),
                    Mtsup[Boot, :],
                    Mwsup,
                    Mb,
                    nc,
                    1.0e-3,
                    LogIter,
                    Status0,
                    MaxIterations,
                    1,
                    0,
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
                )

            for k in range(0, nc):
                MwBlk[:, k * NMFRobustNRuns + iBootstrap] = Mw[:, k]

            Mwn = np.zeros((p, nc))
            for k in range(0, nc):
                ScaleMw = np.linalg.norm(MwBlk[:, k * NMFRobustNRuns + iBootstrap])
                if ScaleMw > 0:
                    MwBlk[:, k * NMFRobustNRuns + iBootstrap] = MwBlk[:, k * NMFRobustNRuns + iBootstrap] / ScaleMw

                Mwn[:, k] = MwBlk[:, k * NMFRobustNRuns + iBootstrap]

            ColClust = np.zeros(p, dtype=int)
            if NMFCalculateLeverage > 0:
                Mwn, AddMessage, ErrMessage, cancel_pressed = Leverage(
                    Mwn, NMFUseRobustLeverage, AddMessage, myStatusBox
                )

            for j in range(0, p):
                ColClust[j] = np.argmax(np.array(Mwn[j, :]))
                MwPct[j, ColClust[j]] = MwPct[j, ColClust[j]] + 1

        MwPct = MwPct / NMFRobustNRuns

        #     Update Mtsup
        MtPct = np.zeros((n, nc))
        for iBootstrap in range(0, NMFRobustNRuns):
            Status0 = f"Step 3 - Boot {iBootstrap + 1}/{NMFRobustNRuns} NTF Ncomp={nc}"
            Mw = np.zeros((p, nc))
            for k in range(0, nc):
                Mw[:, k] = MwBlk[:, k * NMFRobustNRuns + iBootstrap]

            Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = ntf_solve(
                M,
                Mmis,
                Mtsup,
                Mw,
                Mb,
                nc,
                1.0e-3,
                LogIter,
                Status0,
                MaxIterations,
                0,
                1,
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
            )

            RowClust = np.zeros(n, dtype=int)
            if NMFCalculateLeverage > 0:
                Mtn, AddMessage, ErrMessage, cancel_pressed = Leverage(
                    Mt, NMFUseRobustLeverage, AddMessage, myStatusBox
                )
            else:
                Mtn = Mt

            for i in range(0, n):
                RowClust[i] = np.argmax(Mtn[i, :])
                MtPct[i, RowClust[i]] = MtPct[i, RowClust[i]] + 1

        MtPct = MtPct / NMFRobustNRuns

    Mt = Mtsup
    Mw = Mwsup
    Mb = Mbsup
    diff = diff_sup

    # TODO (pcotte) : MtPct and MwPct can be not yet referenced : fix that
    return Mt_conv, Mt, Mw, Mb, MtPct, MwPct, diff, AddMessage, ErrMessage, cancel_pressed


def non_negative_factorization(
        X,
        W=None,
        H=None,
        n_components=None,
        update_W=True,
        update_H=True,
        n_bootstrap=None,
        tol=1e-6,
        max_iter=150,
        regularization=None,
        sparsity=0,
        leverage="standard",
        random_state=None,
        verbose=0,
):
    """Compute Non-negative Matrix Factorization (NMF)

    Find two non-negative matrices (W, H) such as x = W @ H.T + Error.
    This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is minimized with an alternating minimization of W
    and H.

    Parameters
    ----------

    X : array-like, shape (n_samples, n_features)
        Constant matrix.

    W : array-like, shape (n_samples, n_components)
        prior W
        If n_update_W == 0 , it is used as a constant, to solve for H only.

    H : array-like, shape (n_features, n_components)
        prior H
        If n_update_H = 0 , it is used as a constant, to solve for W only.

    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    update_W : boolean, default: True
        Update or keep W fixed

    update_H : boolean, default: True
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

    Estimator (dictionary) with following entries

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

    M = X
    n, p = M.shape
    # Identify missing values
    Mmis = np.array([])
    Mmis = Mmis.astype(np.int)
    ID = np.where(np.isnan(M) is True)
    n_Mmis = ID[0].size
    if n_Mmis > 0:
        Mmis = np.isnan(M) is False
        Mmis = Mmis.astype(np.int)
        M[Mmis == 0] = 0
    else:
        M[Mmis == 0] = 0

    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components

    NMFAlgo = 2
    LogIter = verbose
    myStatusBox = StatusBoxTqdm(verbose=LogIter)
    tolerance = tol
    if (W is None) & (H is None):
        Mt, Mw = nmf_init(M, Mmis, np.array([]), np.array([]), nc)
    else:
        if H is None:
            Mw = np.ones((p, nc))
        elif W is None:
            Mt = np.ones((n, nc))

        for k in range(0, nc):
            # TODO (pcotte) : Mt and Mw can be not yet referenced : fix that
            Mt[:, k] = Mt[:, k] / np.linalg.norm(Mt[:, k])
            Mw[:, k] = Mw[:, k] / np.linalg.norm(Mw[:, k])

    if n_bootstrap is None:
        NMFRobustNRuns = 0
    else:
        NMFRobustNRuns = n_bootstrap

    if NMFRobustNRuns > 1:
        NMFAlgo += 2

    if update_W is True:
        NMFFixUserLHE = 0
    else:
        NMFFixUserLHE = 1

    if update_H is True:
        NMFFixUserRHE = 0
    else:
        NMFFixUserRHE = 1

    MaxIterations = max_iter
    if regularization is None:
        NMFSparseLevel = 0
    else:
        if regularization == "components":
            NMFSparseLevel = sparsity
        elif regularization == "transformation":
            NMFSparseLevel = -sparsity
        else:
            NMFSparseLevel = 0

    if leverage == "standard":
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 0
    elif leverage == "robust":
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 1
    else:
        NMFCalculateLeverage = 0
        NMFUseRobustLeverage = 0

    if random_state is not None:
        RandomSeed = random_state
        np.random.seed(RandomSeed)

    if NMFAlgo <= 2:
        NTFAlgo = 5
    else:
        NTFAlgo = 6

    dummy, Mt, Mw, Mb, MtPct, MwPct, diff, AddMessage, ErrMessage, cancel_pressed = r_ntf_solve(
        M,
        Mmis,
        Mt,
        Mw,
        np.array([]),
        nc,
        tolerance,
        LogIter,
        MaxIterations,
        NMFFixUserLHE,
        NMFFixUserRHE,
        1,
        NTFAlgo,
        NMFRobustNRuns,
        NMFCalculateLeverage,
        NMFUseRobustLeverage,
        NMFSparseLevel,
        0,
        0,
        0,
        0,
        0,
        1,
        np.array([]),
        myStatusBox,
    )
    Mev = np.ones(nc)
    if (NMFFixUserLHE == 0) & (NMFFixUserRHE == 0):
        # Scale
        for k in range(0, nc):
            ScaleMt = np.linalg.norm(Mt[:, k])
            ScaleMw = np.linalg.norm(Mw[:, k])
            Mev[k] = ScaleMt * ScaleMw
            if Mev[k] > 0:
                Mt[:, k] = Mt[:, k] / ScaleMt
                Mw[:, k] = Mw[:, k] / ScaleMw

    volume = NMFDet(Mt, Mw, 1)

    for message in AddMessage:
        print(message)

    myStatusBox.close()

    # Order by decreasing scale
    RMev = np.argsort(-Mev)
    Mev = Mev[RMev]
    Mt = Mt[:, RMev]
    Mw = Mw[:, RMev]
    if isinstance(MtPct, np.ndarray):
        MtPct = MtPct[:, RMev]
        MwPct = MwPct[:, RMev]

    # Scale by max com p
    for k in range(0, nc):
        MaxCol = np.max(Mt[:, k])
        if MaxCol > 0:
            Mt[:, k] /= MaxCol
            Mw[:, k] *= Mev[k] * MaxCol
            Mev[k] = 1
        else:
            Mev[k] = 0

    estimator = {}
    if NMFRobustNRuns <= 1:
        estimator.update([("W", Mt), ("H", Mw), ("volume", volume), ("diff", diff)])
    else:
        estimator.update([("W", Mt), ("H", Mw), ("volume", volume), ("WB", MtPct), ("HB", MwPct), ("diff", diff)])

    return estimator


def nmf_predict(estimator, leverage="robust", blocks=None, cluster_by_stability=False, custom_order=False, verbose=0):
    """Derives ordered sample and feature indexes for future use in ordered heatmaps

    Parameters
    ----------

    estimator : tuplet as returned by non_negative_factorization

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

    Completed estimator with following entries:
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

    Mt = estimator["W"]
    Mw = estimator["H"]
    if "Q" in estimator:
        # X is a 3D tensor, in unfolded form of a 2D array
        # horizontal concatenation of blocks of equal size.
        Mb = estimator["Q"]
        NMFAlgo = 5
        NBlocks = Mb.shape[0]
        BlkSize = Mw.shape[0] * np.ones(NBlocks)
    else:
        Mb = np.array([])
        NMFAlgo = 0
        if blocks is None:
            NBlocks = 1
            BlkSize = np.array([Mw.shape[0]])
        else:
            NBlocks = blocks.shape[0]
            BlkSize = blocks

    if "WB" in estimator:
        MtPct = estimator["WB"]
    else:
        MtPct = None

    if "HB" in estimator:
        MwPct = estimator["HB"]
    else:
        MwPct = None

    if leverage == "standard":
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 0
    elif leverage == "robust":
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 1
    else:
        NMFCalculateLeverage = 0
        NMFUseRobustLeverage = 0

    if cluster_by_stability is True:
        NMFRobustClusterByStability = 1
    else:
        NMFRobustClusterByStability = 0

    if custom_order is True:
        CellPlotOrderedClusters = 1
    else:
        CellPlotOrderedClusters = 0

    AddMessage = []
    myStatusBox = StatusBoxTqdm(verbose=verbose)

    (
        Mtn,
        Mwn,
        Mbn,
        RCt,
        RCw,
        NCt,
        NCw,
        RowClust,
        ColClust,
        BlockClust,
        AddMessage,
        ErrMessage,
        cancel_pressed,
    ) = BuildClusters(
        Mt,
        Mw,
        Mb,
        MtPct,
        MwPct,
        NBlocks,
        BlkSize,
        NMFCalculateLeverage,
        NMFUseRobustLeverage,
        NMFAlgo,
        NMFRobustClusterByStability,
        CellPlotOrderedClusters,
        AddMessage,
        myStatusBox,
    )
    for message in AddMessage:
        print(message)

    myStatusBox.close()
    if "Q" in estimator:
        estimator.update(
            [
                ("WL", Mtn),
                ("HL", Mwn),
                ("WR", RCt),
                ("HR", RCw),
                ("WN", NCt),
                ("HN", NCw),
                ("WC", RowClust),
                ("HC", ColClust),
                ("QL", Mbn),
                ("QC", BlockClust),
            ]
        )
    else:
        estimator.update(
            [
                ("WL", Mtn),
                ("HL", Mwn),
                ("WR", RCt),
                ("HR", RCw),
                ("WN", NCt),
                ("HN", NCw),
                ("WC", RowClust),
                ("HC", ColClust),
                ("QL", None),
                ("QC", None),
            ]
        )
    return estimator


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
    Mt = estimator["W"]
    RCt = estimator["WR"]
    NCt = estimator["WN"]
    RowGroups = y
    uniques, index = np.unique([row for row in RowGroups], return_index=True)
    ListGroups = RowGroups[index]
    nbGroups = ListGroups.shape[0]
    Ngroup = np.zeros(nbGroups)
    for group in range(0, nbGroups):
        Ngroup[group] = np.where(RowGroups == ListGroups[group])[0].shape[0]

    Nrun = n_permutations
    myStatusBox = StatusBoxTqdm(verbose=verbose)
    ClusterSize, Pglob, prun, ClusterProb, ClusterGroup, ClusterNgroup, cancel_pressed = GlobalSign(
        Nrun, nbGroups, Mt, RCt, NCt, RowGroups, ListGroups, Ngroup, myStatusBox
    )

    estimator.update(
        [
            ("score", prun),
            ("pvalue", Pglob),
            ("CS", ClusterSize),
            ("CP", ClusterProb),
            ("CG", ClusterGroup),
            ("CN", ClusterNgroup),
        ]
    )
    return estimator


def non_negative_tensor_factorization(
        X,
        n_blocks,
        W=None,
        H=None,
        Q=None,
        n_components=None,
        update_W=True,
        update_H=True,
        update_Q=True,
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
):
    """Compute Non-negative Tensor Factorization (NTF)

    Find three non-negative matrices (W, H, F) such as x = W @@ H @@ F + Error (@@ = tensor product).
    This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is minimized with an alternating minimization of W
    and H.

    Parameters
    ----------

    X : array-like, shape (n_samples, n_features x n_blocks)
        Constant matrix.
        X is a tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd dimensions.

    n_blocks : integer

    W : array-like, shape (n_samples, n_components)
        prior W

    H : array-like, shape (n_features, n_components)
        prior H

    Q : array-like, shape (n_blocks, n_components)
        prior Q

    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    update_W : boolean, default: True
        Update or keep W fixed

    update_H : boolean, default: True
        Update or keep H fixed

    update_Q : boolean, default: True
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

        Estimator (dictionary) with following entries

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

    M = X
    n, p = M.shape

    # Identify missing values
    Mmis = np.array([])
    Mmis = Mmis.astype(np.int)
    ID = np.where(np.isnan(M) is True)
    n_Mmis = ID[0].size
    if n_Mmis > 0:
        Mmis = np.isnan(M) is False
        Mmis = Mmis.astype(np.int)
        M[Mmis == 0] = 0
    else:
        M[Mmis == 0] = 0

    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components

    NBlocks = n_blocks
    p_block = int(p / NBlocks)
    tolerance = tol
    LogIter = verbose
    if regularization is None:
        NMFSparseLevel = 0
    else:
        if regularization == "components":
            NMFSparseLevel = sparsity
        elif regularization == "transformation":
            NMFSparseLevel = -sparsity
        else:
            NMFSparseLevel = 0
    NTFUnimodal = unimodal
    NTFSmooth = smooth
    NTFLeftComponents = apply_left
    NTFRightComponents = apply_right
    NTFBlockComponents = apply_block
    if random_state is not None:
        RandomSeed = random_state
        np.random.seed(RandomSeed)

    myStatusBox = StatusBoxTqdm(verbose=LogIter)

    if (W is None) & (H is None) & (Q is None):
        Mt0, Mw0, Mb0, AddMessage, ErrMessage, cancel_pressed = ntf_init(
            M,
            Mmis,
            np.array([]),
            np.array([]),
            nc,
            tolerance,
            LogIter,
            NTFUnimodal,
            NTFLeftComponents,
            NTFRightComponents,
            NTFBlockComponents,
            NBlocks,
            init_type,
            myStatusBox
        )
    else:
        if W is None:
            Mt0 = np.ones((n, nc))
        else:
            Mt0 = np.copy(W)

        if H is None:
            Mw0 = np.ones((p_block, nc))
        else:
            Mw0 = np.copy(H)

        if Q is None:
            Mb0 = np.ones((NBlocks, nc))
        else:
            Mb0 = np.copy(Q)

        Mfit = np.zeros((n, p))
        for k in range(0, nc):
            for iBlock in range(0, NBlocks):
                Mfit[:, iBlock * p_block: (iBlock + 1) * p_block] += (
                        Mb0[iBlock, k] * np.reshape(Mt0[:, k], (n, 1)) @ np.reshape(Mw0[:, k], (1, p_block))
                )

        ScaleRatio = (np.linalg.norm(Mfit) / np.linalg.norm(M)) ** (1 / 3)
        for k in range(0, nc):
            Mt0[:, k] /= ScaleRatio
            Mw0[:, k] /= ScaleRatio
            Mb0[:, k] /= ScaleRatio

        Mfit = np.zeros((n, p))
        for k in range(0, nc):
            for iBlock in range(0, NBlocks):
                Mfit[:, iBlock * p_block: (iBlock + 1) * p_block] += (
                        Mb0[iBlock, k] * np.reshape(Mt0[:, k], (n, 1)) @ np.reshape(Mw0[:, k], (1, p_block))
                )

    MaxIterations = max_iter

    if n_bootstrap is None:
        NMFRobustNRuns = 0
    else:
        NMFRobustNRuns = n_bootstrap

    if NMFRobustNRuns <= 1:
        NMFAlgo = 5
    else:
        NMFAlgo = 6

    if leverage == "standard":
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 0
    elif leverage == "robust":
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 1
    else:
        NMFCalculateLeverage = 0
        NMFUseRobustLeverage = 0

    if random_state is not None:
        RandomSeed = random_state
        np.random.seed(RandomSeed)

    if update_W:
        NMFFixUserLHE = 0
    else:
        NMFFixUserLHE = 1

    if update_H:
        NMFFixUserRHE = 0
    else:
        NMFFixUserRHE = 1

    if update_Q:
        NMFFixUserBHE = 0
    else:
        NMFFixUserBHE = 1

    Mt_conv, Mt, Mw, Mb, MtPct, MwPct, diff, AddMessage, ErrMessage, cancel_pressed = r_ntf_solve(
        M,
        Mmis,
        Mt0,
        Mw0,
        Mb0,
        nc,
        tolerance,
        LogIter,
        MaxIterations,
        NMFFixUserLHE,
        NMFFixUserRHE,
        NMFFixUserBHE,
        NMFAlgo,
        NMFRobustNRuns,
        NMFCalculateLeverage,
        NMFUseRobustLeverage,
        NMFSparseLevel,
        NTFUnimodal,
        NTFSmooth,
        NTFLeftComponents,
        NTFRightComponents,
        NTFBlockComponents,
        NBlocks,
        np.array([]),
        myStatusBox,
    )

    volume = NMFDet(Mt, Mw, 1)

    for message in AddMessage:
        print(message)

    myStatusBox.close()

    estimator = {}
    if NMFRobustNRuns <= 1:
        estimator.update([("W", Mt), ("H", Mw), ("Q", Mb), ("volume", volume), ("diff", diff)])
    else:
        estimator.update(
            [("W", Mt), ("H", Mw), ("Q", Mb), ("volume", volume), ("WB", MtPct), ("HB", MwPct), ("diff", diff)]
        )

    return estimator
