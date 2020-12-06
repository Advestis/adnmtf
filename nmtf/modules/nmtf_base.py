"""Non-negative matrix and tensor factorization basic functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20
# Initialize progressbar
import pandas as pd
import math
import numpy as np
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.stats import hypergeom
from scipy.optimize import nnls

from .nmtf_core import *
from .nmtf_utils import *

import sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']

EPSILON = np.finfo(np.float32).eps

def NMFInit(M, Mmis, Mt0, Mw0, nc, tolerance, LogIter, myStatusBox):
    """Initialize NMF components using NNSVD

    Input:
        M: Input matrix
        Mmis: Define missing values (0 = missing cell, 1 = real cell)
        Mt0: Initial left hand matrix (may be empty)
        Mw0: Initial right hand matrix (may be empty)
        nc: NMF rank
    Output:
        Mt: Left hand matrix
        Mw: Right hand matrix
    
    Reference
    ---------

    C. Boutsidis, E. Gallopoulos (2008) SVD based initialization: A head start for nonnegative matrix factorization
    Pattern Recognition Pattern Recognition Volume 41, Issue 4, April 2008, Pages 1350-1362

    """
 
    n, p = M.shape
    Mmis = Mmis.astype(np.int)
    n_Mmis = Mmis.shape[0]
    if n_Mmis == 0:
        ID = np.where(np.isnan(M) == True)
        n_Mmis = ID[0].size
        if n_Mmis > 0:
            Mmis = (np.isnan(M) == False)
            Mmis = Mmis.astype(np.int)
            M[Mmis == 0] = 0

    nc = int(nc)
    Mt = np.copy(Mt0)
    Mw = np.copy(Mw0)
    if (Mt.shape[0] == 0) or (Mw.shape[0] == 0):
        if n_Mmis == 0:
            if nc >= min(n,p):
                # arpack does not accept to factorize at full rank -> need to duplicate in both dimensions to force it work
                t, d, w = svds(np.concatenate((np.concatenate((M, M), axis=1),np.concatenate((M, M), axis=1)), axis=0), k=nc)
                t *= np.sqrt(2)
                w *= np.sqrt(2)
                d /= 2
                # svd causes mem allocation problem with large matrices
                # t, d, w = np.linalg.svd(M)
                # Mt = t
                # Mw = w.T
            else:
                t, d, w = svds(M, k=nc)

            Mt = t[:n,:]
            Mw = w[:,:p].T
            #svds returns singular vectors in reverse order
            Mt = Mt[:,::-1]
            Mw = Mw[:,::-1]
            d = d[::-1]
        else:
            Mt, d, Mw, Mmis, Mmsr, Mmsr2, AddMessage, ErrMessage, cancel_pressed = rSVDSolve(
                M, Mmis, nc, tolerance, LogIter, 0, "", 200,
                1, 1, 1, myStatusBox)
   
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

def rNMFSolve(
        M, Mmis, Mt0, Mw0, nc, tolerance, precision, LogIter, MaxIterations, NMFAlgo, NMFFixUserLHE,
        NMFFixUserRHE, NMFMaxInterm,
        NMFSparseLevel, NMFRobustResampleColumns, NMFRobustNRuns, NMFCalculateLeverage, NMFUseRobustLeverage,
        NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns, NMFPriors, myStatusBox):

    """Estimate left and right hand matrices (robust version)

    Input:
         M: Input matrix
         Mmis: Define missing values (0 = missing cell, 1 = real cell)
         Mt0: Initial left hand matrix
         Mw0: Initial right hand matrix
         nc: NMF rank
         tolerance: Convergence threshold
         precision: Replace 0-values in multiplication rules
         LogIter: Log results through iterations
          MaxIterations: Max iterations
         NMFAlgo: =1,3: Divergence; =2,4: Least squares;
         NMFFixUserLHE: = 1 => fixed left hand matrix columns
         NMFFixUserRHE: = 1 => fixed  right hand matrix columns
         NMFMaxInterm: Max iterations for warmup multiplication rules
         NMFSparseLevel: Requested sparsity in terms of relative number of rows with 0 values in right hand matrix
         NMFRobustResampleColumns: Resample columns during bootstrap
         NMFRobustNRuns: Number of bootstrap runs
         NMFCalculateLeverage: Calculate leverages
         NMFUseRobustLeverage: Calculate leverages based on robust max across factoring columns
         NMFFindParts: Enforce convexity on left hand matrix
         NMFFindCentroids: Enforce convexity on right hand matrix
         NMFKernel: Type of kernel used; 1: linear; 2: quadraitc; 3: radial
         NMFReweighColumns: Reweigh columns in 2nd step of parts-based NMF
         NMFPriors: Priors on right hand matrix
    Output:
         Mt: Left hand matrix
         Mw: Right hand matrix
         MtPct: Percent robust clustered rows
         MwPct: Percent robust clustered columns
         diff: Objective minimum achieved
         Mh: Convexity matrix
         flagNonconvex: Updated non-convexity flag on left hand matrix

    """

    # Check parameter consistency (and correct if needed)
    AddMessage = []
    ErrMessage =''
    cancel_pressed = 0
    nc = int(nc)
    if NMFFixUserLHE*NMFFixUserRHE == 1:
        return Mt0, Mw0, np.array([]), np.array([]), 0, np.array([]), 0, AddMessage, ErrMessage, cancel_pressed

    if (nc == 1) & (NMFAlgo > 2):
        NMFAlgo -= 2

    if NMFAlgo <= 2:
        NMFRobustNRuns = 0

    Mmis = Mmis.astype(np.int)
    n_Mmis = Mmis.shape[0]
    if n_Mmis == 0:
        ID = np.where(np.isnan(M) == True)
        n_Mmis = ID[0].size
        if n_Mmis > 0:
            Mmis = (np.isnan(M) == False)
            Mmis = Mmis.astype(np.int)
            M[Mmis == 0] = 0
    else:
        M[Mmis == 0] = 0

    if NMFRobustResampleColumns > 0:
        M = np.copy(M).T
        if n_Mmis > 0:
            Mmis = np.copy(Mmis).T

        Mtemp = np.copy(Mw0)
        Mw0 = np.copy(Mt0)
        Mt0 = Mtemp
        NMFFixUserLHEtemp = NMFFixUserLHE
        NMFFixUserLHE = NMFFixUserRHE
        NMFFixUserRHE = NMFFixUserLHEtemp

    
    n, p = M.shape
    try:
        n_NMFPriors, nc = NMFPriors.shape
    except:
        n_NMFPriors = 0

    NMFRobustNRuns = int(NMFRobustNRuns)
    MtPct = np.nan
    MwPct = np.nan
    flagNonconvex = 0

    # Step 1: NMF
    Status = "Step 1 - NMF Ncomp=" + str(nc) + ": "
    Mt, Mw, diffsup, Mhsup, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
        M, Mmis, Mt0, Mw0, nc, tolerance, precision, LogIter, Status, MaxIterations, NMFAlgo,
        NMFFixUserLHE, NMFFixUserRHE, NMFMaxInterm, 100, NMFSparseLevel,
        NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns, NMFPriors, flagNonconvex, AddMessage, myStatusBox)
    Mtsup = np.copy(Mt)
    Mwsup = np.copy(Mw)
    if (n_NMFPriors > 0) & (NMFReweighColumns > 0):
        #     Run again with fixed LHE & no priors
        Status = "Step 1bis - NMF (fixed LHE) Ncomp=" + str(nc) + ": "
        Mw = np.ones((p, nc)) / math.sqrt(p)
        Mt, Mw, diffsup, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
            M, Mmis, Mtsup, Mw, nc, tolerance, precision, LogIter, Status, MaxIterations, NMFAlgo, nc, 0, NMFMaxInterm, 100,
            NMFSparseLevel, NMFFindParts, NMFFindCentroids, NMFKernel, 0, NMFPriors, flagNonconvex, AddMessage,
            myStatusBox)
        Mtsup = np.copy(Mt)
        Mwsup = np.copy(Mw)

    # Bootstrap to assess robust clustering
    if NMFRobustNRuns > 1:
        #     Update Mwsup
        MwPct = np.zeros((p, nc))
        MwBlk = np.zeros((p, NMFRobustNRuns * nc))
        for iBootstrap in range(0, NMFRobustNRuns):
            Boot = np.random.randint(n, size=n)
            Status = "Step 2 - " + \
                     "Boot " + str(iBootstrap + 1) + "/" + str(NMFRobustNRuns) + " NMF Ncomp=" + str(nc) + ": "
            if n_Mmis > 0:
                Mt, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
                    M[Boot, :], Mmis[Boot, :], Mtsup[Boot, :], Mwsup, nc, 1.e-3, precision, LogIter, Status, MaxIterations, NMFAlgo, nc, 0,
                    NMFMaxInterm, 20, NMFSparseLevel, NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns,
                    NMFPriors, flagNonconvex, AddMessage, myStatusBox)
            else:
                Mt, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
                    M[Boot, :], Mmis, Mtsup[Boot, :], Mwsup, nc, 1.e-3, precision, LogIter, Status, MaxIterations, NMFAlgo, nc, 0,
                    NMFMaxInterm, 20, NMFSparseLevel, NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns,
                    NMFPriors, flagNonconvex, AddMessage, myStatusBox)

            for k in range(0, nc):
                MwBlk[:, k * NMFRobustNRuns + iBootstrap] = Mw[:, k]

            Mwn = np.zeros((p, nc))
            for k in range(0, nc):
                if (NMFAlgo == 2) | (NMFAlgo == 4):
                    ScaleMw = np.linalg.norm(MwBlk[:, k * NMFRobustNRuns + iBootstrap])
                else:
                    ScaleMw = np.sum(MwBlk[:, k * NMFRobustNRuns + iBootstrap])

                if ScaleMw > 0:
                    MwBlk[:, k * NMFRobustNRuns + iBootstrap] = \
                        MwBlk[:, k * NMFRobustNRuns + iBootstrap] / ScaleMw

                Mwn[:, k] = MwBlk[:, k * NMFRobustNRuns + iBootstrap]

            ColClust = np.zeros(p, dtype=int)
            if NMFCalculateLeverage > 0:
                Mwn, AddMessage, ErrMessage, cancel_pressed = Leverage(Mwn, NMFUseRobustLeverage, AddMessage,
                                                                       myStatusBox)

            for j in range(0, p):
                ColClust[j] = np.argmax(np.array(Mwn[j, :]))
                MwPct[j, ColClust[j]] = MwPct[j, ColClust[j]] + 1

        MwPct = MwPct / NMFRobustNRuns

        #     Update Mtsup
        MtPct = np.zeros((n, nc))
        for iBootstrap in range(0, NMFRobustNRuns):
            Status = "Step 3 - " + \
                     "Boot " + str(iBootstrap + 1) + "/" + str(NMFRobustNRuns) + " NMF Ncomp=" + str(nc) + ": "
            Mw = np.zeros((p, nc))
            for k in range(0, nc):
                Mw[:, k] = MwBlk[:, k * NMFRobustNRuns + iBootstrap]

            Mt, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
                M, Mmis, Mtsup, Mw, nc, 1.e-3, precision, LogIter, Status, MaxIterations, NMFAlgo, 0, nc, NMFMaxInterm, 20,
                NMFSparseLevel, NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns, NMFPriors, flagNonconvex,
                AddMessage, myStatusBox)
            RowClust = np.zeros(n, dtype=int)
            if NMFCalculateLeverage > 0:
                Mtn, AddMessage, ErrMessage, cancel_pressed = Leverage(Mt, NMFUseRobustLeverage, AddMessage,
                                                                       myStatusBox)
            else:
                Mtn = Mt

            for i in range(0, n):
                RowClust[i] = np.argmax(Mtn[i, :])
                MtPct[i, RowClust[i]] = MtPct[i, RowClust[i]] + 1

        MtPct = MtPct / NMFRobustNRuns

    Mt = Mtsup
    Mw = Mwsup
    Mh = Mhsup
    diff = diffsup

    if NMFRobustResampleColumns > 0:
        Mtemp = np.copy(Mt)
        Mt = np.copy(Mw)
        Mw = Mtemp
        Mtemp = np.copy(MtPct)
        MtPct = np.copy(MwPct)
        MwPct = Mtemp

    return Mt, Mw, MtPct, MwPct, diff, Mh, flagNonconvex, AddMessage, ErrMessage, cancel_pressed

def NTFInit(M, Mmis, Mt_nmf, Mw_nmf, nc, tolerance, precision, LogIter, NTFUnimodal,
            NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, init_type, myStatusBox):
    """Initialize NTF components for HALS

     Input:
         M: Input tensor
         Mmis: Define missing values (0 = missing cell, 1 = real cell)
         Mt_nmf: initialization of LHM in NMF(unstacked tensor), may be empty
         Mw_nmf: initialization of RHM of NMF(unstacked tensor), may be empty
         nc: NTF rank
         tolerance: Convergence threshold
         precision: Replace 0-values in multiplication rules
         LogIter: Log results through iterations
         NTFUnimodal: Apply Unimodal constraint on factoring vectors
         NTFLeftComponents: Apply Unimodal/Smooth constraint on left hand matrix
         NTFRightComponents: Apply Unimodal/Smooth constraint on right hand matrix
         NTFBlockComponents: Apply Unimodal/Smooth constraint on block hand matrix
         NBlocks: Number of NTF blocks
         init_type : integer, default 0
             init_type = 0 : NMF initialization applied on the reshaped matrix [1st dim x vectorized (2nd & 3rd dim)] 
             init_type = 1 : NMF initialization applied on the reshaped matrix [vectorized (1st & 2nd dim) x 3rd dim] 
     Output:
         Mt: Left hand matrix
         Mw: Right hand matrix
         Mb: Block hand matrix
     """
    AddMessage = []

    n, p = M.shape
    Mmis = Mmis.astype(np.int)
    n_Mmis = Mmis.shape[0]
    if n_Mmis == 0:
        ID = np.where(np.isnan(M) == True)
        n_Mmis = ID[0].size
        if n_Mmis > 0:
            Mmis = (np.isnan(M) == False)
            Mmis = Mmis.astype(np.int)
            M[Mmis == 0] = 0
  
    nc = int(nc)
    NBlocks = int(NBlocks)
    init_type = int(init_type)

    Status0 = "Step 1 - Quick NMF Ncomp=" + str(nc) + ": "
    
    if init_type == 1:
        #Init legacy
        Mstacked, Mmis_stacked = NTFStack(M, Mmis, NBlocks)
        nc2 = min(nc, NBlocks)  # factorization rank can't be > number of blocks
        if (Mt_nmf.shape[0] == 0) or (Mw_nmf.shape[0] == 0):
            Mt_nmf, Mw_nmf = NMFInit(Mstacked, Mmis_stacked, np.array([]),  np.array([]), nc2, tolerance, LogIter, myStatusBox)
        else:
            Mt_nmf, Mw_nmf = NMFInit(Mstacked, Mmis_stacked, Mt_nmf, Mw_nmf, nc2, tolerance, LogIter, myStatusBox)

        # Quick NMF
        Mt_nmf, Mw_nmf, diff, Mh, dummy1, dummy2, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
            Mstacked, Mmis_stacked, Mt_nmf, Mw_nmf, nc2, tolerance, precision, LogIter, Status0,
            10, 2, 0, 0, 1, 1, 0, 0, 0, 1, 0, np.array([]), 0, AddMessage, myStatusBox)
    
        # Factorize Left vectors and distribute multiple factors if nc2 < nc
        Mt = np.zeros((n, nc))
        Mw = np.zeros((int(p / NBlocks), nc))
        Mb = np.zeros((NBlocks, nc))
        NFact = int(np.ceil(nc / NBlocks))
        for k in range(0, nc2):
            myStatusBox.update_status(delay=1, status="Start SVD...")
            U, d, V = svds(np.reshape(Mt_nmf[:, k], (int(p / NBlocks), n)).T, k=NFact)
            V = V.T
            #svds returns singular vectors in reverse order
            U = U[:,::-1]
            V = V[:,::-1]
            d = d[::-1]

            myStatusBox.update_status(delay=1, status="SVD completed")
            for iFact in range(0, NFact):
                ind = iFact * NBlocks + k
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
                    V1 = np.reshape(V1, (1, int(p / NBlocks)))
                    U2 = np.reshape(U2, (n, 1))
                    V2 = np.reshape(V2, ((1, int(p / NBlocks))))
                    if np.linalg.norm(U1 @ V1) > np.linalg.norm(U2 @ V2):
                        Mt[:, ind] = np.reshape(U1, n)
                        Mw[:, ind] = d[iFact] * np.reshape(V1, int(p / NBlocks))
                    else:
                        Mt[:, ind] = np.reshape(U2, n)
                        Mw[:, ind] = d[iFact] * np.reshape(V2, int(p / NBlocks))

                    Mb[:, ind] = Mw_nmf[:, k]
    else:
        #Init default
        if (Mt_nmf.shape[0] == 0) or (Mw_nmf.shape[0] == 0):
            Mt_nmf, Mw_nmf = NMFInit(M, Mmis, np.array([]),  np.array([]), nc, tolerance, LogIter, myStatusBox)
        else:
            Mt_nmf, Mw_nmf = NMFInit(M, Mmis, Mt_nmf, Mw_nmf, nc, tolerance, LogIter, myStatusBox)

        # Quick NMF
        Mt_nmf, Mw_nmf, diff, Mh, dummy1, dummy2, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
            M, Mmis, Mt_nmf, Mw_nmf, nc, tolerance, precision, LogIter, Status0,
            10, 2, 0, 0, 1, 1, 0, 0, 0, 1, 0, np.array([]), 0, AddMessage, myStatusBox)
    
        #Factorize Left vectors 
        Mt = np.zeros((n, nc))
        Mw = np.zeros((int(p / NBlocks), nc))
        Mb = np.zeros((NBlocks, nc))

        for k in range(0, nc):
            myStatusBox.update_status(delay=1, status="Start SVD...")
            U, d, V = svds(np.reshape(Mw_nmf[:, k], (int(p / NBlocks), NBlocks)), k=1)
            V = V.T
            U = np.abs(U) 
            V = np.abs(V)
            myStatusBox.update_status(delay=1, status="SVD completed")
            Mt[:, k] = Mt_nmf[:, k]
            Mw[:, k] = d[0] * np.reshape(U, int(p / NBlocks))
            Mb[:, k] = np.reshape(V, NBlocks)

        for k in range(0, nc):
            if (NTFUnimodal > 0) & (NTFLeftComponents > 0):
                #                 Enforce unimodal distribution
                tmax = np.argmax(Mt[:, k])
                for i in range(tmax + 1, n):
                    Mt[i, k] = min(Mt[i - 1, k], Mt[i, k])

                for i in range(tmax - 1, -1, -1):
                    Mt[i, k] = min(Mt[i + 1, k], Mt[i, k])

    if (NTFUnimodal > 0) & (NTFRightComponents > 0):
        #                 Enforce unimodal distribution
        wmax = np.argmax(Mw[:, k])
        for j in range(wmax + 1, int(p / NBlocks)):
            Mw[j, k] = min(Mw[j - 1, k], Mw[j, k])

        for j in range(wmax - 1, -1, -1):
            Mw[j, k] = min(Mw[j + 1, k], Mw[j, k])

    if (NTFUnimodal > 0) & (NTFBlockComponents > 0):
        #                 Enforce unimodal distribution
        bmax = np.argmax(Mb[:, k])
        for iBlock in range(bmax + 1, NBlocks):
            Mb[iBlock, k] = min(Mb[iBlock - 1, k], Mb[iBlock, k])

        for iBlock in range(bmax - 1, -1, -1):
            Mb[iBlock, k] = min(Mb[iBlock + 1, k], Mb[iBlock, k])

    return [Mt, Mw, Mb, AddMessage, ErrMessage, cancel_pressed]
  
def rNTFSolve(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, precision, LogIter, MaxIterations, NMFFixUserLHE, NMFFixUserRHE,
              NMFFixUserBHE, NMFAlgo, NMFRobustNRuns, NMFCalculateLeverage, NMFUseRobustLeverage, NTFFastHALS, NTFNIterations,
              NMFSparseLevel, NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv,
              NMFPriors, myStatusBox):
    """Estimate NTF matrices (robust version)

     Input:
         M: Input matrix
         Mmis: Define missing values (0 = missing cell, 1 = real cell)
         Mt0: Initial left hand matrix
         Mw0: Initial right hand matrix
         Mb0: Initial block hand matrix
         nc: NTF rank
         tolerance: Convergence threshold
         precision: Replace 0-values in multiplication rules
         LogIter: Log results through iterations
         MaxIterations: Max iterations
         NMFFixUserLHE: fix left hand matrix columns: = 1, else = 0
         NMFFixUserRHE: fix  right hand matrix columns: = 1, else = 0
         NMFFixUserBHE: fix  block hand matrix columns: = 1, else = 0
         NMFAlgo: =5: Non-robust version, =6: Robust version
         NMFRobustNRuns: Number of bootstrap runs
         NMFCalculateLeverage: Calculate leverages
         NMFUseRobustLeverage: Calculate leverages based on robust max across factoring columns
         NTFFastHALS: Use Fast HALS (does not accept handle missing values and convolution)
         NTFNIterations: Warmup iterations for fast HALS
         NMFSparseLevel : sparsity level (as defined by Hoyer); +/- = make RHE/LHe sparse
         NTFUnimodal: Apply Unimodal constraint on factoring vectors
         NTFSmooth: Apply Smooth constraint on factoring vectors
         NTFLeftComponents: Apply Unimodal/Smooth constraint on left hand matrix
         NTFRightComponents: Apply Unimodal/Smooth constraint on right hand matrix
         NTFBlockComponents: Apply Unimodal/Smooth constraint on block hand matrix
         NBlocks: Number of NTF blocks
         NTFNConv: Half-Size of the convolution window on 3rd-dimension of the tensor
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
    ErrMessage = ''
    cancel_pressed = 0
    n, p0 = M.shape
    nc = int(nc)
    NBlocks = int(NBlocks)
    p = int(p0 / NBlocks)
    NTFNConv = int(NTFNConv)
    if NMFFixUserLHE*NMFFixUserRHE*NMFFixUserBHE == 1:
        return np.zeros((n, nc*(2*NTFNConv+1))), Mt0, Mw0, Mb0, np.zeros((n, p0)), np.ones((n, nc)), np.ones((p, nc)), AddMessage, ErrMessage, cancel_pressed

    Mmis = Mmis.astype(np.int)
    n_Mmis = Mmis.shape[0]
    if n_Mmis == 0:
        ID = np.where(np.isnan(M) == True)
        n_Mmis = ID[0].size
        if n_Mmis > 0:
            Mmis = (np.isnan(M) == False)
            Mmis = Mmis.astype(np.int)
            M[Mmis == 0] = 0
    else:
        M[Mmis == 0] = 0

    NTFNIterations = int(NTFNIterations)
    NMFRobustNRuns = int(NMFRobustNRuns)
    Mt = np.copy(Mt0)
    Mw = np.copy(Mw0)
    Mb = np.copy(Mb0)
    Mt_conv = np.array([])

    # Check parameter consistency (and correct if needed)
    if (nc == 1) | (NMFAlgo == 5):
        NMFRobustNRuns = 0

    if NMFRobustNRuns == 0:
        MtPct = np.nan
        MwPct = np.nan

    if (n_Mmis > 0 or NTFNConv > 0 or NMFSparseLevel != 0) and NTFFastHALS > 0:
        NTFFastHALS = 0
        reverse2HALS = 1
    else:
        reverse2HALS = 0

    # Step 1: NTF
    Status0 = "Step 1 - NTF Ncomp=" + str(nc) + ": "
    if NTFFastHALS > 0:
        if NTFNIterations > 0:
            Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
                M, Mmis, Mt, Mw, Mb, nc, tolerance, LogIter, Status0,
                NTFNIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE, 0, NTFUnimodal, NTFSmooth,
                NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, NMFPriors, myStatusBox)

        Mt, Mw, Mb, diff, cancel_pressed = NTFSolveFast(
            M, Mmis, Mt, Mw, Mb, nc, tolerance, precision, LogIter, Status0,
            MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
            NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox)
    else:
        Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
            M, Mmis, Mt, Mw, Mb, nc, tolerance, LogIter, Status0,
            MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE, NMFSparseLevel, NTFUnimodal, NTFSmooth,
            NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, NMFPriors, myStatusBox)

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
            Status0 = "Step 2 - " + \
                      "Boot " + str(iBootstrap + 1) + "/" + str(NMFRobustNRuns) + " NTF Ncomp=" + str(nc) + ": "
            if NTFFastHALS > 0:
                if n_Mmis > 0:
                    Mt, Mw, Mb, diff, cancel_pressed = NTFSolveFast(
                        M[Boot, :], Mmis[Boot, :], Mtsup[Boot, :], Mwsup, Mb, nc, 1.e-3, precision, LogIter, Status0,
                        MaxIterations, 1, 0, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
                        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox)
                else:
                    Mt, Mw, Mb, diff, cancel_pressed = NTFSolveFast(
                        M[Boot, :], np.array([]), Mtsup[Boot, :], Mwsup, Mb, nc, 1.e-3, precision, LogIter, Status0,
                        MaxIterations, 1, 0, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
                        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox)
            else:
                if n_Mmis > 0:
                    Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
                        M[Boot, :], Mmis[Boot, :], Mtsup[Boot, :], Mwsup, Mb, nc, 1.e-3, LogIter, Status0,
                        MaxIterations, 1, 0, NMFFixUserBHE, NMFSparseLevel, NTFUnimodal, NTFSmooth,
                        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, NMFPriors, myStatusBox)
                else:
                    Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
                        M[Boot, :], np.array([]), Mtsup[Boot, :], Mwsup, Mb, nc, 1.e-3, LogIter, Status0,
                        MaxIterations, 1, 0, NMFFixUserBHE, NMFSparseLevel, NTFUnimodal, NTFSmooth,
                        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, NMFPriors, myStatusBox)

            for k in range(0, nc):
                MwBlk[:, k * NMFRobustNRuns + iBootstrap] = Mw[:, k]

            Mwn = np.zeros((p, nc))
            for k in range(0, nc):
                ScaleMw = np.linalg.norm(MwBlk[:, k * NMFRobustNRuns + iBootstrap])
                if ScaleMw > 0:
                    MwBlk[:, k * NMFRobustNRuns + iBootstrap] = \
                        MwBlk[:, k * NMFRobustNRuns + iBootstrap] / ScaleMw

                Mwn[:, k] = MwBlk[:, k * NMFRobustNRuns + iBootstrap]

            ColClust = np.zeros(p, dtype=int)
            if NMFCalculateLeverage > 0:
                Mwn, AddMessage, ErrMessage, cancel_pressed = Leverage(Mwn, NMFUseRobustLeverage, AddMessage,
                                                                       myStatusBox)

            for j in range(0, p):
                ColClust[j] = np.argmax(np.array(Mwn[j, :]))
                MwPct[j, ColClust[j]] = MwPct[j, ColClust[j]] + 1

        MwPct = MwPct / NMFRobustNRuns

        #     Update Mtsup
        MtPct = np.zeros((n, nc))
        for iBootstrap in range(0, NMFRobustNRuns):
            Status0 = "Step 3 - " + \
                      "Boot " + str(iBootstrap + 1) + "/" + str(NMFRobustNRuns) + " NTF Ncomp=" + str(nc) + ": "
            Mw = np.zeros((p, nc))
            for k in range(0, nc):
                Mw[:, k] = MwBlk[:, k * NMFRobustNRuns + iBootstrap]

            if NTFFastHALS > 0:
                Mt, Mw, Mb, diff, cancel_pressed = NTFSolveFast(
                    M, Mmis, Mtsup, Mw, Mb, nc, 1.e-3, precision, LogIter, Status0, MaxIterations, 0, 1, NMFFixUserBHE,
                    NTFUnimodal, NTFSmooth,
                    NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox)
            else:
                Mt_conv, Mt, Mw, Mb, diff, cancel_pressed = NTFSolve(
                    M, Mmis, Mtsup, Mw, Mb, nc, 1.e-3, LogIter, Status0, MaxIterations, 0, 1, NMFFixUserBHE,
                    NMFSparseLevel, NTFUnimodal, NTFSmooth,
                    NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, NMFPriors, myStatusBox)

            RowClust = np.zeros(n, dtype=int)
            if NMFCalculateLeverage > 0:
                Mtn, AddMessage, ErrMessage, cancel_pressed = Leverage(Mt, NMFUseRobustLeverage, AddMessage,
                                                                       myStatusBox)
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
    if reverse2HALS > 0:
        AddMessage.insert(len(AddMessage), 'Currently, Fast HALS cannot be applied with missing data or convolution window and was reversed to Simple HALS.')

    return Mt_conv, Mt, Mw, Mb, MtPct, MwPct, diff, AddMessage, ErrMessage, cancel_pressed

def rSVDSolve(M, Mmis, nc, tolerance, LogIter, LogTrials, Status0, MaxIterations,
              SVDAlgo, SVDCoverage, SVDNTrials, myStatusBox):
    """Estimate SVD matrices (robust version)

     Input:
         M: Input matrix
         Mmis: Define missing values (0 = missing cell, 1 = real cell)
         nc: SVD rank
         tolerance: Convergence threshold
         LogIter: Log results through iterations
         LogTrials: Log results through trials
         Status0: Initial displayed status to be updated during iterations
         MaxIterations: Max iterations
         SVDAlgo: =1: Non-robust version, =2: Robust version
         SVDCoverage: Coverage non-outliers (robust version)
         SVDNTrials: Number of trials (robust version)
     
     Output:
         Mt: Left hand matrix
         Mev: Scaling factors
         Mw: Right hand matrix
         Mmis: Matrix of missing/flagged outliers
         Mmsr: Vector of Residual SSQ
         Mmsr2: Vector of Reidual variance

     Reference
     ---------

     L. Liu et al (2003) Robust singular value decomposition analysis of microarray data
     PNAS November 11, 2003 vol. 100 no. 23 13167–13172

    """

    AddMessage = []
    ErrMessage = ''
    cancel_pressed = 0

    # M0 is the running matrix (to be factorized, initialized from M)
    M0 = np.copy(M)
    n, p = M0.shape
    Mmis = Mmis.astype(np.bool_)
    n_Mmis = Mmis.shape[0]

    if n_Mmis > 0:
        M0[Mmis == False] = np.nan
    else:
        Mmis = (np.isnan(M0) == False)
        Mmis = Mmis.astype(np.bool_)
        n_Mmis = Mmis.shape[0]

    trace0 = np.sum(M0[Mmis] ** 2)
    nc = int(nc)
    SVDNTrials = int(SVDNTrials)
    nxp = n * p
    nxpcov = int(round(nxp * SVDCoverage, 0))
    Mmsr = np.zeros(nc)
    Mmsr2 = np.zeros(nc)
    Mev = np.zeros(nc)
    if SVDAlgo == 2:
        MaxTrial = SVDNTrials
    else:
        MaxTrial = 1

    Mw = np.zeros((p, nc))
    Mt = np.zeros((n, nc))
    Mdiff = np.zeros((n, p))
    w = np.zeros(p)
    t = np.zeros(n)
    wTrial = np.zeros(p)
    tTrial = np.zeros(n)
    MmisTrial = np.zeros((n, p), dtype=np.bool)
    # Outer-reference M becomes local reference M, which is the running matrix within ALS/LTS loop.
    M = np.zeros((n, p))
    wnorm = np.zeros((p, n))
    tnorm = np.zeros((n, p))
    denomw = np.zeros(n)
    denomt = np.zeros(p)
    StepIter = math.ceil(MaxIterations / 100)
    pbar_step = 100 * StepIter / MaxIterations
    if (n_Mmis == 0) & (SVDAlgo == 1):
        FastCode = 1
    else:
        FastCode = 0

    if (FastCode == 0) and (SVDAlgo == 1):
        denomw[np.count_nonzero(Mmis, axis=1) < 2] = np.nan
        denomt[np.count_nonzero(Mmis, axis=0) < 2] = np.nan

    for k in range(0, nc):
        for iTrial in range(0, MaxTrial):
            myStatusBox.init_bar(delay=1)
            # Copy values of M0 into M
            M[:, :] = M0
            Status1 = Status0 + "Ncomp " + str(k + 1) + " Trial " + str(iTrial + 1) + ": "
            if SVDAlgo == 2:
                #         Select a random subset
                M = np.reshape(M, (nxp, 1))
                M[np.argsort(np.random.rand(nxp))[nxpcov:nxp]] = np.nan
                M = np.reshape(M, (n, p))

            Mmis[:, :] = (np.isnan(M) == False)

            #         Initialize w
            for j in range(0, p):
                w[j] = np.median(M[Mmis[:, j], j])

            if np.where(w > 0)[0].size == 0:
                w[:] = 1

            w /= np.linalg.norm(w)
            # Replace missing values by 0's before regression
            M[Mmis == False] = 0

            #         initialize t (LTS  =stochastic)
            if FastCode == 0:
                wnorm[:, :] = np.repeat(w[:, np.newaxis]**2, n, axis=1) * Mmis.T
                denomw[:] = np.sum(wnorm, axis=0)
                # Request at least 2 non-missing values to perform row regression
                if SVDAlgo == 2:
                    denomw[np.count_nonzero(Mmis, axis=1) < 2] = np.nan

                t[:] = M @ w / denomw
            else:
                t[:] = M @ w / np.linalg.norm(w) ** 2

            t[np.isnan(t) == True] = np.median(t[np.isnan(t) == False])

            if SVDAlgo == 2:
                Mdiff[:, :] = np.abs(M0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
                # Restore missing values instead of 0's
                M[Mmis == False] = M0[Mmis == False]
                M = np.reshape(M, (nxp, 1))
                M[np.argsort(np.reshape(Mdiff, nxp))[nxpcov:nxp]] = np.nan
                M = np.reshape(M, (n, p))
                Mmis[:, :] = (np.isnan(M) == False)
                # Replace missing values by 0's before regression
                M[Mmis == False] = 0

            iIter = 0
            cont = 1
            while (cont > 0) & (iIter < MaxIterations):
                #                 build w
                if FastCode == 0:
                    tnorm[:, :] = np.repeat(t[:, np.newaxis]**2, p, axis=1) * Mmis
                    denomt[:] = np.sum(tnorm, axis=0)
                    #Request at least 2 non-missing values to perform column regression
                    if SVDAlgo == 2:
                        denomt[np.count_nonzero(Mmis, axis=0) < 2] = np.nan

                    w[:] = M.T @ t / denomt
                else:
                    w[:] = M.T @ t / np.linalg.norm(t) ** 2

                w[np.isnan(w) == True] = np.median(w[np.isnan(w) == False])
                #                 normalize w
                w /= np.linalg.norm(w)
                if SVDAlgo == 2:
                    Mdiff[:, :] = np.abs(M0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
                    # Restore missing values instead of 0's
                    M[Mmis == False] = M0[Mmis == False]
                    M = np.reshape(M, (nxp, 1))
                    # Outliers resume to missing values
                    M[np.argsort(np.reshape(Mdiff, nxp))[nxpcov:nxp]] = np.nan
                    M = np.reshape(M, (n, p))
                    Mmis[:, :] = (np.isnan(M) == False)
                    # Replace missing values by 0's before regression
                    M[Mmis == False] = 0

                #                 build t
                if FastCode == 0:
                    wnorm[:, :] = np.repeat(w[:, np.newaxis] ** 2, n, axis=1) * Mmis.T
                    denomw[:] = np.sum(wnorm, axis=0)
                    # Request at least 2 non-missing values to perform row regression
                    if SVDAlgo == 2:
                        denomw[np.count_nonzero(Mmis, axis=1) < 2] = np.nan

                    t[:] = M @ w / denomw
                else:
                    t[:] = M @ w / np.linalg.norm(w) ** 2

                t[np.isnan(t) == True] = np.median(t[np.isnan(t) == False])
                #                 note: only w is normalized within loop, t is normalized after convergence
                if SVDAlgo == 2:
                    Mdiff[:, :] = np.abs(M0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
                    # Restore missing values instead of 0's
                    M[Mmis == False] = M0[Mmis == False]
                    M = np.reshape(M, (nxp, 1))
                    # Outliers resume to missing values
                    M[np.argsort(np.reshape(Mdiff, nxp))[nxpcov:nxp]] = np.nan
                    M = np.reshape(M, (n, p))
                    Mmis[:, :] = (np.isnan(M) == False)
                    # Replace missing values by 0's before regression
                    M[Mmis == False] = 0

                if iIter % StepIter == 0:
                    if SVDAlgo == 1:
                        Mdiff[:, :] = np.abs(M0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))

                    Status = Status1 + 'Iteration: %s' % int(iIter)
                    myStatusBox.update_status(delay=1, status=Status)
                    myStatusBox.update_bar(delay=1, step=pbar_step)
                    if myStatusBox.cancel_pressed:
                        cancel_pressed = 1
                        return [Mt, Mev, Mw, Mmis, Mmsr, Mmsr2, AddMessage, ErrMessage, cancel_pressed]

                    diff = np.linalg.norm(Mdiff[Mmis]) ** 2 / np.where(Mmis)[0].size
                    if LogIter == 1:
                        if SVDAlgo == 2:
                            myStatusBox.myPrint("Ncomp: " + str(k) + " Trial: " + str(iTrial) + " Iter: " + str(
                                iIter) + " MSR: " + str(diff))
                        else:
                            myStatusBox.myPrint("Ncomp: " + str(k) + " Iter: " + str(iIter) + " MSR: " + str(diff))

                    if iIter > 0:
                        if abs(diff - diff0) / diff0 < tolerance:
                            cont = 0

                    diff0 = diff

                iIter += 1

            #         save trial
            if iTrial == 0:
                BestTrial = iTrial
                DiffTrial = diff
                tTrial[:] = t
                wTrial[:] = w
                MmisTrial[:, :] = Mmis
            elif diff < DiffTrial:
                BestTrial = iTrial
                DiffTrial = diff
                tTrial[:] = t
                wTrial[:] = w
                MmisTrial[:, :] = Mmis

            if LogTrials == 1:
                myStatusBox.myPrint("Ncomp: " + str(k) + " Trial: " + str(iTrial) + " MSR: " + str(diff))

        if LogTrials:
            myStatusBox.myPrint("Ncomp: " + str(k) + " Best trial: " + str(BestTrial) + " MSR: " + str(DiffTrial))

        t[:] = tTrial
        w[:] = wTrial
        Mw[:, k] = w
        #         compute eigen value
        if SVDAlgo == 2:
            #             Robust regression of M on tw`
            Mdiff[:, :] = np.abs(M0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
            RMdiff = np.argsort(np.reshape(Mdiff, nxp))
            t /= np.linalg.norm(t)  # Normalize t
            Mt[:, k] = t
            Mmis = np.reshape(Mmis, nxp)
            Mmis[RMdiff[nxpcov:nxp]] = False
            Ycells = np.reshape(M0, (nxp, 1))[Mmis]
            Xcells = np.reshape(np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)), (nxp, 1))[Mmis]
            Mev[k] = Ycells.T @ Xcells / np.linalg.norm(Xcells) ** 2
            Mmis = np.reshape(Mmis, (n, p))
        else:
            Mev[k] = np.linalg.norm(t)
            Mt[:, k] = t / Mev[k]  # normalize t

        if k == 0:
            Mmsr[k] = Mev[k] ** 2
        else:
            Mmsr[k] = Mmsr[k - 1] + Mev[k] ** 2
            Mmsr2[k] = Mmsr[k] - Mev[0] ** 2

        # M0 is deflated before calculating next component
        M0 = M0 - Mev[k] * np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k].T, (1, p))

    trace02 = trace0 - Mev[0] ** 2
    Mmsr = 1 - Mmsr / trace0
    Mmsr[Mmsr > 1] = 1
    Mmsr[Mmsr < 0] = 0
    Mmsr2 = 1 - Mmsr2 / trace02
    Mmsr2[Mmsr2 > 1] = 1
    Mmsr2[Mmsr2 < 0] = 0
    if nc > 1:
        RMev = np.argsort(-Mev)
        Mev = Mev[RMev]
        Mw0 = Mw
        Mt0 = Mt
        for k in range(0, nc):
            Mw[:, k] = Mw0[:, RMev[k]]
            Mt[:, k] = Mt0[:, RMev[k]]

    Mmis[:, :] = True
    Mmis[MmisTrial == False] = False
    #Mmis.astype(dtype=int)

    return [Mt, Mev, Mw, Mmis, Mmsr, Mmsr2, AddMessage, ErrMessage, cancel_pressed]

def non_negative_factorization(X, W=None, H=None, n_components=None,
                               update_W=True,
                               update_H=True,
                               beta_loss='frobenius',
                               use_hals=False,
                               n_bootstrap=None,
                               tol=1e-6,
                               max_iter=150, max_iter_mult=20,
                               regularization=None, sparsity=0,
                               leverage='standard',
                               convex=None, kernel='linear',
                               skewness=False,
                               null_priors=False,
                               random_state=None,
                               verbose=0):
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

    beta_loss : string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss == 'kullback-leibler', the input
        matrix X cannot contain zeros.

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
        Select whether the regularization affects the components (H), the
        transformation (W) or none of them.

    sparsity : float, default: 0
        Sparsity target with 0 <= sparsity <= 1 representing either:
        - the % rows in W or H set to 0 (when use_hals = False)
        - the mean % rows per column in W or H set to 0 (when use_hals = True)

    leverage :  None | 'standard' | 'robust', default 'standard'
        Calculate leverage of W and H rows on each component.

    convex :  None | 'components' | 'transformation', default None
        Apply convex constraint on W or H.

    kernel :  'linear', 'quadratic', 'radial', default 'linear'
        Can be set if convex = 'transformation'.

    null_priors : boolean, default False
        Cells of H with prior cells = 0 will not be updated.
        Can be set only if prior H has been defined.

    skewness : boolean, default False
        When solving mixture problems, columns of X at the extremities of the convex hull will be given largest weights.
        The column weight is a function of the skewness and its sign.
        The expected sign of the skewness is based on the skewness of W components, as returned by the first pass
        of a 2-steps convex NMF. Thus, during the first pass, skewness must be set to False.
        Can be set only if convex = 'transformation' and prior W and H have been defined.

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

    if use_hals:
        #convex and kullback-leibler loss options are not supported
        beta_loss='frobenius'
        convex=None
    
    M = X
    n, p = M.shape
    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components

    if beta_loss == 'frobenius':
        NMFAlgo = 2
    else:
        NMFAlgo = 1

    LogIter = verbose
    myStatusBox = StatusBoxTqdm(verbose=LogIter)
    tolerance = tol
    precision = EPSILON
    if (W is None) & (H is None):
        Mt, Mw = NMFInit(M, np.array([]), np.array([]), np.array([]), nc, tolerance, LogIter, myStatusBox)
        init = 'nndsvd'
    else:
        if H is None:
            Mw = np.ones((p, nc))
            init = 'custom_W'
        elif W is None:
            Mt = np.ones((n, nc))
            init = 'custom_H'
        else:
            init = 'custom'

        for k in range(0, nc):
            if NMFAlgo == 2:
                Mt[:, k] = Mt[:, k] / np.linalg.norm(Mt[:, k])
                Mw[:, k] = Mw[:, k] / np.linalg.norm(Mw[:, k])
            else:
                Mt[:, k] = Mt[:, k] / np.sum(Mt[:, k])
                Mw[:, k] = Mw[:, k] / np.sum(Mw[:, k])

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
    NMFMaxInterm = max_iter_mult
    if regularization is None:
        NMFSparseLevel = 0
    else:
        if regularization == 'components':
            NMFSparseLevel = sparsity
        elif regularization == 'transformation':
            NMFSparseLevel = -sparsity
        else:
            NMFSparseLevel = 0

    NMFRobustResampleColumns = 0

    if leverage == 'standard':
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 0
    elif leverage == 'robust':
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 1
    else:
        NMFCalculateLeverage = 0
        NMFUseRobustLeverage = 0

    if convex is None:
        NMFFindParts = 0
        NMFFindCentroids = 0
        NMFKernel = 1
    elif convex == 'transformation':
        NMFFindParts = 1
        NMFFindCentroids = 0
        NMFKernel = 1
    elif convex == 'components':
        NMFFindParts = 0
        NMFFindCentroids = 1
        if kernel == 'linear':
            NMFKernel = 1
        elif kernel == 'quadratic':
            NMFKernel = 2
        elif kernel == 'radial':
            NMFKernel = 3
        else:
            NMFKernel = 1

    if (null_priors is True) & ((init == 'custom') | (init == 'custom_H')):
        NMFPriors = H
    else:
        NMFPriors = np.array([])

    if convex is None:
        NMFReweighColumns = 0
    else:
        if (convex == 'transformation') & (init == 'custom'):
            if skewness is True:
                NMFReweighColumns = 1
            else:
                NMFReweighColumns = 0

        else:
            NMFReweighColumns = 0

    if random_state is not None:
        RandomSeed = random_state
        np.random.seed(RandomSeed)

    if use_hals:
        if NMFAlgo <=2:
            NTFAlgo = 5
        else:
            NTFAlgo = 6
        
        Mt_conv, Mt, Mw, Mb, MtPct, MwPct, diff, AddMessage, ErrMessage, cancel_pressed = rNTFSolve(
            M, np.array([]), Mt, Mw, np.array([]), nc, tolerance, precision, LogIter, MaxIterations, NMFFixUserLHE, NMFFixUserRHE,
            1, NTFAlgo, NMFRobustNRuns, NMFCalculateLeverage, NMFUseRobustLeverage,
            0, 0, NMFSparseLevel, 0, 0, 0, 0, 0, 1, 0, np.array([]), myStatusBox)
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

    else:
        Mt, Mw, MtPct, MwPct, diff, Mh, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = rNMFSolve(
            M, np.array([]), Mt, Mw, nc, tolerance, precision, LogIter, MaxIterations, NMFAlgo, NMFFixUserLHE,
            NMFFixUserRHE, NMFMaxInterm,
            NMFSparseLevel, NMFRobustResampleColumns, NMFRobustNRuns, NMFCalculateLeverage, NMFUseRobustLeverage,
            NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns, NMFPriors, myStatusBox)

        Mev = np.ones(nc)
        if (NMFFindParts == 0) & (NMFFindCentroids == 0) & (NMFFixUserLHE == 0) & (NMFFixUserRHE == 0):
            # Scale
            for k in range(0, nc):
                if (NMFAlgo == 2) | (NMFAlgo == 4):
                    ScaleMt = np.linalg.norm(Mt[:, k])
                    ScaleMw = np.linalg.norm(Mw[:, k])
                else:
                    ScaleMt = np.sum(Mt[:, k])
                    ScaleMw = np.sum(Mw[:, k])

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

    if (NMFFindParts == 0) & (NMFFindCentroids == 0):
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
        if (NMFFindParts == 0) & (NMFFindCentroids == 0):
            estimator.update([('W', Mt), ('H', Mw), ('volume', volume), ('diff', diff)])
        else:
            estimator.update([('W', Mt), ('H', Mw), ('volume', volume), ('B', Mh), ('diff', diff)])

    else:
        if (NMFFindParts == 0) & (NMFFindCentroids == 0):
            estimator.update([('W', Mt), ('H', Mw), ('volume', volume), ('WB', MtPct), ('HB', MwPct), ('diff', diff)])
        else:
            estimator.update([('W', Mt), ('H', Mw), ('volume', volume), ('B', Mh), ('WB', MtPct), ('HB', MwPct), ('diff', diff)])

    return estimator

def nmf_predict(estimator, leverage='robust', blocks=None, cluster_by_stability=False, custom_order=False, verbose=0):
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

    Mt = estimator['W']
    Mw = estimator['H']
    if 'Q' in estimator:
        # X is a 3D tensor, in unfolded form of a 2D array
        # horizontal concatenation of blocks of equal size.
        Mb = estimator['Q']
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

    if 'WB' in estimator:
        MtPct = estimator['WB']
    else:
        MtPct = None

    if 'HB' in estimator:
        MwPct = estimator['HB']
    else:
        MwPct = None

    if leverage == 'standard':
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 0
    elif leverage == 'robust':
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
    
    Mtn, Mwn, Mbn, RCt, RCw, NCt, NCw, RowClust, ColClust, BlockClust, AddMessage, ErrMessage, cancel_pressed = \
        BuildClusters(Mt, Mw, Mb, MtPct, MwPct, NBlocks, BlkSize, NMFCalculateLeverage, NMFUseRobustLeverage, NMFAlgo,
                      NMFRobustClusterByStability, CellPlotOrderedClusters, AddMessage, myStatusBox)
    for message in AddMessage:
        print(message)

    myStatusBox.close()
    if 'Q' in estimator:
        estimator.update([('WL', Mtn), ('HL', Mwn), ('WR', RCt), ('HR', RCw), ('WN', NCt), ('HN', NCw),
                          ('WC', RowClust), ('HC', ColClust), ('QL', Mbn), ('QC', BlockClust)])
    else:
        estimator.update([('WL', Mtn), ('HL', Mwn), ('WR', RCt), ('HR', RCw), ('WN', NCt), ('HN', NCw),
                          ('WC', RowClust), ('HC', ColClust), ('QL', None), ('QC', None)])
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
    Mt = estimator['W']
    RCt = estimator['WR']
    NCt = estimator['WN']
    RowGroups = y
    uniques, index = np.unique([row for row in RowGroups], return_index=True)
    ListGroups = RowGroups[index]
    nbGroups = ListGroups.shape[0]
    Ngroup = np.zeros(nbGroups)
    for group in range(0, nbGroups):
        Ngroup[group] = np.where(RowGroups == ListGroups[group])[0].shape[0]

    Nrun = n_permutations
    myStatusBox = StatusBoxTqdm(verbose=verbose)
    ClusterSize, Pglob, prun, ClusterProb, ClusterGroup, ClusterNgroup, cancel_pressed = \
        GlobalSign(Nrun, nbGroups, Mt, RCt, NCt, RowGroups, ListGroups, Ngroup, myStatusBox)

    estimator.update(
        [('score', prun), ('pvalue', Pglob), ('CS', ClusterSize), ('CP', ClusterProb), ('CG', ClusterGroup),
         ('CN', ClusterNgroup)])
    return estimator

def non_negative_tensor_factorization(X, n_blocks, W=None, H=None, Q=None, n_components=None,
                                      update_W=True,
                                      update_H=True,
                                      update_Q=True,
                                      fast_hals=True, n_iter_hals=2, n_shift=0,
                                      regularization=None, sparsity=0,
                                      unimodal=False, smooth=False,
                                      apply_left=False, apply_right=False, apply_block=False,
                                      n_bootstrap=None,
                                      tol=1e-6,
                                      max_iter=150,
                                      leverage='standard',
                                      random_state=None,
                                      init_type=0,
                                      verbose=0):
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

    fast_hals : boolean, default: True
        Use fast implementation of HALS

    n_iter_hals : integer, default: 2
        Number of HALS iterations prior to fast HALS
    
    n_shift : integer, default: 0
        max shifting in convolutional NTF

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
    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components

    NBlocks = n_blocks
    p_block = int(p / NBlocks)
    tolerance = tol
    precision = EPSILON
    LogIter = verbose
    if regularization is None:
        NMFSparseLevel = 0
    else:
        if regularization == 'components':
            NMFSparseLevel = sparsity
        elif regularization == 'transformation':
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
        Mt0, Mw0, Mb0, AddMessage, ErrMessage, cancel_pressed = NTFInit(M, np.array([]), np.array([]), np.array([]), nc,
                                                                        tolerance, precision, LogIter, NTFUnimodal,
                                                                        NTFLeftComponents, NTFRightComponents,
                                                                        NTFBlockComponents, NBlocks, init_type, myStatusBox)
    else:
        if W is None:
            Mt0 = np.ones((n, nc))
        else:
            Mt0 = np.copy(W)
            
        if H is None:
            Mw0= np.ones((p_block, nc))
        else:
            Mw0 = np.copy(H)
        
        if Q is None:
            Mb0 = np.ones((NBlocks, nc))
        else:
            Mb0 = np.copy(Q)

        Mfit = np.zeros((n, p))
        for k in range(0, nc):
            for iBlock in range(0, NBlocks):
                Mfit[:, iBlock*p_block:(iBlock+1)*p_block] += Mb0[iBlock, k] * \
                    np.reshape(Mt0[:, k], (n, 1)) @ np.reshape(Mw0[:, k], (1, p_block))

        ScaleRatio = (np.linalg.norm(Mfit) / np.linalg.norm(M))**(1/3)
        for k in range(0, nc):
            Mt0[:, k] /= ScaleRatio
            Mw0[:, k] /= ScaleRatio
            Mb0[:, k] /= ScaleRatio

        Mfit = np.zeros((n, p))
        for k in range(0, nc):
            for iBlock in range(0, NBlocks):
                Mfit[:, iBlock*p_block:(iBlock+1)*p_block] += Mb0[iBlock, k] * \
                    np.reshape(Mt0[:, k], (n, 1)) @ np.reshape(Mw0[:, k], (1, p_block))
        
    NTFFastHALS = fast_hals
    NTFNIterations = n_iter_hals
    MaxIterations = max_iter
    NTFNConv = n_shift
    if n_bootstrap is None:
        NMFRobustNRuns = 0
    else:
        NMFRobustNRuns = n_bootstrap

    if NMFRobustNRuns <= 1:
        NMFAlgo = 5
    else:
        NMFAlgo = 6

    if leverage == 'standard':
        NMFCalculateLeverage = 1
        NMFUseRobustLeverage = 0
    elif leverage == 'robust':
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

    Mt_conv, Mt, Mw, Mb, MtPct, MwPct, diff, AddMessage, ErrMessage, cancel_pressed = rNTFSolve(
        M, np.array([]), Mt0, Mw0, Mb0, nc, tolerance, precision, LogIter, MaxIterations, NMFFixUserLHE, NMFFixUserRHE,
        NMFFixUserBHE, NMFAlgo, NMFRobustNRuns,
        NMFCalculateLeverage, NMFUseRobustLeverage, NTFFastHALS, NTFNIterations, NMFSparseLevel, NTFUnimodal, NTFSmooth,
        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, np.array([]), myStatusBox)

    volume = NMFDet(Mt, Mw, 1)

    for message in AddMessage:
        print(message)

    myStatusBox.close()

    estimator = {}
    if NMFRobustNRuns <= 1:
        estimator.update([('W', Mt), ('H', Mw), ('Q', Mb), ('volume', volume), ('diff', diff)])
    else:
        estimator.update([('W', Mt), ('H', Mw), ('Q', Mb), ('volume', volume), ('WB', MtPct), ('HB', MwPct), ('diff', diff)])

    return estimator
