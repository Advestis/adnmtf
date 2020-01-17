"""Non-negative matrix and tensor factorization core functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20

import math
import numpy as np
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from scipy.stats import hypergeom
from scipy.optimize import nnls

from nmtf_utils import *

def NMFProjGrad(V, Vmis, W, Hinit, NMFAlgo, lambdax, tol, MaxIterations, NMFPriors):
    """Projected gradient
    Code and notations adapted from Matlab code, Chih-Jen Lin
    Input:
        V: Input matrix
        Vmis: Define missing values (0 = missing cell, 1 = real cell)
        W: Left factoring vectors (fixed)
        Hinit: Right factoring vectors (initial values)
        NMFAlgo: =1,3: Divergence; =2,4: Least squares;
        lambdax: Sparseness parameter
            =-1: no penalty
            < 0: Target percent zeroed rows in H
            > 0: Current penalty
        tol: Tolerance
        MaxIterations: max number of iterations to achieve norm(projected gradient) < tol
        NMFPriors: Elements in H that should be updated (others remain 0)
    Output:
        H: Estimated right factoring vectors
        tol: Current level of the tolerance
        lambdax: Current level of the penalty
    
    Reference
    ---------

    C.J. Lin (2007) Projected Gradient Methods for Non-negative Matrix Factorization
    Neural Comput. 2007 Oct;19(10):2756-79.

    """
    H = Hinit
    try:
        n_NMFPriors, nc = NMFPriors.shape
    except:
        n_NMFPriors = 0

    n_Vmis = Vmis.shape[0]
    n, p = np.shape(V)
    n, nc = np.shape(W)
    alpha = 1

    if (NMFAlgo == 2) or (NMFAlgo == 4):
        beta = .1
        if n_Vmis > 0:
            WtV = W.T @ (V * Vmis)
        else:
            WtV = W.T @ V
            WtW = W.T @ W
    else:
        beta = .1
        if n_Vmis > 0:
            WtWH = W.T @ Vmis
        else:
            WtWH = W.T @ np.ones((n, p))

    if (lambdax < 0) & (lambdax != -1):
        H0 = H

    restart = True
    while restart:
        for iIter in range(0, MaxIterations):
            addPenalty = 0
            if lambdax != -1:
                addPenalty = 1

            if (NMFAlgo == 2) or (NMFAlgo == 4):
                if n_Vmis > 0:
                    WtWH = W.T @ ((W @ H) * Vmis)
                else:
                    WtWH = WtW @ H
            else:
                if n_Vmis > 0:
                    WtV = W.T @ ((V * Vmis) / (W @ H))
                else:
                    WtV = W.T @ (V / (W @ H))

            if lambdax > 0:
                grad = WtWH - WtV + lambdax
            else:
                grad = WtWH - WtV

            projgrad = np.linalg.norm(grad[(grad < 0) | (H > 0)])

            if projgrad >= tol:
                # search step size
                for inner_iter in range(1, 21):
                    Hn = H - alpha * grad
                    Hn[np.where(Hn < 0)] = 0
                    if n_NMFPriors > 0:
                        Hn = Hn * NMFPriors

                    d = Hn - H
                    gradd = np.sum(grad * d)
                    if (NMFAlgo == 2) or (NMFAlgo == 4):
                        if n_Vmis > 0:
                            dQd = np.sum((W.T @ ((W @ d) * Vmis)) * d)
                        else:
                            dQd = np.sum((WtW @ d) * d)
                    else:
                        if n_Vmis > 0:
                            dQd = np.sum((W.T @ ((W @ d) * (Vmis / (W @ H)))) * d)
                        else:
                            dQd = np.sum((W.T @ ((W @ d) / (W @ H))) * d)

                    suff_decr = (0.99 * gradd + 0.5 * dQd < 0)
                    if inner_iter == 1:
                        decr_alpha = not suff_decr
                        Hp = H

                    if decr_alpha:
                        if suff_decr:
                            H = Hn
                            break
                        else:
                            alpha = alpha * beta
                    else:
                        if (suff_decr == False) | (np.where(Hp != Hn)[0].size == 0):
                            H = Hp
                            break
                        else:
                            alpha = alpha / beta
                            Hp = Hn
                # End for (inner_iter

                if (lambdax < 0) & addPenalty:
                    # Initialize penalty
                    lambdax = percentile_exc(H[np.where(H > 0)], -lambdax * 100)
                    H = H0
                    alpha = 1
            else:  # projgrad < tol
                if (iIter == 0) & (projgrad > 0):
                    tol /= 10
                else:
                    restart = False

                break
            #       End if projgrad

            if iIter == MaxIterations-1:
                restart = False
        #   End For iIter

    H = H.T
    return [H, tol, lambdax]

def NMFProjGradKernel(Kernel, V, Vmis, W, Hinit, NMFAlgo, tol, MaxIterations, NMFPriors):
    """Projected gradient, kernel version
    Code and notations adapted from Matlab code, Chih-Jen Lin
    Input:
        Kernel: Kernel used
        V: Input matrix
        Vmis: Define missing values (0 = missing cell, 1 = real cell)
        W: Left factoring vectors (fixed)
        Hinit: Right factoring vectors (initial values)
        NMFAlgo: =1,3: Divergence; =2,4: Least squares;
        tol: Tolerance
        MaxIterations: max number of iterations to achieve norm(projected gradient) < tol
        NMFPriors: Elements in H that should be updated (others remain 0)
    Output:
        H: Estimated right factoring vectors
        tol: Current level of the tolerance
    
    Reference
    ---------

    C.J. Lin (2007) Projected Gradient Methods for Non-negative Matrix Factorization
        Neural Comput. 2007 Oct;19(10):2756-79.

    """
    H = Hinit.T
    try:
        n_NMFPriors, nc = NMFPriors.shape
    except:
        n_NMFPriors = 0

    n_Vmis = Vmis.shape[0]
    n, p = np.shape(V)
    p, nc = np.shape(W)
    alpha = 1
    VW = V @ W

    if (NMFAlgo == 2) or (NMFAlgo == 4):
        beta = .1
        if n_Vmis > 0:
            WtV = VW.T @ (V * Vmis)
        else:
            WtV = W.T @ Kernel
            WtW = W.T @ Kernel @ W
    else:
        beta = .1
        MaxIterations = round(MaxIterations/10)
        if n_Vmis > 0:
            WtWH = VW.T @ Vmis
        else:
            WtWH = VW.T @ np.ones((n, p))

    restart = True
    while restart:
        for iIter in range(0, MaxIterations):
            if (NMFAlgo == 2) or (NMFAlgo == 4):
                if n_Vmis > 0:
                    WtWH = VW.T @ ((VW @ H) * Vmis)
                else:
                    WtWH = WtW @ H
            else:
                if n_Vmis > 0:
                    WtV = VW.T @ ((V * Vmis) / (VW @ H))
                else:
                    WtV = VW.T @ (V / (VW @ H))

            grad = WtWH - WtV
            projgrad = np.linalg.norm(grad[(grad < 0) | (H > 0)])
            if projgrad >= tol:
                # search step size
                for inner_iter in range(1, 21):
                    Hn = H - alpha * grad
                    Hn[np.where(Hn < 0)] = 0
                    if n_NMFPriors > 0:
                        Hn = Hn * NMFPriors

                    d = Hn - H
                    gradd = np.sum(grad * d)
                    if (NMFAlgo == 2) or (NMFAlgo == 4):
                        if n_Vmis > 0:
                            dQd = np.sum((VW.T @ ((VW @ d) * Vmis)) * d)
                        else:
                            dQd = np.sum((WtW @ d) * d)
                    else:
                        if n_Vmis > 0:
                            dQd = np.sum((VW.T @ ((VW @ d) * (Vmis / (VW @ H)))) * d)
                        else:
                            dQd = np.sum((VW.T @ ((VW @ d) / (VW @ H))) * d)

                    suff_decr = (0.99 * gradd + 0.5 * dQd < 0)
                    if inner_iter == 1:
                        decr_alpha = not suff_decr
                        Hp = H

                    if decr_alpha:
                        if suff_decr:
                            H = Hn
                            break
                        else:
                            alpha = alpha * beta
                    else:
                        if (suff_decr == False) | (np.where(Hp != Hn)[0].size == 0):
                            H = Hp
                            break
                        else:
                            alpha = alpha / beta
                            Hp = Hn
                # End for (inner_iter
            else:  # projgrad < tol
                if iIter == 0:
                    tol /= 10
                else:
                    restart = False

                break
            #       End if projgrad

            if iIter == MaxIterations-1:
                restart = False
        #   End For iIter

    H = H.T
    return [H, tol]

def NMFApplyKernel(M, NMFKernel, Mt, Mw):
    """Calculate kernel (used with convex NMF)
    Input:
        M: Input matrix
        NMFKernel: Type of kernel
            =-1: linear
            = 2: quadratic
            = 3: radiant
        Mt: Left factoring matrix
        Mw: Right factoring matrix
    Output:
        Kernel
    """

    n, p = M.shape
    try:
        p, nc = Mw.shape
    except:
        nc = 0

    if NMFKernel == 1:
        Kernel = M.T @ M
    elif NMFKernel == 2:
        Kernel = (np.identity(p) + M.T @ M) ** 2
    elif NMFKernel == 3:
        Kernel = np.identity(p)
        # Estimate Sigma2
        Sigma2 = 0

        for k1 in range(1, nc):
            for k2 in range(0, k1):
                Sigma2 = max(Sigma2, np.linalg.norm(Mt[:, k1] - Mt[:, k2]) ** 2)

        Sigma2 /= nc
        for j1 in range(1, p):
            for j2 in range(0, j1):
                Kernel[j1, j2] = math.exp(-np.linalg.norm(M[:, j1] - M[:, j2]) ** 2 / Sigma2)
                Kernel[j2, j1] = Kernel[j1, j2]

    return Kernel

def NMFReweigh(M, Mt, NMFPriors, AddMessage):
    """Overload skewed variables (used with deconvolution only)
    Input:
         M: Input matrix
         Mt: Left hand matrix
         NMFPriors: priors on right hand matrix
    Output:
         NMFPriors: updated priors

    Note: This code is still experimental

    """
    ErrMessage = ""
    n, p = M.shape
    n_NMFPriors, nc = NMFPriors.shape
    NMFPriors[NMFPriors > 0] = 1
    ID = np.where(np.sum(NMFPriors, axis=1) > 1)
    n_ID = ID[0].shape[0]
    if n_ID == p:
        ErrMessage = 'Error! All priors are ambiguous.\nYou may uncheck the option in tab irMF+.'
        return [NMFPriors, AddMessage, ErrMessage]

    NMFPriors[ID, :] = 0
    Mweight = np.zeros((p, nc))
    for k in range(0, nc):
        ID = np.where(NMFPriors[:, k] > 0)
        pk = ID[0].shape[0]
        if pk == 0:
            ErrMessage = 'Error! Null column in NMF priors (' + str(k+1) + ', pre outlier filtering)'
            return [NMFPriors, AddMessage, ErrMessage]

        Mc = np.zeros((n, p))

        # Exclude variables with outliers
        NInterQuart = 1.5
        for j in range(0, pk):
            Quart75 = percentile_exc(M[:, ID[0][j]], 75)
            Quart25 = percentile_exc(M[:, ID[0][j]], 25)
            InterQuart = Quart75 - Quart25
            MaxBound = Quart75 + NInterQuart * InterQuart
            MinBound = Quart25 - NInterQuart * InterQuart
            if np.where((M[:, ID[0][j]] < MinBound) | (M[:, ID[0][j]] > MaxBound))[0].shape[0] == 1:
                NMFPriors[ID[0][j], k] = 0

        ID = np.where(NMFPriors[:, k] > 0)
        pk = ID[0].shape[0]
        if pk == 0:
            ErrMessage = 'Error! Null column in NMF priors (' + str(k+1) + ', post outlier filtering)'
            return [NMFPriors, AddMessage, ErrMessage]

        # Characterize clusters by skewness direction
        Mtc = Mt[:, k] - np.mean(Mt[:, k])
        std = math.sqrt(np.mean(Mtc ** 2))
        skewness = np.mean((Mtc / std) ** 3) * math.sqrt(n * (n - 1)) / (n - 2)

        # Scale columns and initialized weights
        for j in range(0, pk):
            M[:, ID[0][j]] /= np.sum(M[:, ID[0][j]])
            Mc[:, ID[0][j]] = M[:, ID[0][j]] - np.mean(M[:, ID[0][j]])
            std = math.sqrt(np.mean(Mc[:, ID[0][j]] ** 2))
            Mweight[ID[0][j], k] = np.mean((Mc[:, ID[0][j]] / std) ** 3) * math.sqrt(n * (n - 1)) / (n - 2)

        if skewness < 0:
            # Negative skewness => Component identifiable through small proportions
            Mweight[Mweight[:, k] > 0, k] = 0
            Mweight = -Mweight
            IDneg = np.where(Mweight[:, k] > 0)
            Nneg = IDneg[0].shape[0]
            if Nneg == 0:
                ErrMessage = 'Error! No marker variable found in component ' + str(k+1)
                return [NMFPriors, AddMessage, ErrMessage]

            AddMessage.insert(len(AddMessage),
                              'Component ' + str(k+1) + ': compositions are negatively skewed (' + str(
                                  Nneg) + ' active variables)')
        else:
            # Positive skewness => Component identifiable through large proportions
            Mweight[Mweight[:, k] < 0, k] = 0
            IDpos = np.where(Mweight[:, k] > 0)
            Npos = IDpos[0].shape[0]
            if Npos == 0:
                ErrMessage = 'Error! No marker variable found in component ' + str(k+1)
                return [NMFPriors, AddMessage, ErrMessage]

            AddMessage.insert(len(AddMessage),
                              'Component ' + str(k+1) + ': compositions are positively skewed (' + str(
                                  Npos) + ' active variables)')

        # Logistic transform of non-zero weights
        ID2 = np.where(Mweight[:, k] > 0)
        n_ID2 = ID2[0].shape[0]
        if n_ID2 > 1:
            mu = np.mean(Mweight[ID2[0], k])
            std = np.std(Mweight[ID2[0], k])
            Mweight[ID2[0], k] = (Mweight[ID2[0], k] - mu) / std
            Mweight[ID2[0], k] = np.ones(n_ID2) - np.ones(n_ID2) / (np.ones(n_ID2) + np.exp(
                2 * (Mweight[ID2[0], k] - percentile_exc(Mweight[ID2[0], k], 90))))
        else:
            Mweight[ID2[0], k] = 1

        # ReWeigh columns
        M[:, ID[0]] = M[:, ID[0]] * Mweight[ID[0], k].T

        # Update NMF priors (cancel columns with 0 weight & replace non zero values by 1)
        NMFPriors[ID[0], k] = NMFPriors[ID[0], k] * Mweight[ID[0], k]
        ID = np.where(NMFPriors[:, k] > 0)
        if ID[0].shape[0] > 0:
            NMFPriors[ID[0], k] = 1
            # Scale parts
            M[:, ID[0]] /= np.linalg.norm(M[:, ID[0]])
        else:
            ErrMessage = 'Error! Null column in NMF priors (' + str(k+1) + ', post cancelling 0-weight columns)'
            return [NMFPriors, AddMessage, ErrMessage]

    return [NMFPriors, AddMessage, ErrMessage]

def NMFSolve(M, Mmis, Mt0, Mw0, nc, tolerance, precision, LogIter, Status0, MaxIterations, NMFAlgo,
             NMFFixUserLHE, NMFFixUserRHE, NMFMaxInterm, NMFMaxIterProj, NMFSparseLevel,
             NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns, NMFPriors, flagNonconvex, AddMessage,
             myStatusBox):
    """
    Estimate left and right hand matrices
    Input:
         M: Input matrix
         Mmis: Define missing values (0 = missing cell, 1 = real cell)
         Mt0: Initial left hand matrix
         Mw0: Initial right hand matrix
         nc: NMF rank
         tolerance: Convergence threshold
         precision: Replace 0-value in multiplication rules
         LogIter: Log results through iterations
         Status0: Initial displayed status to be updated during iterations
         MaxIterations: Max iterations
         NMFAlgo: =1,3: Divergence; =2,4: Least squares;
         NMFFixUserLHE: = 1 => fixed left hand matrix columns
         NMFFixUserRHE: = 1 => fixed  right hand matrix columns
         NMFMaxInterm: Max iterations for warmup multiplication rules
         NMFMaxIterProj: Max iterations for projected gradient
         NMFSparseLevel: Requested sparsity in terms of relative number of rows with 0 values in right hand matrix
         NMFFindParts: Enforce convexity on left hand matrix
         NMFFindCentroids: Enforce convexity on right hand matrix
         NMFKernel: Type of kernel used; 1: linear; 2: quadratic; 3: radial
         NMFReweighColumns: Reweigh columns in 2nd step of parts-based NMF
         NMFPriors: Priors on right hand matrix
         flagNonconvex: Non-convexity flag on left hand matrix
    Output:
         Mt: Left hand matrix
         Mw: Right hand matrix
         diff: objective cost
         Mh: Convexity matrix
         NMFPriors: Updated priors on right hand matrix
         flagNonconvex: Updated non-convexity flag on left hand matrix
    
    Reference
    ---------

    C. H.Q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
    IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1

    """
    ErrMessage = ''
    cancel_pressed = 0

    n, p = M.shape
    n_Mmis = Mmis.shape[0]
    try:
        n_NMFPriors, nc = NMFPriors.shape
    except:
        n_NMFPriors = 0

    nc = int(nc)
    nxp = int(n * p)
    Mh = np.array([])
    Mt = np.copy(Mt0)
    Mw = np.copy(Mw0)
    diff = 1.e+99

    # Add weights
    if n_NMFPriors > 0:
        if NMFReweighColumns > 0:
            # A local copy of M will be updated
            M = np.copy(M)
            NMFPriors, AddMessage, ErrMessage = NMFReweigh(M, Mt, NMFPriors, AddMessage)
            if ErrMessage != "":
                return [Mt, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed]
        else:
            NMFPriors[np.where(NMFPriors > 0)] = 1

    if (NMFFindParts > 0) & (NMFFixUserLHE > 0):
        NMFFindParts = 0

    if (NMFFindCentroids > 0) & (NMFFixUserRHE > 0):
        NMFFindCentroids = 0
        NMFKernel = 1

    if (NMFFindCentroids > 0) & (NMFKernel > 1):
        if n_Mmis > 0:
            NMFKernel = 1
            AddMessage.insert(len(AddMessage), 'Warning: Non linear kernel canceled due to missing values.')

        if (NMFAlgo == 1) or (NMFAlgo == 3) :
            NMFKernel = 1
            AddMessage.insert(len(AddMessage), 'Warning: Non linear kernel canceled due to divergence minimization.')

    if n_NMFPriors > 0:
        MwUser = NMFPriors
        for k in range(0, nc):
            if (NMFAlgo == 2) | (NMFAlgo == 4):
                Mw[:, k] = MwUser[:, k] / np.linalg.norm(MwUser[:, k])
            else:
                Mw[:, k] = MwUser[:, k] / np.sum(MwUser[:, k])

    MultOrPgrad = 1  # Start with Lee-Seung mult rules
    MaxIterations += NMFMaxInterm  # NMFMaxInterm Li-Seung iterations initialize projected gradient

    StepIter = math.ceil(MaxIterations / 10)
    pbar_step = 100 * StepIter / MaxIterations

    iIter = 0
    cont = 1

    # Initialize penalty
    # lambda = -1: no penalty
    # lambda = -abs(NMFSparselevel) : initialisation by NMFSparselevel (in negative value)
    if NMFSparseLevel > 0:
        lambdaw = -NMFSparseLevel
        lambdat = -1
    elif NMFSparseLevel < 0:
        lambdat = NMFSparseLevel
        lambdaw = -1
    else:
        lambdaw = -1
        lambdat = -1

    PercentZeros = 0
    iterSparse = 0
    NMFConvex = 0
    NLKernelApplied = 0

    myStatusBox.init_bar(delay=1)
    
    # Start loop
    while (cont == 1) & (iIter < MaxIterations):
        # Update RHE
        if NMFFixUserRHE == 0:
            if MultOrPgrad == 1:
                if (NMFAlgo == 2) or (NMFAlgo == 4):
                    if n_Mmis > 0:
                        Mw = \
                            Mw * ((Mt.T @ (M * Mmis)) / (
                                    Mt.T @ ((Mt @ Mw.T) * Mmis) + precision)).T
                    else:
                        Mw = \
                             Mw * ((Mt.T @ M) / (
                                (Mt.T @ Mt) @ Mw.T + precision)).T
                else:
                    if n_Mmis > 0:
                        Mw = Mw * (((M * Mmis) / ((Mt @ Mw.T) * Mmis + precision)).T @ Mt)
                    else:
                        Mw = Mw * ((M / (Mt @ Mw.T + precision)).T @ Mt)

                if n_NMFPriors > 0:
                    Mw = Mw * NMFPriors
            else:
                # Projected gradient
                if (NMFConvex > 0) & (NMFFindParts > 0):
                    Mw, tolMw = NMFProjGradKernel(Kernel, M, Mmis, Mh, Mw, NMFAlgo, tolMw, NMFMaxIterProj, NMFPriors.T)
                elif (NMFConvex > 0) & (NMFFindCentroids > 0):
                    Mh, tolMh, dummy = NMFProjGrad(In, np.array([]), Mt, Mh.T, NMFAlgo, -1, tolMh, NMFMaxIterProj, np.array([]))
                else:
                    Mw, tolMw, lambdaw = NMFProjGrad(M, Mmis, Mt, Mw.T, NMFAlgo, lambdaw, tolMw, \
                                                                NMFMaxIterProj, NMFPriors.T)

            if (NMFConvex > 0) & (NMFFindParts > 0):
                for k in range(0, nc):
                    ScaleMw = np.linalg.norm(Mw[:, k])
                    Mw[:, k] = Mw[:, k] / ScaleMw
                    Mt[:, k] = Mt[:, k] * ScaleMw

        # Update LHE
        if NMFFixUserLHE == 0:
            if MultOrPgrad == 1:
                if (NMFAlgo == 2) | (NMFAlgo == 4):
                    if n_Mmis > 0:
                        Mt = \
                            Mt * ((M * Mmis) @ Mw / (
                                ((Mt @ Mw.T) * Mmis) @ Mw + precision))
                    else:
                        Mt = \
                            Mt * (M @ Mw / (Mt @ (Mw.T @ Mw) + precision))
                else:
                    Mt = Mt * ((M.T / (Mw @ Mt.T + precision)).T @ Mw)
            else:
                # Projected gradient
                if (NMFConvex > 0) & (NMFFindParts > 0):
                    Mh, tolMh, dummy = NMFProjGrad(Ip, np.array([]), Mw, Mh.T, NMFAlgo, -1, tolMh, NMFMaxIterProj, np.array([]))
                elif (NMFConvex > 0) & (NMFFindCentroids > 0):
                    Mt, tolMt = NMFProjGradKernel(Kernel, M.T, Mmis.T, Mh, Mt, NMFAlgo, tolMt, NMFMaxIterProj, np.array([]))
                else:
                    Mt, tolMt, lambdat = NMFProjGrad(M.T, Mmis.T, Mw, Mt.T, NMFAlgo,
                                                            lambdat, tolMt, NMFMaxIterProj, np.array([]))

            # Scaling
            if ((NMFConvex == 0) | (NMFFindCentroids > 0)) & (NMFFixUserLHE == 0) &  (NMFFixUserRHE == 0):
                for k in range(0, nc):
                    if (NMFAlgo == 2) | (NMFAlgo == 4):
                        ScaleMt = np.linalg.norm(Mt[:, k])
                    else:
                        ScaleMt = np.sum(Mt[:, k])

                    if ScaleMt > 0:
                        Mt[:, k] = Mt[:, k] / ScaleMt
                        if MultOrPgrad == 2:
                            Mw[:, k] = Mw[:, k] * ScaleMt

        # Switch to projected gradient
        if iIter == NMFMaxInterm:
            MultOrPgrad = 2
            StepIter = 1
            pbar_step = 100 / MaxIterations
            gradMt = Mt @ (Mw.T @ Mw) - M @ Mw
            gradMw = ((Mt.T @ Mt) @ Mw.T - Mt.T @ M).T
            initgrad = np.linalg.norm(np.concatenate((gradMt, gradMw), axis=0))
            tolMt = 1e-3 * initgrad
            tolMw = tolMt

        if iIter % StepIter == 0:
            if (NMFConvex > 0) & (NMFFindParts > 0):
                MhtKernel = Mh.T @ Kernel
                diff = (TraceKernel + np.trace(-2 * Mw @ MhtKernel + Mw @ (MhtKernel @ Mh) @ Mw.T)) / nxp
            elif (NMFConvex > 0) & (NMFFindCentroids > 0):
                MhtKernel = Mh.T @ Kernel
                diff = (TraceKernel + np.trace(-2 * Mt @ MhtKernel + Mt @ (MhtKernel @ Mh) @ Mt.T)) / nxp
            else:
                if (NMFAlgo == 2) | (NMFAlgo == 4):
                    if n_Mmis > 0:
                        Mdiff = (Mt @ Mw.T - M) * Mmis
                    else:
                        Mdiff = Mt @ Mw.T - M

                    diff = np.linalg.norm(Mdiff) / nxp
                else:
                    MF0 = Mt @ Mw.T
                    if n_Mmis > 0:
                        Mdiff = (M * np.log(M / (MF0 + precision)) + MF0 - M) * Mmis
                    else:
                        Mdiff = M * np.log(M / (MF0 + precision)) + MF0 - M

                    diff = np.sum(Mdiff) / nxp

            Status = Status0 + 'Iteration: %s' % int(iIter)

            if NMFSparseLevel != 0:
                if NMFSparseLevel > 0:
                    lambdax = lambdaw
                else:
                    lambdax = lambdat

                Status = Status + '; Achieved sparsity: ' + str(round(PercentZeros, 2)) + '; Penalty: ' + str(
                    round(lambdax, 2))
                if LogIter == 1:
                    myStatusBox.myPrint(Status) 
            elif (NMFConvex > 0) & (NMFFindParts > 0):
                Status = Status + ' - Find parts'
            elif (NMFConvex > 0) & (NMFFindCentroids > 0) & (NLKernelApplied == 0):
                Status = Status + ' - Find centroids'
            elif NLKernelApplied == 1:
                Status = Status + ' - Apply non linear kernel'

            myStatusBox.update_status(delay=1, status=Status)
            myStatusBox.update_bar(delay=1, step=pbar_step)
            if myStatusBox.cancel_pressed:
                cancel_pressed = 1
                return [Mt, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed]

            if LogIter == 1:
                if (NMFAlgo == 2) | (NMFAlgo == 4):
                    myStatusBox.myPrint(Status0 + " Iter: " + str(iIter) + " MSR: " + str(diff))
                else:
                    myStatusBox.myPrint(Status0 + " Iter: " + str(iIter) + " DIV: " + str(diff))

            if iIter > NMFMaxInterm:
                if (diff0 - diff) / diff0 < tolerance:
                    cont = 0

            diff0 = diff

        iIter += 1

        if (cont == 0) | (iIter == MaxIterations):
            if ((NMFFindParts > 0) | (NMFFindCentroids > 0)) & (NMFConvex == 0):
                # Initialize convexity
                NMFConvex = 1
                diff0 = 1.e+99
                iIter = NMFMaxInterm + 1
                myStatusBox.init_bar(delay=1)
                cont = 1
                if NMFFindParts > 0:
                    Ip = np.identity(p)
                    if (NMFAlgo == 2) or (NMFAlgo == 4):
                        if n_Mmis > 0:
                            Kernel = NMFApplyKernel(Mmis * M, 1, np.array([]), np.array([]))
                        else:
                            Kernel = NMFApplyKernel(M, 1, np.array([]), np.array([]))
                    else:
                        if n_Mmis > 0:
                            Kernel = NMFApplyKernel(Mmis * (M / (Mt @ Mw.T)), 1, np.array([]), np.array([]))
                        else:
                            Kernel = NMFApplyKernel(M / (Mt @ Mw.T), 1, np.array([]), np.array([]))

                    TraceKernel = np.trace(Kernel)
                    try:
                        Mh = Mw @ np.linalg.inv(Mw.T @ Mw)
                    except:
                        Mh = Mw @ np.linalg.pinv(Mw.T @ Mw)

                    Mh[np.where(Mh < 0)] = 0
                    for k in range(0, nc):
                        ScaleMw = np.linalg.norm(Mw[:, k])
                        Mw[:, k] = Mw[:, k] / ScaleMw
                        Mh[:, k] = Mh[:, k] * ScaleMw

                    gradMh = Mh @ (Mw.T @ Mw) - Mw
                    gradMw = ((Mh.T @ Mh) @ Mw.T - Mh.T).T
                    initgrad = np.linalg.norm(np.concatenate((gradMh, gradMw), axis=0))
                    tolMh = 1.e-3 * initgrad
                    tolMw = tolMt
                elif NMFFindCentroids > 0:
                    In = np.identity(n)
                    if (NMFAlgo == 2) or (NMFAlgo == 4):
                        if n_Mmis > 0:
                            Kernel = NMFApplyKernel(Mmis.T * M.T, 1, np.array([]), np.array([]))
                        else:
                            Kernel = NMFApplyKernel(M.T, 1, np.array([]), np.array([]))
                    else:
                        if n_Mmis > 0:
                            Kernel = NMFApplyKernel(Mmis.T * (M.T / (Mt @ Mw.T).T), 1, np.array([]), np.array([]))
                        else:
                            Kernel = NMFApplyKernel(M.T / (Mt @ Mw.T).T, 1, np.array([]), np.array([]))

                    TraceKernel = np.trace(Kernel)
                    try:
                        Mh = Mt @ np.linalg.inv(Mt.T @ Mt)
                    except:
                        Mh = Mt @ np.linalg.pinv(Mt.T @ Mt)

                    Mh[np.where(Mh < 0)] = 0
                    for k in range(0, nc):
                        ScaleMt = np.linalg.norm(Mt[:, k])
                        Mt[:, k] = Mt[:, k] / ScaleMt
                        Mh[:, k] = Mh[:, k] * ScaleMt

                    gradMt = Mt @ (Mh.T @ Mh) - Mh
                    gradMh = ((Mt.T @ Mt) @ Mh.T - Mt.T).T
                    initgrad = np.linalg.norm(np.concatenate((gradMt, gradMh), axis=0))
                    tolMh = 1.e-3 * initgrad
                    tolMt = tolMh

            elif (NMFConvex > 0) & (NMFKernel > 1) & (NLKernelApplied == 0):
                NLKernelApplied = 1
                diff0 = 1.e+99
                iIter = NMFMaxInterm + 1
                myStatusBox.init_bar(delay=1)
                cont = 1
                # Calculate centroids
                for k in range(0, nc):
                    Mh[:, k] = Mh[:, k] / np.sum(Mh[:, k])

                Mw = (Mh.T @ M).T
                if (NMFAlgo == 2) or (NMFAlgo == 4):
                    if n_Mmis > 0:
                        Kernel = NMFApplyKernel(Mmis.T * M.T, NMFKernel, Mw, Mt)
                    else:
                        Kernel = NMFApplyKernel(M.T, NMFKernel, Mw, Mt)
                else:
                    if n_Mmis > 0:
                        Kernel = NMFApplyKernel(Mmis.T * (M.T / (Mt @ Mw.T).T), NMFKernel, Mw, Mt)
                    else:
                        Kernel = NMFApplyKernel(M.T / (Mt @ Mw.T).T, NMFKernel, Mw, Mt)

                TraceKernel = np.trace(Kernel)
                try:
                    Mh = Mt @ np.linalg.inv(Mt.T @ Mt)
                except:
                    Mh = Mt @ np.linalg.pinv(Mt.T @ Mt)

                Mh[np.where(Mh < 0)] = 0
                for k in range(0, nc):
                    ScaleMt = np.linalg.norm(Mt[:, k])
                    Mt[:, k] = Mt[:, k] / ScaleMt
                    Mh[:, k] = Mh[:, k] * ScaleMt

                gradMt = Mt @ (Mh.T @ Mh) - Mh
                gradMh = ((Mt.T @ Mt) @ Mh.T - Mt.T).T
                initgrad = np.linalg.norm(np.concatenate((gradMt, gradMh), axis=0))
                tolMh = 1.e-3 * initgrad
                tolMt = tolMh

            if NMFSparseLevel > 0:
                SparseTest = np.zeros((p, 1))
                for k in range(0, nc):
                    SparseTest[np.where(Mw[:, k] > 0)] = 1

                PercentZeros0 = PercentZeros
                n_SparseTest = np.where(SparseTest == 0)[0].size
                PercentZeros = max(n_SparseTest / p, .01)
                if PercentZeros == PercentZeros0:
                    iterSparse += 1
                else:
                    iterSparse = 0

                if (PercentZeros < 0.99 * NMFSparseLevel) & (iterSparse < 50):
                    lambdaw *= min(1.01 * NMFSparseLevel / PercentZeros, 1.10)
                    iIter = NMFMaxInterm + 1
                    cont = 1

            elif NMFSparseLevel < 0:
                SparseTest = np.zeros((n, 1))
                for k in range(0, nc):
                    SparseTest[np.where(Mt[:, k] > 0)] = 1

                PercentZeros0 = PercentZeros
                n_SparseTest = np.where(SparseTest == 0)[0].size
                PercentZeros = max(n_SparseTest / n, .01)
                if PercentZeros == PercentZeros0:
                    iterSparse += 1
                else:
                    iterSparse = 0

                if (PercentZeros < 0.99 * abs(NMFSparseLevel)) & (iterSparse < 50):
                    lambdat *= min(1.01 * abs(NMFSparseLevel) / PercentZeros, 1.10)
                    iIter = NMFMaxInterm + 1
                    cont = 1

    if NMFFindParts > 0:
        # Make Mt convex
        Mt = M @ Mh
        Mt, Mw, Mh, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFGetConvexScores(Mt, Mw, Mh, flagNonconvex,
                                                                                         AddMessage)
    elif NMFFindCentroids > 0:
        # Calculate row centroids
        for k in range(0, nc):
            ScaleMh = np.sum(Mh[:, k])
            Mh[:, k] = Mh[:, k] / ScaleMh
            Mt[:, k] = Mt[:, k] * ScaleMh

        Mw = (Mh.T @ M).T

    if (NMFKernel > 1) & (NLKernelApplied == 1):
        diff /= TraceKernel / nxp

    return [Mt, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed]

def NTFStack(M, Mmis, NBlocks):
    """Unfold tensor M
        for future use with NMF
    """
    n, p = M.shape
    Mmis = Mmis.astype(np.int)
    n_Mmis = Mmis.shape[0]
    NBlocks = int(NBlocks)

    Mstacked = np.zeros((int(n * p / NBlocks), NBlocks))
    if n_Mmis > 0:
        Mmis_stacked = np.zeros((int(n * p / NBlocks), NBlocks))
    else:
        Mmis_stacked = np.array([])

    for iBlock in range(0, NBlocks):
        for j in range(0, int(p / NBlocks)):
            i1 = j * n
            i2 = i1 + n
            Mstacked[i1:i2, iBlock] = M[:, int(iBlock * p / NBlocks + j)]
            if n_Mmis > 0:
                Mmis_stacked[i1:i2, iBlock] = Mmis[:, int(iBlock * p / NBlocks + j)]

    return [Mstacked, Mmis_stacked]

def NTFSolve(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NMFSparseLevel, NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks,
             NTFNConv, NTFminDiv, myStatusBox):
    """Interface to:
            - NTFSolve_simple
            - NTFSolve_conv
    """
    if NTFNConv > 0:
        return NTFSolve_conv(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NMFSparseLevel, NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks,
             NTFNConv, NTFminDiv, myStatusBox)
    else:
        return NTFSolve_simple(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NMFSparseLevel, NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks,
             NTFminDiv, myStatusBox)

def NTFSolve_simple(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NMFSparseLevel, NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks,
             NTFminDiv, myStatusBox):
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

    # Compute Residual tensor
    Mfit = np.zeros((n, p0))
    for k in range(0, nc):
        if NBlocks > 1:
            for iBlock in range(0, NBlocks):
                Mfit[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] = Mb[iBlock, k] * np.reshape(Mt[:, k], (n, 1)) @ np.reshape(
                    Mw[:, k], (1, p))
        else:
            Mfit[:,:] = np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))

    denomt = np.zeros(n)
    denomw = np.zeros(p)
    denomBlock = np.zeros((NBlocks, nc))
    Mt2 = np.zeros(n)
    Mw2 = np.zeros(p)
    MtMw = np.zeros(nxp)
    denomCutoff = .1

    if n_Mmis > 0:
        Mres = (M - Mfit) * Mmis
    else:
        Mres = M - Mfit

    myStatusBox.init_bar(delay=1)

    # Loop
    cont = 1
    iIter = 0
    diff0 = 1.e+99
    Mpart = np.zeros((n, p0))
    alpha = NMFSparseLevel
    PercentZeros = 0
    iterSparse = 0
    
    while (cont > 0) & (iIter < MaxIterations):
        for k in range(0, nc):
            NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
                NMFFixUserLHE, denomt, Mw2, denomCutoff, alpha ,\
                NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
                denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
                denomBlock, NTFBlockComponents, C, Mfit = \
            NTFUpdate(NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
                NMFFixUserLHE, denomt, Mw2, denomCutoff, alpha ,\
                NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
                denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
                denomBlock, NTFBlockComponents, C, Mfit, NTFminDiv)
                       
        if iIter % StepIter == 0:
            # Check convergence
            if NTFminDiv:
                for k in range(0, nc):
                    if NBlocks > 1:
                        for iBlock in range(0, NBlocks):
                            Mfit[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] = Mb[iBlock, k] * \
                                np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))
                    else:
                        Mfit[:,:] = np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))
            
                diff = np.sum(M * np.log(M / (Mfit + EPSILON)) + Mfit - M) / nxp0
            else:
                diff = np.linalg.norm(Mres) / nxp0

            print(diff0, diff)

            if abs(diff0 - diff) / diff0 < tolerance:
                cont = 0
            else:
                diff0 = diff

            Status = Status0 + 'Iteration: %s' % int(iIter)

            if NMFSparseLevel != 0:
                Status = Status + '; Achieved sparsity: ' + str(round(PercentZeros, 2)) + '; alpha: ' + str(
                    round(alpha, 2))
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

        if (cont == 0) | (iIter == MaxIterations):
            if NMFSparseLevel > 0:
                SparseTest = np.zeros((p, 1))
                for k in range(0, nc):
                    SparseTest[np.where(Mw[:, k] > 0)] = 1

                PercentZeros0 = PercentZeros
                n_SparseTest = np.where(SparseTest == 0)[0].size
                PercentZeros = max(n_SparseTest / p, .01)
                if PercentZeros == PercentZeros0:
                    iterSparse += 1
                else:
                    iterSparse = 0

                if (PercentZeros < 0.99 * NMFSparseLevel) & (iterSparse < 50):
                    alpha *= min(1.01 * NMFSparseLevel / PercentZeros, 1.01)
                    if alpha < .99:
                        iIter = 1
                        cont = 1

            elif NMFSparseLevel < 0:
                SparseTest = np.zeros((n, 1))
                for k in range(0, nc):
                    SparseTest[np.where(Mt[:, k] > 0)] = 1

                PercentZeros0 = PercentZeros
                n_SparseTest = np.where(SparseTest == 0)[0].size
                PercentZeros = max(n_SparseTest / n, .01)
                if PercentZeros == PercentZeros0:
                    iterSparse += 1
                else:
                    iterSparse = 0

                print(iterSparse)
                if (PercentZeros < 0.99 * abs(NMFSparseLevel)) & (iterSparse < 50):
                    alpha *= min(1.01 * abs(NMFSparseLevel) / PercentZeros, 1.01)
                    if abs(alpha) < .99:
                        iIter = 1
                        cont = 1

    if (n_Mmis > 0) & (NMFFixUserBHE == 0):
        Mb *= denomBlock

    return [np.array([]), Mt, Mw, Mb, diff, cancel_pressed]

def NTFSolve_conv(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NMFSparseLevel, NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks,
             NTFNConv, NTFminDiv, myStatusBox):
    """Estimate NTF matrices (HALS)
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
         NTFNConv: Half-Size of the convolution window on 3rd-dimension of the tensor
     Output:
         Mt : if NTFNConv > 0 only otherwise empty. Contains sub-components for each phase in convolution window
         Mt_simple: Left hand matrix (sum of columns Mt_conv for each k)
         Mw_simple: Right hand matrix
         Mb_simple: Block hand matrix
         diff: objective cost
    
     Note: 
         This code extends HALS to allow for shifting on the 3rd dimension of the tensor. Suffix '_simple' is added to 
         the non-convolutional components. Convolutional components are named the usual way.

     """
    cancel_pressed = 0

    n, p0 = M.shape
    n_Mmis = Mmis.shape[0]
    nc = int(nc)
    NBlocks = int(NBlocks)
    NTFNConv = int(NTFNConv)
    p = int(p0 / NBlocks)
    nxp = int(n * p)
    nxp0 = int(n * p0)
    Mt_simple = np.copy(Mt0)
    Mw_simple = np.copy(Mw0)
    Mb_simple = np.copy(Mb0)
    #     StepIter = math.ceil(MaxIterations/10)
    StepIter = 1
    pbar_step = 100 * StepIter / MaxIterations

    IDBlockp = np.arange(0, (NBlocks - 1) * p + 1, p)
    A = np.zeros(n)
    B = np.zeros(p)
    C = np.zeros(NBlocks)
    MtMw = np.zeros(nxp)
    NTFNConv2 = 2*NTFNConv + 1
    
    #Initialize Mt, Mw, Mb
    Mt = np.repeat(Mt_simple, NTFNConv2, axis=1) / NTFNConv2
    Mw = np.repeat(Mw_simple, NTFNConv2, axis=1)
    Mb = np.repeat(Mb_simple, NTFNConv2, axis=1)

    for k3 in range(0, nc):
        n_shift = -NTFNConv - 1
        for k2 in range(0, NTFNConv2):
            n_shift += 1
            k = k3*NTFNConv2+k2
            Mb[:,k] = shift(Mb_simple[:, k3], n_shift)

    # Initialize Residual tensor
    Mfit = np.zeros((n, p0))
    for k3 in range(0, nc):
        for k2 in range(0, NTFNConv2):
            k = k3*NTFNConv2+k2
            for iBlock in range(0, NBlocks):
                Mfit[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] += Mb[iBlock,k] * \
                    np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))

    denomt = np.zeros(n)
    denomw = np.zeros(p)
    denomBlock = np.zeros((NBlocks, nc))
    Mt2 = np.zeros(n)
    Mw2 = np.zeros(p)
    denomCutoff = .1

    if n_Mmis > 0:
        Mres = (M - Mfit) * Mmis
    else:
        Mres = M - Mfit

    myStatusBox.init_bar(delay=1)

    # Loop
    cont = 1
    iIter = 0
    diff0 = 1.e+99
    Mpart = np.zeros((n, p0))
    alpha = NMFSparseLevel
    PercentZeros = 0
    iterSparse = 0

    while (cont > 0) & (iIter < MaxIterations):
        for k3 in range(0, nc):
            for k2 in range(0, NTFNConv2):
                k = k3*NTFNConv2+k2
                NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
                    NMFFixUserLHE, denomt, Mw2, denomCutoff, alpha ,\
                    NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
                    denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
                    denomBlock, NTFBlockComponents, C, Mfit = \
                NTFUpdate(NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
                    NMFFixUserLHE, denomt, Mw2, denomCutoff, alpha, \
                    NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
                    denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
                    denomBlock, NTFBlockComponents, C, Mfit, NTFminDiv)
            
            #Update Mt_simple, Mw_simple & Mb_simple
            k = k3*NTFNConv2+NTFNConv
            Mt_simple[:, k3] = Mt[:, k]
            Mw_simple[:, k3] = Mw[:, k]
            Mb_simple[:, k3] = Mb[:, k]

            # Update Mw & Mb
            Mw[:,:] = np.repeat(Mw_simple, NTFNConv2, axis=1)
            n_shift = -NTFNConv - 1
            for k2 in range(0, NTFNConv2):
                n_shift += 1
                k = k3*NTFNConv2+k2
                Mb[:,k] = shift(Mb_simple[:, k3], n_shift)
            
        if iIter % StepIter == 0:
            # Check convergence
            diff = np.linalg.norm(Mres) ** 2 / nxp0
            if (diff0 - diff) / diff0 < tolerance:
                cont = 0
            else:
                diff0 = diff

            Status = Status0 + 'Iteration: %s' % int(iIter)

            if NMFSparseLevel != 0:
                Status = Status + '; Achieved sparsity: ' + str(round(PercentZeros, 2)) + '; alpha: ' + str(
                    round(alpha, 2))
                if LogIter == 1:
                    myStatusBox.myPrint(Status)

            myStatusBox.update_status(delay=1, status=Status)
            myStatusBox.update_bar(delay=1, step=pbar_step)
            if myStatusBox.cancel_pressed:
                cancel_pressed = 1
                return [Mt, Mt_simple, Mw_simple, Mb_simple, cancel_pressed]

            if LogIter == 1:
                myStatusBox.myPrint(Status0 + " Iter: " + str(iIter) + " MSR: " + str(diff))

        iIter += 1

        if (cont == 0) | (iIter == MaxIterations):
            if NMFSparseLevel > 0:
                SparseTest = np.zeros((p, 1))
                for k in range(0, nc):
                    SparseTest[np.where(Mw[:, k] > 0)] = 1

                PercentZeros0 = PercentZeros
                n_SparseTest = np.where(SparseTest == 0)[0].size
                PercentZeros = max(n_SparseTest / p, .01)
                if PercentZeros == PercentZeros0:
                    iterSparse += 1
                else:
                    iterSparse = 0

                if (PercentZeros < 0.99 * NMFSparseLevel) & (iterSparse < 50):
                    alpha *= min(1.01 * NMFSparseLevel / PercentZeros, 1.01)
                    alpha = min(alpha, .99)
                    if alpha < .99:
                        iIter = 1
                        cont = 1

            elif NMFSparseLevel < 0:
                SparseTest = np.zeros((n, 1))
                for k in range(0, nc):
                    SparseTest[np.where(Mt[:, k] > 0)] = 1

                PercentZeros0 = PercentZeros
                n_SparseTest = np.where(SparseTest == 0)[0].size
                PercentZeros = max(n_SparseTest / n, .01)
                if PercentZeros == PercentZeros0:
                    iterSparse += 1
                else:
                    iterSparse = 0

                if (PercentZeros < 0.99 * abs(NMFSparseLevel)) & (iterSparse < 50):
                    alpha *= min(1.01 * abs(NMFSparseLevel) / PercentZeros, 1.01)
                    if abs(alpha) < .99:
                        iIter = 1
                        cont = 1

    if (n_Mmis > 0) & (NMFFixUserBHE == 0):
        Mb *= denomBlock

    return [Mt, Mt_simple, Mw_simple, Mb_simple, diff, cancel_pressed]

def NTFSolveFast(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, precision, LogIter, Status0, MaxIterations, NMFFixUserLHE,
                 NMFFixUserRHE, NMFFixUserBHE, NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents,
                 NBlocks, myStatusBox):
    """Estimate NTF matrices (fast HALS)
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
         Status0: Initial displayed status to be updated during iterations
         MaxIterations: Max iterations
         NMFFixUserLHE: fix left hand matrix columns: = 1, else = 0
         NMFFixUserRHE: fix  right hand matrix columns: = 1, else = 0
         NMFFixUserBHE: fix  block hand matrix columns: = 1, else = 0
         NTFUnimodal: Apply Unimodal constraint on factoring vectors
         NTFSmooth: Apply Smooth constraint on factoring vectors
         NTFLeftComponents: Apply Unimodal/Smooth constraint on left hand matrix
         NTFRightComponents: Apply Unimodal/Smooth constraint on right hand matrix
         NTFBlockComponents: Apply Unimodal/Smooth constraint on block hand matrix
         NBlocks: Number of NTF blocks
     Output:
         Mt: Left hand matrix
         Mw: Right hand matrix
         Mb: Block hand matrix
         diff: objective cost

     Note: This code does not support missing values, nor sparsity constraint

     """
    Mres = np.array([])
    cancel_pressed = 0

    n, p0 = M.shape
    n_Mmis = Mmis.shape[0]
    nc = int(nc)
    NBlocks = int(NBlocks)
    p = int(p0 / NBlocks)
    n0 = int(n * NBlocks)
    nxp = int(n * p)
    nxp0 = int(n * p0)
    Mt = np.copy(Mt0)
    Mw = np.copy(Mw0)
    Mb = np.copy(Mb0)
    StepIter = math.ceil(MaxIterations / 10)
    pbar_step = 100 * StepIter / MaxIterations

    IDBlockn = np.arange(0, (NBlocks - 1) * n + 1, n)
    IDBlockp = np.arange(0, (NBlocks - 1) * p + 1, p)
    A = np.zeros(n)
    B = np.zeros(p)
    C = np.zeros(NBlocks)

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

    for k in range(0, nc):
        if (NMFFixUserLHE > 0) & NormLHE:
            norm = np.linalg.norm(Mt[:, k])
            if norm > 0:
                Mt[:, k] /= norm
    
        if (NMFFixUserRHE > 0) & NormRHE:
            norm = np.linalg.norm(Mw[:, k])
            if norm > 0:
                Mw[:, k] /= norm
            
        if (NMFFixUserBHE > 0) & NormBHE:
            norm = np.linalg.norm(Mb[:, k])
            if norm > 0:
                Mb[:, k] /= norm
    
    # Normalize factors to unit length
    #    for k in range(0, nc):
    #        ScaleMt = np.linalg.norm(Mt[:, k])
    #        Mt[:, k] /= ScaleMt
    #        ScaleMw = np.linalg.norm(Mw[:, k])
    #        Mw[:, k] /= ScaleMw
    #        Mb[:, k] *= (ScaleMt * ScaleMw)

    # Initialize T1
    Mt2 = Mt.T @ Mt
    Mt2[Mt2 == 0] = precision
    Mw2 = Mw.T @ Mw
    Mw2[Mw2 == 0] = precision
    Mb2 = Mb.T @ Mb
    Mb2[Mb2 == 0] = precision
    T1 = Mt2 * Mw2 * Mb2
    T2t = np.zeros((n, nc))
    T2w = np.zeros((p, nc))
    T2Block = np.zeros((NBlocks, nc))

    # Transpose M by block once for all
    M2 = np.zeros((p, n0))

    Mfit = np.zeros((n, p0))
    if n_Mmis > 0:
        denomt = np.zeros(n)
        denomw = np.zeros(p)
        denomBlock = np.ones((NBlocks, nc))
        MxMmis2 = np.zeros((p, n0))
        denomCutoff = .1

    myStatusBox.init_bar(delay=1)

    # Loop
    cont = 1
    iIter = 0
    diff0 = 1.e+99


    for iBlock in range(0, NBlocks):
        M2[:, IDBlockn[iBlock]:IDBlockn[iBlock] + n] = M[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p].T
        if n_Mmis > 0:
            MxMmis2[:, IDBlockn[iBlock]:IDBlockn[iBlock] + n] = (M[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] * \
                                                                 Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p]).T

    if n_Mmis > 0:
        MxMmis = M * Mmis

    while (cont > 0) & (iIter < MaxIterations):
        if n_Mmis > 0:
            Gamma = np.diag((denomBlock*Mb).T @ (denomBlock*Mb))
        else:
            Gamma = np.diag(Mb.T @ Mb)

        if NMFFixUserLHE == 0:
            # Update Mt
            T2t[:,:] = 0
            for k in range(0, nc):
                if n_Mmis > 0:
                    denomt[:] = 0
                    Mwn = np.repeat(Mw[:, k, np.newaxis] ** 2, n, axis=1)
                    for iBlock in range(0, NBlocks):
                        # Broadcast missing cells into Mw to calculate Mw.T * Mw
                        denomt += Mb[iBlock, k]**2 * np.sum(Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p].T * Mwn, axis = 0)

                    denomt /= np.max(denomt)
                    denomt[denomt < denomCutoff] = denomCutoff
                    for iBlock in range(0, NBlocks):
                        T2t[:, k] += MxMmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] @ Mw[:, k] * Mb[iBlock, k]

                    T2t[:, k] /= denomt
                else:
                    for iBlock in range(0, NBlocks):
                        T2t[:, k] += M[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] @ Mw[:, k] * Mb[iBlock, k]

            Mt2 = Mt.T @ Mt
            Mt2[Mt2 == 0] = precision
            T3 = T1 / Mt2

            for k in range(0, nc):
                Mt[:, k] = Gamma[k] * Mt[:, k] + T2t[:, k] - Mt @ T3[:, k]
                Mt[np.where(Mt[:, k] < 0), k] = 0

                if (NTFUnimodal > 0) & (NTFLeftComponents > 0):
                    #                 Enforce unimodal distribution
                    tmax = np.argmax(Mt[:, k])
                    for i in range(tmax + 1, n):
                        Mt[i, k] = min(Mt[i - 1, k], Mt[i, k])

                    for i in range(tmax - 1, -1, -1):
                        Mt[i, k] = min(Mt[i + 1, k], Mt[i, k])

                if (NTFSmooth > 0) & (NTFLeftComponents > 0):
                    #             Smooth distribution
                    A[0] = .75 * Mt[0, k] + .25 * Mt[1, k]
                    A[n - 1] = .25 * Mt[n - 2, k] + .75 * Mt[n - 1, k]
                    for i in range(1, n - 1):
                        A[i] = .25 * Mt[i - 1, k] + .5 * Mt[i, k] + .25 * Mt[i + 1, k]

                    Mt[:, k] = A

                if NormLHE:
                    Mt[:, k] /= np.linalg.norm(Mt[:, k])

            Mt2 = Mt.T @ Mt
            Mt2[Mt2 == 0] = precision
            T1 = T3 * Mt2

        if NMFFixUserRHE == 0:
            # Update Mw
            T2w[:,:] = 0
            for k in range(0, nc):
                if n_Mmis > 0:
                    denomw[:] = 0
                    Mtp = np.repeat(Mt[:, k, np.newaxis] ** 2, p, axis=1)
                    for iBlock in range(0, NBlocks):
                        # Broadcast missing cells into Mw to calculate Mt.T * Mt
                        denomw += Mb[iBlock, k]**2 * np.sum(Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] * Mtp, axis = 0)

                    denomw /= np.max(denomw)
                    denomw[denomw < denomCutoff] = denomCutoff
                    for iBlock in range(0, NBlocks):
                        T2w[:, k] += MxMmis2[:, IDBlockn[iBlock]:IDBlockn[iBlock] + n] @ Mt[:, k] * Mb[iBlock, k]

                    T2w[:, k] /= denomw
                else:
                    for iBlock in range(0, NBlocks):
                        T2w[:, k] += M2[:, IDBlockn[iBlock]:IDBlockn[iBlock] + n] @ Mt[:, k] * Mb[iBlock, k]

            Mw2 = Mw.T @ Mw
            Mw2[Mw2 == 0] = precision
            T3 = T1 / Mw2

            for k in range(0, nc):
                Mw[:, k] = Gamma[k] * Mw[:, k] + T2w[:, k] - Mw @ T3[:, k]
                Mw[np.where(Mw[:, k] < 0), k] = 0

                if (NTFUnimodal > 0) & (NTFRightComponents > 0):
                    #                 Enforce unimodal distribution
                    wmax = np.argmax(Mw[:, k])
                    for j in range(wmax + 1, p):
                        Mw[j, k] = min(Mw[j - 1, k], Mw[j, k])

                    for j in range(wmax - 1, -1, -1):
                        Mw[j, k] = min(Mw[j + 1, k], Mw[j, k])

                if (NTFSmooth > 0) & (NTFLeftComponents > 0):
                    #             Smooth distribution
                    B[0] = .75 * Mw[0, k] + .25 * Mw[1, k]
                    B[p - 1] = .25 * Mw[p - 2, k] + .75 * Mw[p - 1, k]
                    for j in range(1, p - 1):
                        B[j] = .25 * Mw[j - 1, k] + .5 * Mw[j, k] + .25 * Mw[j + 1, k]

                    Mw[:, k] = B

                if NormRHE:
                    Mw[:, k] /= np.linalg.norm(Mw[:, k])

            Mw2 = Mw.T @ Mw
            Mw2[Mw2 == 0] = precision
            T1 = T3 * Mw2

        if NMFFixUserBHE == 0:
            # Update Mb
            for k in range(0, nc):
                if n_Mmis > 0:
                    for iBlock in range(0, NBlocks):
                        # Broadcast missing cells into Mb to calculate Mb.T * Mb
                        denomBlock[iBlock, k] = np.sum(np.reshape(Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], nxp) *
                                np.reshape((np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))), nxp)**2, axis=0)

                    maxdenomBlock = np.max(denomBlock[:, k])
                    denomBlock[denomBlock[:, k] < denomCutoff * maxdenomBlock] = denomCutoff * maxdenomBlock
                    for iBlock in range(0, NBlocks):
                        T2Block[iBlock, k] = np.reshape(MxMmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], nxp).T @ \
                                        (np.reshape((np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))), nxp)) / denomBlock[iBlock, k]

                else:
                    for iBlock in range(0, NBlocks):
                        T2Block[iBlock, k] = np.reshape(M[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], nxp).T @ \
                                        (np.reshape((np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))), nxp))

            Mb2 = Mb.T @ Mb
            Mb2[Mb2 == 0] = precision
            T3 = T1 / Mb2

            for k in range(0, nc):
                Mb[:, k] = Mb[:, k] + T2Block[:, k] - Mb @ T3[:, k]
                Mb[np.where(Mb[:, k] < 0), k] = 0

                if (NTFUnimodal > 0) & (NTFBlockComponents > 0):
                    #                 Enforce unimodal distribution
                    bmax = np.argmax(Mb[:, k])
                    for iBlock in range(bmax + 1, NBlocks):
                        Mb[iBlock, k] = min(Mb[iBlock - 1, k], Mb[iBlock, k])

                    for iBlock in range(bmax - 1, -1, -1):
                        Mb[iBlock, k] = min(Mb[iBlock + 1, k], Mb[iBlock, k])

                if (NTFSmooth > 0) & (NTFLeftComponents > 0):
                    #             Smooth distribution
                    C[0] = .75 * Mb[0, k] + .25 * Mb[1, k]
                    C[NBlocks - 1] = .25 * Mb[NBlocks - 2, k] + .75 * Mb[NBlocks - 1, k]
                    for iBlock in range(1, NBlocks - 1):
                        C[iBlock] = .25 * Mb[iBlock - 1, k] + .5 * Mb[iBlock, k] + .25 * Mb[iBlock + 1, k]

                    Mb[:, k] = C

            Mb2 = Mb.T @ Mb
            Mb2[Mb2 == 0] = precision
            T1 = T3 * Mb2

        if iIter % StepIter == 0:
            # Update residual tensor
            Mfit[:,:] = 0

            for k in range(0, nc):
                if n_Mmis > 0:
                    for iBlock in range(0, NBlocks):
                        #Mfit[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] += denomBlock[iBlock, k] * Mb[iBlock, k] * (
                        Mfit[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] += Mb[iBlock, k] * (
                        np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p)))

                    Mres = (M - Mfit) * Mmis
                else:
                    for iBlock in range(0, NBlocks):
                        Mfit[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] += Mb[iBlock, k] * (
                                np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p)))

                    Mres = (M - Mfit)

            # Check convergence
            diff = np.linalg.norm(Mres) ** 2 / nxp0
            if (diff0 - diff) / diff0 < tolerance:
                cont = 0
            else:
                diff0 = diff

            Status = Status0 + 'Iteration: %s' % int(iIter)
            myStatusBox.update_status(delay=1, status=Status)
            myStatusBox.update_bar(delay=1, step=pbar_step)
            if myStatusBox.cancel_pressed:
                cancel_pressed = 1
                return [Mt, Mw, Mb, Mres, cancel_pressed]

            if LogIter == 1:
                myStatusBox.myPrint(Status0 + " Iter: " + str(iIter) + " MSR: " + str(diff))

        iIter += 1

    if n_Mmis > 0:
        Mb *= denomBlock

    return [Mt, Mw, Mb, diff, cancel_pressed]

def NTFSetWeights (Mt, Mw, Mb, NBlocks, k, n, p, IDBlockp, Mweight):
    """Weigh part by inverse component-wise approximation 
       to iteratively minimize divergence
    """
    # Update pure part and derive weights
    if NBlocks > 1:
        for iBlock in range(0, NBlocks):
            Mweight[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] = Mb[iBlock, k] * \
                np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))
    else:
        Mweight[:,:] = np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))

    #Update weights
    Mweight[Mweight == 0] = EPSILON
    Mweight[:,:] = 1 / Mweight
 
    return Mweight

def NTFUpdate(NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
        NMFFixUserLHE, denomt, Mw2, denomCutoff, alpha, \
        NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
        denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
        denomBlock, NTFBlockComponents, C, Mfit, NTFMinDiv):
    """Core updating code called by NTFSolve_simple & NTF Solve_conv
    Input:
        All variables in the calling function used in the function 
    Output:
        Same as Input
    """

    n_iter = 100 #Hard-coded number of iterations for divergence minimization through irls
    # Compute kth-part
    if NBlocks > 1:
        for iBlock in range(0, NBlocks):
            Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] = Mb[iBlock, k] * \
                np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))
    else:
        Mpart[:,:] = np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))
  
    Mpart += Mres
    if n_Mmis > 0:
        Mpart *= Mmis

    if NTFMinDiv:
        Mweight = np.zeros_like(Mpart)
 
    if (NMFFixUserBHE > 0) & (NMFFixUserLHE > 0):
        NormLHE = False
        NormRHE = False
        NormBHE = False
    elif (NMFFixUserBHE > 0) & (NMFFixUserRHE > 0):
        NormLHE = False
        NormRHE = False
        NormBHE = False
    elif (NMFFixUserLHE > 0) & (NMFFixUserRHE > 0):
        NormLHE = False
        NormRHE = False
        NormBHE = False
    elif NMFFixUserBHE > 0:
        NormLHE = True
        NormRHE = False
        NormBHE = False
    elif NMFFixUserRHE > 0:
        NormLHE = True
        NormRHE = False
        NormBHE = False
    elif NMFFixUserLHE > 0:
        NormLHE = False
        NormRHE = False
        NormBHE = True
    else:
        NormLHE = True
        NormRHE = True
        NormBHE = False

    """
    if NormLHE:
        if NTFMinDiv:
            norm = np.sum(Mt[:, k])
        else:
            norm = np.linalg.norm(Mt[:, k])

        if norm > 0:
            Mt[:, k] /= norm

    if NormRHE:
        if NTFMinDiv:
            norm = np.sum(Mw[:, k])
        else:
            norm = np.linalg.norm(Mw[:, k])
        
        if norm > 0:
            Mw[:, k] /= norm
        
    if NormBHE:
        if NTFMinDiv:
            norm = np.sum(Mb[:, k])
        else:
            norm = np.linalg.norm(Mb[:, k])
        
        if norm > 0:
            Mb[:, k] /= norm
    """

    if NMFFixUserLHE == 0:
        t0 = np.full(n, EPSILON)
        iIter = 0
        cont = True
        while cont:
            if NTFMinDiv:
                Mweight = NTFSetWeights (Mt, Mw, Mb, NBlocks, k, n, p, IDBlockp, Mweight)
                if n_Mmis > 0:
                    Mweight *= Mmis

                t0[:] = Mt[:, k]

            # Update Mt
            if NBlocks > 1:
                Mt[:, k] = 0
                for iBlock in range(0, NBlocks):
                    if NTFMinDiv:
                        X = Mweight[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p]
                        Mt[:, k] += Mb[iBlock, k] * \
                            (X * Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p]) @ Mw[:, k]
                    else:
                        Mt[:, k] += Mb[iBlock, k] * \
                            Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] @ Mw[:, k]
            else: 
                if NTFMinDiv:
                    Mt[:, k] = Mweight * Mpart @ Mw[:, k]
                else:
                    Mt[:, k] = Mpart @ Mw[:, k]

            if (n_Mmis > 0) or NTFMinDiv:
                Mw2[:] = Mw[:, k] ** 2
                if NBlocks > 1:
                    denomt[:] = 0
                    for iBlock in range(0, NBlocks):
                        if NTFMinDiv:
                            denomt += Mb[iBlock, k]**2 * Mweight[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] @ Mw2
                        else:
                            denomt += Mb[iBlock, k]**2 * Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] @ Mw2

                else:
                    if NTFMinDiv:
                        denomt = Mweight[:, IDBlockp[0]:IDBlockp[0] + p] @ Mw2
                    else:
                        denomt = Mmis[:, IDBlockp[0]:IDBlockp[0] + p] @ Mw2

                max_denomt = np.max(denomt)
                denomt[denomt < denomCutoff*max_denomt] = denomCutoff*max_denomt
                Mt[:, k] /= denomt

            Mt[Mt[:, k] < 0, k] = 0
            if NormLHE:
                if NTFMinDiv:
                    norm = np.sum(Mt[:, k])
                else:
                    norm = np.linalg.norm(Mt[:, k])

                if norm > 0:
                    Mt[:, k] /= norm

            if not NTFMinDiv:
                cont = False
            else:
                iIter += 1
                cont = (np.linalg.norm(Mt[:, k]-t0)/np.linalg.norm(t0) > EPSILON) & (iIter < n_iter)

        if alpha < 0:
            Mt[:, k] = sparse_opt(Mt[:, k], -alpha)

        if (NTFUnimodal > 0) & (NTFLeftComponents > 0):
            #                 Enforce unimodal distribution
            tmax = np.argmax(Mt[:, k])
            for i in range(tmax + 1, n):
                Mt[i, k] = min(Mt[i - 1, k], Mt[i, k])

            for i in range(tmax - 1, -1, -1):
                Mt[i, k] = min(Mt[i + 1, k], Mt[i, k])

        if (NTFSmooth > 0) & (NTFLeftComponents > 0):
            #             Smooth distribution
            A[0] = .75 * Mt[0, k] + .25 * Mt[1, k]
            A[n - 1] = .25 * Mt[n - 2, k] + .75 * Mt[n - 1, k]
            for i in range(1, n - 1):
                A[i] = .25 * Mt[i - 1, k] + .5 * Mt[i, k] + .25 * Mt[i + 1, k]

            Mt[:, k] = A

    if NMFFixUserRHE == 0:
        w0 = np.full(p, EPSILON)
        iIter = 0
        cont = True
        while cont:
            if NTFMinDiv:
                Mweight = NTFSetWeights (Mt, Mw, Mb, NBlocks, k, n, p, IDBlockp, Mweight)
                if n_Mmis > 0:
                    Mweight *= Mmis

                w0[:] = Mw[:, k]

            # Update Mw
            if NBlocks > 1:
                Mw[:, k] = 0
                for iBlock in range(0, NBlocks):
                    if NTFMinDiv:
                        X = Mweight[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p].T
                        Mw[:, k] += X * Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p].T @ Mt[:, k] * Mb[iBlock, k]
                    else:
                        Mw[:, k] += Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p].T @ Mt[:, k] * Mb[iBlock, k]
            else:
                if NTFMinDiv:
                    X = Mweight.T
                    Mw[:, k] = X * Mpart.T @ Mt[:, k]
                else:
                    Mw[:, k] = Mpart.T @ Mt[:, k]

            if (n_Mmis > 0) or NTFMinDiv:
                Mt2[:] = Mt[:, k] ** 2
                if NBlocks > 1:
                    denomw[:] = 0
                    for iBlock in range(0, NBlocks):
                        if NTFMinDiv:
                            denomw += Mb[iBlock, k] ** 2 * Mweight[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p].T @ Mt2
                        else:
                            denomw += Mb[iBlock, k] ** 2 * Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p].T @ Mt2

                else:
                    if NTFMinDiv:
                        denomw = Mweight[:, IDBlockp[0]:IDBlockp[0] + p].T @ Mt2
                    else:
                        denomw = Mmis[:, IDBlockp[0]:IDBlockp[0] + p].T @ Mt2

                max_denomw = np.max(denomw)
                denomw[denomw < max_denomw*denomCutoff] = max_denomw*denomCutoff
                Mw[:, k] /= denomw

            Mw[Mw[:, k] < 0, k] = 0
            if NormRHE:
                if NTFMinDiv:
                    norm = np.sum(Mw[:, k])
                else:
                    norm = np.linalg.norm(Mw[:, k])

                if norm > 0:
                    Mw[:, k] /= norm

            if not NTFMinDiv:
                cont = False
            else:
                iIter += 1
                cont = (np.linalg.norm(Mw[:, k]-w0)/np.linalg.norm(w0) > EPSILON) & (iIter < n_iter)
        if alpha > 0:
            Mw[:, k] = sparse_opt(Mw[:, k], alpha)

        if (NTFUnimodal > 0) & (NTFRightComponents > 0):
            #Enforce unimodal distribution
            wmax = np.argmax(Mw[:, k])
            for j in range(wmax + 1, p):
                Mw[j, k] = min(Mw[j - 1, k], Mw[j, k])

            for j in range(wmax - 1, -1, -1):
                Mw[j, k] = min(Mw[j + 1, k], Mw[j, k])

        if (NTFSmooth > 0) & (NTFLeftComponents > 0):
            #             Smooth distribution
            B[0] = .75 * Mw[0, k] + .25 * Mw[1, k]
            B[p - 1] = .25 * Mw[p - 2, k] + .75 * Mw[p - 1, k]
            for j in range(1, p - 1):
                B[j] = .25 * Mw[j - 1, k] + .5 * Mw[j, k] + .25 * Mw[j + 1, k]

            Mw[:, k] = B

    if NMFFixUserBHE == 0:
        b0 = np.full(NBlocks, EPSILON)
        iIter = 0
        cont = True
        while cont:
            if NTFMinDiv:
                Mweight = NTFSetWeights (Mt, Mw, Mb, NBlocks, k, n, p, IDBlockp, Mweight)
                if n_Mmis > 0:
                    Mweight *= Mmis

                b0[:] = Mb[:, k]

            # Update Mb
            MtMw[:] = np.reshape((np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))), nxp)

            for iBlock in range(0, NBlocks):
                if NTFMinDiv:
                    X = np.reshape(Mweight[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], nxp).T
                    Mb[iBlock, k] = X * np.reshape(Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], nxp).T @ MtMw
                else:
                    Mb[iBlock, k] = np.reshape(Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], nxp).T @ MtMw

            if (n_Mmis > 0) or NTFMinDiv:                          
                MtMw[:] = MtMw[:] ** 2
                for iBlock in range(0, NBlocks):
                    if NTFMinDiv:
                        denomBlock[iBlock, k] = np.reshape(Mweight[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], (1, nxp)) @ MtMw
                    else:
                        denomBlock[iBlock, k] = np.reshape(Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], (1, nxp)) @ MtMw

                max_denomBlock = np.max(denomBlock[:, k])
                denomBlock[denomBlock[:, k] < denomCutoff * max_denomBlock] = denomCutoff * max_denomBlock
                Mb[:, k] /= denomBlock[:, k]

            Mb[Mb[:, k] < 0, k] = 0
            if NormBHE:
                if NTFMinDiv:
                    norm = np.sum(Mb[:, k])
                else:
                    norm = np.linalg.norm(Mb[:, k])

                if norm > 0:
                    Mb[:, k] /= norm

            if not NTFMinDiv:
                cont = False
            else:
                iIter += 1
                cont = (np.linalg.norm(Mb[:, k]-b0)/np.linalg.norm(b0) > EPSILON) & (iIter < n_iter)
      
        if (NTFUnimodal > 0) & (NTFBlockComponents > 0):
            #             Enforce unimodal distribution
            bmax = np.argmax(Mb[:, k])
            for iBlock in range(bmax + 1, NBlocks):
                Mb[iBlock, k] = min(Mb[iBlock - 1, k], Mb[iBlock, k])

            for iBlock in range(bmax - 1, -1, -1):
                Mb[iBlock, k] = min(Mb[iBlock + 1, k], Mb[iBlock, k])

        if (NTFSmooth > 0) & (NTFLeftComponents > 0):
            #             Smooth distribution
            C[0] = .75 * Mb[0, k] + .25 * Mb[1, k]
            C[NBlocks - 1] = .25 * Mb[NBlocks - 2, k] + .75 * Mb[NBlocks - 1, k]
            for iBlock in range(1, NBlocks - 1):
                C[iBlock] = .25 * Mb[iBlock - 1, k] + .5 * Mb[iBlock, k] + .25 * Mb[iBlock + 1, k]

            Mb[:, k] = C
  
    # Update residual tensor
    if NBlocks > 1:
        for iBlock in range(0, NBlocks):
            Mfit[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] = Mb[iBlock, k] * \
                np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))
    else:
        Mfit[:,:] = np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))

    if n_Mmis > 0:
        Mres[:,:] = (Mpart - Mfit) * Mmis
    else:
        Mres[:,:] = Mpart - Mfit

    return NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
            NMFFixUserLHE, denomt, Mw2, denomCutoff, alpha ,\
            NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
            denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
            denomBlock, NTFBlockComponents, C, Mfit
 