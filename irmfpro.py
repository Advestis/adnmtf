""" Non-negative matrix and tensor factorization

"""

# Author: Paul Fogel

# License: MIT
# # Dec 28, '19
# Initialize progressbar
from tkinter import *
from tkinter import ttk
import math
import numpy as np
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from scipy.stats import hypergeom
from scipy.optimize import nnls
import sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']

EPSILON = np.finfo(np.float32).eps
GALDERMA_FLAG = False

class StatusBox:
    def __init__(self):
        self.root = Tk()
        self.root.title('irMF status - Python kernel')
        self.root.minsize(width=230, height=60)
        self.frame = Frame(self.root, borderwidth=6)
        self.frame.pack()
        self.var = StringVar()
        self.status = Label(self.frame, textvariable=self.var, width=60, height=1)
        self.status.pack(fill=NONE, padx=6, pady=6)
        self.pbar = ttk.Progressbar(self.frame, orient=HORIZONTAL, max=100, mode='determinate')
        self.pbar.pack(fill=NONE, padx=6, pady=6)
        Button(self.frame, text='Cancel', command=self.close_dialog).pack(fill=NONE, padx=6, pady=6)
        self.cancel_pressed = False
        self.n_steps = 0

    def close_dialog(self):
        self.cancel_pressed = True

    def update_bar(self, delay=1, step=1):
        self.n_steps += step
        self.pbar.step(step)
        self.pbar.after(delay, lambda: self.root.quit())
        self.root.mainloop()

    def init_bar(self, delay=1):
        self.update_bar(delay=1, step=100 - self.n_steps)
        self.n_steps = 0

    def update_status(self, delay=1, status=''):
        self.var.set(status)
        self.status.after(delay, lambda: self.root.quit())
        self.root.mainloop()

    def close(self):
        self.root.destroy()

    def myPrint(self, status=''):
        print(status)


class StatusBoxTqdm:
    def __init__(self, verbose=0):
        self.LogIter = verbose
        if self.LogIter == 0:
            self.pbar = tqdm(total=100)

        self.cancel_pressed = False

    def update_bar(self, delay=0, step=1):
        if self.LogIter == 0:
            self.pbar.update(n=step)

    def init_bar(self, delay=0):
        if self.LogIter == 0:
            self.pbar.n = 0

    def update_status(self, delay=0, status=''):
        if self.LogIter == 0:
            self.pbar.set_description(status, refresh=False)
            self.pbar.refresh()

    def close(self):
        if self.LogIter == 0:
            self.pbar.clear()
            self.pbar.close()

    def myPrint(self, status=''):
       if self.LogIter == 1:
            print(status, end='\n')


def NMFDet(Mt, Mw, NMFExactDet):
    """
    Volume occupied by Left and Right factoring vectors
    Input:
        Mt: Left hand matrix
        Mw: Right hand matrix
        NMFExactDet if = 0 compute an approximate determinant in reduced space n x n or p x p
        through random sampling in the largest dimension
    Output:
        detXcells: determinant
    """
     
    n, nc = Mt.shape
    p, nc = Mw.shape
    nxp = n * p
    if (NMFExactDet > 0) | (n == p):
        Xcells = np.zeros((nxp, nc))
        for k in range(0, nc):
            Xcells[:, k] = np.reshape(np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p)), nxp)
            norm_k = np.linalg.norm(Xcells[:, k])
            if norm_k > 0 :
                Xcells[:, k] = Xcells[:, k] / norm_k
            else:
                Xcells[:, k] = 0
    else:
        if n > p:
            Xcells = np.zeros((p ** 2, nc))
            ID = np.arange(n)
            np.random.shuffle(ID)
            ID = ID[0:p]
            for k in range(0, nc):
                Xcells[:, k] = np.reshape(np.reshape(Mt[ID, k], (p, 1)) @ np.reshape(Mw[:, k], (1, p)), p ** 2)
                norm_k = np.linalg.norm(Xcells[:, k])
                if norm_k > 0 :
                    Xcells[:, k] = Xcells[:, k] / norm_k
                else:
                    Xcells[:, k] = 0
        else:
            Xcells = np.zeros((n ** 2, nc))
            ID = np.arange(p)
            np.random.shuffle(ID)
            ID = ID[0:n]
            for k in range(0, nc):
                Xcells[:, k] = np.reshape(np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[ID, k], (1, n)), n ** 2)
                norm_k = np.linalg.norm(Xcells[:, k])
                if norm_k > 0 :
                    Xcells[:, k] = Xcells[:, k] / norm_k
                else:
                    Xcells[:, k] = 0

    detXcells = np.linalg.det(Xcells.T @ Xcells)
    return detXcells


def ProjGrad(V, Vmis, W, Hinit, NMFAlgo, lambdax, tol, MaxIterations, NMFPriors):
    """
    Projected gradient
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
        for iIter in range(1, MaxIterations + 1):
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
                if (iIter == 1) & (projgrad > 0):
                    tol /= 10
                else:
                    restart = False

                break
            #       End if projgrad

            if iIter == MaxIterations:
                restart = False
        #   End For iIter

    H = H.T
    return [H, tol, lambdax]


def ProjGradKernel(Kernel, V, Vmis, W, Hinit, NMFAlgo, tol, MaxIterations, NMFPriors):
    """
    Projected gradient, kernel version
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
        for iIter in range(1, MaxIterations + 1):
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
                if iIter == 1:
                    tol /= 10
                else:
                    restart = False

                break
            #       End if projgrad

            if iIter == MaxIterations:
                restart = False
        #   End For iIter

    H = H.T
    return [H, tol]


def ApplyKernel(M, NMFKernel, Mt, Mw):
    """
    Calculate kernel
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


def GetConvexScores(Mt, Mw, Mh, flag, AddMessage):
    """
    Reweigh scores to sum up to 1
    Input:
        Mt: Left factoring matrix
        Mw: Right factoring matrix
        flag:  Current value
    Output:

       Mt: Left factoring matrix
        Mw: Right factoring matrix
        flag: += 1: Negative weights found
    """
    ErrMessage = ''
    cancel_pressed = 0

    n, nc = Mt.shape
    n_Mh = Mh.shape[0]
    try:
        Malpha = np.linalg.inv(Mt.T @ Mt) @ (Mt.T @ np.ones(n))
    except:
        Malpha = np.linalg.pinv(Mt.T @ Mt) @ (Mt.T @ np.ones(n))

    if np.where(Malpha < 0)[0].size > 0:
        flag += 1
        Malpha = nnls(Mt, np.ones(n))[0]

    n_zeroed = 0
    for k in range(0, nc):
        Mt[:, k] *= Malpha[k]
        if n_Mh > 0:
            Mh[:, k] *= Malpha[k]
        if Malpha[k] > 0:
            Mw[:, k] /= Malpha[k]
        else:
            n_zeroed += 1
        
    if n_zeroed > 0:
        AddMessage.insert(len(AddMessage), 'Ncomp=' + str(nc) + ': ' + str(n_zeroed) + ' components were zeroed')

    # Goodness of fit
    R2 = 1 - np.linalg.norm(np.sum(Mt.T, axis=0).T - np.ones(n)) ** 2 / n
    AddMessage.insert(len(AddMessage), 'Ncomp=' + str(nc) + ': Goodness of mixture fit before adjustement = ' + str(round(R2, 2)))

    for i in range(0, n):
        Mt[i, :] /= np.sum(Mt[i, :])

    return [Mt, Mw, Mh, flag, AddMessage, ErrMessage, cancel_pressed]


def percentile_exc(a, q):
    """
    Percentile, exclusive
    Input:
        a: Matrix
        q: Percentile
    Output:
        Percentile
    """
    return np.percentile(np.concatenate((np.array([np.min(a)]), a.flatten(), np.array([np.max(a)]))), q)


def RobustMax(V0, AddMessage, myStatusBox):
    """
    Robust max of column vectors
    For each column:
         = weighted mean of column elements larger than 95% percentile
        for each row, weight = specificity of the column value wrt other columns
    Input:
        V0: column vectors
    Output: Robust max by column
    """
    ErrMessage = ''
    cancel_pressed = 0

    V = V0.copy()
    n, nc = V.shape
    if nc > 1:
        ncI = 1 / nc
        lnncI = 1 / math.log(nc)

    ind = max(math.ceil(n * .05) - 1, min(n - 1, 2))
    Scale = np.max(V, axis=0)
    for k in range(0, nc):
        V[:, k] = V[:, k] / Scale[k]

    RobMax = np.max(V, axis=0)
    RobMax0 = 1e+99 * np.ones(nc)
    iIter = 0
    maxIterations = 100
    pbar_step = 100 / maxIterations
    myStatusBox.init_bar(delay=1)

    while ((np.linalg.norm(RobMax - RobMax0) / np.linalg.norm(RobMax)) ** 2 > 1e-6) & (iIter < maxIterations):
        for k in range(0, nc):
            V = V[np.argsort(-V[:, k]), :]
            if nc > 1:
                den = np.repeat(np.sum(V, axis=1), nc).reshape((n, nc))
                den[den == 0] = 1.e-10
                Prob = V / den
                Prob[Prob == 0] = 1.e-10
                Specificity = (np.ones(n) + np.sum(Prob * np.log(Prob), axis=1) * lnncI)
                Specificity[Prob[:, k] < ncI] = 0
            else:
                Specificity = np.ones(n)

            Specificity[ind:n] = 0
            RobMax0[k] = RobMax[k]
            RobMax[k] = np.sum(V[:, k] * Specificity) / np.sum(Specificity)
            V[V[:, k] > RobMax[k], k] = RobMax[k]

        myStatusBox.update_bar(delay=1, step=pbar_step)
        if myStatusBox.cancel_pressed:
            cancel_pressed = 1
            return RobMax * Scale, AddMessage, ErrMessage, cancel_pressed

        iIter += 1

    if iIter == maxIterations:
        AddMessage.insert(len(AddMessage),
                          'Warning: Max iterations reached while calculating robust max (N = ' + str(n) + ').')

    return [RobMax * Scale, AddMessage, ErrMessage, cancel_pressed]


def Leverage(V, NMFUseRobustLeverage, AddMessage, myStatusBox):
    """
    Calculate leverages
    Input:
        V: Input column vectors
        NMFUseRobustLeverage: Estimate robust through columns of V
    Output:
        Vn: Leveraged column vectors
    """

    ErrMessage = ''
    cancel_pressed = 0

    n, nc = V.shape
    Vn = np.zeros((n, nc))
    Vr = np.zeros((n, nc))
    if NMFUseRobustLeverage > 0:
        MaxV, AddMessage, ErrMessage, cancel_pressed = RobustMax(V, AddMessage, myStatusBox)
        if cancel_pressed == 1:
            return Vn, AddMessage, ErrMessage, cancel_pressed
    else:
        MaxV = np.max(V, axis=0)

    pbar_step = 100 / nc
    myStatusBox.init_bar(delay=1)
    for k in range(0, nc):
        Vr[V[:, k] > 0, k] = 1
        Vn[:, k] = MaxV[k] - V[:, k]
        Vn[Vn[:, k] < 0, k] = 0
        Vn[:, k] = Vn[:, k] ** 2
        for k2 in range(0, nc):
            if k2 != k:
                Vn[:, k] = Vn[:, k] + V[:, k2] ** 2

        Status = 'Leverage: Comp ' + str(k+1)
        myStatusBox.update_status(delay=1, status=Status)
        myStatusBox.update_bar(delay=1, step=pbar_step)
        if myStatusBox.cancel_pressed:
            cancel_pressed = 1
            return Vn, AddMessage, ErrMessage, cancel_pressed

    Vn = 10 ** (-Vn / (2 * np.mean(Vn))) * Vr
    return [Vn, AddMessage, ErrMessage, cancel_pressed]

def BuildClusters(Mt, Mw, Mb, MtPct, MwPct, NBlocks, BlkSize, NMFCalculateLeverage, NMFUseRobustLeverage, NMFAlgo,
                  NMFRobustClusterByStability, CellPlotOrderedClusters, AddMessage, myStatusBox):
    NBlocks = int(NBlocks)
    myStatusBox.update_status(delay=1, status='Build clusters...')
    ErrMessage = ''
    cancel_pressed = 0
    n, nc = np.shape(Mt)
    p = np.shape(Mw)[0]
    if NMFAlgo >= 5:
        BlockClust = np.zeros(NBlocks)
    else:
        BlockClust = np.array([])
        Mbn = np.array([])

    RCt = np.zeros(n)
    RCw = np.zeros(p)
    NCt = np.zeros(nc)
    NCw = np.zeros(NBlocks * nc)
    RowClust = np.zeros(n)
    ColClust = np.zeros(p)
    ilast = 0
    jlast = 0
    if NMFCalculateLeverage == 1:
        myStatusBox.update_status(delay=1, status="Leverages - Left components...")
        Mtn, AddMessage, ErrMessage, cancel_pressed = Leverage(Mt, NMFUseRobustLeverage, AddMessage, myStatusBox)
        myStatusBox.update_status(delay=1, status="Leverages - Right components...")
        Mwn, AddMessage, ErrMessage, cancel_pressed = Leverage(Mw, NMFUseRobustLeverage, AddMessage, myStatusBox)
        if NMFAlgo >= 5:
            myStatusBox.update_status(delay=1, status="Leverages - Block components...")
            Mbn, AddMessage, ErrMessage, cancel_pressed = Leverage(Mb, NMFUseRobustLeverage, AddMessage, myStatusBox)
    else:
        Mtn = Mt
        Mwn = Mw
        if NMFAlgo >= 5:
            Mbn = Mb

    if NMFAlgo >= 5:
        for iBlock in range(0, NBlocks):
            if nc > 1:
                BlockClust[iBlock] = np.argmax(Mbn[iBlock, :]) + 1
            else:
                BlockClust[iBlock] = 1

    for i in range(0, n):
        if nc > 1:
            if (isinstance(MtPct, np.ndarray)) & (NMFRobustClusterByStability > 0):
                RowClust[i] = np.argmax(MtPct[i, :]) + 1
            else:
                RowClust[i] = np.argmax(Mtn[i, :]) + 1
        else:
            RowClust[i] = 1

    for j in range(0, p):
        if nc > 1:
            if (isinstance(MwPct, np.ndarray)) & (NMFRobustClusterByStability > 0):
                ColClust[j] = np.argmax(MwPct[j, :]) + 1
            else:
                ColClust[j] = np.argmax(Mwn[j, :]) + 1
        else:
            ColClust[j] = 1

    if (CellPlotOrderedClusters == 1) & (nc >= 3):
        MtS = np.zeros(n)
        MwS = np.zeros(p)
        for i in range(0, n):
            if RowClust[i] == 1:
                MtS[i] = sum(k * Mtn[i, k] for k in range(0, 2)) / \
                         max(sum(Mtn[i, k] for k in range(0, 2)), 1.e-10)
            elif RowClust[i] == nc:
                MtS[i] = sum(k * Mtn[i, k] for k in range(nc - 2, nc)) / \
                         max(sum(Mtn[i, k] for k in range(nc - 2, nc)), 1.e-10)
            else:
                MtS[i] = sum(k * Mtn[i, k] for k in range(int(RowClust[i] - 2), int(RowClust[i] + 1))) / \
                         max(sum(Mtn[i, k] for k in range(int(RowClust[i] - 2), int(RowClust[i] + 1))), 1.e-10)

        for j in range(0, p):
            if ColClust[j] == 1:
                MwS[j] = sum(k * Mwn[j, k] for k in range(0, 2)) / \
                         max(sum(Mwn[j, k] for k in range(0, 2)), 1.e-10)
            elif ColClust[j] == nc:
                MwS[j] = sum(k * Mwn[j, k] for k in range(nc - 2, nc)) / \
                         max(sum(Mwn[j, k] for k in range(nc - 2, nc)), 1.e-10)
            else:
                MwS[j] = sum(k * Mwn[j, k] for k in range(int(ColClust[j] - 2), int(ColClust[j] + 1))) / \
                         max(sum(Mwn[j, k] for k in range(int(ColClust[j] - 2), int(ColClust[j] + 1))), 1.e-10)

    for k in range(0, nc):
        Mindex1 = np.where(RowClust == k + 1)[0]
        if len(Mindex1) > 0:
            if len(Mindex1) == 1:
                Mindex = Mindex1,
            elif (nc == 2) & (k == 1):
                Mindex = Mindex1[np.argsort(Mtn[Mindex1, k])]
            elif (CellPlotOrderedClusters == 1) & (nc >= 3):
                Mindex = Mindex1[np.argsort(MtS[Mindex1])]
            else:
                Mindex = Mindex1[np.argsort(-Mtn[Mindex1, k])]

            RCt[ilast:len(Mindex) + ilast] = Mindex
            ilast += len(Mindex)

        NCt[k] = ilast

    for iBlock in range(0, NBlocks):
        if iBlock == 0:
            j1 = 0
            j2 = int(abs(BlkSize[iBlock]))
        else:
            j1 = j2
            j2 += int(abs(BlkSize[iBlock]))

        for k in range(0, nc):
            Mindex2 = np.where(ColClust[j1:j2] == k + 1)[0]
            if len(Mindex2) > 0:
                Mindex2 = Mindex2 + j1
                if len(Mindex2) == 1:
                    Mindex = Mindex2
                elif (nc == 2) & (k == 1):
                    Mindex = Mindex2[np.argsort(Mwn[Mindex2, k])]
                elif (CellPlotOrderedClusters == 1) & (nc >= 3):
                    Mindex = Mindex2[np.argsort(MwS[Mindex2])]
                else:
                    Mindex = Mindex2[np.argsort(-Mwn[Mindex2, k])]

                RCw[jlast:len(Mindex) + jlast] = Mindex
                jlast += len(Mindex)

            NCw[iBlock * nc + k] = jlast

    return [Mtn, Mwn, Mbn, RCt, RCw, NCt, NCw, RowClust, ColClust, BlockClust, AddMessage, ErrMessage, cancel_pressed]

def ClusterPvalues(ClusterSize, nbGroups, Mt, RCt, NCt,RowGroups, ListGroups, Ngroup):
    n, nc = Mt.shape
    ClusterSize = ClusterSize.astype(np.int)
    nbGroups = int(nbGroups)
    RCt = RCt.astype(np.int)
    NCt = NCt.astype(np.int)
    ClusterSize = np.reshape(ClusterSize, nc)
    RCt = np.reshape(RCt, (n,))
    NCt = np.reshape(NCt, (nc,))
    RowGroups = np.reshape(RowGroups, (n,))

    ClusterGroup = np.zeros(nc)
    ClusterProb = np.zeros(nc)
    ClusterNgroup = np.zeros((nc,nbGroups))
    ClusterNWgroup = np.zeros((nc,nbGroups))
    prun = 0

    for k in range(0, nc):
        if ClusterSize[k] > 0:
            # Find main group (only if clustersize>2)
            kfound0 = 0
            for iGroup in range(0, nbGroups):
                if k == 0:
                    MX = np.where(RowGroups[RCt[0:NCt[0]]] == ListGroups[iGroup])[0]
                    if len(MX) >= 1:
                        ClusterNWgroup[k, iGroup] = np.sum(
                            Mt[RCt[0:NCt[0]][MX], k]
                        )
                        ClusterNgroup[k, iGroup] = len(MX)
                else:
                    MX = np.where(RowGroups[RCt[NCt[k-1]:NCt[k]]] == ListGroups[iGroup])[0]
                    if len(MX) >= 1:
                        ClusterNWgroup[k, iGroup] = np.sum(
                            Mt[RCt[NCt[k-1]:NCt[k]][MX], k]
                        )
                        ClusterNgroup[k, iGroup] = len(MX)

                if ClusterNgroup[k, iGroup] > kfound0:
                    kfound0 = ClusterNgroup[k, iGroup]
                    ClusterGroup[k] = iGroup

            SumClusterNWgroup = sum(ClusterNWgroup[k, :]);
            for iGroup in range(0, nbGroups):
                ClusterNWgroup[k, iGroup] = ClusterSize[k] * ClusterNWgroup[k, iGroup] / SumClusterNWgroup

        else:
            for iGroup in range(0, nbGroups):
                ClusterNgroup[k, iGroup] = 0
                ClusterNWgroup[k, iGroup] = 0

            ClusterGroup[k] = 1

    for k in range(0, nc):
        if ClusterSize[k] > 2:
            ClusterProb[k] = hypergeom.sf(ClusterNgroup[k, int(ClusterGroup[k])], n, Ngroup[int(ClusterGroup[k])], ClusterSize[k], loc=0) + \
                             hypergeom.pmf(ClusterNgroup[k, int(ClusterGroup[k])], n, Ngroup[int(ClusterGroup[k])], ClusterSize[k], loc=0)
        else:
            ClusterProb[k] = 1

    for k in range(0, nc):
        for iGroup in range(0, nbGroups):
            if ClusterNWgroup[k, iGroup]:
                prun += ClusterNWgroup[k, iGroup] * math.log(
                    ClusterNWgroup[k, iGroup] / (ClusterSize[k] * Ngroup[iGroup] / n))

    return [prun, ClusterGroup, ClusterProb, ClusterNgroup, ClusterNWgroup]


def GlobalSign(Nrun, nbGroups, Mt, RCt, NCt, RowGroups, ListGroups, Ngroup, myStatusBox):
    n, nc = Mt.shape
    Nrun = int(Nrun)
    nbGroups = int(nbGroups)
    RCt = RCt.astype(np.int)
    NCt = NCt.astype(np.int)
    ClusterSize = np.zeros(nc)
    RCt = np.reshape(RCt, n)
    NCt = np.reshape(NCt, nc)
    cancel_pressed = 0
    for k in range(0, nc):
        if k == 0:
            ClusterSize[k] = NCt[0]
        else:
            ClusterSize[k] = NCt[k] - NCt[k - 1]

    if nbGroups > 1:
        RowGroups = np.reshape(RowGroups, (n,))
        StepIter = np.round(Nrun / 10)
        pbar_step = 10
        Pglob = 1
        for irun in range(0, Nrun):
            if irun % StepIter == 0:
                myStatusBox.update_status(delay=1,
                                          status='Calculating global significance: ' + str(irun) + ' / ' + str(Nrun))
                myStatusBox.update_bar(delay=1, step=pbar_step)
                if myStatusBox.cancel_pressed:
                    cancel_pressed = 1
                    return [ClusterSize, Pglob, prun, ClusterProb, ClusterGroup, ClusterNgroup, cancel_pressed]

            prun, ClusterGroup, ClusterProb, ClusterNgroup, ClusterNWgroup = ClusterPvalues(ClusterSize, nbGroups, Mt,
                                                                                            RCt, NCt, RowGroups,
                                                                                            ListGroups, Ngroup)
            if irun == 0:
                ClusterProb0 = np.copy(ClusterProb)
                ClusterGroup0 = np.copy(ClusterGroup)
                ClusterNgroup0 = np.copy(ClusterNgroup)
                RowGroups0 = np.copy(RowGroups)
                prun0 = prun;
            else:
                if prun >= prun0:
                    Pglob += 1

            if irun < Nrun - 1:
                # permute row groups
                Boot = np.random.permutation
                RowGroups = RowGroups0[np.random.permutation(n)]
            else:
                # Restore
                ClusterProb = ClusterProb0
                ClusterGroup = ClusterGroup0
                ClusterNgroup = ClusterNgroup0
                RowGroups = RowGroups0
                prun = prun0
                Pglob /= Nrun
    else:
        Pglob = np.NaN
        prun = np.NaN
        ClusterProb = np.array([])
        ClusterGroup = np.array([])
        ClusterNgroup = np.array([])

    return [ClusterSize, Pglob, prun, ClusterProb, ClusterGroup, ClusterNgroup, cancel_pressed]

def NMFInit(M, Mmis, Mt0, Mw0, nc, tolerance, LogIter, myStatusBox):
    """
    NMF initialization using NNSVD (Boutsisdis)
    Input:
        M: Input matrix
        Mmis: Define missing values (0 = missing cell, 1 = real cell)
        Mt0: Initial left hand matrix (may be empty)
        Mw0: Initial right hand matrix (may be empty)
        nc: NMF rank
    Output:
        Mt: Left hand matrix
        Mw: Right hand matrix
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
        if (n >= nc) and (p >= nc):
            Msvd = M
        else:
            if n < nc:
                # Replicate rows until > nc 
                Msvd = np.repeat(M, np.ceil(nc/n), axis=0)
            
            if p < nc:
                # Replicate rows until > nc
                Msvd = np.repeat(Msvd, np.ceil(nc/p), axis=1)

        if n_Mmis == 0:
            t, d, w = randomized_svd(Msvd,
                                     n_components=nc,
                                     n_iter='auto',
                                     random_state=None)         
            Mt = t
            Mw = w.T
        else:
            Mt, d, Mw, Mmis, Mmsr, Mmsr2, AddMessage, ErrMessage, cancel_pressed = rSVDSolve(
                Msvd, Mmis, nc, tolerance, LogIter, 0, "", 200,
                1, 1, 1, myStatusBox)

        if n < nc:
            Mt = np.concatenate(Mt, np.ones((n,nc-n)))
        
        if p < nc:
            Mw = np.concatenate(Mw, np.ones((p,nc-p)))
    
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

def NMFReweigh(M, Mt, NMFPriors, AddMessage):
    """
    Cancel variables that load on more than one component
    Input:
         M: Input matrix
         Mt: Left hand matrix
         NMFPriors: priors on right hand matrix
    Output:
         NMFPriors: updated priors
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
         Mev: Scaling matrix
         Mw: Right hand matrix
         diff: objective cost
         Mh: Convexity matrix
         NMFPriors: Updated priors on right hand matrix
         flagNonconvex: Updated non-convexity flag on left hand matrix
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
    Mev = np.ones(nc)
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
                return [Mt, Mev, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed]
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
                    Mw, tolMw = ProjGradKernel(Kernel, M, Mmis, Mh, Mw, NMFAlgo, tolMw, NMFMaxIterProj, NMFPriors.T)
                elif (NMFConvex > 0) & (NMFFindCentroids > 0):
                    Mh, tolMh, dummy = ProjGrad(In, np.array([]), Mt, Mh.T, NMFAlgo, -1, tolMh, NMFMaxIterProj, np.array([]))
                else:
                    Mw, tolMw, lambdaw = ProjGrad(M, Mmis, Mt, Mw.T, NMFAlgo, lambdaw, tolMw, \
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
                    Mh, tolMh, dummy = ProjGrad(Ip, np.array([]), Mw, Mh.T, NMFAlgo, -1, tolMh, NMFMaxIterProj, np.array([]))
                elif (NMFConvex > 0) & (NMFFindCentroids > 0):
                    Mt, tolMt = ProjGradKernel(Kernel, M.T, Mmis.T, Mh, Mt, NMFAlgo, tolMt, NMFMaxIterProj, np.array([]))
                else:
                    Mt, tolMt, lambdat = ProjGrad(M.T, Mmis.T, Mw, Mt.T, NMFAlgo,
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

                else:
                    MF0 = Mt @ Mw.T
                    Mdiff = M * np.log(M / MF0 + precision) + MF0 - M

                diff = (np.linalg.norm(Mdiff)) ** 2 / nxp

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
                return [Mt, Mev, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed]

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
                            Kernel = ApplyKernel(Mmis * M, 1, np.array([]), np.array([]))
                        else:
                            Kernel = ApplyKernel(M, 1, np.array([]), np.array([]))
                    else:
                        if n_Mmis > 0:
                            Kernel = ApplyKernel(Mmis * (M / (Mt @ Mw.T)), 1, np.array([]), np.array([]))
                        else:
                            Kernel = ApplyKernel(M / (Mt @ Mw.T), 1, np.array([]), np.array([]))

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
                            Kernel = ApplyKernel(Mmis.T * M.T, 1, np.array([]), np.array([]))
                        else:
                            Kernel = ApplyKernel(M.T, 1, np.array([]), np.array([]))
                    else:
                        if n_Mmis > 0:
                            Kernel = ApplyKernel(Mmis.T * (M.T / (Mt @ Mw.T).T), 1, np.array([]), np.array([]))
                        else:
                            Kernel = ApplyKernel(M.T / (Mt @ Mw.T).T, 1, np.array([]), np.array([]))

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
                        Kernel = ApplyKernel(Mmis.T * M.T, NMFKernel, Mw, Mt)
                    else:
                        Kernel = ApplyKernel(M.T, NMFKernel, Mw, Mt)
                else:
                    if n_Mmis > 0:
                        Kernel = ApplyKernel(Mmis.T * (M.T / (Mt @ Mw.T).T), NMFKernel, Mw, Mt)
                    else:
                        Kernel = ApplyKernel(M.T / (Mt @ Mw.T).T, NMFKernel, Mw, Mt)

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

                if (PercentZeros < 0.99 * NMFSparseLevel) & (iterSparse < 50):
                    lambdat *= min(1.01 * NMFSparseLevel / PercentZeros, 1.10)
                    iIter = NMFMaxInterm + 1
                    cont = 1

    if NMFFindParts > 0:
        # Make Mt convex
        Mt = M @ Mh
        Mt, Mw, Mh, flagNonconvex2, AddMessage, ErrMessage, cancel_pressed = GetConvexScores(Mt, Mw, Mh, flagNonconvex,
                                                                                         AddMessage)
#        if flagNonconvex2 > flagNonconvex:
#            flagNonconvex = flagNonconvex2
#            # Calculate column centroids
#            for k in range(0, nc):
#                ScaleMh = np.sum(Mh[:, k])
#                Mh[:, k] = Mh[:, k] / ScaleMh
#                Mw[:, k] = Mw[:, k] * ScaleMh
#
#            Mt = M @ Mh
    elif NMFFindCentroids > 0:
        # Calculate row centroids
        for k in range(0, nc):
            ScaleMh = np.sum(Mh[:, k])
            Mh[:, k] = Mh[:, k] / ScaleMh
            Mt[:, k] = Mt[:, k] * ScaleMh

        Mw = (Mh.T @ M).T

    if (NMFConvex == 0) & (NMFFixUserLHE == 0) & (NMFFixUserRHE == 0):
        # Scale
        for k in range(0, nc):
            if (NMFAlgo == 2) | (NMFAlgo == 4):
                ScaleMt = np.linalg.norm(Mt[:, k])
                ScaleMw = np.linalg.norm(Mw[:, k])
            else:
                ScaleMt = np.sum(Mt[:, k])
                ScaleMw = np.sum(Mw[:, k])

            if ScaleMt > 0:
                Mt[:, k] = Mt[:, k] / ScaleMt

            if ScaleMw > 0:
                Mw[:, k] = Mw[:, k] / ScaleMw

            Mev[k] = ScaleMt * ScaleMw

    if (NMFKernel > 1) & (NLKernelApplied == 1):
        diff /= TraceKernel / nxp

    return [Mt, Mev, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed]


def rNMFSolve(
        M, Mmis, Mt0, Mw0, nc, tolerance, precision, LogIter, MaxIterations, NMFAlgo, NMFFixUserLHE,
        NMFFixUserRHE, NMFMaxInterm,
        NMFSparseLevel, NMFRobustResampleColumns, NMFRobustNRuns, NMFCalculateLeverage, NMFUseRobustLeverage,
        NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns, NMFPriors, myStatusBox):

    """
    Estimate left and right hand matrices (robust version)
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
         Mev: Scaling matrix
         Mw: Right hand matrix
         MtPct: Percent robust clustered rows
         MwPct: Percent robust clustered columns
         diff: Objective cost
         Mh: Convexity matrix
         flagNonconvex: Updated non-convexity flag on left hand matrix
    """

    # Check parameter consistency (and correct if needed)
    AddMessage = []
    ErrMessage =''
    cancel_pressed = 0
    nc = int(nc)
    if NMFFixUserLHE*NMFFixUserRHE == 1:
        Mev = np.ones(nc)
        return Mt0, Mev, Mw0, np.array([]), np.array([]), 0, np.array([]), 0, AddMessage, ErrMessage, cancel_pressed

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
    Mt, Mev, Mw, diffsup, Mhsup, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
        M, Mmis, Mt0, Mw0, nc, tolerance, precision, LogIter, Status, MaxIterations, NMFAlgo,
        NMFFixUserLHE, NMFFixUserRHE, NMFMaxInterm, 100, NMFSparseLevel,
        NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns, NMFPriors, flagNonconvex, AddMessage, myStatusBox)
    Mtsup = np.copy(Mt)
    Mevsup = np.copy(Mev)
    Mwsup = np.copy(Mw)
    if (n_NMFPriors > 0) & (NMFReweighColumns > 0):
        #     Run again with fixed LHE & no priors
        Status = "Step 1bis - NMF (fixed LHE) Ncomp=" + str(nc) + ": "
        Mw = np.ones((p, nc)) / math.sqrt(p)
        Mt, Mev, Mw, diffsup, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
            M, Mmis, Mtsup, Mw, nc, tolerance, precision, LogIter, Status, MaxIterations, NMFAlgo, nc, 0, NMFMaxInterm, 100,
            NMFSparseLevel, NMFFindParts, NMFFindCentroids, NMFKernel, 0, NMFPriors, flagNonconvex, AddMessage,
            myStatusBox)
        Mtsup = np.copy(Mt)
        Mevsup = np.copy(Mev)
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
                Mt, Mev, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
                    M[Boot, :], Mmis[Boot, :], Mtsup[Boot, :], Mwsup, nc, 1.e-3, precision, LogIter, Status, MaxIterations, NMFAlgo, nc, 0,
                    NMFMaxInterm, 20, NMFSparseLevel, NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns,
                    NMFPriors, flagNonconvex, AddMessage, myStatusBox)
            else:
                Mt, Mev, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
                    M[Boot, :], Mmis, Mtsup[Boot, :], Mwsup, nc, 1.e-3, precision, LogIter, Status, MaxIterations, NMFAlgo, nc, 0,
                    NMFMaxInterm, 20, NMFSparseLevel, NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns,
                    NMFPriors, flagNonconvex, AddMessage, myStatusBox)

            for k in range(0, nc):
                MwBlk[:, k * NMFRobustNRuns + iBootstrap] = Mw[:, k] * Mev[k]

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

            Mt, Mev, Mw, diff, Mh, NMFPriors, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
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
    Mev = Mevsup
    diff = diffsup

    if NMFRobustResampleColumns > 0:
        Mtemp = np.copy(Mt)
        Mt = np.copy(Mw)
        Mw = Mtemp
        Mtemp = np.copy(MtPct)
        MtPct = np.copy(MwPct)
        MwPct = Mtemp

    return Mt, Mev, Mw, MtPct, MwPct, diff, Mh, flagNonconvex, AddMessage, ErrMessage, cancel_pressed

def NTFStack(M, Mmis, NBlocks):
    # Unfold M
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


def NTFInit(M, Mmis, MtxMw, Mb2, nc, tolerance, precision, LogIter, NTFUnimodal,
            NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox):
    """
     Estimate NTF matrices (HALS)
     Input:
         M: Input tensor
         Mmis: Define missing values (0 = missing cell, 1 = real cell)
         MtxMw: initialization of LHM in NMF(unstacked tensor), may be empty
         Mb2: initialization of RHM of NMF(unstacked tensor), may be empty
         NBlocks: Number of NTF blocks
         nc: NTF rank
         tolerance: Convergence threshold
         precision: Replace 0-values in multiplication rules
         LogIter: Log results through iterations
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
    Status0 = "Step 1 - Quick NMF Ncomp=" + str(nc) + ": "
    Mstacked, Mmis_stacked = NTFStack(M, Mmis, NBlocks)
    nc2 = min(nc, NBlocks)  # factorization rank can't be > number of blocks
    if (MtxMw.shape[0] == 0) or (Mb2.shape[0] == 0):
        MtxMw, Mb2 = NMFInit(Mstacked, Mmis_stacked, np.array([]),  np.array([]), nc2, tolerance, LogIter, myStatusBox)
    # NOTE: NMFInit (NNSVD) should always be called to prevent initializing NMF with signed components.
    # Creates minor differences in AD clustering, correction non implemented in Galderma version
    if not GALDERMA_FLAG:
        MtxMw, Mb2 = NMFInit(Mstacked, Mmis_stacked, MtxMw, Mb2, nc2, tolerance, LogIter, myStatusBox)
    else:
        print("Galderma version! In NTFInit, NNSVD has been superseded by SVD prior to NMF initialization")

    # Quick NMF
    MtxMw, Mev2, Mb2, diff, Mh, dummy1, dummy2, AddMessage, ErrMessage, cancel_pressed = NMFSolve(
        Mstacked, Mmis_stacked, MtxMw, Mb2, nc2, tolerance, precision, LogIter, Status0, 10, 2, 0, 0, 1, 1, 0, 0, 0, 1, 0, np.array([]), 0,
        AddMessage,
        myStatusBox)
    for k in range(0, nc2):
        Mb2[:, k] = Mb2[:, k] * Mev2[k]

    # Factorize Left vectors and distribute multiple factors if nc2 < nc
    Mt = np.zeros((n, nc))
    Mw = np.zeros((int(p / NBlocks), nc))
    Mb = np.zeros((NBlocks, nc))
    NFact = int(np.ceil(nc / NBlocks))
    for k in range(0, nc2):
        myStatusBox.update_status(delay=1, status="Start SVD...")
        U, d, V = randomized_svd(np.reshape(MtxMw[:, k], (int(p / NBlocks), n)).T,
                                     n_components=NFact,
                                     n_iter='auto',
                                     random_state=None)
        V = V.T
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

                Mb[:, ind] = Mb2[:, k]

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

def shift(arr, num, fill_value=EPSILON):
    """
    Shift a vector
    Input:
        arr: Input column vector
        num: number of indexs to shift ( < 0: To the left )
    Output:
        result: shifted column vector
    """

    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def NTFUpdate(NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
        NMFFixUserLHE, denomt, Mw2, denomCutoff, \
        NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
        denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
        denomBlock, NTFBlockComponents, C, Mfit):
    """
    Core updating code called by NTFSolve_simple & NTF Solve_conv
    Input:
        All variables in the calling function used in the function 
    Output:
        Same as Input
    """
    
    # Compute kth-part
    for iBlock in range(0, NBlocks):
        Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] = Mb[iBlock, k] * \
            np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))

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
        
    if (NMFFixUserBHE > 0) & NormBHE:
        norm = np.linalg.norm(Mb[:, k])
        if norm > 0:
            Mb[:, k] /= norm

    if NMFFixUserLHE == 0:
        # Update Mt
        Mt[:, k] = 0
        for iBlock in range(0, NBlocks):
            Mt[:, k] += Mb[iBlock, k] * Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] @ Mw[:, k]

        if n_Mmis > 0:
            denomt[:] = 0
            Mw2[:] = Mw[:, k] ** 2
            for iBlock in range(0, NBlocks):
                # Broadcast missing cells into Mw to calculate Mw.T * Mw
                denomt += Mb[iBlock, k]**2 * Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] @ Mw2

            denomt /= np.max(denomt)
            denomt[denomt < denomCutoff] = denomCutoff
            Mt[:, k] /= denomt               

        Mt[Mt[:, k] < 0, k] = 0

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
            norm = np.linalg.norm(Mt[:, k])
            if norm > 0:
                Mt[:, k] /= norm

    if NMFFixUserRHE == 0:
        # Update Mw
        Mw[:, k] = 0
        for iBlock in range(0, NBlocks):
            Mw[:, k] += Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p].T @ Mt[:, k] * Mb[iBlock, k]

        if n_Mmis > 0:
            denomw[:] = 0
            Mt2[:] = Mt[:, k] ** 2
            for iBlock in range(0, NBlocks):
                # Broadcast missing cells into Mw to calculate Mt.T * Mt
                denomw += Mb[iBlock, k] ** 2 * Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p].T @ Mt2

            denomw /= np.max(denomw)
            denomw[denomw < denomCutoff] = denomCutoff
            Mw[:, k] /= denomw

        Mw[Mw[:, k] < 0, k] = 0

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

        if NormRHE:
            norm = np.linalg.norm(Mw[:, k])
            if norm > 0:
                Mw[:, k] /= norm

    if NMFFixUserBHE == 0:
        # Update Mb
        Mb[:, k] = 0
        MtMw[:] = np.reshape((np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))), nxp)

        for iBlock in range(0, NBlocks):
            Mb[iBlock, k] = np.reshape(Mpart[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], nxp).T @ MtMw

        if n_Mmis > 0:                          
            MtMw[:] = MtMw[:] ** 2
            for iBlock in range(0, NBlocks):
                # Broadcast missing cells into Mb to calculate Mb.T * Mb
                denomBlock[iBlock, k] = np.reshape(Mmis[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p], (1, nxp)) @ MtMw

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

        if (NTFSmooth > 0) & (NTFLeftComponents > 0):
            #             Smooth distribution
            C[0] = .75 * Mb[0, k] + .25 * Mb[1, k]
            C[NBlocks - 1] = .25 * Mb[NBlocks - 2, k] + .75 * Mb[NBlocks - 1, k]
            for iBlock in range(1, NBlocks - 1):
                C[iBlock] = .25 * Mb[iBlock - 1, k] + .5 * Mb[iBlock, k] + .25 * Mb[iBlock + 1, k]

            Mb[:, k] = C
  
        if NormBHE:
            norm = np.linalg.norm(Mb[:, k])
            if norm > 0:
                Mb[:, k] /= norm

    # Update residual tensor
    Mfit[:,:] = 0
    for iBlock in range(0, NBlocks):
        Mfit[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] += Mb[iBlock, k] * \
            np.reshape(Mt[:, k], (n, 1)) @ np.reshape(Mw[:, k], (1, p))

    if n_Mmis > 0:
        Mres[:,:] = (Mpart - Mfit) * Mmis
    else:
        Mres[:,:] = Mpart - Mfit

    return NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
            NMFFixUserLHE, denomt, Mw2, denomCutoff, \
            NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
            denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
            denomBlock, NTFBlockComponents, C, Mfit
   
def NTFSolve(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox):
    """
    Interface to NTFSolve_simple & NTFSolve_conv
    """

    if NTFNConv > 0:
        return NTFSolve_conv(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox)
    else:
        return NTFSolve_simple(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox)

def NTFSolve_simple(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox):
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
         Mres: Residual tensor
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
        for iBlock in range(0, NBlocks):
            Mfit[:, IDBlockp[iBlock]:IDBlockp[iBlock] + p] += Mb[iBlock, k] * np.reshape(Mt[:, k], (n, 1)) @ np.reshape(
                Mw[:, k], (1, p))

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
    
    #for k in range (0, nc):
    #    print(k, Mb[:, k])


    while (cont > 0) & (iIter < MaxIterations):
        for k in range(0, nc):
            NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
                NMFFixUserLHE, denomt, Mw2, denomCutoff, \
                NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
                denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
                denomBlock, NTFBlockComponents, C, Mfit = \
            NTFUpdate(NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
                NMFFixUserLHE, denomt, Mw2, denomCutoff, \
                NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
                denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
                denomBlock, NTFBlockComponents, C, Mfit)
                       
        if iIter % StepIter == 0:
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
                return [np.array([]), Mt, Mw, Mb, Mres, cancel_pressed]

            if LogIter == 1:
                myStatusBox.myPrint(Status0 + " Iter: " + str(iIter) + " MSR: " + str(diff))

        iIter += 1

    if (n_Mmis > 0) & (NMFFixUserBHE == 0):
        Mb *= denomBlock

    return [np.array([]), Mt, Mw, Mb, Mres, cancel_pressed]

def NTFSolve_conv(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, LogIter, Status0, MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE,
             NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox):
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
         NTFUnimodal: Apply Unimodal constraint on factoring vectors
         NTFSmooth: Apply Smooth constraint on factoring vectors
         NTFLeftComponents: Apply Unimodal/Smooth constraint on left hand matrix
         NTFRightComponents: Apply Unimodal/Smooth constraint on right hand matrix
         NTFBlockComponents: Apply Unimodal/Smooth constraint on block hand matrix
         NBlocks: Number of NTF blocks
         NTFNConv: Half-Size of the convolution window on 3rd-dimension of the tensor
     Output:
         Mt: Left hand matrix (sum of columns Mt_conv for each k)
         Mt_conv : if NTFNConv > 0 only otherwise empty. Contains sub-components for each phase in convolution window
         Mw: Right hand matrix
         Mb: Block hand matrix
         Mres: Residual tensor
c     """
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

    while (cont > 0) & (iIter < MaxIterations):
        for k3 in range(0, nc):
            for k2 in range(0, NTFNConv2):
                k = k3*NTFNConv2+k2
                NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
                    NMFFixUserLHE, denomt, Mw2, denomCutoff, \
                    NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
                    denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
                    denomBlock, NTFBlockComponents, C, Mfit = \
                NTFUpdate(NBlocks, Mpart, IDBlockp, p, Mb, k, Mt, n, Mw, n_Mmis, Mmis, Mres, \
                    NMFFixUserLHE, denomt, Mw2, denomCutoff, \
                    NTFUnimodal, NTFLeftComponents, NTFSmooth, A, NMFFixUserRHE, \
                    denomw, Mt2, NTFRightComponents, B, NMFFixUserBHE, MtMw, nxp, \
                    denomBlock, NTFBlockComponents, C, Mfit)
            
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
            myStatusBox.update_status(delay=1, status=Status)
            myStatusBox.update_bar(delay=1, step=pbar_step)
            if myStatusBox.cancel_pressed:
                cancel_pressed = 1
                return [Mt, Mt_simple, Mw_simple, Mb_simple, Mres, cancel_pressed]

            if LogIter == 1:
                myStatusBox.myPrint(Status0 + " Iter: " + str(iIter) + " MSR: " + str(diff))

        iIter += 1
    
    if (n_Mmis > 0) & (NMFFixUserBHE == 0):
        Mb *= denomBlock

    return [Mt, Mt_simple, Mw_simple, Mb_simple, Mres, cancel_pressed]


def NTFSolveFast(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, precision, LogIter, Status0, MaxIterations, NMFFixUserLHE,
                 NMFFixUserRHE, NMFFixUserBHE, NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents,
                 NBlocks, myStatusBox):
    """
     Estimate NTF matrices (fast HALS)
     Input:
         M: Input matrix
         Mmis: Define missing values (0 = missing cell, 1 = real cell)
            NOTE: Still not workable version
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
         Mres: Residual tensor
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

    return [Mt, Mw, Mb, Mres, cancel_pressed]


def rNTFSolve(M, Mmis, Mt0, Mw0, Mb0, nc, tolerance, precision, LogIter, MaxIterations, NMFFixUserLHE, NMFFixUserRHE,
              NMFFixUserBHE, NMFAlgo, NMFRobustNRuns, NMFCalculateLeverage, NMFUseRobustLeverage, NTFFastHALS, NTFNIterations,
              NTFUnimodal, NTFSmooth, NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox):
    """
     Estimate NTF matrices (robust version)
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
         NTFFastHALS: Use Fast HALS
         NTFNIterations: Warmup iterations for fast HALS
         NTFUnimodal: Apply Unimodal constraint on factoring vectors
         NTFSmooth: Apply Smooth constraint on factoring vectors
         NTFLeftComponents: Apply Unimodal/Smooth constraint on left hand matrix
         NTFRightComponents: Apply Unimodal/Smooth constraint on right hand matrix
         NTFBlockComponents: Apply Unimodal/Smooth constraint on block hand matrix
         NBlocks: Number of NTF blocks
         NTFNConv: Half-Size of the convolution window on 3rd-dimension of the tensor
         
     Output:
         Mt_conv: Convolutional Left hand matrix
         Mt: Left hand matrix
         Mw: Right hand matrix
         Mb: Block hand matrix
         Mres: Residual tensor
         MtPct: Percent robust clustered rows
         MwPct: Percent robust clustered columns
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

    if (n_Mmis > 0 or NTFNConv > 0) and NTFFastHALS > 0:
        NTFFastHALS = 0
        reverse2HALS = 1
    else:
        reverse2HALS = 0

    # Step 1: NTF
    Status0 = "Step 1 - NTF Ncomp=" + str(nc) + ": "
    if NTFFastHALS > 0:
        if NTFNIterations > 0:
            Mt_conv, Mt, Mw, Mb, Mres, cancel_pressed = NTFSolve(
                M, Mmis, Mt, Mw, Mb, nc, tolerance, LogIter, Status0,
                NTFNIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
                NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox)

        Mt, Mw, Mb, Mres, cancel_pressed = NTFSolveFast(
            M, Mmis, Mt, Mw, Mb, nc, tolerance, precision, LogIter, Status0,
            MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
            NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox)
    else:
        Mt_conv, Mt, Mw, Mb, Mres, cancel_pressed = NTFSolve(
            M, Mmis, Mt, Mw, Mb, nc, tolerance, LogIter, Status0,
            MaxIterations, NMFFixUserLHE, NMFFixUserRHE, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
            NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox)

    Mtsup = np.copy(Mt)
    Mwsup = np.copy(Mw)
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
                    Mt, Mw, Mb, Mres, cancel_pressed = NTFSolveFast(
                        M[Boot, :], Mmis[Boot, :], Mtsup[Boot, :], Mwsup, Mb, nc, 1.e-3, precision, LogIter, Status0,
                        MaxIterations, 1, 0, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
                        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox)
                else:
                    Mt, Mw, Mb, Mres, cancel_pressed = NTFSolveFast(
                        M[Boot, :], np.array([]), Mtsup[Boot, :], Mwsup, Mb, nc, 1.e-3, precision, LogIter, Status0,
                        MaxIterations, 1, 0, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
                        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox)
            else:
                if n_Mmis > 0:
                    Mt_conv, Mt, Mw, Mb, Mres, cancel_pressed = NTFSolve(
                        M[Boot, :], Mmis[Boot, :], Mtsup[Boot, :], Mwsup, Mb, nc, 1.e-3, LogIter, Status0,
                        MaxIterations, 1, 0, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
                        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox)
                else:
                    Mt_conv, Mt, Mw, Mb, Mres, cancel_pressed = NTFSolve(
                        M[Boot, :], np.array([]), Mtsup[Boot, :], Mwsup, Mb, nc, 1.e-3, LogIter, Status0,
                        MaxIterations, 1, 0, NMFFixUserBHE, NTFUnimodal, NTFSmooth,
                        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox)

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
                Mt, Mw, Mb, Mres, cancel_pressed = NTFSolveFast(
                    M, Mmis, Mtsup, Mw, Mb, nc, 1.e-3, precision, LogIter, Status0, MaxIterations, 0, 1, NMFFixUserBHE,
                    NTFUnimodal, NTFSmooth,
                    NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, myStatusBox)
            else:
                Mt_conv, Mt, Mw, Mb, Mres, cancel_pressed = NTFSolve(
                    M, Mmis, Mtsup, Mw, Mb, nc, 1.e-3, LogIter, Status0, MaxIterations, 0, 1, NMFFixUserBHE,
                    NTFUnimodal, NTFSmooth,
                    NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox)

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
    if reverse2HALS > 0:
        AddMessage.insert(len(AddMessage), 'Currently, Fast HALS cannot be applied with missing data or convolution window and was reversed to Simple HALS.')

    return Mt_conv, Mt, Mw, Mb, Mres, MtPct, MwPct, AddMessage, ErrMessage, cancel_pressed


def rSVDSolve(M, Mmis, nc, tolerance, LogIter, LogTrials, Status0, MaxIterations,
              SVDAlgo, SVDCoverage, SVDNTrials, myStatusBox):
    """
     Estimate SVD matrices (robust version)
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
         Mev:
         Mw: Right hand matrix
         Mmis: Matrix of missing/flagged outliers
         Mmsr: Vector of Residual SSQ
         Mmsr2: Vector of Reidual variance
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
                               n_bootstrap=None,
                               tol=1e-6,
                               max_iter=150, max_iter_mult=20,
                               regularization=None, sparsity=None,
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

    sparsity : double, default: 0.
        Sparsity target with 0 <= sparsity <= 1 representing the % rows in W or H set to 0.

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

    V : scalar, volume occupied by W and H

    WB : array-like, shape (n_samples, n_components)
        Percent consistently clustered rows for each component.
        only if n_bootstrap > 0.

    HB : array-like, shape (n_features, n_components)
        Percent consistently clustered columns for each component.
        only if n_bootstrap > 0.

    B : array-like, shape (n_observations, n_components) or (n_features, n_components)
        only if active convex variant, H = B.T @ X or W = X @ B


    Examples
    --------

    >>> import numpy as np

    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])

    >>> from sklearn.decomposition import non_negative_factorization

    >>> W, H, n_iter = non_negative_factorization(X, n_components=2, \

        random_state=0)


    References
    ----------

    Fogel

    Lin
    """

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

    Mt, Mev, Mw, MtPct, MwPct, diff0, Mh, flagNonconvex, AddMessage, ErrMessage, cancel_pressed = rNMFSolve(
        M, np.array([]), Mt, Mw, nc, tolerance, precision, LogIter, MaxIterations, NMFAlgo, NMFFixUserLHE,
        NMFFixUserRHE, NMFMaxInterm,
        NMFSparseLevel, NMFRobustResampleColumns, NMFRobustNRuns, NMFCalculateLeverage, NMFUseRobustLeverage,
        NMFFindParts, NMFFindCentroids, NMFKernel, NMFReweighColumns, NMFPriors, myStatusBox)
    
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
            estimator.update([('W', Mt), ('H', Mw), ('V', volume)])
        else:
            estimator.update([('W', Mt), ('H', Mw), ('V', volume), ('B', Mh)])

    else:
        if (NMFFindParts == 0) & (NMFFindCentroids == 0):
            estimator.update([('W', Mt), ('H', Mw), ('V', volume), ('WB', MtPct), ('HB', MwPct)])
        else:
            estimator.update([('W', Mt), ('H', Mw), ('V', volume), ('B', Mh), ('WB', MtPct), ('HB', MwPct)])

    return estimator


def nmf_predict(estimator, leverage='robust', blocks=None, cluster_by_stability=False, custom_order=False, verbose=0):
    """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

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

    FL : array-like, shape (n_blocks, n_components)
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

    FC : vector-like, shape (size(blocks))
         Block assigned cluster (NTF only)

    Examples
    --------

    >>> import numpy as np

    """

    Mt = estimator['W']
    Mw = estimator['H']
    if 'F' in estimator:
        # X is a 3D tensor, in unfolded form of a 2D array
        # horizontal concatenation of blocks of equal size.
        Mb = estimator['F']
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
    if 'F' in estimator:
        estimator.update([('WL', Mtn), ('HL', Mwn), ('WR', RCt), ('HR', RCw), ('WN', NCt), ('HN', NCw),
                          ('WC', RowClust), ('HC', ColClust), ('FL', Mbn), ('FC', BlockClust)])
    else:
        estimator.update([('WL', Mtn), ('HL', Mwn), ('WR', RCt), ('HR', RCw), ('WN', NCt), ('HN', NCw),
                          ('WC', RowClust), ('HC', ColClust), ('FL', None), ('FC', None)])
    return estimator


def nmf_permutation_test_score(estimator, y, n_permutations=100, verbose=0):
    """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

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

    Examples
    --------

    >>> import numpy as np

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

def non_negative_tensor_factorization(X, n_blocks, W=None, H=None, F=None, n_components=None,
                                      update_W=True,
                                      update_H=True,
                                      update_F=True,
                                      fast_hals=True, n_iter_hals=2, n_shift=0,
                                      unimodal=False, smooth=False,
                                      apply_left=False, apply_right=False, apply_block=False,
                                      n_bootstrap=None,
                                      tol=1e-6,
                                      max_iter=150,
                                      leverage='standard',
                                      random_state=None,
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

    F : array-like, shape (n_blocks, n_components)
        prior H

    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    update_W : boolean, default: True
        Update or keep W fixed

    update_H : boolean, default: True
        Update or keep H fixed

    update_F : boolean, default: True
        Update or keep F fixed

    fast_hals : boolean, default: True
        Use fast implementation of HALS

    n_iter_hals : integer, default: 2
        Number of HALS iterations prior to fast HALS
    
    n_shift : integer, default: 0
        max shifting in convolutional NTF

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

    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------

        Estimator (dictionary) with following entries

        W : array-like, shape (n_samples, n_components)
            Solution to the non-negative least squares problem.

        H : array-like, shape (n_features, n_components)
            Solution to the non-negative least squares problem.

        F : array-like, shape (n_blocks, n_components)
            Solution to the non-negative least squares problem.
        
        E : array-like, shape (n_samples, n_features x n_blocks)
            E is the residual tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd dimensions.
        
        V : scalar, volume occupied by W and H
        
        WB : array-like, shape (n_samples, n_components)
            Percent consistently clustered rows for each component.
            only if n_bootstrap > 0.

        HB : array-like, shape (n_features, n_components)
            Percent consistently clustered columns for each component.
            only if n_bootstrap > 0.

    Examples
    --------

    >>> import numpy as np

    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])

    >>> from sklearn.decomposition import non_negative_factorization

    >>> W, H, n_iter = non_negative_factorization(X, n_components=2, \

        random_state=0)


    References
    ----------

    Fogel

    Lin
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
    NTFUnimodal = unimodal
    NTFSmooth = smooth
    NTFLeftComponents = apply_left
    NTFRightComponents = apply_right
    NTFBlockComponents = apply_block
    if random_state is not None:
        RandomSeed = random_state
        np.random.seed(RandomSeed)

    myStatusBox = StatusBoxTqdm(verbose=LogIter)

    if (W is None) & (H is None) & (F is None):
        Mt0, Mw0, Mb0, AddMessage, ErrMessage, cancel_pressed = NTFInit(M, np.array([]), np.array([]), np.array([]), nc,
                                                                        tolerance, precision, LogIter, NTFUnimodal,
                                                                        NTFLeftComponents, NTFRightComponents,
                                                                        NTFBlockComponents, NBlocks, myStatusBox)
    else:
        if W is None:
            Mt0 = np.ones((n, nc))
        else:
            Mt0 = np.copy(W)
            
        if H is None:
            Mw0= np.ones((p_block, nc))
        else:
            Mw0 = np.copy(H)
        
        if F is None:
            Mb0 = np.ones((NBlocks, nc))
        else:
            Mb0 = np.copy(F)

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

    if update_F:
        NMFFixUserBHE = 0
    else:
        NMFFixUserBHE = 1

    Mt_conv, Mt, Mw, Mb, Mres, MtPct, MwPct, AddMessage, ErrMessage, cancel_pressed = rNTFSolve(
        M, np.array([]), Mt0, Mw0, Mb0, nc, tolerance, precision, LogIter, MaxIterations, NMFFixUserLHE, NMFFixUserRHE,
        NMFFixUserBHE, NMFAlgo, NMFRobustNRuns,
        NMFCalculateLeverage, NMFUseRobustLeverage, NTFFastHALS, NTFNIterations, NTFUnimodal, NTFSmooth,
        NTFLeftComponents, NTFRightComponents, NTFBlockComponents, NBlocks, NTFNConv, myStatusBox)

    volume = NMFDet(Mt, Mw, 1)

    for message in AddMessage:
        print(message)

    myStatusBox.close()

    estimator = {}
    if NMFRobustNRuns <= 1:
        estimator.update([('W', Mt), ('H', Mw), ('F', Mb), ('E', Mres), ('V', volume)])
    else:
        estimator.update([('W', Mt), ('H', Mw), ('F', Mb), ('E', Mres), ('V', volume), ('WB', MtPct), ('HB', MwPct)])

    return estimator
