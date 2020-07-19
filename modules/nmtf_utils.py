"""Non-negative matrix and tensor factorization utility functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20

from tkinter import *
from tkinter import ttk
from tqdm import tqdm
import math
from scipy.stats import hypergeom
from scipy.optimize import nnls
import numpy as np

EPSILON = np.finfo(np.float32).eps

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
    """Volume occupied by Left and Right factoring vectors

    Input:
        Mt: Left hand matrix
        Mw: Right hand matrix
        NMFExactDet if = 0 compute an approximate determinant in reduced space n x n or p x p
        through random sampling in the largest dimension
    Output:
        detXcells: determinant

    Reference
    ---------
    
    P. Fogel et al (2016) Applications of a Novel Clustering Approach Using Non-Negative Matrix Factorization to Environmental
        Research in Public Health Int. J. Environ. Res. Public Health 2016, 13, 509; doi:10.3390/ijerph13050509

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

def NMFGetConvexScores(Mt, Mw, Mh, flag, AddMessage):
    """Rescale scores to sum up to 1 (used with deconvolution)
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
    AddMessage.insert(len(AddMessage), 'Ncomp=' + str(nc) + ': Goodness of mixture fit = ' + str(round(R2, 2)))
    # AddMessage.insert(len(AddMessage), 'Ncomp=' + str(nc) + ': Goodness of mixture fit before adjustement = ' + str(round(R2, 2)))

    # for i in range(0, n):
    #     Mt[i, :] /= np.sum(Mt[i, :])

    return [Mt, Mw, Mh, flag, AddMessage, ErrMessage, cancel_pressed]

def percentile_exc(a, q):
    """Percentile, exclusive

    Input:
        a: Matrix
        q: Percentile
    Output:
        Percentile
    """
    return np.percentile(np.concatenate((np.array([np.min(a)]), a.flatten(), np.array([np.max(a)]))), q)

def RobustMax(V0, AddMessage, myStatusBox):
    """Robust max of column vectors

    For each column:
         = weighted mean of column elements larger than 95% percentile
        for each row, weight = specificity of the column value wrt other columns
    Input:
        V0: column vectors
    Output: Robust max by column

    Reference
    ---------

    P. Fogel et al (2016) Applications of a Novel Clustering Approach Using Non-Negative Matrix Factorization to Environmental
        Research in Public Health Int. J. Environ. Res. Public Health 2016, 13, 509; doi:10.3390/ijerph13050509

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
    """Calculate leverages

    Input:
        V: Input column vectors
        NMFUseRobustLeverage: Estimate robust through columns of V
    Output:
        Vn: Leveraged column vectors
    
    Reference
    ---------
    
    P. Fogel et al (2016) Applications of a Novel Clustering Approach Using Non-Negative Matrix Factorization to Environmental
        Research in Public Health Int. J. Environ. Res. Public Health 2016, 13, 509; doi:10.3390/ijerph13050509

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
    """Builder clusters from leverages

    """
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
    """Calculate Pvalue of each group versus cluster

    """
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
    """Calculate global significance of association with a covariate
        following multiple factorization trials
    """

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

def shift(arr, num, fill_value=EPSILON):
    """Shift a vector

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

def sparse_opt(b, alpha, two_sided):
    """Return the L2-closest vector with sparsity alpha 
    
    Input:
        b: original vector
    Output:
        x: sparse vector
    
    Reference
    ---------

    V. K. Potluru & all (2013) Block Coordinate Descent for Sparse NMF arXiv:1301.3527v2 [cs.LG]
    
    """
    m = b.size
    if two_sided is False:
        m_alpha = (np.sqrt(m) - np.linalg.norm(b, ord=1)/np.linalg.norm(b, ord=2))/(np.sqrt(m)-1)
        if (alpha == 0) or (alpha <= m_alpha):
            return b
    
    b_rank = np.argsort(-b)
    ranks = np.empty_like(b_rank)
    ranks[b_rank] = np.arange(m)
    b_norm= np.linalg.norm(b)
    a = b[b_rank] / b_norm
    k = math.sqrt(m) - alpha * (math.sqrt(m)-1)
    p0 = m
    mylambda0 = np.nan
    mu0 = np.nan
    mylambda = mylambda0
    mu = mu0
    
    for p in range(int(np.ceil(k**2)), m+1):
        mylambda0 = mylambda
        mu0 = mu
        mylambda = -np.sqrt((p * np.linalg.norm(a[0:p])**2 - np.linalg.norm(a[0:p], ord=1)**2)/(p-k**2))
        mu = -(np.linalg.norm(a[0:p], ord=1) + k*mylambda) / p
        if a[p-1] < -mu:
            p0 = p-1
            mylambda = mylambda0
            mu = mu0
            break
    
    x = np.zeros(m)
    x[0:p0] = -b_norm * (a[0:p0] + mu) / mylambda
    return x[ranks]