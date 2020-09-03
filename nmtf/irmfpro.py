""" Non-negative matrix and tensor factorization

"""

# Author: Paul Fogel

# License: MIT
# # Dec 28, '19
# Initialize progressbar
from tkinter import Tk, Frame, StringVar, Label, NONE, HORIZONTAL, Button
from tkinter import ttk
import math
import numpy as np
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from scipy.stats import hypergeom
from scipy.optimize import nnls
import sys

if not hasattr(sys, "argv"):
    sys.argv = [""]

EPSILON = np.finfo(np.float32).eps
GALDERMA_FLAG = False


class StatusBox:
    def __init__(self):
        self.root = Tk()
        self.root.title("irMF status - Python kernel")
        self.root.minsize(width=230, height=60)
        self.frame = Frame(self.root, borderwidth=6)
        self.frame.pack()
        self.var = StringVar()
        self.status = Label(self.frame, textvariable=self.var, width=60, height=1)
        self.status.pack(fill=NONE, padx=6, pady=6)
        self.pbar = ttk.Progressbar(self.frame, orient=HORIZONTAL, max=100, mode="determinate")
        self.pbar.pack(fill=NONE, padx=6, pady=6)
        Button(self.frame, text="Cancel", command=self.close_dialog).pack(fill=NONE, padx=6, pady=6)
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
        self.update_bar(delay=delay, step=100 - self.n_steps)
        self.n_steps = 0

    def update_status(self, delay=1, status=""):
        self.var.set(status)
        self.status.after(delay, lambda: self.root.quit())
        self.root.mainloop()

    def close(self):
        self.root.destroy()

    @staticmethod
    def my_print(status=""):
        print(status)


class StatusBoxTqdm:
    def __init__(self, verbose=0):
        self.log_iter = verbose
        if self.log_iter == 0:
            self.pbar = tqdm(total=100)

        self.cancel_pressed = False

    def update_bar(self, step=1):
        if self.log_iter == 0:
            self.pbar.update(n=step)

    def init_bar(self):
        if self.log_iter == 0:
            self.pbar.n = 0

    def update_status(self, status=""):
        if self.log_iter == 0:
            self.pbar.set_description(status, refresh=False)
            self.pbar.refresh()

    def close(self):
        if self.log_iter == 0:
            self.pbar.clear()
            self.pbar.close()

    def my_print(self, status=""):
        if self.log_iter == 1:
            print(status, end="\n")


def nmf_det(mt, mw, nmf_exact_det):
    """
    Volume occupied by Left and Right factoring vectors
    Input:
        m: Left hand matrix
        mw: Right hand matrix
        nmf_exact_det if = 0 compute an approximate determinant in reduced space n x n or p x p
        through random sampling in the largest dimension
    Output:
        det_xcells: determinant
    """

    n, nc = mt.shape
    p, nc = mw.shape
    nxp = n * p
    if (nmf_exact_det > 0) | (n == p):
        xcells = np.zeros((nxp, nc))
        for k in range(0, nc):
            xcells[:, k] = np.reshape(np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p)), nxp)
            norm_k = np.linalg.norm(xcells[:, k])
            if norm_k > 0:
                xcells[:, k] = xcells[:, k] / norm_k
            else:
                xcells[:, k] = 0
    else:
        if n > p:
            xcells = np.zeros((p ** 2, nc))
            the_id = np.arange(n)
            np.random.shuffle(the_id)
            the_id = the_id[0:p]
            for k in range(0, nc):
                xcells[:, k] = np.reshape(np.reshape(mt[the_id, k], (p, 1)) @ np.reshape(mw[:, k], (1, p)), p ** 2)
                norm_k = np.linalg.norm(xcells[:, k])
                if norm_k > 0:
                    xcells[:, k] = xcells[:, k] / norm_k
                else:
                    xcells[:, k] = 0
        else:
            xcells = np.zeros((n ** 2, nc))
            the_id = np.arange(p)
            np.random.shuffle(the_id)
            the_id = the_id[0:n]
            for k in range(0, nc):
                xcells[:, k] = np.reshape(np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[the_id, k], (1, n)), n ** 2)
                norm_k = np.linalg.norm(xcells[:, k])
                if norm_k > 0:
                    xcells[:, k] = xcells[:, k] / norm_k
                else:
                    xcells[:, k] = 0

    det_xcells = np.linalg.det(xcells.T @ xcells)
    return det_xcells


def proj_grad(v, vmis, w, hinit, nmf_algo, lambdax, tol, max_iterations, nmf_priors):
    """
    Projected gradient
    Code and notations adapted from Matlab code, Chih-Jen Lin
    Input:
        v: Input matrix
        vmis: Define missing values (0 = missing cell, 1 = real cell)
        w: Left factoring vectors (fixed)
        hinit: Right factoring vectors (initial values)
        nmf_algo: =1,3: Divergence; =2,4: Least squares;
        lambdax: Sparseness parameter
            =-1: no penalty
            < 0: Target percent zeroed rows in h
            > 0: Current penalty
        tol: Tolerance
        max_iterations: max number of iterations to achieve norm(projected gradient) < tol
        nmf_priors: Elements in h that should be updated (others remain 0)
    Output:
        h: Estimated right factoring vectors
        tol: Current level of the tolerance
        lambdax: Current level of the penalty
    """
    h = hinit
    # noinspection PyBroadException
    try:
        n_nmf_priors, nc = nmf_priors.shape
    except BaseException:
        n_nmf_priors = 0

    n_vmis = vmis.shape[0]
    n, p = np.shape(v)
    n, nc = np.shape(w)
    alpha = 1
    wt_w = None
    wt_wh = None
    wt_v = None
    h0 = None

    if (nmf_algo == 2) or (nmf_algo == 4):
        beta = 0.1
        if n_vmis > 0:
            wt_v = w.T @ (v * vmis)
        else:
            wt_v = w.T @ v
            wt_w = w.T @ w
    else:
        beta = 0.1
        if n_vmis > 0:
            wt_wh = w.T @ vmis
        else:
            wt_wh = w.T @ np.ones((n, p))

    if (lambdax < 0) & (lambdax != -1):
        h0 = h

    restart = True
    while restart:
        for i_iter in range(1, max_iterations + 1):
            add_penalty = 0
            if lambdax != -1:
                add_penalty = 1

            if (nmf_algo == 2) or (nmf_algo == 4):
                if n_vmis > 0:
                    wt_wh = w.T @ ((w @ h) * vmis)
                else:
                    wt_wh = wt_w @ h
            else:
                if n_vmis > 0:
                    wt_v = w.T @ ((v * vmis) / (w @ h))
                else:
                    wt_v = w.T @ (v / (w @ h))

            if lambdax > 0:
                grad = wt_wh - wt_v + lambdax
            else:
                grad = wt_wh - wt_v

            projgrad = np.linalg.norm(grad[(grad < 0) | (h > 0)])

            if projgrad >= tol:
                # search step size
                for inner_iter in range(1, 21):
                    hn = h - alpha * grad
                    hn[np.where(hn < 0)] = 0
                    if n_nmf_priors > 0:
                        hn = hn * nmf_priors

                    d = hn - h
                    gradd = np.sum(grad * d)
                    if (nmf_algo == 2) or (nmf_algo == 4):
                        if n_vmis > 0:
                            d_qd = np.sum((w.T @ ((w @ d) * vmis)) * d)
                        else:
                            d_qd = np.sum((wt_w @ d) * d)
                    else:
                        if n_vmis > 0:
                            d_qd = np.sum((w.T @ ((w @ d) * (vmis / (w @ h)))) * d)
                        else:
                            d_qd = np.sum((w.T @ ((w @ d) / (w @ h))) * d)

                    suff_decr = 0.99 * gradd + 0.5 * d_qd < 0
                    decr_alpha = False
                    hp = None
                    if inner_iter == 1:
                        decr_alpha = not suff_decr
                        hp = h

                    if decr_alpha:
                        if suff_decr:
                            h = hn
                            break
                        else:
                            alpha = alpha * beta
                    else:
                        if (suff_decr is False) | (np.where(hp != hn)[0].size == 0):
                            h = hp
                            break
                        else:
                            alpha = alpha / beta
                            # hp = hn  # Unused ?
                # End for (inner_iter

                if (lambdax < 0) & add_penalty:
                    # Initialize penalty
                    lambdax = percentile_exc(h[np.where(h > 0)], -lambdax * 100)
                    h = h0
                    alpha = 1
            else:  # projgrad < tol
                if (i_iter == 1) & (projgrad > 0):
                    tol /= 10
                else:
                    restart = False

                break
            #       End if projgrad

            if i_iter == max_iterations:
                restart = False
        #   End For i_iter

    h = h.T
    return [h, tol, lambdax]


def proj_grad_kernel(kernel, v, vmis, w, hinit, nmf_algo, tol, max_iterations, nmf_priors):
    """
        Projected gradient, kernel version
        Code and notations adapted from Matlab code, Chih-Jen Lin
        Input:
            kernel: kernel used
            v: Input matrix
            vmis: Define missing values (0 = missing cell, 1 = real cell)
            w: Left factoring vectors (fixed)
            hinit: Right factoring vectors (initial values)
            nmf_algo: =1,3: Divergence; =2,4: Least squares;
            tol: Tolerance
            max_iterations: max number of iterations to achieve norm(projected gradient) < tol
            nmf_priors: Elements in h that should be updated (others remain 0)
        Output:
            h: Estimated right factoring vectors
            tol: Current level of the tolerance
        """
    h = hinit.T
    # noinspection PyBroadException
    try:
        n_nmf_priors, nc = nmf_priors.shape
    except BaseException:
        n_nmf_priors = 0

    n_vmis = vmis.shape[0]
    n, p = np.shape(v)
    p, nc = np.shape(w)
    alpha = 1
    wt_w = None
    wt_wh = None
    wt_v = None
    hp = None
    vw = v @ w

    if (nmf_algo == 2) or (nmf_algo == 4):
        beta = 0.1
        if n_vmis > 0:
            wt_v = vw.T @ (v * vmis)
        else:
            wt_v = w.T @ kernel
            wt_w = w.T @ kernel @ w
    else:
        beta = 0.1
        max_iterations = round(max_iterations / 10)
        if n_vmis > 0:
            wt_wh = vw.T @ vmis
        else:
            wt_wh = vw.T @ np.ones((n, p))

    restart = True
    while restart:
        for i_iter in range(1, max_iterations + 1):
            if (nmf_algo == 2) or (nmf_algo == 4):
                if n_vmis > 0:
                    wt_wh = vw.T @ ((vw @ h) * vmis)
                else:
                    wt_wh = wt_w @ h
            else:
                if n_vmis > 0:
                    wt_v = vw.T @ ((v * vmis) / (vw @ h))
                else:
                    wt_v = vw.T @ (v / (vw @ h))

            grad = wt_wh - wt_v
            projgrad = np.linalg.norm(grad[(grad < 0) | (h > 0)])
            if projgrad >= tol:
                # search step size
                for inner_iter in range(1, 21):
                    hn = h - alpha * grad
                    hn[np.where(hn < 0)] = 0
                    if n_nmf_priors > 0:
                        hn = hn * nmf_priors

                    d = hn - h
                    gradd = np.sum(grad * d)
                    if (nmf_algo == 2) or (nmf_algo == 4):
                        if n_vmis > 0:
                            d_qd = np.sum((vw.T @ ((vw @ d) * vmis)) * d)
                        else:
                            d_qd = np.sum((wt_w @ d) * d)
                    else:
                        if n_vmis > 0:
                            d_qd = np.sum((vw.T @ ((vw @ d) * (vmis / (vw @ h)))) * d)
                        else:
                            d_qd = np.sum((vw.T @ ((vw @ d) / (vw @ h))) * d)

                    suff_decr = 0.99 * gradd + 0.5 * d_qd < 0
                    decr_alpha = False
                    if inner_iter == 1:
                        decr_alpha = not suff_decr
                        hp = h

                    if decr_alpha:
                        if suff_decr:
                            h = hn
                            break
                        else:
                            alpha = alpha * beta
                    else:
                        if (suff_decr is False) | (np.where(hp != hn)[0].size == 0):
                            h = hp
                            break
                        else:
                            alpha = alpha / beta
                            hp = hn
                # End for (inner_iter
            else:  # projgrad < tol
                if i_iter == 1:
                    tol /= 10
                else:
                    restart = False

                break
            #       End if projgrad

            if i_iter == max_iterations:
                restart = False
        #   End For i_iter

    h = h.T
    return [h, tol]


def apply_kernel(m, nmf_kernel, mt, mw):
    """
    Calculate kernel
    Input:
        m: Input matrix
        nmf_kernel: Type of kernel
            =-1: linear
            = 2: quadratic
            = 3: radiant
        m: Left factoring matrix
        mw: Right factoring matrix
    Output:
        kernel
    """

    n, p = m.shape
    # noinspection PyBroadException
    try:
        p, nc = mw.shape
    except BaseException:
        nc = 0

    if nmf_kernel == 1:
        kernel = m.T @ m
    elif nmf_kernel == 2:
        kernel = (np.identity(p) + m.T @ m) ** 2
    elif nmf_kernel == 3:
        kernel = np.identity(p)
        # Estimate sigma2
        sigma2 = 0

        for k1 in range(1, nc):
            for k2 in range(0, k1):
                sigma2 = max(sigma2, np.linalg.norm(mt[:, k1] - mt[:, k2]) ** 2)

        sigma2 /= nc
        for j1 in range(1, p):
            for j2 in range(0, j1):
                kernel[j1, j2] = math.exp(-np.linalg.norm(m[:, j1] - m[:, j2]) ** 2 / sigma2)
                kernel[j2, j1] = kernel[j1, j2]
    else:
        raise ValueError(f"nmf_kernel was set to {nmf_kernel}, but can only be 1 2 or 3.")

    return kernel


def get_convex_scores(mt, mw, mh, flag, add_message):
    """
    Reweigh scores to sum up to 1
    Input:
        m: Left factoring matrix
        mw: Right factoring matrix
        flag:  Current value
    Output:

       m: Left factoring matrix
        mw: Right factoring matrix
        flag: += 1: Negative weights found
    """
    err_message = ""
    cancel_pressed = 0

    n, nc = mt.shape
    n_mh = mh.shape[0]
    # noinspection PyBroadException
    try:
        malpha = np.linalg.inv(mt.T @ mt) @ (mt.T @ np.ones(n))
    except BaseException:
        malpha = np.linalg.pinv(mt.T @ mt) @ (mt.T @ np.ones(n))

    if np.where(malpha < 0)[0].size > 0:
        flag += 1
        malpha = nnls(mt, np.ones(n))[0]

    n_zeroed = 0
    for k in range(0, nc):
        mt[:, k] *= malpha[k]
        if n_mh > 0:
            mh[:, k] *= malpha[k]
        if malpha[k] > 0:
            mw[:, k] /= malpha[k]
        else:
            n_zeroed += 1

    if n_zeroed > 0:
        add_message.insert(len(add_message), "Ncomp=" + str(nc) + ": " + str(n_zeroed) + " components were zeroed")

    # Goodness of fit
    r2 = 1 - np.linalg.norm(np.sum(mt.T, axis=0).T - np.ones(n)) ** 2 / n
    add_message.insert(
        len(add_message), "Ncomp=" + str(nc) + ": Goodness of mixture fit before adjustement = " + str(round(r2, 2))
    )

    for i in range(0, n):
        mt[i, :] /= np.sum(mt[i, :])

    return [mt, mw, mh, flag, add_message, err_message, cancel_pressed]


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


def robust_max(v0, add_message, my_status_box):
    """
    Robust max of column vectors
    For each column:
         = weighted mean of column elements larger than 95% percentile
        for each row, weight = specificity of the column value wrt other columns
    Input:
        v0: column vectors
    Output: Robust max by column
    """
    err_message = ""
    cancel_pressed = 0

    v = v0.copy()
    n, nc = v.shape
    lnnc_i = None
    nc_i = None
    if nc > 1:
        nc_i = 1 / nc
        lnnc_i = 1 / math.log(nc)

    ind = max(math.ceil(n * 0.05) - 1, min(n - 1, 2))
    scale = np.max(v, axis=0)
    for k in range(0, nc):
        v[:, k] = v[:, k] / scale[k]

    rob_max = np.max(v, axis=0)
    rob_max0 = 1e99 * np.ones(nc)
    i_iter = 0
    max_iterations = 100
    pbar_step = 100 / max_iterations
    my_status_box.init_bar(delay=1)

    while ((np.linalg.norm(rob_max - rob_max0) / np.linalg.norm(rob_max)) ** 2 > 1e-6) & (i_iter < max_iterations):
        for k in range(0, nc):
            v = v[np.argsort(-v[:, k]), :]
            if nc > 1:
                den = np.repeat(np.sum(v, axis=1), nc).reshape((n, nc))
                den[den == 0] = 1.0e-10
                prob = v / den
                prob[prob == 0] = 1.0e-10
                specificity = np.ones(n) + np.sum(prob * np.log(prob), axis=1) * lnnc_i
                specificity[prob[:, k] < nc_i] = 0
            else:
                specificity = np.ones(n)

            specificity[ind:n] = 0
            rob_max0[k] = rob_max[k]
            rob_max[k] = np.sum(v[:, k] * specificity) / np.sum(specificity)
            v[v[:, k] > rob_max[k], k] = rob_max[k]

        my_status_box.update_bar(delay=1, step=pbar_step)
        if my_status_box.cancel_pressed:
            cancel_pressed = 1
            return rob_max * scale, add_message, err_message, cancel_pressed

        i_iter += 1

    if i_iter == max_iterations:
        add_message.insert(
            len(add_message), "Warning: Max iterations reached while calculating robust max (N = " + str(n) + ")."
        )

    return [rob_max * scale, add_message, err_message, cancel_pressed]


def leverage(v, nmf_use_robust_leverage, add_message, my_status_box):
    """
    Calculate leverages
    Input:
        v: Input column vectors
        nmf_use_robust_leverage: Estimate robust through columns of v
    Output:
        vn: Leveraged column vectors
    """

    err_message = ""
    cancel_pressed = 0

    n, nc = v.shape
    vn = np.zeros((n, nc))
    vr = np.zeros((n, nc))
    if nmf_use_robust_leverage > 0:
        max_v, add_message, err_message, cancel_pressed = robust_max(v, add_message, my_status_box)
        if cancel_pressed == 1:
            return vn, add_message, err_message, cancel_pressed
    else:
        max_v = np.max(v, axis=0)

    pbar_step = 100 / nc
    my_status_box.init_bar(delay=1)
    for k in range(0, nc):
        vr[v[:, k] > 0, k] = 1
        vn[:, k] = max_v[k] - v[:, k]
        vn[vn[:, k] < 0, k] = 0
        vn[:, k] = vn[:, k] ** 2
        for k2 in range(0, nc):
            if k2 != k:
                vn[:, k] = vn[:, k] + v[:, k2] ** 2

        status = "the_leverage: Comp " + str(k + 1)
        my_status_box.update_status(delay=1, status=status)
        my_status_box.update_bar(delay=1, step=pbar_step)
        if my_status_box.cancel_pressed:
            cancel_pressed = 1
            return vn, add_message, err_message, cancel_pressed

    vn = 10 ** (-vn / (2 * np.mean(vn))) * vr
    return [vn, add_message, err_message, cancel_pressed]


def build_clusters(
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
):
    n_blocks = int(n_blocks)
    my_status_box.update_status(delay=1, status="Build clusters...")
    err_message = ""
    cancel_pressed = 0
    n, nc = np.shape(mt)
    p = np.shape(mw)[0]
    mbn = None
    mt_s = None
    mw_s = None
    if nmf_algo >= 5:
        block_clust = np.zeros(n_blocks)
    else:
        block_clust = np.array([])
        mbn = np.array([])

    r_ct = np.zeros(n)
    r_cw = np.zeros(p)
    n_ct = np.zeros(nc)
    n_cw = np.zeros(n_blocks * nc)
    row_clust = np.zeros(n)
    col_clust = np.zeros(p)
    ilast = 0
    jlast = 0
    if nmf_calculate_leverage == 1:
        my_status_box.update_status(delay=1, status="Leverages - Left components...")
        mtn, add_message, err_message, cancel_pressed = leverage(
            mt, nmf_use_robust_leverage, add_message, my_status_box
        )
        my_status_box.update_status(delay=1, status="Leverages - Right components...")
        mwn, add_message, err_message, cancel_pressed = leverage(
            mw, nmf_use_robust_leverage, add_message, my_status_box
        )
        if nmf_algo >= 5:
            my_status_box.update_status(delay=1, status="Leverages - Block components...")
            mbn, add_message, err_message, cancel_pressed = leverage(
                mb, nmf_use_robust_leverage, add_message, my_status_box
            )
    else:
        mtn = mt
        mwn = mw
        if nmf_algo >= 5:
            mbn = mb

    if nmf_algo >= 5:
        for i_block in range(0, n_blocks):
            if nc > 1:
                block_clust[i_block] = np.argmax(mbn[i_block, :]) + 1
            else:
                block_clust[i_block] = 1

    for i in range(0, n):
        if nc > 1:
            if (isinstance(mt_pct, np.ndarray)) & (nmf_robust_cluster_by_stability > 0):
                row_clust[i] = np.argmax(mt_pct[i, :]) + 1
            else:
                row_clust[i] = np.argmax(mtn[i, :]) + 1
        else:
            row_clust[i] = 1

    for j in range(0, p):
        if nc > 1:
            if (isinstance(mw_pct, np.ndarray)) & (nmf_robust_cluster_by_stability > 0):
                col_clust[j] = np.argmax(mw_pct[j, :]) + 1
            else:
                col_clust[j] = np.argmax(mwn[j, :]) + 1
        else:
            col_clust[j] = 1

    if (cell_plot_ordered_clusters == 1) & (nc >= 3):
        mt_s = np.zeros(n)
        mw_s = np.zeros(p)
        for i in range(0, n):
            if row_clust[i] == 1:
                mt_s[i] = sum(k * mtn[i, k] for k in range(0, 2)) / max(sum(mtn[i, k] for k in range(0, 2)), 1.0e-10)
            elif row_clust[i] == nc:
                mt_s[i] = sum(k * mtn[i, k] for k in range(nc - 2, nc)) / max(
                    sum(mtn[i, k] for k in range(nc - 2, nc)), 1.0e-10
                )
            else:
                mt_s[i] = sum(k * mtn[i, k] for k in range(int(row_clust[i] - 2), int(row_clust[i] + 1))) / max(
                    sum(mtn[i, k] for k in range(int(row_clust[i] - 2), int(row_clust[i] + 1))), 1.0e-10
                )

        for j in range(0, p):
            if col_clust[j] == 1:
                mw_s[j] = sum(k * mwn[j, k] for k in range(0, 2)) / max(sum(mwn[j, k] for k in range(0, 2)), 1.0e-10)
            elif col_clust[j] == nc:
                mw_s[j] = sum(k * mwn[j, k] for k in range(nc - 2, nc)) / max(
                    sum(mwn[j, k] for k in range(nc - 2, nc)), 1.0e-10
                )
            else:
                mw_s[j] = sum(k * mwn[j, k] for k in range(int(col_clust[j] - 2), int(col_clust[j] + 1))) / max(
                    sum(mwn[j, k] for k in range(int(col_clust[j] - 2), int(col_clust[j] + 1))), 1.0e-10
                )

    for k in range(0, nc):
        mindex1 = np.where(row_clust == k + 1)[0]
        if len(mindex1) > 0:
            if len(mindex1) == 1:
                mindex = (mindex1,)
            elif (nc == 2) & (k == 1):
                mindex = mindex1[np.argsort(mtn[mindex1, k])]
            elif (cell_plot_ordered_clusters == 1) & (nc >= 3):
                mindex = mindex1[np.argsort(mt_s[mindex1])]
            else:
                mindex = mindex1[np.argsort(-mtn[mindex1, k])]

            r_ct[ilast: len(mindex) + ilast] = mindex
            ilast += len(mindex)

        n_ct[k] = ilast

    for i_block in range(0, n_blocks):
        j2 = None
        if i_block == 0:
            j1 = 0
            j2 = int(abs(blk_size[i_block]))
        else:
            j1 = j2
            j2 += int(abs(blk_size[i_block]))

        for k in range(0, nc):
            mindex2 = np.where(col_clust[j1:j2] == k + 1)[0]
            if len(mindex2) > 0:
                mindex2 = mindex2 + j1
                if len(mindex2) == 1:
                    mindex = mindex2
                elif (nc == 2) & (k == 1):
                    mindex = mindex2[np.argsort(mwn[mindex2, k])]
                elif (cell_plot_ordered_clusters == 1) & (nc >= 3):
                    mindex = mindex2[np.argsort(mw_s[mindex2])]
                else:
                    mindex = mindex2[np.argsort(-mwn[mindex2, k])]

                r_cw[jlast: len(mindex) + jlast] = mindex
                jlast += len(mindex)

            n_cw[i_block * nc + k] = jlast

    return [
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
    ]


def cluster_pvalues(cluster_size, nb_groups, mt, r_ct, n_ct, row_groups, list_groups, ngroup):
    n, nc = mt.shape
    cluster_size = cluster_size.astype(np.int)
    nb_groups = int(nb_groups)
    r_ct = r_ct.astype(np.int)
    n_ct = n_ct.astype(np.int)
    cluster_size = np.reshape(cluster_size, nc)
    r_ct = np.reshape(r_ct, (n,))
    n_ct = np.reshape(n_ct, (nc,))
    row_groups = np.reshape(row_groups, (n,))

    cluster_group = np.zeros(nc)
    cluster_prob = np.zeros(nc)
    cluster_ngroup = np.zeros((nc, nb_groups))
    cluster_n_wgroup = np.zeros((nc, nb_groups))
    prun = 0

    for k in range(0, nc):
        if cluster_size[k] > 0:
            # Find main group (only if clustersize>2)
            kfound0 = 0
            for i_group in range(0, nb_groups):
                if k == 0:
                    mx = np.where(row_groups[r_ct[0: n_ct[0]]] == list_groups[i_group])[0]
                    if len(mx) >= 1:
                        cluster_n_wgroup[k, i_group] = np.sum(mt[r_ct[0: n_ct[0]][mx], k])
                        cluster_ngroup[k, i_group] = len(mx)
                else:
                    mx = np.where(row_groups[r_ct[n_ct[k - 1]: n_ct[k]]] == list_groups[i_group])[0]
                    if len(mx) >= 1:
                        cluster_n_wgroup[k, i_group] = np.sum(mt[r_ct[n_ct[k - 1]: n_ct[k]][mx], k])
                        cluster_ngroup[k, i_group] = len(mx)

                if cluster_ngroup[k, i_group] > kfound0:
                    kfound0 = cluster_ngroup[k, i_group]
                    cluster_group[k] = i_group

            sum_cluster_n_wgroup = sum(cluster_n_wgroup[k, :])
            for i_group in range(0, nb_groups):
                cluster_n_wgroup[k, i_group] = cluster_size[k] * cluster_n_wgroup[k, i_group] / sum_cluster_n_wgroup

        else:
            for i_group in range(0, nb_groups):
                cluster_ngroup[k, i_group] = 0
                cluster_n_wgroup[k, i_group] = 0

            cluster_group[k] = 1

    for k in range(0, nc):
        if cluster_size[k] > 2:
            cluster_prob[k] = hypergeom.sf(
                cluster_ngroup[k, int(cluster_group[k])], n, ngroup[int(cluster_group[k])], cluster_size[k], loc=0
            ) + hypergeom.pmf(
                cluster_ngroup[k, int(cluster_group[k])], n, ngroup[int(cluster_group[k])], cluster_size[k], loc=0
            )
        else:
            cluster_prob[k] = 1

    for k in range(0, nc):
        for i_group in range(0, nb_groups):
            if cluster_n_wgroup[k, i_group]:
                prun += cluster_n_wgroup[k, i_group] * math.log(
                    cluster_n_wgroup[k, i_group] / (cluster_size[k] * ngroup[i_group] / n)
                )

    return [prun, cluster_group, cluster_prob, cluster_ngroup, cluster_n_wgroup]


def global_sign(nrun, nb_groups, mt, r_ct, n_ct, row_groups, list_groups, ngroup, my_status_box):
    n, nc = mt.shape
    nrun = int(nrun)
    nb_groups = int(nb_groups)
    r_ct = r_ct.astype(np.int)
    n_ct = n_ct.astype(np.int)
    cluster_size = np.zeros(nc)
    r_ct = np.reshape(r_ct, n)
    n_ct = np.reshape(n_ct, nc)
    cancel_pressed = 0
    prun = cluster_prob = cluster_group = cluster_ngroup = prun0 = None
    row_groups0 = cluster_prob0 = cluster_group0 = cluster_ngroup0 = None
    for k in range(0, nc):
        if k == 0:
            cluster_size[k] = n_ct[0]
        else:
            cluster_size[k] = n_ct[k] - n_ct[k - 1]

    if nb_groups > 1:
        row_groups = np.reshape(row_groups, (n,))
        step_iter = np.round(nrun / 10)
        pbar_step = 10
        pglob = 1
        for irun in range(0, nrun):
            if irun % step_iter == 0:
                my_status_box.update_status(
                    delay=1, status="Calculating global significance: " + str(irun) + " / " + str(nrun)
                )
                my_status_box.update_bar(delay=1, step=pbar_step)
                if my_status_box.cancel_pressed:
                    cancel_pressed = 1
                    return [cluster_size, pglob, prun, cluster_prob, cluster_group, cluster_ngroup, cancel_pressed]

            prun, cluster_group, cluster_prob, cluster_ngroup, cluster_n_wgroup = cluster_pvalues(
                cluster_size, nb_groups, mt, r_ct, n_ct, row_groups, list_groups, ngroup
            )
            if irun == 0:
                cluster_prob0 = np.copy(cluster_prob)
                cluster_group0 = np.copy(cluster_group)
                cluster_ngroup0 = np.copy(cluster_ngroup)
                row_groups0 = np.copy(row_groups)
                prun0 = prun
            else:
                if prun >= prun0:
                    pglob += 1

            if irun < nrun - 1:
                # permute row groups
                # boot = np.random.permutation  # Unused ?
                row_groups = row_groups0[np.random.permutation(n)]
            else:
                # Restore
                cluster_prob = cluster_prob0
                cluster_group = cluster_group0
                cluster_ngroup = cluster_ngroup0
                row_groups = row_groups0
                prun = prun0
                pglob /= nrun
    else:
        pglob = np.NaN
        prun = np.NaN
        cluster_prob = np.array([])
        cluster_group = np.array([])
        cluster_ngroup = np.array([])

    return [cluster_size, pglob, prun, cluster_prob, cluster_group, cluster_ngroup, cancel_pressed]


def nmf_init(m, mmis, mt0, mw0, nc, tolerance, log_iter, my_status_box):
    """
    NMF initialization using NNSVD (Boutsisdis)
    Input:
        m: Input matrix
        mmis: Define missing values (0 = missing cell, 1 = real cell)
        mt0: Initial left hand matrix (may be empty)
        mw0: Initial right hand matrix (may be empty)
        nc: NMF rank
    Output:
        m: Left hand matrix
        mw: Right hand matrix
    """

    n, p = m.shape
    mmis = mmis.astype(np.int)
    n_mmis = mmis.shape[0]
    if n_mmis == 0:
        the_id = np.where(np.isnan(m) is True)
        n_mmis = the_id[0].size
        if n_mmis > 0:
            mmis = np.isnan(m) is False
            mmis = mmis.astype(np.int)
            m[mmis == 0] = 0

    nc = int(nc)
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    msvd = None
    if (mt.shape[0] == 0) or (mw.shape[0] == 0):
        if (n >= nc) and (p >= nc):
            msvd = m
        else:
            if n < nc:
                # Replicate rows until > nc
                msvd = np.repeat(m, np.ceil(nc / n), axis=0)

            if p < nc:
                # Replicate rows until > nc
                msvd = np.repeat(msvd, np.ceil(nc / p), axis=1)

        if n_mmis == 0:
            t, d, w = randomized_svd(msvd, n_components=nc, n_iter="auto", random_state=None)
            mt = t
            mw = w.T
        else:
            mt, d, mw, mmis, mmsr, mmsr2, add_message, err_message, cancel_pressed = r_svd_solve(
                msvd, mmis, nc, tolerance, log_iter, 0, "", 200, 1, 1, 1, my_status_box
            )

        if n < nc:
            mt = np.concatenate((mt, np.ones((n, nc - n))))

        if p < nc:
            mw = np.concatenate((mw, np.ones((p, nc - p))))

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

    return [mt, mw]


def nmf_reweigh(m, mt, nmf_priors, add_message):
    """
    Cancel variables that load on more than one component
    Input:
         m: Input matrix
         m: Left hand matrix
         nmf_priors: priors on right hand matrix
    Output:
         nmf_priors: updated priors
    """
    err_message = ""
    n, p = m.shape
    n_nmf_priors, nc = nmf_priors.shape
    nmf_priors[nmf_priors > 0] = 1
    the_id = np.where(np.sum(nmf_priors, axis=1) > 1)
    n_id = the_id[0].shape[0]
    if n_id == p:
        err_message = "Error! All priors are ambiguous.\nYou may uncheck the option in tab irMF+."
        return [nmf_priors, add_message, err_message]

    nmf_priors[the_id, :] = 0
    mweight = np.zeros((p, nc))
    for k in range(0, nc):
        the_id = np.where(nmf_priors[:, k] > 0)
        pk = the_id[0].shape[0]
        if pk == 0:
            err_message = "Error! Null column in NMF priors (" + str(k + 1) + ", pre outlier filtering)"
            return [nmf_priors, add_message, err_message]

        mc = np.zeros((n, p))

        # Exclude variables with outliers
        n_inter_quart = 1.5
        for j in range(0, pk):
            quart75 = percentile_exc(m[:, the_id[0][j]], 75)
            quart25 = percentile_exc(m[:, the_id[0][j]], 25)
            inter_quart = quart75 - quart25
            max_bound = quart75 + n_inter_quart * inter_quart
            min_bound = quart25 - n_inter_quart * inter_quart
            if np.where((m[:, the_id[0][j]] < min_bound) | (m[:, the_id[0][j]] > max_bound))[0].shape[0] == 1:
                nmf_priors[the_id[0][j], k] = 0

        the_id = np.where(nmf_priors[:, k] > 0)
        pk = the_id[0].shape[0]
        if pk == 0:
            err_message = "Error! Null column in NMF priors (" + str(k + 1) + ", post outlier filtering)"
            return [nmf_priors, add_message, err_message]

        # Characterize clusters by skewness direction
        mtc = mt[:, k] - np.mean(mt[:, k])
        std = math.sqrt(np.mean(mtc ** 2))
        skewness = np.mean((mtc / std) ** 3) * math.sqrt(n * (n - 1)) / (n - 2)

        # Scale columns and initialized weights
        for j in range(0, pk):
            m[:, the_id[0][j]] /= np.sum(m[:, the_id[0][j]])
            mc[:, the_id[0][j]] = m[:, the_id[0][j]] - np.mean(m[:, the_id[0][j]])
            std = math.sqrt(np.mean(mc[:, the_id[0][j]] ** 2))
            mweight[the_id[0][j], k] = np.mean((mc[:, the_id[0][j]] / std) ** 3) * math.sqrt(n * (n - 1)) / (n - 2)

        if skewness < 0:
            # Negative skewness => Component identifiable through small
            # proportions
            mweight[mweight[:, k] > 0, k] = 0
            mweight = -mweight
            i_dneg = np.where(mweight[:, k] > 0)
            nneg = i_dneg[0].shape[0]
            if nneg == 0:
                err_message = "Error! No marker variable found in component " + str(k + 1)
                return [nmf_priors, add_message, err_message]

            add_message.insert(
                len(add_message),
                "Component " + str(k + 1) + ": compositions are negatively skewed (" + str(nneg) + " active variables)",
            )
        else:
            # Positive skewness => Component identifiable through large
            # proportions
            mweight[mweight[:, k] < 0, k] = 0
            i_dpos = np.where(mweight[:, k] > 0)
            npos = i_dpos[0].shape[0]
            if npos == 0:
                err_message = "Error! No marker variable found in component " + str(k + 1)
                return [nmf_priors, add_message, err_message]

            add_message.insert(
                len(add_message),
                "Component " + str(k + 1) + ": compositions are positively skewed (" + str(npos) + " active variables)",
            )

        # Logistic transform of non-zero weights
        id2 = np.where(mweight[:, k] > 0)
        n_id2 = id2[0].shape[0]
        if n_id2 > 1:
            mu = np.mean(mweight[id2[0], k])
            std = np.std(mweight[id2[0], k])
            mweight[id2[0], k] = (mweight[id2[0], k] - mu) / std
            mweight[id2[0], k] = np.ones(n_id2) - np.ones(n_id2) / (
                np.ones(n_id2) + np.exp(2 * (mweight[id2[0], k] - percentile_exc(mweight[id2[0], k], 90)))
            )
        else:
            mweight[id2[0], k] = 1

        # ReWeigh columns
        m[:, the_id[0]] = m[:, the_id[0]] * mweight[the_id[0], k].T

        # Update NMF priors (cancel columns with 0 weight & replace non zero
        # values by 1)
        nmf_priors[the_id[0], k] = nmf_priors[the_id[0], k] * mweight[the_id[0], k]
        the_id = np.where(nmf_priors[:, k] > 0)
        if the_id[0].shape[0] > 0:
            nmf_priors[the_id[0], k] = 1
            # Scale parts
            m[:, the_id[0]] /= np.linalg.norm(m[:, the_id[0]])
        else:
            err_message = "Error! Null column in NMF priors (" + str(k + 1) + ", post cancelling 0-weight columns)"
            return [nmf_priors, add_message, err_message]

    return [nmf_priors, add_message, err_message]


def nmf_solve(
        m,
        mmis,
        mt0,
        mw0,
        nc,
        tolerance,
        precision,
        log_iter,
        status0,
        max_iterations,
        nmf_algo,
        nmf_fix_user_lhe,
        nmf_fix_user_rhe,
        nmf_max_interm,
        nmf_max_iter_proj,
        nmf_sparse_level,
        nmf_find_parts,
        nmf_find_centroids,
        nmf_kernel,
        nmf_reweigh_columns,
        nmf_priors,
        flag_nonconvex,
        add_message,
        my_status_box,
):
    """
    Estimate left and right hand matrices
    Input:
         m: Input matrix
         mmis: Define missing values (0 = missing cell, 1 = real cell)
         mt0: Initial left hand matrix
         mw0: Initial right hand matrix
         nc: NMF rank
         tolerance: Convergence threshold
         precision: Replace 0-value in multiplication rules
         log_iter: Log results through iterations
         status0: Initial displayed status to be updated during iterations
         max_iterations: Max iterations
         nmf_algo: =1,3: Divergence; =2,4: Least squares;
         nmf_fix_user_lhe: = 1 => fixed left hand matrix columns
         nmf_fix_user_rhe: = 1 => fixed  right hand matrix columns
         nmf_max_interm: Max iterations for warmup multiplication rules
         nmf_max_iter_proj: Max iterations for projected gradient
         nmf_sparse_level: Requested sparsity in terms of relative number of rows with 0 values in right hand matrix
         nmf_find_parts: Enforce convexity on left hand matrix
         nmf_find_centroids: Enforce convexity on right hand matrix
         nmf_kernel: Type of kernel used; 1: linear; 2: quadratic; 3: radial
         nmf_reweigh_columns: Reweigh columns in 2nd step of parts-based NMF
         nmf_priors: Priors on right hand matrix
         flag_nonconvex: Non-convexity flag on left hand matrix
    Output:
         m: Left hand matrix
         mev: Scaling matrix
         mw: Right hand matrix
         diff: objective cost
         mh: Convexity matrix
         nmf_priors: Updated priors on right hand matrix
         flag_nonconvex: Updated non-convexity flag on left hand matrix
    """
    err_message = ""
    cancel_pressed = 0
    kernel = None
    trace_kernel = None
    tol_mw = None
    tol_mh = None
    tol_mt = None
    the_in = None
    ip = None
    diff0 = None

    n, p = m.shape
    n_mmis = mmis.shape[0]
    # noinspection PyBroadException
    try:
        n_nmf_priors, nc = nmf_priors.shape
    except BaseException:
        n_nmf_priors = 0

    nc = int(nc)
    nxp = int(n * p)
    mev = np.ones(nc)
    mh = np.array([])
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    diff = 1.0e99

    # Add weights
    if n_nmf_priors > 0:
        if nmf_reweigh_columns > 0:
            # a local copy of m will be updated
            m = np.copy(m)
            nmf_priors, add_message, err_message = nmf_reweigh(m, mt, nmf_priors, add_message)
            if err_message != "":
                return [mt, mev, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed]
        else:
            nmf_priors[np.where(nmf_priors > 0)] = 1

    if (nmf_find_parts > 0) & (nmf_fix_user_lhe > 0):
        nmf_find_parts = 0

    if (nmf_find_centroids > 0) & (nmf_fix_user_rhe > 0):
        nmf_find_centroids = 0
        nmf_kernel = 1

    if (nmf_find_centroids > 0) & (nmf_kernel > 1):
        if n_mmis > 0:
            nmf_kernel = 1
            add_message.insert(len(add_message), "Warning: Non linear kernel canceled due to missing values.")

        if (nmf_algo == 1) or (nmf_algo == 3):
            nmf_kernel = 1
            add_message.insert(len(add_message), "Warning: Non linear kernel canceled due to divergence minimization.")

    if n_nmf_priors > 0:
        mw_user = nmf_priors
        for k in range(0, nc):
            if (nmf_algo == 2) | (nmf_algo == 4):
                mw[:, k] = mw_user[:, k] / np.linalg.norm(mw_user[:, k])
            else:
                mw[:, k] = mw_user[:, k] / np.sum(mw_user[:, k])

    mult_or_pgrad = 1  # Start with Lee-Seung mult rules
    # nmf_max_interm Li-Seung iterations initialize projected gradient
    max_iterations += nmf_max_interm

    step_iter = math.ceil(max_iterations / 10)
    pbar_step = 100 * step_iter / max_iterations

    i_iter = 0
    cont = 1

    # Initialize penalty
    # lambda = -1: no penalty
    # lambda = -abs(NMFSparselevel) : initialisation by NMFSparselevel (in
    # negative value)
    if nmf_sparse_level > 0:
        lambdaw = -nmf_sparse_level
        lambdat = -1
    elif nmf_sparse_level < 0:
        lambdat = nmf_sparse_level
        lambdaw = -1
    else:
        lambdaw = -1
        lambdat = -1

    percent_zeros = 0
    iter_sparse = 0
    nmf_convex = 0
    nl_kernel_applied = 0

    my_status_box.init_bar(delay=1)

    # Start loop
    while (cont == 1) & (i_iter < max_iterations):
        # Update RHE
        if nmf_fix_user_rhe == 0:
            if mult_or_pgrad == 1:
                if (nmf_algo == 2) or (nmf_algo == 4):
                    if n_mmis > 0:
                        mw = mw * ((mt.T @ (m * mmis)) / (mt.T @ ((mt @ mw.T) * mmis) + precision)).T
                    else:
                        mw = mw * ((mt.T @ m) / ((mt.T @ mt) @ mw.T + precision)).T
                else:
                    if n_mmis > 0:
                        mw = mw * (((m * mmis) / ((mt @ mw.T) * mmis + precision)).T @ mt)
                    else:
                        mw = mw * ((m / (mt @ mw.T + precision)).T @ mt)

                if n_nmf_priors > 0:
                    mw = mw * nmf_priors
            else:
                # Projected gradient
                if (nmf_convex > 0) & (nmf_find_parts > 0):
                    mw, tol_mw = proj_grad_kernel(
                        kernel, m, mmis, mh, mw, nmf_algo, tol_mw, nmf_max_iter_proj, nmf_priors.T
                    )
                elif (nmf_convex > 0) & (nmf_find_centroids > 0):
                    mh, tol_mh, dummy = proj_grad(
                        the_in, np.array([]), mt, mh.T, nmf_algo, -1, tol_mh, nmf_max_iter_proj, np.array([])
                    )
                else:
                    mw, tol_mw, lambdaw = proj_grad(
                        m, mmis, mt, mw.T, nmf_algo, lambdaw, tol_mw, nmf_max_iter_proj, nmf_priors.T
                    )

            if (nmf_convex > 0) & (nmf_find_parts > 0):
                for k in range(0, nc):
                    scale_mw = np.linalg.norm(mw[:, k])
                    mw[:, k] = mw[:, k] / scale_mw
                    mt[:, k] = mt[:, k] * scale_mw

        # Update LHE
        if nmf_fix_user_lhe == 0:
            if mult_or_pgrad == 1:
                if (nmf_algo == 2) | (nmf_algo == 4):
                    if n_mmis > 0:
                        mt = mt * ((m * mmis) @ mw / (((mt @ mw.T) * mmis) @ mw + precision))
                    else:
                        mt = mt * (m @ mw / (mt @ (mw.T @ mw) + precision))
                else:
                    mt = mt * ((m.T / (mw @ mt.T + precision)).T @ mw)
            else:
                # Projected gradient
                if (nmf_convex > 0) & (nmf_find_parts > 0):
                    mh, tol_mh, dummy = proj_grad(
                        ip, np.array([]), mw, mh.T, nmf_algo, -1, tol_mh, nmf_max_iter_proj, np.array([])
                    )
                elif (nmf_convex > 0) & (nmf_find_centroids > 0):
                    mt, tol_mt = proj_grad_kernel(
                        kernel, m.T, mmis.T, mh, mt, nmf_algo, tol_mt, nmf_max_iter_proj, np.array([])
                    )
                else:
                    mt, tol_mt, lambdat = proj_grad(
                        m.T, mmis.T, mw, mt.T, nmf_algo, lambdat, tol_mt, nmf_max_iter_proj, np.array([])
                    )

            # Scaling
            if ((nmf_convex == 0) | (nmf_find_centroids > 0)) & (nmf_fix_user_lhe == 0) & (nmf_fix_user_rhe == 0):
                for k in range(0, nc):
                    if (nmf_algo == 2) | (nmf_algo == 4):
                        scale_mt = np.linalg.norm(mt[:, k])
                    else:
                        scale_mt = np.sum(mt[:, k])

                    if scale_mt > 0:
                        mt[:, k] = mt[:, k] / scale_mt
                        if mult_or_pgrad == 2:
                            mw[:, k] = mw[:, k] * scale_mt

        # Switch to projected gradient
        if i_iter == nmf_max_interm:
            mult_or_pgrad = 2
            step_iter = 1
            pbar_step = 100 / max_iterations
            grad_mt = mt @ (mw.T @ mw) - m @ mw
            grad_mw = ((mt.T @ mt) @ mw.T - mt.T @ m).T
            initgrad = np.linalg.norm(np.concatenate((grad_mt, grad_mw), axis=0))
            tol_mt = 1e-3 * initgrad
            tol_mw = tol_mt

        if i_iter % step_iter == 0:
            if (nmf_convex > 0) & (nmf_find_parts > 0):
                mht_kernel = mh.T @ kernel
                diff = (trace_kernel + np.trace(-2 * mw @ mht_kernel + mw @ (mht_kernel @ mh) @ mw.T)) / nxp
            elif (nmf_convex > 0) & (nmf_find_centroids > 0):
                mht_kernel = mh.T @ kernel
                diff = (trace_kernel + np.trace(-2 * mt @ mht_kernel + mt @ (mht_kernel @ mh) @ mt.T)) / nxp
            else:
                if (nmf_algo == 2) | (nmf_algo == 4):
                    if n_mmis > 0:
                        mdiff = (mt @ mw.T - m) * mmis
                    else:
                        mdiff = mt @ mw.T - m

                else:
                    mf0 = mt @ mw.T
                    mdiff = m * np.log(m / mf0 + precision) + mf0 - m

                diff = (np.linalg.norm(mdiff)) ** 2 / nxp

            status = status0 + "Iteration: %s" % int(i_iter)

            if nmf_sparse_level != 0:
                if nmf_sparse_level > 0:
                    lambdax = lambdaw
                else:
                    lambdax = lambdat

                status = (
                    status
                    + "; Achieved sparsity: "
                    + str(round(percent_zeros, 2))
                    + "; Penalty: "
                    + str(round(lambdax, 2))
                )
                if log_iter == 1:
                    my_status_box.my_print(status)
            elif (nmf_convex > 0) & (nmf_find_parts > 0):
                status = status + " - Find parts"
            elif (nmf_convex > 0) & (nmf_find_centroids > 0) & (nl_kernel_applied == 0):
                status = status + " - Find centroids"
            elif nl_kernel_applied == 1:
                status = status + " - Apply non linear kernel"

            my_status_box.update_status(delay=1, status=status)
            my_status_box.update_bar(delay=1, step=pbar_step)
            if my_status_box.cancel_pressed:
                cancel_pressed = 1
                return [mt, mev, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed]

            if log_iter == 1:
                if (nmf_algo == 2) | (nmf_algo == 4):
                    my_status_box.my_print(status0 + " Iter: " + str(i_iter) + " MSR: " + str(diff))
                else:
                    my_status_box.my_print(status0 + " Iter: " + str(i_iter) + " DIV: " + str(diff))

            if i_iter > nmf_max_interm and (diff0 - diff) / diff0 < tolerance:
                cont = 0

            diff0 = diff

        i_iter += 1

        if (cont == 0) | (i_iter == max_iterations):
            if ((nmf_find_parts > 0) | (nmf_find_centroids > 0)) & (nmf_convex == 0):
                # Initialize convexity
                nmf_convex = 1
                diff0 = 1.0e99
                i_iter = nmf_max_interm + 1
                my_status_box.init_bar(delay=1)
                cont = 1
                if nmf_find_parts > 0:
                    ip = np.identity(p)
                    if (nmf_algo == 2) or (nmf_algo == 4):
                        if n_mmis > 0:
                            kernel = apply_kernel(mmis * m, 1, np.array([]), np.array([]))
                        else:
                            kernel = apply_kernel(m, 1, np.array([]), np.array([]))
                    else:
                        if n_mmis > 0:
                            kernel = apply_kernel(mmis * (m / (mt @ mw.T)), 1, np.array([]), np.array([]))
                        else:
                            kernel = apply_kernel(m / (mt @ mw.T), 1, np.array([]), np.array([]))

                    trace_kernel = np.trace(kernel)
                    # noinspection PyBroadException
                    try:
                        mh = mw @ np.linalg.inv(mw.T @ mw)
                    except BaseException:
                        mh = mw @ np.linalg.pinv(mw.T @ mw)

                    mh[np.where(mh < 0)] = 0
                    for k in range(0, nc):
                        scale_mw = np.linalg.norm(mw[:, k])
                        mw[:, k] = mw[:, k] / scale_mw
                        mh[:, k] = mh[:, k] * scale_mw

                    grad_mh = mh @ (mw.T @ mw) - mw
                    grad_mw = ((mh.T @ mh) @ mw.T - mh.T).T
                    initgrad = np.linalg.norm(np.concatenate((grad_mh, grad_mw), axis=0))
                    tol_mh = 1.0e-3 * initgrad
                    tol_mw = tol_mt
                elif nmf_find_centroids > 0:
                    the_in = np.identity(n)
                    if (nmf_algo == 2) or (nmf_algo == 4):
                        if n_mmis > 0:
                            kernel = apply_kernel(mmis.T * m.T, 1, np.array([]), np.array([]))
                        else:
                            kernel = apply_kernel(m.T, 1, np.array([]), np.array([]))
                    else:
                        if n_mmis > 0:
                            kernel = apply_kernel(mmis.T * (m.T / (mt @ mw.T).T), 1, np.array([]), np.array([]))
                        else:
                            kernel = apply_kernel(m.T / (mt @ mw.T).T, 1, np.array([]), np.array([]))

                    trace_kernel = np.trace(kernel)
                    # noinspection PyBroadException
                    try:
                        mh = mt @ np.linalg.inv(mt.T @ mt)
                    except BaseException:
                        mh = mt @ np.linalg.pinv(mt.T @ mt)

                    mh[np.where(mh < 0)] = 0
                    for k in range(0, nc):
                        scale_mt = np.linalg.norm(mt[:, k])
                        mt[:, k] = mt[:, k] / scale_mt
                        mh[:, k] = mh[:, k] * scale_mt

                    grad_mt = mt @ (mh.T @ mh) - mh
                    grad_mh = ((mt.T @ mt) @ mh.T - mt.T).T
                    initgrad = np.linalg.norm(np.concatenate((grad_mt, grad_mh), axis=0))
                    tol_mh = 1.0e-3 * initgrad
                    tol_mt = tol_mh

            elif (nmf_convex > 0) & (nmf_kernel > 1) & (nl_kernel_applied == 0):
                nl_kernel_applied = 1
                diff0 = 1.0e99
                i_iter = nmf_max_interm + 1
                my_status_box.init_bar(delay=1)
                cont = 1
                # Calculate centroids
                for k in range(0, nc):
                    mh[:, k] = mh[:, k] / np.sum(mh[:, k])

                mw = (mh.T @ m).T
                if (nmf_algo == 2) or (nmf_algo == 4):
                    if n_mmis > 0:
                        kernel = apply_kernel(mmis.T * m.T, nmf_kernel, mw, mt)
                    else:
                        kernel = apply_kernel(m.T, nmf_kernel, mw, mt)
                else:
                    if n_mmis > 0:
                        kernel = apply_kernel(mmis.T * (m.T / (mt @ mw.T).T), nmf_kernel, mw, mt)
                    else:
                        kernel = apply_kernel(m.T / (mt @ mw.T).T, nmf_kernel, mw, mt)

                trace_kernel = np.trace(kernel)
                # noinspection PyBroadException
                try:
                    mh = mt @ np.linalg.inv(mt.T @ mt)
                except BaseException:
                    mh = mt @ np.linalg.pinv(mt.T @ mt)

                mh[np.where(mh < 0)] = 0
                for k in range(0, nc):
                    scale_mt = np.linalg.norm(mt[:, k])
                    mt[:, k] = mt[:, k] / scale_mt
                    mh[:, k] = mh[:, k] * scale_mt

                grad_mt = mt @ (mh.T @ mh) - mh
                grad_mh = ((mt.T @ mt) @ mh.T - mt.T).T
                initgrad = np.linalg.norm(np.concatenate((grad_mt, grad_mh), axis=0))
                tol_mh = 1.0e-3 * initgrad
                tol_mt = tol_mh

            if nmf_sparse_level > 0:
                sparse_test = np.zeros((p, 1))
                for k in range(0, nc):
                    sparse_test[np.where(mw[:, k] > 0)] = 1

                percent_zeros0 = percent_zeros
                n_sparse_test = np.where(sparse_test == 0)[0].size
                percent_zeros = max(n_sparse_test / p, 0.01)
                if percent_zeros == percent_zeros0:
                    iter_sparse += 1
                else:
                    iter_sparse = 0

                if (percent_zeros < 0.99 * nmf_sparse_level) & (iter_sparse < 50):
                    lambdaw *= min(1.01 * nmf_sparse_level / percent_zeros, 1.10)
                    i_iter = nmf_max_interm + 1
                    cont = 1

            elif nmf_sparse_level < 0:
                sparse_test = np.zeros((n, 1))
                for k in range(0, nc):
                    sparse_test[np.where(mt[:, k] > 0)] = 1

                percent_zeros0 = percent_zeros
                n_sparse_test = np.where(sparse_test == 0)[0].size
                percent_zeros = max(n_sparse_test / n, 0.01)
                if percent_zeros == percent_zeros0:
                    iter_sparse += 1
                else:
                    iter_sparse = 0

                if (percent_zeros < 0.99 * nmf_sparse_level) & (iter_sparse < 50):
                    lambdat *= min(1.01 * nmf_sparse_level / percent_zeros, 1.10)
                    i_iter = nmf_max_interm + 1
                    cont = 1

    if nmf_find_parts > 0:
        # Make m convex
        mt = m @ mh
        mt, mw, mh, flag_nonconvex2, add_message, err_message, cancel_pressed = get_convex_scores(
            mt, mw, mh, flag_nonconvex, add_message
        )
    #        if flag_nonconvex2 > flag_nonconvex:
    #            flag_nonconvex = flag_nonconvex2
    #            # Calculate column centroids
    #            for k in range(0, nc):
    #                scale_mh = np.sum(mh[:, k])
    #                mh[:, k] = mh[:, k] / scale_mh
    #                mw[:, k] = mw[:, k] * scale_mh
    #
    #            m = m @ mh
    elif nmf_find_centroids > 0:
        # Calculate row centroids
        for k in range(0, nc):
            scale_mh = np.sum(mh[:, k])
            mh[:, k] = mh[:, k] / scale_mh
            mt[:, k] = mt[:, k] * scale_mh

        mw = (mh.T @ m).T

    if (nmf_convex == 0) & (nmf_fix_user_lhe == 0) & (nmf_fix_user_rhe == 0):
        # Scale
        for k in range(0, nc):
            if (nmf_algo == 2) | (nmf_algo == 4):
                scale_mt = np.linalg.norm(mt[:, k])
                scale_mw = np.linalg.norm(mw[:, k])
            else:
                scale_mt = np.sum(mt[:, k])
                scale_mw = np.sum(mw[:, k])

            if scale_mt > 0:
                mt[:, k] = mt[:, k] / scale_mt

            if scale_mw > 0:
                mw[:, k] = mw[:, k] / scale_mw

            mev[k] = scale_mt * scale_mw

    if (nmf_kernel > 1) & (nl_kernel_applied == 1):
        diff /= trace_kernel / nxp

    return [mt, mev, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed]


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
    """
    Estimate left and right hand matrices (robust version)
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
         mev: Scaling matrix
         mw: Right hand matrix
         mt_pct: Percent robust clustered rows
         mw_pct: Percent robust clustered columns
         diff: Objective cost
         mh: Convexity matrix
         flag_nonconvex: Updated non-convexity flag on left hand matrix
    """

    # Check parameter consistency (and correct if needed)
    add_message = []
    err_message = ""
    cancel_pressed = 0
    nc = int(nc)
    if nmf_fix_user_lhe * nmf_fix_user_rhe == 1:
        mev = np.ones(nc)
        return mt0, mev, mw0, np.array([]), np.array([]), 0, np.array([]), 0, add_message, err_message, cancel_pressed

    if (nc == 1) & (nmf_algo > 2):
        nmf_algo -= 2

    if nmf_algo <= 2:
        nmf_robust_n_runs = 0

    mmis = mmis.astype(np.int)
    n_mmis = mmis.shape[0]
    if n_mmis == 0:
        the_id = np.where(np.isnan(m))
        n_mmis = the_id[0].size
        if n_mmis > 0:
            mmis = np.isnan(m) is False
            mmis = mmis.astype(np.int)
            m[mmis == 0] = 0
    else:
        m[mmis == 0] = 0

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
    mt, mev, mw, diffsup, mhsup, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
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
    mevsup = np.copy(mev)
    mwsup = np.copy(mw)
    if (n_nmf_priors > 0) & (nmf_reweigh_columns > 0):
        #     Run again with fixed LHE & no priors
        status = "Step 1bis - NMF (fixed LHE) Ncomp=" + str(nc) + ": "
        mw = np.ones((p, nc)) / math.sqrt(p)
        mt, mev, mw, diffsup, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
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
        mevsup = np.copy(mev)
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
                mt, mev, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
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
                mt, mev, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
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
                mw_blk[:, k * nmf_robust_n_runs + i_bootstrap] = mw[:, k] * mev[k]

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

            col_clust = np.zeros(p, dtype=int)
            if nmf_calculate_leverage > 0:
                mwn, add_message, err_message, cancel_pressed = leverage(
                    mwn, nmf_use_robust_leverage, add_message, my_status_box
                )

            for j in range(0, p):
                col_clust[j] = np.argmax(np.array(mwn[j, :]))
                mw_pct[j, col_clust[j]] = mw_pct[j, col_clust[j]] + 1

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

            mt, mev, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_solve(
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
            row_clust = np.zeros(n, dtype=int)
            if nmf_calculate_leverage > 0:
                mtn, add_message, err_message, cancel_pressed = leverage(
                    mt, nmf_use_robust_leverage, add_message, my_status_box
                )
            else:
                mtn = mt

            for i in range(0, n):
                row_clust[i] = np.argmax(mtn[i, :])
                mt_pct[i, row_clust[i]] = mt_pct[i, row_clust[i]] + 1

        mt_pct = mt_pct / nmf_robust_n_runs

    mt = mtsup
    mw = mwsup
    mh = mhsup
    mev = mevsup
    diff = diffsup

    if nmf_robust_resample_columns > 0:
        mtemp = np.copy(mt)
        mt = np.copy(mw)
        mw = mtemp
        mtemp = np.copy(mt_pct)
        mt_pct = np.copy(mw_pct)
        mw_pct = mtemp

    return mt, mev, mw, mt_pct, mw_pct, diff, mh, flag_nonconvex, add_message, err_message, cancel_pressed


def ntf_stack(m, mmis, n_blocks):
    # Unfold m
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
    """
     Estimate NTF matrices (HALS)
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
    n, p = m.shape
    mmis = mmis.astype(np.int)
    n_mmis = mmis.shape[0]
    if n_mmis == 0:
        the_id = np.where(np.isnan(m))
        n_mmis = the_id[0].size
        if n_mmis > 0:
            mmis = np.isnan(m) is False
            mmis = mmis.astype(np.int)
            m[mmis == 0] = 0

    nc = int(nc)
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
    if not GALDERMA_FLAG:
        mtx_mw, mb2 = nmf_init(mstacked, mmis_stacked, mtx_mw, mb2, nc2, tolerance, log_iter, my_status_box)
    else:
        print("Galderma version! the_in ntf_init, NNSVD has been superseded by SVD prior to NMF initialization")

    # Quick NMF
    mtx_mw, mev2, mb2, diff, mh, dummy1, dummy2, add_message, err_message, cancel_pressed = nmf_solve(
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
    for k in range(0, nc2):
        mb2[:, k] = mb2[:, k] * mev2[k]

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

                mb[:, ind] = mb2[:, k]

    for k in range(0, nc):
        if (ntf_unimodal > 0) & (ntf_left_components > 0):
            # Enforce unimodal distribution
            tmax = np.argmax(mt[:, k])
            for i in range(tmax + 1, n):
                mt[i, k] = min(mt[i - 1, k], mt[i, k])

            for i in range(tmax - 1, -1, -1):
                mt[i, k] = min(mt[i + 1, k], mt[i, k])

        if (ntf_unimodal > 0) & (ntf_right_components > 0):
            #                 Enforce unimodal distribution
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

    return [mt, mw, mb, add_message, err_message, cancel_pressed]


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
):
    """
    Core updating code called by ntf_solve_simple & NTF Solve_conv
    Input:
        All variables in the calling function used in the function
    Output:
        Same as Input
    """

    # Compute kth-part
    for i_block in range(0, n_blocks):
        mpart[:, id_blockp[i_block]: id_blockp[i_block] + p] = (
            mb[i_block, k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
        )

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

    if (nmf_fix_user_bhe > 0) & norm_bhe:
        norm = np.linalg.norm(mb[:, k])
        if norm > 0:
            mb[:, k] /= norm

    if nmf_fix_user_lhe == 0:
        # Update m
        mt[:, k] = 0
        for i_block in range(0, n_blocks):
            mt[:, k] += mb[i_block, k] * mpart[:, id_blockp[i_block]: id_blockp[i_block] + p] @ mw[:, k]

        if n_mmis > 0:
            denomt[:] = 0
            mw2[:] = mw[:, k] ** 2
            for i_block in range(0, n_blocks):
                # Broadcast missing cells into mw to calculate mw.T * mw
                denomt += mb[i_block, k] ** 2 * mmis[:, id_blockp[i_block]: id_blockp[i_block] + p] @ mw2

            denomt /= np.max(denomt)
            denomt[denomt < denom_cutoff] = denom_cutoff
            mt[:, k] /= denomt

        mt[mt[:, k] < 0, k] = 0

        if (ntf_unimodal > 0) & (ntf_left_components > 0):
            #                 Enforce unimodal distribution
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
        # Update mw
        mw[:, k] = 0
        for i_block in range(0, n_blocks):
            mw[:, k] += mpart[:, id_blockp[i_block]: id_blockp[i_block] + p].T @ mt[:, k] * mb[i_block, k]

        if n_mmis > 0:
            denomw[:] = 0
            mt2[:] = mt[:, k] ** 2
            for i_block in range(0, n_blocks):
                # Broadcast missing cells into mw to calculate m.T * m
                denomw += mb[i_block, k] ** 2 * mmis[:, id_blockp[i_block]: id_blockp[i_block] + p].T @ mt2

            denomw /= np.max(denomw)
            denomw[denomw < denom_cutoff] = denom_cutoff
            mw[:, k] /= denomw

        mw[mw[:, k] < 0, k] = 0

        if (ntf_unimodal > 0) & (ntf_right_components > 0):
            # Enforce unimodal distribution
            wmax = np.argmax(mw[:, k])
            for j in range(wmax + 1, p):
                mw[j, k] = min(mw[j - 1, k], mw[j, k])

            for j in range(wmax - 1, -1, -1):
                mw[j, k] = min(mw[j + 1, k], mw[j, k])

        if (ntf_smooth > 0) & (ntf_left_components > 0):
            #             Smooth distribution
            b[0] = 0.75 * mw[0, k] + 0.25 * mw[1, k]
            b[p - 1] = 0.25 * mw[p - 2, k] + 0.75 * mw[p - 1, k]
            for j in range(1, p - 1):
                b[j] = 0.25 * mw[j - 1, k] + 0.5 * mw[j, k] + 0.25 * mw[j + 1, k]

            mw[:, k] = b

        if norm_rhe:
            norm = np.linalg.norm(mw[:, k])
            if norm > 0:
                mw[:, k] /= norm

    if nmf_fix_user_bhe == 0:
        # Update mb
        mb[:, k] = 0
        mt_mw[:] = np.reshape((np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))), nxp)

        for i_block in range(0, n_blocks):
            mb[i_block, k] = np.reshape(mpart[:, id_blockp[i_block]: id_blockp[i_block] + p], nxp).T @ mt_mw

        if n_mmis > 0:
            mt_mw[:] = mt_mw[:] ** 2
            for i_block in range(0, n_blocks):
                # Broadcast missing cells into mb to calculate mb.T * mb
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

        if (ntf_smooth > 0) & (ntf_left_components > 0):
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
    for i_block in range(0, n_blocks):
        mfit[:, id_blockp[i_block]: id_blockp[i_block] + p] += (
            mb[i_block, k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
        )

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
    )


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
        ntf_unimodal,
        ntf_smooth,
        ntf_left_components,
        ntf_right_components,
        ntf_block_components,
        n_blocks,
        ntfn_conv,
        my_status_box,
):
    """
    Interface to ntf_solve_simple & ntf_solve_conv
    """

    if ntfn_conv > 0:
        return ntf_solve_conv(
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
            ntf_unimodal,
            ntf_smooth,
            ntf_left_components,
            ntf_right_components,
            ntf_block_components,
            n_blocks,
            ntfn_conv,
            my_status_box,
        )
    else:
        return ntf_solve_simple(
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
            ntf_unimodal,
            ntf_smooth,
            ntf_left_components,
            ntf_right_components,
            ntf_block_components,
            n_blocks,
            my_status_box,
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
        ntf_unimodal,
        ntf_smooth,
        ntf_left_components,
        ntf_right_components,
        ntf_block_components,
        n_blocks,
        my_status_box,
):
    """
    Estimate NTF matrices (HALS)
    Input:
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
         ntf_unimodal: Apply Unimodal constraint on factoring vectors
         ntf_smooth: Apply Smooth constraint on factoring vectors
         compo: Apply Unimodal/Smooth constraint on left hand matrix
         ntf_right_components: Apply Unimodal/Smooth constraint on right hand matrix
         compo1: Apply Unimodal/Smooth constraint on block hand matrix
         n_blocks: Number of NTF blocks
    Output:
         m: Left hand matrix
         mw: Right hand matrix
         mb: Block hand matrix
         mres: Residual tensor
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
    #     step_iter = math.ceil(max_iterations/10)
    step_iter = 1
    pbar_step = 100 * step_iter / max_iterations

    id_blockp = np.arange(0, (n_blocks - 1) * p + 1, p)
    a = np.zeros(n)
    b = np.zeros(p)
    c = np.zeros(n_blocks)

    # Compute Residual tensor
    mfit = np.zeros((n, p0))
    for k in range(0, nc):
        for i_block in range(0, n_blocks):
            mfit[:, id_blockp[i_block]: id_blockp[i_block] + p] += (
                mb[i_block, k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
            )

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

    my_status_box.init_bar(delay=1)

    # Loop
    cont = 1
    i_iter = 0
    diff0 = 1.0e99
    mpart = np.zeros((n, p0))

    # for k in range (0, nc):
    #    print(k, mb[:, k])

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
            ) = ntf_update(
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
            )

        if i_iter % step_iter == 0:
            # Check convergence
            diff = np.linalg.norm(mres) ** 2 / nxp0
            if (diff0 - diff) / diff0 < tolerance:
                cont = 0
            else:
                diff0 = diff

            status = status0 + "Iteration: %s" % int(i_iter)
            my_status_box.update_status(delay=1, status=status)
            my_status_box.update_bar(delay=1, step=pbar_step)
            if my_status_box.cancel_pressed:
                cancel_pressed = 1
                return [np.array([]), mt, mw, mb, mres, cancel_pressed]

            if log_iter == 1:
                my_status_box.my_print(status0 + " Iter: " + str(i_iter) + " MSR: " + str(diff))

        i_iter += 1

    if (n_mmis > 0) & (nmf_fix_user_bhe == 0):
        mb *= denom_block

    return [np.array([]), mt, mw, mb, mres, cancel_pressed]


def ntf_solve_conv(
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
        ntf_unimodal,
        ntf_smooth,
        ntf_left_components,
        ntf_right_components,
        ntf_block_components,
        n_blocks,
        ntfn_conv,
        my_status_box,
):
    """
     Estimate NTF matrices (HALS)
     Input:
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
         ntf_unimodal: Apply Unimodal constraint on factoring vectors
         ntf_smooth: Apply Smooth constraint on factoring vectors
         compo: Apply Unimodal/Smooth constraint on left hand matrix
         ntf_right_components: Apply Unimodal/Smooth constraint on right hand matrix
         compo1: Apply Unimodal/Smooth constraint on block hand matrix
         n_blocks: Number of NTF blocks
         ntfn_conv: Half-Size of the convolution window on 3rd-dimension of the tensor
     Output:
         m: Left hand matrix (sum of columns Mt_conv for each k)
         Mt_conv : if ntfn_conv > 0 only otherwise empty. Contains sub-components for each phase in convolution window
         mw: Right hand matrix
         mb: Block hand matrix
         mres: Residual tensor
x     """
    cancel_pressed = 0

    n, p0 = m.shape
    n_mmis = mmis.shape[0]
    nc = int(nc)
    n_blocks = int(n_blocks)
    ntfn_conv = int(ntfn_conv)
    p = int(p0 / n_blocks)
    nxp = int(n * p)
    nxp0 = int(n * p0)
    mt_simple = np.copy(mt0)
    mw_simple = np.copy(mw0)
    mb_simple = np.copy(mb0)
    #     step_iter = math.ceil(max_iterations/10)
    step_iter = 1
    pbar_step = 100 * step_iter / max_iterations

    id_blockp = np.arange(0, (n_blocks - 1) * p + 1, p)
    a = np.zeros(n)
    b = np.zeros(p)
    c = np.zeros(n_blocks)
    mt_mw = np.zeros(nxp)
    ntfn_conv2 = 2 * ntfn_conv + 1

    # Initialize m, mw, mb
    mt = np.repeat(mt_simple, ntfn_conv2, axis=1) / ntfn_conv2
    mw = np.repeat(mw_simple, ntfn_conv2, axis=1)
    mb = np.repeat(mb_simple, ntfn_conv2, axis=1)

    for k3 in range(0, nc):
        n_shift = -ntfn_conv - 1
        for k2 in range(0, ntfn_conv2):
            n_shift += 1
            k = k3 * ntfn_conv2 + k2
            mb[:, k] = shift(mb_simple[:, k3], n_shift)

    # Initialize Residual tensor
    mfit = np.zeros((n, p0))
    for k3 in range(0, nc):
        for k2 in range(0, ntfn_conv2):
            k = k3 * ntfn_conv2 + k2
            for i_block in range(0, n_blocks):
                mfit[:, id_blockp[i_block]: id_blockp[i_block] + p] += (
                    mb[i_block, k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
                )

    denomt = np.zeros(n)
    denomw = np.zeros(p)
    denom_block = np.zeros((n_blocks, nc))
    mt2 = np.zeros(n)
    mw2 = np.zeros(p)
    denom_cutoff = 0.1

    if n_mmis > 0:
        mres = (m - mfit) * mmis
    else:
        mres = m - mfit

    my_status_box.init_bar(delay=1)

    # Loop
    cont = 1
    i_iter = 0
    diff0 = 1.0e99
    mpart = np.zeros((n, p0))

    while (cont > 0) & (i_iter < max_iterations):
        for k3 in range(0, nc):
            for k2 in range(0, ntfn_conv2):
                k = k3 * ntfn_conv2 + k2
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
                ) = ntf_update(
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
                )

            # Update mt_simple, mw_simple & mb_simple
            k = k3 * ntfn_conv2 + ntfn_conv
            mt_simple[:, k3] = mt[:, k]
            mw_simple[:, k3] = mw[:, k]
            mb_simple[:, k3] = mb[:, k]

            # Update mw & mb
            mw[:, :] = np.repeat(mw_simple, ntfn_conv2, axis=1)
            n_shift = -ntfn_conv - 1
            for k2 in range(0, ntfn_conv2):
                n_shift += 1
                k = k3 * ntfn_conv2 + k2
                mb[:, k] = shift(mb_simple[:, k3], n_shift)

        if i_iter % step_iter == 0:
            # Check convergence
            diff = np.linalg.norm(mres) ** 2 / nxp0
            if (diff0 - diff) / diff0 < tolerance:
                cont = 0
            else:
                diff0 = diff

            status = status0 + "Iteration: %s" % int(i_iter)
            my_status_box.update_status(delay=1, status=status)
            my_status_box.update_bar(delay=1, step=pbar_step)
            if my_status_box.cancel_pressed:
                cancel_pressed = 1
                return [mt, mt_simple, mw_simple, mb_simple, mres, cancel_pressed]

            if log_iter == 1:
                my_status_box.my_print(status0 + " Iter: " + str(i_iter) + " MSR: " + str(diff))

        i_iter += 1

    if (n_mmis > 0) & (nmf_fix_user_bhe == 0):
        mb *= denom_block

    return [mt, mt_simple, mw_simple, mb_simple, mres, cancel_pressed]


def ntf_solve_fast(
        m,
        mmis,
        mt0,
        mw0,
        mb0,
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
):
    """
     Estimate NTF matrices (fast HALS)
     Input:
         m: Input matrix
         mmis: Define missing values (0 = missing cell, 1 = real cell)
            NOTE: Still not workable version
         mt0: Initial left hand matrix
         mw0: Initial right hand matrix
         mb0: Initial block hand matrix
         nc: NTF rank
         tolerance: Convergence threshold
         precision: Replace 0-values in multiplication rules
         log_iter: Log results through iterations
         status0: Initial displayed status to be updated during iterations
         max_iterations: Max iterations
         nmf_fix_user_lhe: fix left hand matrix columns: = 1, else = 0
         nmf_fix_user_rhe: fix  right hand matrix columns: = 1, else = 0
         nmf_fix_user_bhe: fix  block hand matrix columns: = 1, else = 0
         ntf_unimodal: Apply Unimodal constraint on factoring vectors
         ntf_smooth: Apply Smooth constraint on factoring vectors
         compo: Apply Unimodal/Smooth constraint on left hand matrix
         ntf_right_components: Apply Unimodal/Smooth constraint on right hand matrix
         compo1: Apply Unimodal/Smooth constraint on block hand matrix
         n_blocks: Number of NTF blocks
     Output:
         m: Left hand matrix
         mw: Right hand matrix
         mb: Block hand matrix
         mres: Residual tensor
     """
    mres = np.array([])
    cancel_pressed = 0

    n, p0 = m.shape
    n_mmis = mmis.shape[0]
    nc = int(nc)
    n_blocks = int(n_blocks)
    p = int(p0 / n_blocks)
    n0 = int(n * n_blocks)
    nxp = int(n * p)
    nxp0 = int(n * p0)
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    mb = np.copy(mb0)
    step_iter = math.ceil(max_iterations / 10)
    pbar_step = 100 * step_iter / max_iterations
    mx_mmis2 = None
    denom_block = None
    denomt = None
    mx_mmis = None
    denom_cutoff = None
    denomw = None

    id_blockn = np.arange(0, (n_blocks - 1) * n + 1, n)
    id_blockp = np.arange(0, (n_blocks - 1) * p + 1, p)
    a = np.zeros(n)
    b = np.zeros(p)
    c = np.zeros(n_blocks)

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

    for k in range(0, nc):
        if (nmf_fix_user_lhe > 0) & norm_lhe:
            norm = np.linalg.norm(mt[:, k])
            if norm > 0:
                mt[:, k] /= norm

        if (nmf_fix_user_rhe > 0) & norm_rhe:
            norm = np.linalg.norm(mw[:, k])
            if norm > 0:
                mw[:, k] /= norm

        if (nmf_fix_user_bhe > 0) & norm_bhe:
            norm = np.linalg.norm(mb[:, k])
            if norm > 0:
                mb[:, k] /= norm

    # Normalize factors to unit length
    #    for k in range(0, nc):
    #        ScaleMt = np.linalg.norm(m[:, k])
    #        m[:, k] /= ScaleMt
    #        ScaleMw = np.linalg.norm(mw[:, k])
    #        mw[:, k] /= ScaleMw
    #        mb[:, k] *= (ScaleMt * ScaleMw)

    # Initialize t1
    mt2 = mt.T @ mt
    mt2[mt2 == 0] = precision
    mw2 = mw.T @ mw
    mw2[mw2 == 0] = precision
    mb2 = mb.T @ mb
    mb2[mb2 == 0] = precision
    t1 = mt2 * mw2 * mb2
    t2t = np.zeros((n, nc))
    t2w = np.zeros((p, nc))
    t2_block = np.zeros((n_blocks, nc))

    # Transpose m by block once for all
    m2 = np.zeros((p, n0))

    mfit = np.zeros((n, p0))
    if n_mmis > 0:
        denomt = np.zeros(n)
        denomw = np.zeros(p)
        denom_block = np.ones((n_blocks, nc))
        mx_mmis2 = np.zeros((p, n0))
        denom_cutoff = 0.1

    my_status_box.init_bar(delay=1)

    # Loop
    cont = 1
    i_iter = 0
    diff0 = 1.0e99

    for i_block in range(0, n_blocks):
        m2[:, id_blockn[i_block]: id_blockn[i_block] + n] = m[:, id_blockp[i_block]: id_blockp[i_block] + p].T
        if n_mmis > 0:
            mx_mmis2[:, id_blockn[i_block]: id_blockn[i_block] + n] = (
                m[:, id_blockp[i_block]: id_blockp[i_block] + p]
                * mmis[:, id_blockp[i_block]: id_blockp[i_block] + p]
            ).T

    if n_mmis > 0:
        mx_mmis = m * mmis

    while (cont > 0) & (i_iter < max_iterations):
        if n_mmis > 0:
            gamma = np.diag((denom_block * mb).T @ (denom_block * mb))
        else:
            gamma = np.diag(mb.T @ mb)

        if nmf_fix_user_lhe == 0:
            # Update m
            t2t[:, :] = 0
            for k in range(0, nc):
                if n_mmis > 0:
                    denomt[:] = 0
                    mwn = np.repeat(mw[:, k, np.newaxis] ** 2, n, axis=1)
                    for i_block in range(0, n_blocks):
                        # Broadcast missing cells into mw to calculate mw.T *
                        # mw
                        denomt += mb[i_block, k] ** 2 * np.sum(
                            mmis[:, id_blockp[i_block]: id_blockp[i_block] + p].T * mwn, axis=0
                        )

                    denomt /= np.max(denomt)
                    denomt[denomt < denom_cutoff] = denom_cutoff
                    for i_block in range(0, n_blocks):
                        t2t[:, k] += mx_mmis[:, id_blockp[i_block]: id_blockp[i_block] + p] @ mw[:, k] * mb[i_block, k]

                    t2t[:, k] /= denomt
                else:
                    for i_block in range(0, n_blocks):
                        t2t[:, k] += m[:, id_blockp[i_block]: id_blockp[i_block] + p] @ mw[:, k] * mb[i_block, k]

            mt2 = mt.T @ mt
            mt2[mt2 == 0] = precision
            t3 = t1 / mt2

            for k in range(0, nc):
                mt[:, k] = gamma[k] * mt[:, k] + t2t[:, k] - mt @ t3[:, k]
                mt[np.where(mt[:, k] < 0), k] = 0

                if (ntf_unimodal > 0) & (ntf_left_components > 0):
                    #                 Enforce unimodal distribution
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
                    mt[:, k] /= np.linalg.norm(mt[:, k])

            mt2 = mt.T @ mt
            mt2[mt2 == 0] = precision
            t1 = t3 * mt2

        if nmf_fix_user_rhe == 0:
            # Update mw
            t2w[:, :] = 0
            for k in range(0, nc):
                if n_mmis > 0:
                    denomw[:] = 0
                    mtp = np.repeat(mt[:, k, np.newaxis] ** 2, p, axis=1)
                    for i_block in range(0, n_blocks):
                        # Broadcast missing cells into mw to calculate m.T *
                        # m
                        denomw += mb[i_block, k] ** 2 * np.sum(
                            mmis[:, id_blockp[i_block]: id_blockp[i_block] + p] * mtp, axis=0
                        )

                    denomw /= np.max(denomw)
                    denomw[denomw < denom_cutoff] = denom_cutoff
                    for i_block in range(0, n_blocks):
                        t2w[:, k] += (
                            mx_mmis2[:, id_blockn[i_block]: id_blockn[i_block] + n] @ mt[:, k] * mb[i_block, k]
                        )

                    t2w[:, k] /= denomw
                else:
                    for i_block in range(0, n_blocks):
                        t2w[:, k] += m2[:, id_blockn[i_block]: id_blockn[i_block] + n] @ mt[:, k] * mb[i_block, k]

            mw2 = mw.T @ mw
            mw2[mw2 == 0] = precision
            t3 = t1 / mw2

            for k in range(0, nc):
                mw[:, k] = gamma[k] * mw[:, k] + t2w[:, k] - mw @ t3[:, k]
                mw[np.where(mw[:, k] < 0), k] = 0

                if (ntf_unimodal > 0) & (ntf_right_components > 0):
                    #                 Enforce unimodal distribution
                    wmax = np.argmax(mw[:, k])
                    for j in range(wmax + 1, p):
                        mw[j, k] = min(mw[j - 1, k], mw[j, k])

                    for j in range(wmax - 1, -1, -1):
                        mw[j, k] = min(mw[j + 1, k], mw[j, k])

                if (ntf_smooth > 0) & (ntf_left_components > 0):
                    #             Smooth distribution
                    b[0] = 0.75 * mw[0, k] + 0.25 * mw[1, k]
                    b[p - 1] = 0.25 * mw[p - 2, k] + 0.75 * mw[p - 1, k]
                    for j in range(1, p - 1):
                        b[j] = 0.25 * mw[j - 1, k] + 0.5 * mw[j, k] + 0.25 * mw[j + 1, k]

                    mw[:, k] = b

                if norm_rhe:
                    mw[:, k] /= np.linalg.norm(mw[:, k])

            mw2 = mw.T @ mw
            mw2[mw2 == 0] = precision
            t1 = t3 * mw2

        if nmf_fix_user_bhe == 0:
            # Update mb
            for k in range(0, nc):
                if n_mmis > 0:
                    for i_block in range(0, n_blocks):
                        # Broadcast missing cells into mb to calculate mb.T *
                        # mb
                        denom_block[i_block, k] = np.sum(
                            np.reshape(mmis[:, id_blockp[i_block]: id_blockp[i_block] + p], nxp)
                            * np.reshape((np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))), nxp) ** 2,
                            axis=0,
                        )

                    maxdenom_block = np.max(denom_block[:, k])
                    denom_block[denom_block[:, k] < denom_cutoff * maxdenom_block] = denom_cutoff * maxdenom_block
                    for i_block in range(0, n_blocks):
                        t2_block[i_block, k] = (
                            np.reshape(mx_mmis[:, id_blockp[i_block]: id_blockp[i_block] + p], nxp).T
                            @ (np.reshape((np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))), nxp))
                            / denom_block[i_block, k]
                        )

                else:
                    for i_block in range(0, n_blocks):
                        t2_block[i_block, k] = np.reshape(m[:, id_blockp[i_block]: id_blockp[i_block] + p], nxp).T @ (
                            np.reshape((np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))), nxp)
                        )

            mb2 = mb.T @ mb
            mb2[mb2 == 0] = precision
            t3 = t1 / mb2

            for k in range(0, nc):
                mb[:, k] = mb[:, k] + t2_block[:, k] - mb @ t3[:, k]
                mb[np.where(mb[:, k] < 0), k] = 0

                if (ntf_unimodal > 0) & (ntf_block_components > 0):
                    #                 Enforce unimodal distribution
                    bmax = np.argmax(mb[:, k])
                    for i_block in range(bmax + 1, n_blocks):
                        mb[i_block, k] = min(mb[i_block - 1, k], mb[i_block, k])

                    for i_block in range(bmax - 1, -1, -1):
                        mb[i_block, k] = min(mb[i_block + 1, k], mb[i_block, k])

                if (ntf_smooth > 0) & (ntf_left_components > 0):
                    #             Smooth distribution
                    c[0] = 0.75 * mb[0, k] + 0.25 * mb[1, k]
                    c[n_blocks - 1] = 0.25 * mb[n_blocks - 2, k] + 0.75 * mb[n_blocks - 1, k]
                    for i_block in range(1, n_blocks - 1):
                        c[i_block] = 0.25 * mb[i_block - 1, k] + 0.5 * mb[i_block, k] + 0.25 * mb[i_block + 1, k]

                    mb[:, k] = c

            mb2 = mb.T @ mb
            mb2[mb2 == 0] = precision
            t1 = t3 * mb2

        if i_iter % step_iter == 0:
            # Update residual tensor
            mfit[:, :] = 0

            for k in range(0, nc):
                if n_mmis > 0:
                    for i_block in range(0, n_blocks):
                        # mfit[:, id_blockp[i_block]:id_blockp[i_block] + p] +=
                        # denom_block[i_block, k] * mb[i_block, k] * (
                        mfit[:, id_blockp[i_block]: id_blockp[i_block] + p] += mb[i_block, k] * (
                            np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
                        )

                    mres = (m - mfit) * mmis
                else:
                    for i_block in range(0, n_blocks):
                        mfit[:, id_blockp[i_block]: id_blockp[i_block] + p] += mb[i_block, k] * (
                            np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
                        )

                    mres = m - mfit

            # Check convergence
            diff = np.linalg.norm(mres) ** 2 / nxp0
            if (diff0 - diff) / diff0 < tolerance:
                cont = 0
            else:
                diff0 = diff

            status = status0 + "Iteration: %s" % int(i_iter)
            my_status_box.update_status(delay=1, status=status)
            my_status_box.update_bar(delay=1, step=pbar_step)
            if my_status_box.cancel_pressed:
                cancel_pressed = 1
                return [mt, mw, mb, mres, cancel_pressed]

            if log_iter == 1:
                my_status_box.my_print(status0 + " Iter: " + str(i_iter) + " MSR: " + str(diff))

        i_iter += 1

    if n_mmis > 0:
        mb *= denom_block

    return [mt, mw, mb, mres, cancel_pressed]


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
        ntf_unimodal,
        ntf_smooth,
        ntf_left_components,
        ntf_right_components,
        ntf_block_components,
        n_blocks,
        ntfn_conv,
        my_status_box,
):
    """
     Estimate NTF matrices (robust version)
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
         ntf_fast_hals: Use Fast HALS
         ntfn_iterations: Warmup iterations for fast HALS
         ntf_unimodal: Apply Unimodal constraint on factoring vectors
         ntf_smooth: Apply Smooth constraint on factoring vectors
         compo: Apply Unimodal/Smooth constraint on left hand matrix
         ntf_right_components: Apply Unimodal/Smooth constraint on right hand matrix
         compo1: Apply Unimodal/Smooth constraint on block hand matrix
         n_blocks: Number of NTF blocks
         ntfn_conv: Half-Size of the convolution window on 3rd-dimension of the tensor

     Output:
         mt_conv: Convolutional Left hand matrix
         m: Left hand matrix
         mw: Right hand matrix
         mb: Block hand matrix
         mres: Residual tensor
         mt_pct: Percent robust clustered rows
         mw_pct: Percent robust clustered columns
     """
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

    mmis = mmis.astype(np.int)
    n_mmis = mmis.shape[0]
    if n_mmis == 0:
        the_id = np.where(np.isnan(m))
        n_mmis = the_id[0].size
        if n_mmis > 0:
            mmis = np.isnan(m) is False
            mmis = mmis.astype(np.int)
            m[mmis == 0] = 0
    else:
        m[mmis == 0] = 0

    ntfn_iterations = int(ntfn_iterations)
    nmf_robust_n_runs = int(nmf_robust_n_runs)
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    mb = np.copy(mb0)
    mt_conv = np.array([])

    # Check parameter consistency (and correct if needed)
    if (nc == 1) | (nmf_algo == 5):
        nmf_robust_n_runs = 0

    # Unused ?
    # if nmf_robust_n_runs == 0:
    #     mt_pct = np.nan
    #     mw_pct = np.nan

    if (n_mmis > 0 or ntfn_conv > 0) and ntf_fast_hals > 0:
        ntf_fast_hals = 0
        reverse2_hals = 1
    else:
        reverse2_hals = 0

    # Step 1: NTF
    status0 = "Step 1 - NTF Ncomp=" + str(nc) + ": "
    if ntf_fast_hals > 0:
        if ntfn_iterations > 0:
            mt_conv, mt, mw, mb, mres, cancel_pressed = ntf_solve(
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
                ntf_unimodal,
                ntf_smooth,
                ntf_left_components,
                ntf_right_components,
                ntf_block_components,
                n_blocks,
                ntfn_conv,
                my_status_box,
            )

        mt, mw, mb, mres, cancel_pressed = ntf_solve_fast(
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
        mt_conv, mt, mw, mb, mres, cancel_pressed = ntf_solve(
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
            ntf_unimodal,
            ntf_smooth,
            ntf_left_components,
            ntf_right_components,
            ntf_block_components,
            n_blocks,
            ntfn_conv,
            my_status_box,
        )

    mtsup = np.copy(mt)
    mwsup = np.copy(mw)
    mt_pct = None
    mw_pct = None
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
                    mt, mw, mb, mres, cancel_pressed = ntf_solve_fast(
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
                    mt, mw, mb, mres, cancel_pressed = ntf_solve_fast(
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
                    mt_conv, mt, mw, mb, mres, cancel_pressed = ntf_solve(
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
                        ntf_unimodal,
                        ntf_smooth,
                        ntf_left_components,
                        ntf_right_components,
                        ntf_block_components,
                        n_blocks,
                        ntfn_conv,
                        my_status_box,
                    )
                else:
                    mt_conv, mt, mw, mb, mres, cancel_pressed = ntf_solve(
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
                        ntf_unimodal,
                        ntf_smooth,
                        ntf_left_components,
                        ntf_right_components,
                        ntf_block_components,
                        n_blocks,
                        ntfn_conv,
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

            col_clust = np.zeros(p, dtype=int)
            if nmf_calculate_leverage > 0:
                mwn, add_message, err_message, cancel_pressed = leverage(
                    mwn, nmf_use_robust_leverage, add_message, my_status_box
                )

            for j in range(0, p):
                col_clust[j] = np.argmax(np.array(mwn[j, :]))
                mw_pct[j, col_clust[j]] = mw_pct[j, col_clust[j]] + 1

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
                mt, mw, mb, mres, cancel_pressed = ntf_solve_fast(
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
                mt_conv, mt, mw, mb, mres, cancel_pressed = ntf_solve(
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
                    ntf_unimodal,
                    ntf_smooth,
                    ntf_left_components,
                    ntf_right_components,
                    ntf_block_components,
                    n_blocks,
                    ntfn_conv,
                    my_status_box,
                )

            row_clust = np.zeros(n, dtype=int)
            if nmf_calculate_leverage > 0:
                mtn, add_message, err_message, cancel_pressed = leverage(
                    mt, nmf_use_robust_leverage, add_message, my_status_box
                )
            else:
                mtn = mt

            for i in range(0, n):
                row_clust[i] = np.argmax(mtn[i, :])
                mt_pct[i, row_clust[i]] = mt_pct[i, row_clust[i]] + 1

        mt_pct = mt_pct / nmf_robust_n_runs

    mt = mtsup
    mw = mwsup
    if reverse2_hals > 0:
        add_message.insert(
            len(add_message),
            "Currently, Fast HALS cannot be applied with missing data or convolution window and was reversed to "
            "Simple HALS.",
        )

    return mt_conv, mt, mw, mb, mres, mt_pct, mw_pct, add_message, err_message, cancel_pressed


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
    """
     Estimate SVD matrices (robust version)
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
         mev:
         mw: Right hand matrix
         mmis: Matrix of missing/flagged outliers
         mmsr: Vector of Residual SSQ
         mmsr2: Vector of Reidual variance
    """

    add_message = []
    err_message = ""
    cancel_pressed = 0
    diff0 = None
    diff = None
    diff_trial = None
    best_trial = None

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

            #         initialize t (LTS  =stochastic)
            if fast_code == 0:
                wnorm[:, :] = np.repeat(w[:, np.newaxis] ** 2, n, axis=1) * mmis.T
                denomw[:] = np.sum(wnorm, axis=0)
                # Request at least 2 non-missing values to perform row
                # regression
                if svd_algo == 2:
                    denomw[np.count_nonzero(mmis, axis=1) < 2] = np.nan

                t[:] = m @ w / denomw
            else:
                t[:] = m @ w / np.linalg.norm(w) ** 2

            t[np.isnan(t)] = np.median(t[np.isnan(t) is False])

            if svd_algo == 2:
                mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
                # Restore missing values instead of 0's
                m[mmis is False] = m0[mmis is False]
                m = np.reshape(m, (nxp, 1))
                m[np.argsort(np.reshape(mdiff, nxp))[nxpcov:nxp]] = np.nan
                m = np.reshape(m, (n, p))
                mmis[:, :] = np.isnan(m) is False
                # Replace missing values by 0's before regression
                m[mmis is False] = 0

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
                #                 normalize w
                w /= np.linalg.norm(w)
                if svd_algo == 2:
                    mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
                    # Restore missing values instead of 0's
                    m[mmis is False] = m0[mmis is False]
                    m = np.reshape(m, (nxp, 1))
                    # Outliers resume to missing values
                    m[np.argsort(np.reshape(mdiff, nxp))[nxpcov:nxp]] = np.nan
                    m = np.reshape(m, (n, p))
                    mmis[:, :] = np.isnan(m) is False
                    # Replace missing values by 0's before regression
                    m[mmis is False] = 0

                #                 build t
                if fast_code == 0:
                    wnorm[:, :] = np.repeat(w[:, np.newaxis] ** 2, n, axis=1) * mmis.T
                    denomw[:] = np.sum(wnorm, axis=0)
                    # Request at least 2 non-missing values to perform row
                    # regression
                    if svd_algo == 2:
                        denomw[np.count_nonzero(mmis, axis=1) < 2] = np.nan

                    t[:] = m @ w / denomw
                else:
                    t[:] = m @ w / np.linalg.norm(w) ** 2

                t[np.isnan(t)] = np.median(t[np.isnan(t) is False])
                # note: only w is normalized within loop, t is normalized after
                # convergence
                if svd_algo == 2:
                    mdiff[:, :] = np.abs(m0 - np.reshape(t, (n, 1)) @ np.reshape(w, (1, p)))
                    # Restore missing values instead of 0's
                    m[mmis is False] = m0[mmis is False]
                    m = np.reshape(m, (nxp, 1))
                    # Outliers resume to missing values
                    m[np.argsort(np.reshape(mdiff, nxp))[nxpcov:nxp]] = np.nan
                    m = np.reshape(m, (n, p))
                    mmis[:, :] = np.isnan(m) is False
                    # Replace missing values by 0's before regression
                    m[mmis is False] = 0

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
            # This does not make sense : both if and elif produce the same result, and diff_trial does not exist yet
            # anyway
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
        n_bootstrap=None,
        tol=1e-6,
        max_iter=150,
        max_iter_mult=20,
        regularization=None,
        sparsity=None,
        the_leverage="standard",
        convex=None,
        kernel="linear",
        skewness=False,
        null_priors=False,
        random_state=None,
        verbose=0,
):
    # noinspection PyShadowingNames
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

        sparsity : double, default: 0.
            Sparsity target with 0 <= sparsity <= 1 representing the % rows in w or h set to 0.

        the_leverage :  None | 'standard' | 'robust', default 'standard'
            Calculate the_leverage of w and h rows on each component.

        convex :  None | 'components' | 'transformation', default None
            Apply convex constraint on w or h.

        kernel :  'linear', 'quadratic', 'radial', default 'linear'
            Can be set if convex = 'transformation'.

        null_priors : boolean, default False
            Cells of h with prior cells = 0 will not be updated.
            Can be set only if prior h has been defined.

        skewness : boolean, default False
            When solving mixture problems, columns of x at the extremities of the convex hull will be given largest
            weights.
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

        v : scalar, volume occupied by w and h

        WB : array-like, shape (n_samples, n_components)
            Percent consistently clustered rows for each component.
            only if n_bootstrap > 0.

        HB : array-like, shape (n_features, n_components)
            Percent consistently clustered columns for each component.
            only if n_bootstrap > 0.

        b : array-like, shape (n_observations, n_components) or (n_features, n_components)
            only if active convex variant, h = b.T @ x or w = x @ b


        Examples
        --------

        >>> import numpy as np
        >>> x = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
        >>> from sklearn.decomposition import non_negative_factorization
        >>> w, h, n_iter = non_negative_factorization(x, n_components=2, random_state=0)


        References
        ----------

        Fogel

        Lin
        """

    m = x
    mt = None
    mw = None
    nmf_find_parts = None
    nmf_find_centroids = None
    nmf_kernel = None
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

    if the_leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif the_leverage == "robust":
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

    mt, mev, mw, mt_pct, mw_pct, diff0, mh, flag_nonconvex, add_message, err_message, cancel_pressed = r_nmf_solve(
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
            estimator.update([("w", mt), ("h", mw), ("v", volume)])
        else:
            estimator.update([("w", mt), ("h", mw), ("v", volume), ("b", mh)])

    else:
        if (nmf_find_parts == 0) & (nmf_find_centroids == 0):
            estimator.update([("w", mt), ("h", mw), ("v", volume), ("WB", mt_pct), ("HB", mw_pct)])
        else:
            estimator.update([("w", mt), ("h", mw), ("v", volume), ("b", mh), ("WB", mt_pct), ("HB", mw_pct)])

    return estimator


def nmf_predict(
        estimator, the_leverage="robust", blocks=None, cluster_by_stability=False, custom_order=False, verbose=0
):
    """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

    Parameters
    ----------

    estimator : tuplet as returned by non_negative_factorization

    the_leverage :  None | 'standard' | 'robust', default 'robust'
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

    FL : array-like, shape (n_blocks, n_components)
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

    FC : vector-like, shape (size(blocks))
         Block assigned cluster (NTF only)

    """

    mt = estimator["w"]
    mw = estimator["h"]
    if "f" in estimator:
        # x is a 3D tensor, in unfolded form of a 2D array
        # horizontal concatenation of blocks of equal size.
        mb = estimator["f"]
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

    if the_leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif the_leverage == "robust":
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
    if "f" in estimator:
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
                ("FL", mbn),
                ("FC", block_clust),
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
                ("FL", None),
                ("FC", None),
            ]
        )
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
        f=None,
        n_components=None,
        update_w=True,
        update_h=True,
        update_f=True,
        fast_hals=True,
        n_iter_hals=2,
        n_shift=0,
        unimodal=False,
        smooth=False,
        apply_left=False,
        apply_right=False,
        apply_block=False,
        n_bootstrap=None,
        tol=1e-6,
        max_iter=150,
        the_leverage="standard",
        random_state=None,
        verbose=0,
):
    # noinspection PyShadowingNames
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

        f : array-like, shape (n_blocks, n_components)
            prior h

        n_components : integer
            Number of components, if n_components is not set : n_components = min(n_samples, n_features)

        update_w : boolean, default: True
            Update or keep w fixed

        update_h : boolean, default: True
            Update or keep h fixed

        update_f : boolean, default: True
            Update or keep f fixed

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

        the_leverage :  None | 'standard' | 'robust', default 'standard'
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

            f : array-like, shape (n_blocks, n_components)
                Solution to the non-negative least squares problem.

            E : array-like, shape (n_samples, n_features x n_blocks)
                E is the residual tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and
                3rd dimensions.

            v : scalar, volume occupied by w and h

            WB : array-like, shape (n_samples, n_components)
                Percent consistently clustered rows for each component.
                only if n_bootstrap > 0.

            HB : array-like, shape (n_features, n_components)
                Percent consistently clustered columns for each component.
                only if n_bootstrap > 0.

        Examples
        --------

        >>> import numpy as np
        >>> x = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
        >>> from sklearn.decomposition import non_negative_factorization
        >>> w, h, n_iter = non_negative_factorization(x, n_components=2, random_state=0)


        References
        ----------

        Fogel

        Lin
        """

    m = x
    n, p = m.shape
    if n_components is None:
        nc = min(n, p)
    else:
        nc = n_components

    # n_blocks = n_blocks  # the fuck ?
    p_block = int(p / n_blocks)
    tolerance = tol
    precision = EPSILON
    log_iter = verbose
    ntf_unimodal = unimodal
    ntf_smooth = smooth
    ntf_left_components = apply_left
    ntf_right_components = apply_right
    ntf_block_components = apply_block
    if random_state is not None:
        random_seed = random_state
        np.random.seed(random_seed)

    my_status_box = StatusBoxTqdm(verbose=log_iter)

    if (w is None) & (h is None) & (f is None):
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

        if f is None:
            mb0 = np.ones((n_blocks, nc))
        else:
            mb0 = np.copy(f)

        mfit = np.zeros((n, p))
        for k in range(0, nc):
            for i_block in range(0, n_blocks):
                mfit[:, i_block * p_block: (i_block + 1) * p_block] += (
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
                mfit[:, i_block * p_block: (i_block + 1) * p_block] += (
                    mb0[i_block, k] * np.reshape(mt0[:, k], (n, 1)) @ np.reshape(mw0[:, k], (1, p_block))
                )

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

    if the_leverage == "standard":
        nmf_calculate_leverage = 1
        nmf_use_robust_leverage = 0
    elif the_leverage == "robust":
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

    if update_f:
        nmf_fix_user_bhe = 0
    else:
        nmf_fix_user_bhe = 1

    mt_conv, mt, mw, mb, mres, mt_pct, mw_pct, add_message, err_message, cancel_pressed = r_ntf_solve(
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
        ntf_unimodal,
        ntf_smooth,
        ntf_left_components,
        ntf_right_components,
        ntf_block_components,
        n_blocks,
        ntfn_conv,
        my_status_box,
    )

    volume = nmf_det(mt, mw, 1)

    for message in add_message:
        print(message)

    my_status_box.close()

    estimator = {}
    if nmf_robust_n_runs <= 1:
        estimator.update([("w", mt), ("h", mw), ("f", mb), ("E", mres), ("v", volume)])
    else:
        estimator.update([("w", mt), ("h", mw), ("f", mb), ("E", mres), ("v", volume), ("WB", mt_pct), ("HB", mw_pct)])

    return estimator
