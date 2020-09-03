"""Non-negative matrix and tensor factorization utility functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20

from tkinter import Tk, Frame, StringVar, Label, NONE, HORIZONTAL, Button
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
        self.status = Label(
            self.frame,
            textvariable=self.var,
            width=60,
            height=1)
        self.status.pack(fill=NONE, padx=6, pady=6)
        self.pbar = ttk.Progressbar(
            self.frame,
            orient=HORIZONTAL,
            max=100,
            mode='determinate')
        self.pbar.pack(fill=NONE, padx=6, pady=6)
        Button(
            self.frame,
            text='Cancel',
            command=self.close_dialog).pack(
            fill=NONE,
            padx=6,
            pady=6)
        self.cancel_pressed = False
        self.n_steps = 0

    def close_dialog(self):
        self.cancel_pressed = True

    def update_bar(self, delay=1, step=1):
        self.n_steps += step
        self.pbar.step(step)
        self.pbar.after(delay, lambda: self.root.quit())
        self.root.mainloop()

    def init_bar(self):
        self.update_bar(delay=1, step=100 - self.n_steps)
        self.n_steps = 0

    def update_status(self, delay=1, status=''):
        self.var.set(status)
        self.status.after(delay, lambda: self.root.quit())
        self.root.mainloop()

    def close(self):
        self.root.destroy()

    @staticmethod
    def my_print(status=''):
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

    def my_print(self, status=''):
        if self.log_iter == 1:
            print(status, end='\n')


def nmf_det(mt, mw, nmf_exact_det):
    """Volume occupied by Left and Right factoring vectors

    Input:
        m: Left hand matrix
        mw: Right hand matrix
        nmf_exact_det if = 0 compute an approximate determinant in reduced space n x n or p x p
        through random sampling in the largest dimension
    Output:
        det_xcells: determinant

    Reference
    ---------

    P. Fogel et al (2016) Applications of a Novel Clustering Approach Using Non-Negative Matrix Factorization to
    Environmental Research in Public Health Int. J. Environ. Res. Public Health 2016, 13,
    509; doi:10.3390/ijerph13050509

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


def nmf_get_convex_scores(mt, mw, mh, flag, add_message):
    """Rescale scores to sum up to 1 (used with deconvolution)
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
    add_message.insert(len(add_message), "Ncomp=" + str(nc) + ": Goodness of mixture fit = " + str(round(r2, 2)))
    # add_message.insert(len(add_message), 'Ncomp=' + str(nc) + ': Goodness of mixture fit before adjustement = ' +
    # str(round(r2, 2)))

    # for i in range(0, n):
    #     m[i, :] /= np.sum(m[i, :])

    return [mt, mw, mh, flag, add_message, err_message, cancel_pressed]


def percentile_exc(a, q):
    """Percentile, exclusive

    Input:
        a: Matrix
        q: Percentile
    Output:
        Percentile
    """
    return np.percentile(np.concatenate((np.array([np.min(a)]), a.flatten(), np.array([np.max(a)]))), q)


def robust_max(v0, add_message, my_status_box):
    """Robust max of column vectors

    For each column:
         = weighted mean of column elements larger than 95% percentile
        for each row, weight = specificity of the column value wrt other columns
    Input:
        v0: column vectors
    Output: Robust max by column

    Reference
    ---------

    P. Fogel et al (2016) Applications of a Novel Clustering Approach Using Non-Negative Matrix Factorization to
    Environmental Research in Public Health Int. J. Environ. Res. Public Health 2016, 13,
    509; doi:10.3390/ijerph13050509

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


def do_leverage(v, nmf_use_robust_leverage, add_message, my_status_box):
    """Calculate leverages

    Input:
        v: Input column vectors
        nmf_use_robust_leverage: Estimate robust through columns of v
    Output:
        vn: Leveraged column vectors

    Reference
    ---------

    P. Fogel et al (2016) Applications of a Novel Clustering Approach Using Non-Negative Matrix Factorization to
    Environmental Research in Public Health Int. J. Environ. Res. Public Health 2016, 13,
    509; doi:10.3390/ijerph13050509

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
    """Builder clusters from leverages

    """
    mbn = None
    mt_s = None
    mw_s = None
    j2 = None
    n_blocks = int(n_blocks)
    my_status_box.update_status(delay=1, status="Build clusters...")
    err_message = ""
    cancel_pressed = 0
    n, nc = np.shape(mt)
    p = np.shape(mw)[0]
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
        mtn, add_message, err_message, cancel_pressed = do_leverage(
            mt, nmf_use_robust_leverage, add_message, my_status_box
        )
        my_status_box.update_status(delay=1, status="Leverages - Right components...")
        mwn, add_message, err_message, cancel_pressed = do_leverage(
            mw, nmf_use_robust_leverage, add_message, my_status_box
        )
        if nmf_algo >= 5:
            my_status_box.update_status(delay=1, status="Leverages - Block components...")
            mbn, add_message, err_message, cancel_pressed = do_leverage(
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

    def set_clust(row_or_col, the_n, m_pct, mn):
        ret_row_or_col = row_or_col
        for the_i in range(0, the_n):
            if nc > 1:
                if (isinstance(m_pct, np.ndarray)) & (nmf_robust_cluster_by_stability > 0):
                    ret_row_or_col[the_i] = np.argmax(mt_pct[the_i, :]) + 1
                else:
                    ret_row_or_col[the_i] = np.argmax(mn[the_i, :]) + 1
            else:
                ret_row_or_col[the_i] = 1
        return ret_row_or_col

    row_clust = set_clust(row_clust, n, mt_pct, mtn)
    col_clust = set_clust(col_clust, p, mw_pct, mwn)

    if (cell_plot_ordered_clusters == 1) & (nc >= 3):
        mt_s = np.zeros(n)
        mw_s = np.zeros(p)

        def set_m(the_m, the_n, mn, clust):
            ret_m = the_m
            for the_i in range(0, the_n):
                if clust[the_i] == 1:
                    ret_m[the_i] = sum(the_k * mn[the_i, the_k] for the_k in range(0, 2)) / max(
                        sum(mn[the_i, the_k] for the_k in range(0, 2)), 1.0e-10
                    )
                elif clust[the_i] == nc:
                    ret_m[the_i] = sum(the_k * mn[the_i, the_k] for the_k in range(nc - 2, nc)) / max(
                        sum(mn[the_i, the_k] for the_k in range(nc - 2, nc)), 1.0e-10
                    )
                else:
                    ret_m[the_i] = sum(
                        the_k * mn[the_i, the_k] for the_k in range(int(clust[the_i] - 2), int(clust[the_i] + 1))
                    ) / max(
                        sum(mn[the_i, the_k] for the_k in range(int(clust[the_i] - 2), int(clust[the_i] + 1))), 1.0e-10
                    )
            return ret_m

        mt_s = set_m(mt_s, n, mtn, row_clust)
        mw_s = set_m(mw_s, p, mwn, col_clust)

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
    """Calculate Pvalue of each group versus cluster

    """
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
    """Calculate global significance of association with a covariate
        following multiple factorization trials
    """

    n, nc = mt.shape
    nrun = int(nrun)
    nb_groups = int(nb_groups)
    r_ct = r_ct.astype(np.int)
    n_ct = n_ct.astype(np.int)
    cluster_size = np.zeros(nc)
    r_ct = np.reshape(r_ct, n)
    n_ct = np.reshape(n_ct, nc)
    cancel_pressed = 0
    prun = cluster_prob = cluster_group = cluster_ngroup = None
    prun0 = row_groups0 = cluster_prob0 = cluster_group0 = cluster_ngroup0 = None
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

    v. K. Potluru & all (2013) Block Coordinate Descent for Sparse NMF arXiv:1301.3527v2 [cs.LG]

    """
    m = b.size
    if two_sided is False:
        m_alpha = (np.sqrt(m) - np.linalg.norm(b, ord=1) / np.linalg.norm(b, ord=2)) / (np.sqrt(m) - 1)
        if (alpha == 0) or (alpha <= m_alpha):
            return b

    b_rank = np.argsort(-b)
    ranks = np.empty_like(b_rank)
    ranks[b_rank] = np.arange(m)
    b_norm = np.linalg.norm(b)
    a = b[b_rank] / b_norm
    k = math.sqrt(m) - alpha * (math.sqrt(m) - 1)
    p0 = m
    mylambda0 = np.nan
    mu0 = np.nan
    mylambda = mylambda0
    mu = mu0

    for p in range(int(np.ceil(k ** 2)), m + 1):
        mylambda0 = mylambda
        mu0 = mu
        mylambda = -np.sqrt((p * np.linalg.norm(a[0:p]) ** 2 - np.linalg.norm(a[0:p], ord=1) ** 2) / (p - k ** 2))
        mu = -(np.linalg.norm(a[0:p], ord=1) + k * mylambda) / p
        if a[p - 1] < -mu:
            p0 = p - 1
            mylambda = mylambda0
            mu = mu0
            break

    x = np.zeros(m)
    x[0:p0] = -b_norm * (a[0:p0] + mu) / mylambda
    return x[ranks]
