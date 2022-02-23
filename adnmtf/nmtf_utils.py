"""Non-negative matrix and tensor factorization utility functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20

from tqdm import tqdm
import math
from scipy.stats import hypergeom
import logging
import numpy as np

EPSILON = np.finfo(np.float32).eps
logger = logging.getLogger(__name__)

# TODO (pcotte) typing


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
            logger.info(status)


def nmf_det(mt, mw, nmf_exact_det):
    """Volume occupied by Left and Right factoring vectors

    Parameters
    ----------
    mt:
       Left hand matrix
    mw:
       Right hand matrix
    nmf_exact_det:
       if = 0 compute an approximate determinant in reduced space n x n or p x p through random sampling in the largest
       dimension


    Returns
    -------
    determinant

    Reference
    ---------
    P. Fogel et al (2016) Applications of a Novel Clustering Approach Using Non-Negative Matrix Factorization to
    Environmental Research in Public Health Int. J. Environ. Res. Public Health 2016, 13, 509
    doi:10.3390/ijerph13050509

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
            theid = np.arange(n)
            np.random.shuffle(theid)
            theid = theid[0:p]
            for k in range(0, nc):
                xcells[:, k] = np.reshape(np.reshape(mt[theid, k], (p, 1)) @ np.reshape(mw[:, k], (1, p)), p ** 2)
                norm_k = np.linalg.norm(xcells[:, k])
                if norm_k > 0:
                    xcells[:, k] = xcells[:, k] / norm_k
                else:
                    xcells[:, k] = 0
        else:
            xcells = np.zeros((n ** 2, nc))
            theid = np.arange(p)
            np.random.shuffle(theid)
            theid = theid[0:n]
            for k in range(0, nc):
                xcells[:, k] = np.reshape(np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[theid, k], (1, n)), n ** 2)
                norm_k = np.linalg.norm(xcells[:, k])
                if norm_k > 0:
                    xcells[:, k] = xcells[:, k] / norm_k
                else:
                    xcells[:, k] = 0

    det_xcells = np.linalg.det(xcells.T @ xcells)
    return det_xcells


def robust_max(v0, add_message, my_status_box):
    """Robust max of column vectors

    For each column:
         = weighted mean of column elements larger than 95% percentile
        for each row, weight = specificity of the column value wrt other columns

    Parameters
    ----------
    v0: column vectors
    add_message: List[str]
    my_status_box

    Returns
    -------
    Robust max by column

    Reference
    ---------
    P. Fogel et al (2016) Applications of a Novel Clustering Approach Using Non-Negative Matrix Factorization to
    Environmental Research in Public Health Int. J. Environ. Res. Public Health 2016, 13, 509 doi:10.3390/ijerph13050509

    """
    err_message = ""
    cancel_pressed = 0

    v = v0.copy()
    n, nc = v.shape
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
    my_status_box.init_bar()

    while ((np.linalg.norm(rob_max - rob_max0) / np.linalg.norm(rob_max)) ** 2 > 1e-6) & (i_iter < max_iterations):
        for k in range(0, nc):
            v = v[np.argsort(-v[:, k]), :]
            if nc > 1:
                den = np.repeat(np.sum(v, axis=1), nc).reshape((n, nc))
                den[den == 0] = 1.0e-10
                prob = v / den
                prob[prob == 0] = 1.0e-10
                # TODO (pcotte) lnnc_i and nc_i could have not been assigned yet. Fix that.
                specificity = np.ones(n) + np.sum(prob * np.log(prob), axis=1) * lnnc_i
                specificity[prob[:, k] < nc_i] = 0
            else:
                specificity = np.ones(n)

            specificity[ind:n] = 0
            rob_max0[k] = rob_max[k]
            rob_max[k] = np.sum(v[:, k] * specificity) / np.sum(specificity)
            v[v[:, k] > rob_max[k], k] = rob_max[k]

        my_status_box.update_bar(step=pbar_step)
        if my_status_box.cancel_pressed:
            cancel_pressed = 1
            return rob_max * scale, add_message, err_message, cancel_pressed

        i_iter += 1

    if i_iter == max_iterations:
        add_message.insert(len(add_message), f"Warning: Max iterations reached while calculating robust max (N = {n})")

    return rob_max * scale, add_message, err_message, cancel_pressed


def calc_leverage(v, nmf_use_robust_leverage, add_message, my_status_box):
    """Calculate leverages

    Parameter
    ---------
    v: Input column vectors
    nmf_use_robust_leverage: Estimate robust through columns of V
    add_message
    my_status_box

    Returns
    -------
    vn: Leveraged column vectors

    Reference
    ---------
    P. Fogel et al (2016) Applications of a Novel Clustering Approach Using Non-Negative Matrix Factorization to
    Environmental Research in Public Health Int. J. Environ. Res. Public Health 2016, 13, 509 doi:10.3390/ijerph13050509

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
    my_status_box.init_bar()
    for k in range(0, nc):
        vr[v[:, k] > 0, k] = 1
        vn[:, k] = max_v[k] - v[:, k]
        vn[vn[:, k] < 0, k] = 0
        vn[:, k] = vn[:, k] ** 2
        for k2 in range(0, nc):
            if k2 != k:
                vn[:, k] = vn[:, k] + v[:, k2] ** 2

        status = f"Leverage: Comp {k + 1}"
        my_status_box.update_status(status=status)
        my_status_box.update_bar(step=pbar_step)
        if my_status_box.cancel_pressed:
            cancel_pressed = 1
            return vn, add_message, err_message, cancel_pressed

    vn = 10 ** (-vn / (2 * np.mean(vn))) * vr
    return vn, add_message, err_message, cancel_pressed


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
    """Builder clusters from leverages"""
    n_blocks = int(n_blocks)
    my_status_box.update_status(status="Build clusters...")
    err_message = ""
    cancel_pressed = 0
    n, nc = np.shape(mt)
    p = np.shape(mw)[0]
    if nmf_algo == "ntf":
        block_clust = np.zeros(n_blocks)
    elif nmf_algo == "nmf":
        block_clust = np.array([])
    else:
        raise ValueError(f"Unknown algo '{nmf_algo}'")

    r_ct = np.zeros(n)
    r_cw = np.zeros(p)
    n_ct = np.zeros(nc)
    n_cw = np.zeros(n_blocks * nc)
    row_clust = np.zeros(n)
    col_clust = np.zeros(p)
    ilast = 0
    jlast = 0

    if nmf_calculate_leverage == 1:
        my_status_box.update_status(status="Leverages - Left components...")
        mtn, add_message, err_message, cancel_pressed = calc_leverage(
            mt, nmf_use_robust_leverage, add_message, my_status_box
        )
        my_status_box.update_status(status="Leverages - Right components...")
        mwn, add_message, err_message, cancel_pressed = calc_leverage(
            mw, nmf_use_robust_leverage, add_message, my_status_box
        )
        if nmf_algo == "ntf":
            my_status_box.update_status(status="Leverages - Block components...")
            mbn, add_message, err_message, cancel_pressed = calc_leverage(
                mb, nmf_use_robust_leverage, add_message, my_status_box
            )
        else:
            mbn = None
    else:
        mtn = mt
        mwn = mw
        if nmf_algo == "ntf":
            mbn = mb
        else:
            mbn = None

    if nmf_algo == "ntf":
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
                # TODO (pcotte): mt_s could have not been assigned yet. Fix that.
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
            # TODO (pcotte): j2 could have not been assigned yet. Fix that.
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
                    # TODO (pcotte): mw_s could have not been assigned yet. Fix that.
                    mindex = mindex2[np.argsort(mw_s[mindex2])]
                else:
                    mindex = mindex2[np.argsort(-mwn[mindex2, k])]

                r_cw[jlast: len(mindex) + jlast] = mindex
                jlast += len(mindex)

            n_cw[i_block * nc + k] = jlast

    return (
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
    )


def cluster_pvalues(cluster_size, nb_groups, mt, r_ct, n_ct, row_groups, list_groups, ngroup):
    """Calculate Pvalue of each group versus cluster"""
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

    return prun, cluster_group, cluster_prob, cluster_ngroup, cluster_n_wgroup


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
                my_status_box.update_status(status=f"Calculating global significance: {irun}/{nrun}")
                my_status_box.update_bar(step=pbar_step)
                if my_status_box.cancel_pressed:
                    cancel_pressed = 1
                    # TODO (pcotte): prun, cluster_prob, cluster_group, cluster_ngroup could have
                    #  not been assigned yet. Fix that.
                    return cluster_size, pglob, prun, cluster_prob, cluster_group, cluster_ngroup, cancel_pressed

            prun, cluster_group, cluster_prob, cluster_ngroup, ClusterNWgroup = cluster_pvalues(
                cluster_size, nb_groups, mt, r_ct, n_ct, row_groups, list_groups, ngroup
            )
            if irun == 0:
                cluster_prob0 = np.copy(cluster_prob)
                cluster_group0 = np.copy(cluster_group)
                cluster_ngroup0 = np.copy(cluster_ngroup)
                row_groups0 = np.copy(row_groups)
                prun0 = prun
            else:
                # TODO (pcotte): prun0 could have not been assigned yet. Fix that.
                if prun >= prun0:
                    pglob += 1

            if irun < nrun - 1:
                # permute row groups
                # TODO (pcotte): row_groups0 could have not been assigned yet. Fix that.
                row_groups = row_groups0[np.random.permutation(n)]
            else:
                # Restore
                # TODO (pcotte): cluster_prob0, cluster_group0, cluster_ngroup0 could have not been assigned yet.
                # Fix that.
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

    # TODO (pcotte): prun, cluster_prob, cluster_group, cluster_ngroup could have
    #  not been assigned yet. Fix that.
    return cluster_size, pglob, prun, cluster_prob, cluster_group, cluster_ngroup, cancel_pressed


def sparse_opt(b, alpha, two_sided):
    """Return the L2-closest vector with sparsity alpha

    Parameters
    ----------
    b: original vector
    alpha
    two_sided

    Returns
    -------
    x: sparse vector

    Reference
    ---------
    V. K. Potluru & all (2013) Block Coordinate Descent for Sparse NMF arXiv:1301.3527v2 [cs.LG]

    Examples
    --------
    >>> from adnmtf.nmtf_utils import sparse_opt
    >>> b_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> alpha_ = 1
    >>> two_sided_ = True
    >>> sparse_opt(b_, alpha_, two_sided_)
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., nan])
    >>> two_sided_ = False
    >>> sparse_opt(b_, alpha_, two_sided_)
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., nan])
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
