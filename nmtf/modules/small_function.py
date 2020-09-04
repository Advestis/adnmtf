import numpy as np
from .nmtf_utils import do_leverage


def get_mmis(mmis):
    return mmis.astype(np.int)


def get_id(m):
    return np.where(np.isnan(m))


def update_m_mmis(m, mmis):
    mmis = get_mmis(mmis)
    n_mmis = mmis.shape[0]
    if n_mmis == 0:
        the_id = get_id(m)
        n_mmis = the_id[0].size
        if n_mmis > 0:
            mmis = np.isnan(m) is False
            mmis = get_mmis(mmis)
            m[mmis == 0] = 0
    else:
        m[mmis == 0] = 0
    return m, mmis, n_mmis


def update_col_clust(p, nmf_calculate_leverage, mwn, mw_pct, nmf_use_robust_leverage, add_message, my_status_box):
    err_message = cancel_pressed = None
    col_clust = np.zeros(p, dtype=int)
    if nmf_calculate_leverage > 0:
        mwn, add_message, err_message, cancel_pressed = do_leverage(
            mwn, nmf_use_robust_leverage, add_message, my_status_box
        )

    for j in range(0, p):
        col_clust[j] = np.argmax(np.array(mwn[j, :]))
        mw_pct[j, col_clust[j]] = mw_pct[j, col_clust[j]] + 1
    return col_clust, mwn, mw_pct, add_message, err_message, cancel_pressed


def update_row_clust(n, nmf_calculate_leverage, mt, mt_pct, nmf_use_robust_leverage, add_message, my_status_box):
    err_message = cancel_pressed = None
    row_clust = np.zeros(n, dtype=int)
    if nmf_calculate_leverage > 0:
        mtn, add_message, err_message, cancel_pressed = do_leverage(
            mt, nmf_use_robust_leverage, add_message, my_status_box
        )
    else:
        mtn = mt

    for i in range(0, n):
        row_clust[i] = np.argmax(mtn[i, :])
        mt_pct[i, row_clust[i]] = mt_pct[i, row_clust[i]] + 1
    return row_clust, mtn, mt_pct, add_message, err_message, cancel_pressed


def update_m(ntf_unimodal, compo, m, k, n):
    if (ntf_unimodal > 0) & (compo > 0):
        #                 Enforce unimodal distribution
        tmax = np.argmax(m[:, k])
        for i in range(tmax + 1, n):
            m[i, k] = min(m[i - 1, k], m[i, k])

        for i in range(tmax - 1, -1, -1):
            m[i, k] = min(m[i + 1, k], m[i, k])
    return m


def in_loop_update_m(m, k, ntf_unimodal, compo1, n, ntf_smooth, x, compo2=None):
    if compo2 is None:
        compo2 = compo1
    m = update_m(ntf_unimodal, compo1, m, k, n)

    if (ntf_smooth > 0) & (compo2 > 0):
        #             Smooth distribution
        x[0] = 0.75 * m[0, k] + 0.25 * m[1, k]
        x[n - 1] = 0.25 * m[n - 2, k] + 0.75 * m[n - 1, k]
        for i_block in range(1, n - 1):
            x[i_block] = 0.25 * m[i_block - 1, k] + 0.5 * m[i_block, k] + 0.25 * m[i_block + 1, k]

        m[:, k] = x
    return m, x


def set_nmf_attributes(m, mmis, nc):
    n, p = m.shape
    mmis = get_mmis(mmis)
    n_mmis = mmis.shape[0]
    if n_mmis == 0:
        the_id = get_id(m)
        n_mmis = the_id[0].size
        if n_mmis > 0:
            mmis = np.isnan(m) is False
            mmis = get_mmis(mmis)
            m[mmis == 0] = 0

    nc = int(nc)
    return n, p, n_mmis, m, mmis, nc


def set_uv(mt, mw, k, n):
    u1 = mt[:, k]
    u2 = -mt[:, k]
    u1[u1 < 0] = 0
    u2[u2 < 0] = 0
    v1 = mw[:, k]
    v2 = -mw[:, k]
    v1[v1 < 0] = 0
    v2[v2 < 0] = 0
    u1 = np.reshape(u1, (n, 1))
    return u1, u2, v1, v2


def do_svd_algo(w, n, mmis, svd_algo, t, m, mdiff, m0, nxp, nxpcov, p):
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
    return mdiff, m, mmis


def fast_code_and_svd_algo(fast_code, wnorm, w, n, mmis, denomw, svd_algo, t, m, mdiff, m0, nxp, nxpcov, p):
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
    mdiff, m, mmis = do_svd_algo(w, n, mmis, svd_algo, t, m, mdiff, m0, nxp, nxpcov, p)
    return wnorm, denomw, t, mdiff, m, mmis


def do_mfit(n, p, nc, n_blocks, p_block, mb0, mt0, mw0):
    mfit = np.zeros((n, p))
    for k in range(0, nc):
        for i_block in range(0, n_blocks):
            mfit[:, i_block * p_block: (i_block + 1) * p_block] += (
                mb0[i_block, k] * np.reshape(mt0[:, k], (n, 1)) @ np.reshape(mw0[:, k], (1, p_block))
            )
    return mfit


def do_inner_iter(h, alpha, grad, n_nmf_priors, nmf_priors, nmf_algo, n_vmis, vw, vmis, wt_w, decr_alpha, beta, hp):
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
    return h, alpha


def do_if_nmf_algo(nmf_algo, n_vmis, w, vmis, h, v, wt_wh, wt_v, wt_w):
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
    return wt_wh, wt_v


def do_initgrad(kernel, nc, m2, m3, m4, tol=None):

    trace_kernel = np.trace(kernel)
    # noinspection PyBroadException
    try:
        m1 = m2 @ np.linalg.inv(m2.T @ m2)
    except BaseException:
        m1 = m2 @ np.linalg.pinv(m2.T @ m2)

    m1[np.where(m1 < 0)] = 0
    for k in range(0, nc):
        scale = np.linalg.norm(m2[:, k])
        m2[:, k] = m2[:, k] / scale
        m1[:, k] = m1[:, k] * scale

    grad_1 = m3 @ (m4.T @ m4) - m4
    grad_2 = ((m3.T @ m3) @ m4.T - m3.T).T
    initgrad = np.linalg.norm(np.concatenate((grad_1, grad_2), axis=0))
    tol_1 = 1.0e-3 * initgrad
    if tol is not None:
        tol_2 = tol
    else:
        tol_2 = tol_1
    return m1, m2, grad_1, grad_2, initgrad, tol_1, tol_2, trace_kernel


def do_nmf_sparse(x, m, nc, percent_zeros, iter_sparse):
    sparse_test = np.zeros((x, 1))
    for k in range(0, nc):
        sparse_test[np.where(m[:, k] > 0)] = 1

    percent_zeros0 = percent_zeros
    n_sparse_test = np.where(sparse_test == 0)[0].size
    percent_zeros = max(n_sparse_test / x, 0.01)
    if percent_zeros == percent_zeros0:
        iter_sparse += 1
    else:
        iter_sparse = 0
    return percent_zeros0, iter_sparse, percent_zeros


def preinit_ntf_solve(m, mmis, nc, n_blocks):
    n, p0 = m.shape
    n_mmis = mmis.shape[0]
    nc = int(nc)
    n_blocks = int(n_blocks)
    p = int(p0 / n_blocks)
    n0 = int(n * n_blocks)
    nxp = int(n * p)
    nxp0 = int(n * p0)
    return n, p0, n_mmis, nc, n_blocks, p, nxp, nxp0, n0


def init_ntf_solve(m, mmis, nc, n_blocks, mt0, mw0, mb0, max_iterations):
    n, p0, n_mmis, nc, n_blocks, p, nxp, nxp0, n0 = preinit_ntf_solve(
        m, mmis, nc, n_blocks
    )
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
    return n, p0, n_mmis, nc, n_blocks, p, nxp, nxp0, mt, mw, mb, step_iter, pbar_step, id_blockp, a, b, c


def second_init_ntf_solve(nxp, n_mmis, m, mfit, mmis, my_status_box, n, p0):
    mtw = np.zeros(nxp)
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
    return mtw, denom_cutoff, mres, cont, i_iter, diff0, mpart


def reshape_mfit(n_blocks, mfit, id_blockp, mt, k, mw, n, mb, p):
    if n_blocks > 1:
        mfit = reshape_fit_if_nblocks(n_blocks, id_blockp, mfit, p, mb, k, mt, n, mw)
    else:
        mfit[:, id_blockp[0]: id_blockp[0] + p] += np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
    return mfit


def reshape_fit_if_nblocks(n_blocks, id_blockp, mfit, p, mb, k, mt, n, mw):
    for i_block in range(0, n_blocks):
        mfit[:, id_blockp[i_block]: id_blockp[i_block] + p] += (
            mb[i_block, k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
        )
    return mfit


def update_status(status0, i_iter, nmf_sparse_level, percent_zeros, alpha, log_iter, my_status_box, pbar_step, m1, m2,
                  m3, m4, m5, diff):
    status = status0 + "Iteration: %s" % int(i_iter)

    if nmf_sparse_level != 0:
        status = status + "; Achieved sparsity: " + str(round(percent_zeros, 2)) + "; alpha: " + str(round(alpha, 2))
        if log_iter == 1:
            my_status_box.my_print(status)

    my_status_box.update_status(delay=1, status=status)
    my_status_box.update_bar(delay=1, step=pbar_step)
    if my_status_box.cancel_pressed:
        cancel_pressed = 1
        if m1 is None:
            return [m2, m3, m4, m5, cancel_pressed]
        else:
            return [m1, m2, m3, m4, m5, cancel_pressed]

    if log_iter == 1:
        my_status_box.my_print(status0 + " Iter: " + str(i_iter) + " MSR: " + str(diff))


def denom_and_t2(denom, cutoff, id_block, k, t2, n_blocks, mtw, mb, mx_mmis, x):
    denom /= np.max(denom)
    denom[denom < cutoff] = cutoff
    for i_block in range(0, n_blocks):
        t2[:, k] += mx_mmis[:, id_block[i_block]: id_block[i_block] + x] @ mtw[:, k] * mb[i_block, k]

    t2[:, k] /= denom
    return denom, t2


def three_ifs(nmf_fix_user_lhe, norm_lhe, mt, k, nmf_fix_user_rhe, norm_rhe, mw, nmf_fix_user_bhe, norm_bhe, mb,
              n_blocks=None):
    norm = None
    if (nmf_fix_user_lhe > 0) & norm_lhe:
        norm = np.linalg.norm(mt[:, k])
        if norm > 0:
            mt[:, k] /= norm

    if (nmf_fix_user_rhe > 0) & norm_rhe:
        norm = np.linalg.norm(mw[:, k])
        if norm > 0:
            mw[:, k] /= norm

    if (nmf_fix_user_bhe > 0) & norm_bhe & ((n_blocks > 1) if n_blocks is not None else True):
        norm = np.linalg.norm(mb[:, k])
        if norm > 0:
            mb[:, k] /= norm
    return norm, mt, mw, mb
