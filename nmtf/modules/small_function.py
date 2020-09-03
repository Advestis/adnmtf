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
