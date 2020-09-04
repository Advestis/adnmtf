"""Non-negative matrix and tensor factorization core functions

"""

# Author: Paul Fogel

# License: MIT
# Jan 4, '20

import math
import numpy as np

from .nmtf_utils import percentile_exc, nmf_get_convex_scores, shift, sparse_opt
from .small_function import in_loop_update_m, do_inner_iter, do_if_nmf_algo, do_initgrad, do_nmf_sparse, \
    reshape_mfit, init_ntf_solve, second_init_ntf_solve, reshape_fit_if_nblocks, update_status, denom_and_t2, \
    three_ifs, preinit_ntf_solve


def nmf_proj_grad(v, vmis, w, hinit, nmf_algo, lambdax, tol, max_iterations, nmf_priors):
    """Projected gradient
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

    Reference
    ---------

    x.J. Lin (2007) Projected Gradient Methods for Non-negative Matrix Factorization
    Neural Comput. 2007 Oct;19(10):2756-79.

    """
    h = hinit
    wt_w = wt_wh = wt_v = decr_alpha = hp = h0 = None
    # noinspection PyBroadException
    try:
        n_nmf_priors, nc = nmf_priors.shape
    except BaseException:
        n_nmf_priors = 0

    n_vmis = vmis.shape[0]
    n, p = np.shape(v)
    n, nc = np.shape(w)
    alpha = 1

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

            wt_wh, wt_v = do_if_nmf_algo(nmf_algo, n_vmis, w, vmis, h, v, wt_wh, wt_v, wt_w)

            if lambdax > 0:
                grad = wt_wh - wt_v + lambdax
            else:
                grad = wt_wh - wt_v

            projgrad = np.linalg.norm(grad[(grad < 0) | (h > 0)])

            if projgrad >= tol:
                do_inner_iter(h, alpha, grad, n_nmf_priors, nmf_priors, nmf_algo, n_vmis, w, vmis, wt_w, decr_alpha,
                              beta, hp)

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


def nmf_proj_grad_kernel(kernel, v, vmis, w, hinit, nmf_algo, tol, max_iterations, nmf_priors):
    """Projected gradient, kernel version
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

    Reference
    ---------

    x.J. Lin (2007) Projected Gradient Methods for Non-negative Matrix Factorization
        Neural Comput. 2007 Oct;19(10):2756-79.

    """
    wt_w = wt_wh = wt_v = decr_alpha = hp = None
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
            wt_wh, wt_v = do_if_nmf_algo(nmf_algo, n_vmis, vw, vmis, h, v, wt_wh, wt_v, wt_w)

            grad = wt_wh - wt_v
            projgrad = np.linalg.norm(grad[(grad < 0) | (h > 0)])
            if projgrad >= tol:
                h, alpha = do_inner_iter(h, alpha, grad, n_nmf_priors, nmf_priors, nmf_algo, n_vmis, vw, vmis, wt_w,
                                         decr_alpha, beta, hp)
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


def nmf_apply_kernel(m, nmf_kernel, mt, mw):
    """Calculate kernel (used with convex NMF)
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
    kernel = None
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

    return kernel


def nmf_reweigh(m, mt, nmf_priors, add_message):
    """Overload skewed variables (used with deconvolution only)
    Input:
         m: Input matrix
         m: Left hand matrix
         nmf_priors: priors on right hand matrix
    Output:
         nmf_priors: updated priors

    Note: This code is still experimental

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
         mw: Right hand matrix
         diff: objective cost
         mh: Convexity matrix
         nmf_priors: Updated priors on right hand matrix
         flag_nonconvex: Updated non-convexity flag on left hand matrix

    Reference
    ---------

    x. h.q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
    IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1

    """
    err_message = ""
    cancel_pressed = 0
    kernel = tol_mw = the_ip = the_in = tol_mh = tol_mt = trace_kernel = diff0 = None
    n, p = m.shape
    n_mmis = mmis.shape[0]
    # noinspection PyBroadException
    try:
        n_nmf_priors, nc = nmf_priors.shape
    except BaseException:
        n_nmf_priors = 0

    nc = int(nc)
    nxp = int(n * p)
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
                return [mt, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed]
        else:
            nmf_priors[nmf_priors > 0] = 1

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
                    mw, tol_mw = nmf_proj_grad_kernel(kernel, m, mmis, mh, mw, nmf_algo, tol_mw, nmf_max_iter_proj,
                                                      nmf_priors.T)
                elif (nmf_convex > 0) & (nmf_find_centroids > 0):
                    mh, tol_mh, dummy = nmf_proj_grad(
                        the_in, np.array([]), mt, mh.T, nmf_algo, -1, tol_mh, nmf_max_iter_proj, np.array([])
                    )
                else:
                    mw, tol_mw, lambdaw = nmf_proj_grad(
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
                    if n_mmis > 0:
                        mt = mt * (((m * mmis).T / (mw @ mt.T + precision)).T @ mw)
                    else:
                        mt = mt * ((m.T / (mw @ mt.T + precision)).T @ mw)
            else:
                # Projected gradient
                if (nmf_convex > 0) & (nmf_find_parts > 0):
                    mh, tol_mh, dummy = nmf_proj_grad(
                        the_ip, np.array([]), mw, mh.T, nmf_algo, -1, tol_mh, nmf_max_iter_proj, np.array([])
                    )
                elif (nmf_convex > 0) & (nmf_find_centroids > 0):
                    mt, tol_mt = nmf_proj_grad_kernel(
                        kernel, m.T, mmis.T, mh, mt, nmf_algo, tol_mt, nmf_max_iter_proj, np.array([])
                    )
                else:
                    mt, tol_mt, lambdat = nmf_proj_grad(
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
                return [mt, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed]

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
                    the_ip = np.identity(p)
                    if (nmf_algo == 2) or (nmf_algo == 4):
                        if n_mmis > 0:
                            kernel = nmf_apply_kernel(mmis * m, 1, np.array([]), np.array([]))
                        else:
                            kernel = nmf_apply_kernel(m, 1, np.array([]), np.array([]))
                    else:
                        if n_mmis > 0:
                            kernel = nmf_apply_kernel(mmis * (m / (mt @ mw.T)), 1, np.array([]), np.array([]))
                        else:
                            kernel = nmf_apply_kernel(m / (mt @ mw.T), 1, np.array([]), np.array([]))

                    mh, mw, grad_mh, grad_mw, initgrad, tol_mh, tol_mw, trace_kernel = do_initgrad(
                        kernel, nc, mw, mh, mw, tol=tol_mt
                    )
                elif nmf_find_centroids > 0:
                    the_in = np.identity(n)
                    if (nmf_algo == 2) or (nmf_algo == 4):
                        if n_mmis > 0:
                            kernel = nmf_apply_kernel(mmis.T * m.T, 1, np.array([]), np.array([]))
                        else:
                            kernel = nmf_apply_kernel(m.T, 1, np.array([]), np.array([]))
                    else:
                        if n_mmis > 0:
                            kernel = nmf_apply_kernel(mmis.T * (m.T / (mt @ mw.T).T), 1, np.array([]), np.array([]))
                        else:
                            kernel = nmf_apply_kernel(m.T / (mt @ mw.T).T, 1, np.array([]), np.array([]))

                    mh, mt, grad_mt, grad_mh, initgrad, tol_mh, tol_mt, trace_kernel = do_initgrad(
                        kernel, nc, mt, mt, mh
                    )

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
                        kernel = nmf_apply_kernel(mmis.T * m.T, nmf_kernel, mw, mt)
                    else:
                        kernel = nmf_apply_kernel(m.T, nmf_kernel, mw, mt)
                else:
                    if n_mmis > 0:
                        kernel = nmf_apply_kernel(mmis.T * (m.T / (mt @ mw.T).T), nmf_kernel, mw, mt)
                    else:
                        kernel = nmf_apply_kernel(m.T / (mt @ mw.T).T, nmf_kernel, mw, mt)

                mh, mt, grad_mt, grad_mh, initgrad, tol_mh, tol_mt, trace_kernel = do_initgrad(
                    kernel, nc, mt, mt, mh
                )

            if nmf_sparse_level > 0:
                percent_zeros0, iter_sparse, percent_zeros = do_nmf_sparse(
                    p, mw, nc, percent_zeros, iter_sparse
                )

                if (percent_zeros < 0.99 * nmf_sparse_level) & (iter_sparse < 50):
                    lambdaw *= min(1.01 * nmf_sparse_level / percent_zeros, 1.10)
                    i_iter = nmf_max_interm + 1
                    cont = 1

            elif nmf_sparse_level < 0:
                percent_zeros0, iter_sparse, percent_zeros = do_nmf_sparse(
                    n, mt, nc, percent_zeros, iter_sparse
                )

                if (percent_zeros < 0.99 * abs(nmf_sparse_level)) & (iter_sparse < 50):
                    lambdat *= min(1.01 * abs(nmf_sparse_level) / percent_zeros, 1.10)
                    i_iter = nmf_max_interm + 1
                    cont = 1

    if nmf_find_parts > 0:
        # Make m convex
        mt = m @ mh
        mt, mw, mh, flag_nonconvex, add_message, err_message, cancel_pressed = nmf_get_convex_scores(
            mt, mw, mh, flag_nonconvex, add_message
        )
    elif nmf_find_centroids > 0:
        # Calculate row centroids
        for k in range(0, nc):
            scale_mh = np.sum(mh[:, k])
            mh[:, k] = mh[:, k] / scale_mh
            mt[:, k] = mt[:, k] * scale_mh

        mw = (mh.T @ m).T

    if (nmf_kernel > 1) & (nl_kernel_applied == 1):
        diff /= trace_kernel / nxp

    return [mt, mw, diff, mh, nmf_priors, flag_nonconvex, add_message, err_message, cancel_pressed]


def ntf_stack(m, mmis, n_blocks):
    """Unfold tensor m
        for future use with NMF
    """
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
        nmf_sparse_level,
        ntf_unimodal,
        ntf_smooth,
        ntf_left_components,
        ntf_right_components,
        ntf_block_components,
        n_blocks,
        ntfn_conv,
        nmf_priors,
        my_status_box,
):
    """Interface to:
            - ntf_solve_simple
            - ntf_solve_conv
    """

    # noinspection PyBroadException
    try:
        n_nmf_priors, nc = nmf_priors.shape
    except BaseException:
        n_nmf_priors = 0

    if n_nmf_priors > 0:
        nmf_priors[nmf_priors > 0] = 1

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
            nmf_sparse_level,
            ntf_unimodal,
            ntf_smooth,
            ntf_left_components,
            ntf_right_components,
            ntf_block_components,
            n_blocks,
            ntfn_conv,
            nmf_priors,
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
            nmf_sparse_level,
            ntf_unimodal,
            ntf_smooth,
            ntf_left_components,
            ntf_right_components,
            ntf_block_components,
            n_blocks,
            nmf_priors,
            my_status_box,
        )


# noinspection DuplicatedCode
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
        nmf_sparse_level,
        ntf_unimodal,
        ntf_smooth,
        ntf_left_components,
        ntf_right_components,
        ntf_block_components,
        n_blocks,
        nmf_priors,
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
         nmf_sparse_level : sparsity level (as defined by Hoyer); +/- = make RHE/LHe sparse
         ntf_unimodal: Apply Unimodal constraint on factoring vectors
         ntf_smooth: Apply Smooth constraint on factoring vectors
         compo: Apply Unimodal/Smooth constraint on left hand matrix
         ntf_right_components: Apply Unimodal/Smooth constraint on right hand matrix
         compo1: Apply Unimodal/Smooth constraint on block hand matrix
         n_blocks: Number of NTF blocks
         nmf_priors: Elements in mw that should be updated (others remain 0)

    Output:
         m: Left hand matrix
         mw: Right hand matrix
         mb: Block hand matrix
         diff: objective cost

    Reference
    ---------

    a. Cichocki, P.h.a.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

    """

    diff = None
    cancel_pressed = 0

    (n,
     p0,
     n_mmis,
     nc, n_blocks,
     p,
     nxp,
     nxp0,
     mt,
     mw,
     mb,
     step_iter,
     pbar_step,
     id_blockp,
     a,
     b,
     c) = init_ntf_solve(m, mmis, nc, n_blocks, mt0, mw0, mb0, max_iterations)

    # Compute Residual tensor
    mfit = np.zeros((n, p0))
    for k in range(0, nc):
        mfit = reshape_mfit(n_blocks, mfit, id_blockp, mt, k, mw, n, mb, p)

    denomt = np.zeros(n)
    denomw = np.zeros(p)
    denom_block = np.zeros((n_blocks, nc))
    mt2 = np.zeros(n)
    mw2 = np.zeros(p)
    mt_mw, denom_cutoff, mres, cont, i_iter, diff0, mpart = second_init_ntf_solve(
        nxp, n_mmis, m, mfit, mmis, my_status_box, n, p0
    )
    # alpha = nmf_sparse_level
    alpha = nmf_sparse_level / 2
    percent_zeros = 0
    iter_sparse = 0

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
                alpha,
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
                nmf_priors,
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
                alpha,
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
                nmf_priors,
            )

        if i_iter % step_iter == 0:
            # Check convergence
            diff = np.linalg.norm(mres) ** 2 / nxp0
            if (diff0 - diff) / diff0 < tolerance:
                cont = 0
            else:
                if diff > diff0:
                    my_status_box.my_print(status0 + " Iter: " + str(i_iter) + " MSR does not improve")

                diff0 = diff

            ret = update_status(status0, i_iter, nmf_sparse_level, percent_zeros, alpha, log_iter, my_status_box,
                                pbar_step, np.array([]), mt, mw, mb, mres, diff)
            if ret is not None:
                return ret
            status = status0 + "Iteration: %s" % int(i_iter)

            if nmf_sparse_level != 0:
                status = (
                    status + "; Achieved sparsity: " + str(round(percent_zeros, 2)) + "; alpha: " + str(
                        round(alpha, 2))
                )
                if log_iter == 1:
                    my_status_box.my_print(status)

            my_status_box.update_status(delay=1, status=status)
            my_status_box.update_bar(delay=1, step=pbar_step)
            if my_status_box.cancel_pressed:
                cancel_pressed = 1
                return [np.array([]), mt, mw, mb, mres, cancel_pressed]

            if log_iter == 1:
                my_status_box.my_print(status0 + " Iter: " + str(i_iter) + " MSR: " + str(diff))

        i_iter += 1

        if (cont == 0) | (i_iter == max_iterations):
            # if nmf_sparse_level > 0:
            #     sparse_test = np.zeros((p, 1))
            #     for k in range(0, nc):
            #         sparse_test[np.where(mw[:, k] > 0)] = 1

            #     percent_zeros0 = percent_zeros
            #     n_SparseTest = np.where(sparse_test == 0)[0].size
            #     percent_zeros = max(n_SparseTest / p, .01)
            #     if percent_zeros == percent_zeros0:
            #         iter_sparse += 1
            #     else:
            #         iter_sparse = 0

            #     if (percent_zeros < 0.99 * nmf_sparse_level) & (iter_sparse < 50):
            #         alpha *= min(1.01 * nmf_sparse_level / percent_zeros, 1.01)
            #         if alpha < .99:
            #             i_iter = 1
            #             cont = 1

            # elif nmf_sparse_level < 0:
            #     sparse_test = np.zeros((n, 1))
            #     for k in range(0, nc):
            #         sparse_test[np.where(m[:, k] > 0)] = 1

            #     percent_zeros0 = percent_zeros
            #     n_SparseTest = np.where(sparse_test == 0)[0].size
            #     percent_zeros = max(n_SparseTest / n, .01)
            #     if percent_zeros == percent_zeros0:
            #         iter_sparse += 1
            #     else:
            #         iter_sparse = 0

            #     if (percent_zeros < 0.99 * abs(nmf_sparse_level)) & (iter_sparse < 50):
            #         alpha *= min(1.01 * abs(nmf_sparse_level) / percent_zeros, 1.01)
            #         if abs(alpha) < .99:
            #             i_iter = 1
            #             cont = 1

            def a_function(_nc, _percent_zeros, _mw, x, _iter_sparse):
                _sparse_test = np.zeros((_nc, 1))
                _percent_zeros0 = _percent_zeros
                for _k in range(0, _nc):
                    sparse_test[_k] = np.where(_mw[:, _k] == 0)[0].size

                _percent_zeros = np.mean(sparse_test) / x
                if _percent_zeros < percent_zeros0:
                    _iter_sparse += 1
                else:
                    _iter_sparse = 0
                return _sparse_test, _percent_zeros0, _percent_zeros, _iter_sparse

            if nmf_sparse_level > 0:
                sparse_test, percent_zeros0, percent_zeros, iter_sparse = a_function(nc, percent_zeros, mw, p,
                                                                                     iter_sparse)

                if (percent_zeros < 0.99 * nmf_sparse_level) & (iter_sparse < 50):
                    alpha *= min(1.01 * nmf_sparse_level / percent_zeros, 1.1)
                    if alpha < 1:
                        i_iter = 1
                        cont = 1

            elif nmf_sparse_level < 0:
                sparse_test, percent_zeros0, percent_zeros, iter_sparse = a_function(nc, percent_zeros, mw, n,
                                                                                     iter_sparse)

                if (percent_zeros < 0.99 * abs(nmf_sparse_level)) & (iter_sparse < 50):
                    alpha *= min(1.01 * abs(nmf_sparse_level) / percent_zeros, 1.1)
                    if abs(alpha) < 1:
                        i_iter = 1
                        cont = 1

    if (n_mmis > 0) & (nmf_fix_user_bhe == 0):
        mb *= denom_block

    return [np.array([]), mt, mw, mb, diff, cancel_pressed]


# noinspection DuplicatedCode
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
        nmf_sparse_level,
        ntf_unimodal,
        ntf_smooth,
        ntf_left_components,
        ntf_right_components,
        ntf_block_components,
        n_blocks,
        ntfn_conv,
        nmf_priors,
        my_status_box,
):
    """Estimate NTF matrices (HALS)
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
         nmf_sparse_level : sparsity level (as defined by Hoyer); +/- = make RHE/LHe sparse
         ntf_unimodal: Apply Unimodal constraint on factoring vectors
         ntf_smooth: Apply Smooth constraint on factoring vectors
         compo: Apply Unimodal/Smooth constraint on left hand matrix
         ntf_right_components: Apply Unimodal/Smooth constraint on right hand matrix
         compo1: Apply Unimodal/Smooth constraint on block hand matrix
         n_blocks: Number of NTF blocks
         ntfn_conv: Half-Size of the convolution window on 3rd-dimension of the tensor
         nmf_priors: Elements in mw that should be updated (others remain 0)

     Output:
         m : if ntfn_conv > 0 only otherwise empty. Contains sub-components for each phase in convolution window
         mt_simple: Left hand matrix (sum of columns Mt_conv for each k)
         mw_simple: Right hand matrix
         mb_simple: Block hand matrix
         diff: objective cost

     Note:
         This code extends HALS to allow for shifting on the 3rd dimension of the tensor. Suffix '_simple' is added to
         the non-convolutional components. Convolutional components are named the usual way.

     """

    ntfn_conv = int(ntfn_conv)
    diff = None
    cancel_pressed = 0

    (n,
     p0,
     n_mmis,
     nc, n_blocks,
     p,
     nxp,
     nxp0,
     mt_simple,
     mw_simple,
     mb_simple,
     step_iter,
     pbar_step,
     id_blockp,
     a,
     b,
     c) = init_ntf_solve(m, mmis, nc, n_blocks, mt0, mw0, mb0, max_iterations)

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
            mfit = reshape_fit_if_nblocks(n_blocks, id_blockp, mfit, p, mb, k, mt, n, mw)

    denomt = np.zeros(n)
    denomw = np.zeros(p)
    denom_block = np.zeros((n_blocks, nc))
    mt2 = np.zeros(n)
    mw2, denom_cutoff, mres, cont, i_iter, diff0, mpart = second_init_ntf_solve(
        nxp, n_mmis, m, mfit, mmis, my_status_box, n, p0
    )
    alpha = nmf_sparse_level
    # alpha_blocks = 0  # Unused ?
    percent_zeros = 0
    iter_sparse = 0

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
                    alpha,
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
                    nmf_priors,
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
                    alpha,
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
                    nmf_priors,
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

            ret = update_status(status0, i_iter, nmf_sparse_level, percent_zeros, alpha, log_iter, my_status_box,
                                pbar_step, None, mt, mt_simple, mw_simple, mb_simple, diff)
            if ret is not None:
                return ret

        i_iter += 1

        if (cont == 0) | (i_iter == max_iterations):
            if nmf_sparse_level > 0:
                percent_zeros0, iter_sparse, percent_zeros = do_nmf_sparse(
                    p, mw, nc, percent_zeros, iter_sparse
                )

                if (percent_zeros < 0.99 * nmf_sparse_level) & (iter_sparse < 50):
                    alpha *= min(1.01 * nmf_sparse_level / percent_zeros, 1.01)
                    if alpha < 0.99:
                        i_iter = 1
                        cont = 1

            elif nmf_sparse_level < 0:
                percent_zeros0, iter_sparse, percent_zeros = do_nmf_sparse(
                    n, mt, nc, percent_zeros, iter_sparse
                )

                if (percent_zeros < 0.99 * abs(nmf_sparse_level)) & (iter_sparse < 50):
                    alpha *= min(1.01 * abs(nmf_sparse_level) / percent_zeros, 1.01)
                    if abs(alpha) < 0.99:
                        i_iter = 1
                        cont = 1

    if (n_mmis > 0) & (nmf_fix_user_bhe == 0):
        mb *= denom_block

    return [mt, mt_simple, mw_simple, mb_simple, diff, cancel_pressed]


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
    """Estimate NTF matrices (fast HALS)
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
         diff: objective cost

     Note: This code does not support missing values, nor sparsity constraint

     """
    mres = np.array([])
    mx_mmis2 = denom_block = denomt = denom_cutoff = mx_mmis = denomw = diff = None
    cancel_pressed = 0

    n, p0, n_mmis, nc, n_blocks, p, nxp, nxp0, n0 = preinit_ntf_solve(
        m, mmis, nc, n_blocks
    )
    mt = np.copy(mt0)
    mw = np.copy(mw0)
    mb = np.copy(mb0)
    step_iter = math.ceil(max_iterations / 10)
    pbar_step = 100 * step_iter / max_iterations

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
        norm, mt, mw, mb = three_ifs(nmf_fix_user_lhe, norm_lhe, mt, k, nmf_fix_user_rhe, norm_rhe, mw,
                                     nmf_fix_user_bhe, norm_bhe, mb)

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
                m[:, id_blockp[i_block]: id_blockp[i_block] + p] * mmis[
                    :, id_blockp[i_block]: id_blockp[i_block] + p]
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

                    denomt, t2t = denom_and_t2(denomt, denom_cutoff, id_blockp, k, t2t, n_blocks, mw, mb, mx_mmis, p)
                else:
                    for i_block in range(0, n_blocks):
                        t2t[:, k] += m[:, id_blockp[i_block]: id_blockp[i_block] + p] @ mw[:, k] * mb[i_block, k]

            mt2 = mt.T @ mt
            mt2[mt2 == 0] = precision
            t3 = t1 / mt2

            for k in range(0, nc):
                mt[:, k] = gamma[k] * mt[:, k] + t2t[:, k] - mt @ t3[:, k]
                m[np.where(m[:, k] < 0), k] = 0
                mt, a = in_loop_update_m(mt, k, ntf_unimodal, ntf_left_components, n, ntf_smooth, a)

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

                    denomw, t2w = denom_and_t2(denomw, denom_cutoff, id_blockn, k, t2w, n_blocks, mt, mb, mx_mmis2, n)
                else:
                    for i_block in range(0, n_blocks):
                        t2w[:, k] += m2[:, id_blockn[i_block]: id_blockn[i_block] + n] @ mt[:, k] * mb[i_block, k]

            mw2 = mw.T @ mw
            mw2[mw2 == 0] = precision
            t3 = t1 / mw2

            for k in range(0, nc):
                mw[:, k] = gamma[k] * mw[:, k] + t2w[:, k] - mw @ t3[:, k]
                m[np.where(m[:, k] < 0), k] = 0
                mw, b = in_loop_update_m(mw, k, ntf_unimodal, ntf_right_components, p, ntf_smooth,
                                         b, ntf_left_components)

                if norm_rhe:
                    mw[:, k] /= np.linalg.norm(mw[:, k])

            mw2 = mw.T @ mw
            mw2[mw2 == 0] = precision
            t1 = t3 * mw2

        if nmf_fix_user_bhe == 0:
            def a_reshape(_mx_mmis, _id_blockp, _i_block, _p, _nxp, _mt, _k, _n, _mw):
                return np.reshape(_mx_mmis[:, _id_blockp[_i_block]: _id_blockp[_i_block] + _p], _nxp).T @ (np.reshape((
                    np.reshape(_mt[:, _k], (_n, 1)) @ np.reshape(_mw[:, _k], (1, _p))), _nxp))

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
                            a_reshape(mx_mmis, id_blockp, i_block, p, nxp, mt, k, n, mw) / denom_block[i_block, k]
                        )

                else:
                    for i_block in range(0, n_blocks):
                        t2_block[i_block, k] = a_reshape(
                            m, id_blockp, i_block, p, nxp, mt, k, n, mw) / denom_block[i_block, k]

            mb2 = mb.T @ mb
            mb2[mb2 == 0] = precision
            t3 = t1 / mb2

            for k in range(0, nc):
                mb[:, k] = mb[:, k] + t2_block[:, k] - mb @ t3[:, k]
                m[np.where(m[:, k] < 0), k] = 0
                mb, c = in_loop_update_m(mb, k, ntf_unimodal, ntf_block_components, n_blocks, ntf_smooth,
                                         c, ntf_left_components)

            mb2 = mb.T @ mb
            mb2[mb2 == 0] = precision
            t1 = t3 * mb2

        if i_iter % step_iter == 0:
            # Update residual tensor
            mfit[:, :] = 0

            for k in range(0, nc):
                for i_block in range(0, n_blocks):
                    mfit[:, id_blockp[i_block]: id_blockp[i_block] + p] += mb[i_block, k] * (
                        np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
                    )
                if n_mmis > 0:
                    mres = (m - mfit) * mmis
                else:
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

    return [mt, mw, mb, diff, cancel_pressed]


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
        alpha,
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
        nmf_priors,
):
    """Core updating code called by ntf_solve_simple & NTF Solve_conv
    Input:
        All variables in the calling function used in the function
    Output:
        Same as Input
    """

    # noinspection PyBroadException
    try:
        n_nmf_priors, nc = nmf_priors.shape
    except BaseException:
        n_nmf_priors = 0

    # Compute kth-part
    if n_blocks > 1:
        for i_block in range(0, n_blocks):
            mpart[:, id_blockp[i_block]: id_blockp[i_block] + p] = (
                mb[i_block, k] * np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))
            )
    else:
        mpart[:, id_blockp[0]: id_blockp[0] + p] = np.reshape(mt[:, k], (n, 1)) @ np.reshape(mw[:, k], (1, p))

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

    norm, mt, mw, mb = three_ifs(nmf_fix_user_lhe, norm_lhe, mt, k, nmf_fix_user_rhe, norm_rhe, mw, nmf_fix_user_bhe,
                                 norm_bhe, mb, n_blocks)

    if nmf_fix_user_lhe == 0:
        # Update m
        mt[:, k] = 0
        if n_blocks > 1:
            for i_block in range(0, n_blocks):
                mt[:, k] += mb[i_block, k] * mpart[:, id_blockp[i_block]: id_blockp[i_block] + p] @ mw[:, k]
        else:
            mt[:, k] += mpart[:, id_blockp[0]: id_blockp[0] + p] @ mw[:, k]

        if n_mmis > 0:
            denomt[:] = 0
            mw2[:] = mw[:, k] ** 2
            if n_blocks > 1:
                for i_block in range(0, n_blocks):
                    # Broadcast missing cells into mw to calculate mw.T * mw
                    denomt += mb[i_block, k] ** 2 * mmis[:, id_blockp[i_block]: id_blockp[i_block] + p] @ mw2
            else:
                denomt += mmis[:, id_blockp[0]: id_blockp[0] + p] @ mw2

            denomt /= np.max(denomt)
            denomt[denomt < denom_cutoff] = denom_cutoff
            mt[:, k] /= denomt

        mt[mt[:, k] < 0, k] = 0
        if alpha < 0:
            mt[:, k] = sparse_opt(mt[:, k], -alpha, False)

        mt, a = in_loop_update_m(mt, k, ntf_unimodal, ntf_left_components, n, ntf_smooth, a)

        if norm_lhe:
            norm = np.linalg.norm(mt[:, k])
            if norm > 0:
                mt[:, k] /= norm

    if nmf_fix_user_rhe == 0:
        # Update mw

        mw[:, k] = 0
        if n_blocks > 1:
            for i_block in range(0, n_blocks):
                mw[:, k] += mpart[:, id_blockp[i_block]: id_blockp[i_block] + p].T @ mt[:, k] * mb[i_block, k]
        else:
            mw[:, k] += mpart[:, id_blockp[0]: id_blockp[0] + p].T @ mt[:, k]

        if n_mmis > 0:
            denomw[:] = 0
            mt2[:] = mt[:, k] ** 2
            if n_blocks > 1:
                for i_block in range(0, n_blocks):
                    # Broadcast missing cells into mw to calculate m.T * m
                    denomw += mb[i_block, k] ** 2 * mmis[:, id_blockp[i_block]: id_blockp[i_block] + p].T @ mt2
            else:
                denomw += mmis[:, id_blockp[0]: id_blockp[0] + p].T @ mt2

            denomw /= np.max(denomw)
            denomw[denomw < denom_cutoff] = denom_cutoff
            mw[:, k] /= denomw

        mw[mw[:, k] < 0, k] = 0

        if alpha > 0:
            mw[:, k] = sparse_opt(mw[:, k], alpha, False)

        mw, b = in_loop_update_m(mw, k, ntf_unimodal, p, ntf_right_components, ntf_smooth, b)

        if n_nmf_priors > 0:
            mw[:, k] = mw[:, k] * nmf_priors[:, k]

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
                denom_block[i_block, k] = np.reshape(mmis[:, id_blockp[i_block]: id_blockp[i_block] + p],
                                                     (1, nxp)) @ mt_mw

            maxdenom_block = np.max(denom_block[:, k])
            denom_block[denom_block[:, k] < denom_cutoff * maxdenom_block] = denom_cutoff * maxdenom_block
            mb[:, k] /= denom_block[:, k]

        mb[mb[:, k] < 0, k] = 0
        mb, c = in_loop_update_m(mb, k, ntf_unimodal, ntf_block_components, n, ntf_smooth, c)

        if norm_bhe:
            norm = np.linalg.norm(mb[:, k])
            if norm > 0:
                mb[:, k] /= norm

    # Update residual tensor
    mfit[:, :] = 0
    mfit = reshape_mfit(n_blocks, mfit, id_blockp, mt, k, mw, n, mb, p)

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
        alpha,
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
        nmf_priors,
    )
