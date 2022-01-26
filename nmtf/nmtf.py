""" Classes accessing Non-negative matrix and tensor factorization functions
"""

# Author: Paul Fogel

# License: MIT
# Dec 28, '19
import numpy as np
import logging
from .nmtf_base import init_factorization, nmf_init, r_ntf_solve, ntf_init
from .nmtf_utils import nmf_det, StatusBoxTqdm, build_clusters, global_sign

logger = logging.getLogger(__name__)

# TODO (pcotte): typing


class NMTF:
    """Abstract class overleaded by `nmtf.nmft.NMF` and `nmtf.nmft.NTF`"""
    def __init__(
        self, n_components=None, tol=1e-6, max_iter=150, leverage="standard", random_state=None, verbose=0, **ntf_kwargs
    ):
        """Initialize NMF or NTF model

        Parameters
        ----------
        n_components: integer
            Number of components, if n_components is not set: n_components = min(n_samples, n_features)
        tol: float, default: 1e-6
            Tolerance of the stopping condition.
        max_iter: integer, default: 200
            Maximum number of iterations.
        leverage:  None | 'standard' | 'robust', default 'standard'
            Calculate leverage of W and H rows on each component.
        random_state: int, RandomState instance or None, optional, default: None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        verbose: integer, default: 0
            The verbosity level (0/1).
        ntf_kwargs: dict
            Additional keyword arguments for NTF

        Returns
        -------
        NMF model

        Example
        -------
        >>> from nmtf import NMF
        >>> myNMFmodel = NMF(n_components=4)

        References
        ----------
        P. Fogel, D.M. Hawkins, C. Beecher, G. Luta, S. S. Young (2013). A Tale of Two Matrix Factorizations.
        The American Statistician, Vol. 67, Issue 4.
        C. H.Q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
        IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1
        """
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.leverage = leverage
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(
        self,
        m,
        w=None,
        h=None,
        update_w=True,
        update_h=True,
        n_bootstrap=None,
        regularization=None,
        sparsity=0,
        **ntf_kwargs
    ) -> dict:
        """To implement in daughter class"""
        pass

    def predict(self, estimator, blocks=None, cluster_by_stability=False, custom_order=False) -> dict:
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------
        estimator: tuplet as returned by fit_transform
        blocks: array-like, shape(n_blocks), default None
            Size of each block (if any) in ordered heatmap.
        cluster_by_stability: boolean, default False
             Use stability instead of leverage to assign samples/features to clusters
        custom_order:  boolean, default False
             if False samples/features with highest leverage or stability appear on top of each cluster
             if True within cluster ordering is modified to suggest a continuum  between adjacent clusters

        Returns
        -------
        dict: Completed estimator with following entries:\n
          * WL: array-like, shape (n_samples, n_components)\n
             Sample leverage on each component\n
          * HL: array-like, shape (n_features, n_components)\n
             Feature leverage on each component\n
          * QL: array-like, shape (n_blocks, n_components)\n
             Block leverage on each component (NTF only)\n
          * WR: vector-like, shape (n_samples)\n
             Ranked sample indexes (by cluster and leverage or stability)
             Used to produce ordered heatmaps\n
          * HR: vector-like, shape (n_features)\n
             Ranked feature indexes (by cluster and leverage or stability)
             Used to produce ordered heatmaps\n
          * WN: vector-like, shape (n_components)\n
             Sample cluster bounds in ordered heatmap\n
          * HN: vector-like, shape (n_components)\n
             Feature cluster bounds in ordered heatmap\n
          * WC: vector-like, shape (n_samples)\n
             Sample assigned cluster\n
          * HC: vector-like, shape (n_features)\n
             Feature assigned cluster\n
          * QC: vector-like, shape (size(blocks))\n
             Block assigned cluster (NTF only)

        Example
        -------
        >>> from nmtf import NMF
        >>> myNMFmodel = NMF(n_components=4)
        >>> m = ...  # matrix to be factorized
        >>> myestimator = myNMFmodel.fit_transform(m)
        >>> myestimator = myNMFmodel.predict(myestimator)
        """
        mt = estimator["W"]
        mw = estimator["H"]
        if "Q" in estimator:
            # X is a 3D tensor, in unfolded form of a 2D array
            # horizontal concatenation of blocks of equal size.
            mb = estimator["Q"]
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

        if self.leverage == "standard":
            nmf_calculate_leverage = 1
            nmf_use_robust_leverage = 0
        elif self.leverage == "robust":
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
        my_status_box = StatusBoxTqdm(verbose=self.verbose)

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
            mt=mt,
            mw=mw,
            mb=mb,
            mt_pct=mt_pct,
            mw_pct=mw_pct,
            n_blocks=n_blocks,
            blk_size=blk_size,
            nmf_calculate_leverage=nmf_calculate_leverage,
            nmf_use_robust_leverage=nmf_use_robust_leverage,
            nmf_algo=nmf_algo,
            nmf_robust_cluster_by_stability=nmf_robust_cluster_by_stability,
            cell_plot_ordered_clusters=cell_plot_ordered_clusters,
            add_message=add_message,
            my_status_box=my_status_box,
        )
        for message in add_message:
            logger.info(message)

        my_status_box.close()
        if "Q" in estimator:
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
                    ("QL", mbn),
                    ("QC", block_clust),
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
                    ("QL", None),
                    ("QC", None),
                ]
            )
        return estimator

    # TODO (pcotte): this function is not called by any pytest. Make a pytest calling it.
    def permutation_test_score(self, estimator, y, n_permutations=100) -> dict:
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------
        estimator: tuplet as returned by fit_transform
        y:  array-like, group to be predicted
        n_permutations:  integer, default: 100

        Returns
        -------
        dict: Completed estimator with following entries:\n
          * score: float\n
             The true score without permuting targets.\n
          * pvalue: float\n
             The p-value, which approximates the probability that the score would be obtained by chance.\n
          * CS: array-like, shape(n_components)\n
             The size of each cluster\n
          * CP: array-like, shape(n_components)\n
             The pvalue of the most significant group within each cluster\n
          * CG: array-like, shape(n_components)\n
             The index of the most significant group within each cluster\n
          * CN: array-like, shape(n_components, n_groups)\n
             The size of each group within each cluster

        Example
        -------
        >>> from nmtf import NMF
        >>> myNMFmodel = NMF(n_components=4)
        >>> m = ...  # matrix to be factorized
        >>> myestimator = myNMFmodel.fit_transform(m)
        >>> sample_group = ...  # the group each sample is associated with
        >>> myestimator = myNMFmodel.permutation_test_score(myestimator, sample_group, n_permutations=100)
        """
        mt = estimator["W"]
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
        my_status_box = StatusBoxTqdm(verbose=self.verbose)
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


class NMF(NMTF):
    """Overloads `nmtf.nmft.NMTF`."""
    def fit_transform(
        self,
        m,
        w=None,
        h=None,
        update_w=True,
        update_h=True,
        n_bootstrap=None,
        regularization=None,
        sparsity=0,
        **ntf_kwargs
    ) -> dict:
        """Compute Non-negative Matrix Factorization (NMF)

        Find two non-negative matrices (W, H) such as x = W @ H.T + Error.
        This factorization can be used for example for
        dimensionality reduction, source separation or topic extraction.

        The objective function is minimized with an alternating minimization of W
        and H.

        Parameters
        ----------
        m: array-like, shape (n_samples, n_features)
            Constant matrix.
        w: array-like, shape (n_samples, n_components)
            prior W
        h: array-like, shape (n_features, n_components)
            prior H
        update_w: boolean, default: True
            Update or keep W fixed
        update_h: boolean, default: True
            Update or keep H fixed
        n_bootstrap: integer, default: 0
            Number of bootstrap runs.
        regularization:  None | 'components' | 'transformation'
            Select whether the regularization affects the components (H), the
            transformation (W) or none of them.
        sparsity: float, default: 0
            Sparsity target with 0 <= sparsity < 1 representing either:
            - the % rows in W or H set to 0 (when use_hals = False)
            - the mean % rows per column in W or H set to 0 (when use_hals = True)
            sparsity == 1: adaptive sparsity through hard thresholding and hhi
        ntf_kwargs: dict
            Should be empty

        Returns
        -------
        dict: Estimator (dictionary) with following entries\n
          * W: array-like, shape (n_samples, n_components)\n
            Solution to the non-negative least squares problem.\n
          * H: array-like, shape (n_components, n_features)\n
            Solution to the non-negative least squares problem.\n
          * volume: scalar, volume occupied by W and H\n
          * WB: array-like, shape (n_samples, n_components)\n
            A sample is clustered in cluster k if its leverage on component k is higher than on any other
            components. During each run of the bootstrap, samples are re-clustered.
            Each row of WB contains the frequencies of the n_components clusters following the bootstrap.
            Only if n_bootstrap > 0.\n
          * HB: array-like, shape (n_components, n_features)\n
            A feature is clustered in cluster k if its leverage on component k is higher than on any other
            components. During each run of the bootstrap, features are re-clustered.
            Each row of HB contains the frequencies of the n_components clusters following the bootstrap.
            Only if n_bootstrap > 0.\n
          * B: array-like, shape (n_observations, n_components) or (n_features, n_components)\n
            Only if active convex variant, H = B.T @ X or W = X @ B\n
          * diff: scalar, objective minimum achieved

        Example
        -------
        >>> from nmtf import NMF
        >>> myNMFmodel = NMF(n_components=4)
        >>> mm = ...  # matrix to be factorized
        >>> est = myNMFmodel.fit_transform(mm)

        References
        ----------
        P. Fogel, D.M. Hawkins, C. Beecher, G. Luta, S. S. Young (2013). A Tale of Two Matrix Factorizations.
        The American Statistician, Vol. 67, Issue 4.
        C. H.Q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
        IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1
        """
        if len(ntf_kwargs) > 0:
            raise ValueError("You gave NTF keyword arguments to NMF 'fit_transform'. Are you using the correct class ?")

        m, n, p, mmis, nc = init_factorization(m, self.n_components)

        nmf_algo = 2
        log_iter = self.verbose
        my_status_box = StatusBoxTqdm(verbose=log_iter)
        tolerance = self.tol
        if (w is None) & (h is None):
            mt, mw = nmf_init(m, mmis, np.array([]), np.array([]), nc)
        elif h is None:
            mw = np.ones((p, nc))
            mt = w.copy()
        else:
            mt = np.ones((n, nc))
            mw = h.copy()

            for k in range(0, nc):
                mt[:, k] = mt[:, k] / np.linalg.norm(mt[:, k])
                mw[:, k] = mw[:, k] / np.linalg.norm(mw[:, k])

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

        max_iterations = self.max_iter
        if regularization is None:
            nmf_sparse_level = 0
        else:
            if regularization == "components":
                nmf_sparse_level = sparsity
            elif regularization == "transformation":
                nmf_sparse_level = -sparsity
            else:
                nmf_sparse_level = 0

        if self.leverage == "standard":
            nmf_calculate_leverage = 1
            nmf_use_robust_leverage = 0
        elif self.leverage == "robust":
            nmf_calculate_leverage = 1
            nmf_use_robust_leverage = 1
        else:
            nmf_calculate_leverage = 0
            nmf_use_robust_leverage = 0

        if self.random_state is not None:
            random_seed = self.random_state
            np.random.seed(random_seed)

        if nmf_algo <= 2:
            ntf_algo = 5
        else:
            ntf_algo = 6

        dummy, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed = r_ntf_solve(
            m=m,
            mmis=mmis,
            mt0=mt,
            mw0=mw,
            mb0=np.array([]),
            nc=nc,
            tolerance=tolerance,
            log_iter=log_iter,
            max_iterations=max_iterations,
            nmf_fix_user_lhe=nmf_fix_user_lhe,
            nmf_fix_user_rhe=nmf_fix_user_rhe,
            nmf_fix_user_bhe=1,
            nmf_algo=ntf_algo,
            nmf_robust_n_runs=nmf_robust_n_runs,
            nmf_calculate_leverage=nmf_calculate_leverage,
            nmf_use_robust_leverage=nmf_use_robust_leverage,
            nmf_sparse_level=nmf_sparse_level,
            ntf_unimodal=0,
            ntf_smooth=0,
            ntf_left_components=0,
            ntf_right_components=0,
            ntf_block_components=0,
            n_blocks=1,
            nmf_priors=np.array([]),
            my_status_box=my_status_box,
        )
        mev = np.ones(nc)
        if (nmf_fix_user_lhe == 0) & (nmf_fix_user_rhe == 0):
            # Scale
            for k in range(0, nc):
                scale_mt = np.linalg.norm(mt[:, k])
                scale_mw = np.linalg.norm(mw[:, k])
                mev[k] = scale_mt * scale_mw
                if mev[k] > 0:
                    mt[:, k] = mt[:, k] / scale_mt
                    mw[:, k] = mw[:, k] / scale_mw

        volume = nmf_det(mt, mw, 1)

        for message in add_message:
            logger.info(message)

        my_status_box.close()

        # Order by decreasing scale
        r_mev = np.argsort(-mev)
        mev = mev[r_mev]
        mt = mt[:, r_mev]
        mw = mw[:, r_mev]
        if isinstance(mt_pct, np.ndarray):
            mt_pct = mt_pct[:, r_mev]
            mw_pct = mw_pct[:, r_mev]

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
            estimator.update([("W", mt), ("H", mw), ("volume", volume), ("diff", diff)])
        else:
            estimator.update([("W", mt), ("H", mw), ("volume", volume), ("WB", mt_pct), ("HB", mw_pct), ("diff", diff)])

        return estimator


class NTF(NMTF):
    """Overloads `nmtf.nmft.NMTF`."""
    def __init__(
        self,
        n_components=None,
        tol=1e-6,
        max_iter=150,
        leverage="standard",
        random_state=None,
        verbose=0,
        unimodal=False,
        smooth=False,
        apply_left=False,
        apply_right=False,
        apply_block=False,
        init_type=1,
    ):
        """Initialize NTF model

        Parameters
        ----------
        n_components: integer
            Number of components, if n_components is not set: n_components = min(n_samples, n_features)
        tol: float, default: 1e-6
            Tolerance of the stopping condition.
        max_iter: integer, default: 200
            Maximum number of iterations.
        leverage:  None | 'standard' | 'robust', default 'standard'
            Calculate leverage of W and H rows on each component.
        random_state: int, RandomState instance or None, optional, default: None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        verbose: integer, default: 0
            The verbosity level (0/1).
        unimodal: Boolean, default: False
        smooth: Boolean, default: False
        apply_left: Boolean, default: False
        apply_right: Boolean, default: False
        apply_block: Boolean, default: False
        init_type: integer, default 1
            init_type = 1: NMF initialization applied on the reshaped matrix [vectorized (1st & 2nd dim) x 3rd dim]
            init_type = 2: NMF initialization applied on the reshaped matrix [1st dim x vectorized (2nd & 3rd dim)]

        Returns
        -------
        NTF model

        Example
        -------
        >>> from nmtf import NTF
        >>> myNTFmodel = NTF(n_components=4)

        Reference
        ---------
        A. Cichocki, P.H.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor
        factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

        """
        super().__init__(
            n_components=n_components,
            tol=tol,
            max_iter=max_iter,
            leverage=leverage,
            random_state=random_state,
            verbose=verbose,
        )
        self.unimodal = unimodal
        self.smooth = smooth
        self.apply_left = apply_left
        self.apply_right = apply_right
        self.apply_block = apply_block
        self.init_type = init_type
        
    def fit_transform(
        self,
        m,
        w=None,
        h=None,
        update_w=True,
        update_h=True,
        regularization=None,
        sparsity=0,
        n_bootstrap=None,
        n_blocks=None,
        q=None,
        update_q=True,
    ):
        """Compute Non-negative Tensor Factorization (NTF)

        Find three non-negative matrices (W, H, Q) such as x = W @@ H @@ Q + Error (@@ = tensor product).
        This factorization can be used for example for
        dimensionality reduction, source separation or topic extraction.

        The objective function is minimized with an alternating minimization of W
        and H.

        Parameters
        ----------
        m: array-like, shape (n_samples, n_features x n_blocks)
            Constant matrix.
            X is a tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd
            dimensions.
        n_blocks: integer
            Number of blocks defining the 3rd dimension of the tensor
        n_bootstrap: integer
            Number of bootstrap runs
        regularization:  None | 'components' | 'transformation'
            Select whether the regularization affects the components (H), the
            transformation (W) or none of them.
        sparsity: float, default: 0
            Sparsity target with 0 <= sparsity < 1 representing either:
            - the % rows in W or H set to 0 (when use_hals = False)
            - the mean % rows per column in W or H set to 0 (when use_hals = True)
            sparsity == 1: adaptive sparsity through hard thresholding and hhi
        w: array-like, shape (n_samples, n_components)
            Prior W
        h: array-like, shape (n_features, n_components)
            Prior H
        q: array-like, shape (n_blocks, n_components)
            Prior Q
        update_w: boolean, default: True
            Update or keep W fixed
        update_h: boolean, default: True
            Update or keep H fixed
        update_q: boolean, default: True
            Update or keep Q fixed

        Returns
        -------
        dict: Estimator with following entries\n
          * W: array-like, shape (n_samples, n_components)\n
            Solution to the non-negative least squares problem.\n
          * H: array-like, shape (n_features, n_components)\n
            Solution to the non-negative least squares problem.\n
          * Q: array-like, shape (n_blocks, n_components)\n
            Solution to the non-negative least squares problem.\n
          * volume: scalar, volume occupied by W and H\n
          * WB: array-like, shape (n_samples, n_components)\n
            Percent consistently clustered rows for each component.
            only if n_bootstrap > 0.\n
          * HB: array-like, shape (n_features, n_components)\n
            Percent consistently clustered columns for each component.
            only if n_bootstrap > 0.\n
          * diff: scalar, objective minimum achieved

        Example
        -------
        >>> from nmtf import NTF
        >>> myNTFmodel = NTF(n_components=4)
        >>> t = ...  # tensor with 5 blocks to be factorized
        >>> est = myNTFmodel.fit_transform(t, 5)

        Reference
        ---------

        A. Cichocki, P.H.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor
        factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.
        """
        if n_blocks is None:
            raise ValueError("Argument 'n_blocks' can not be None")
        m, n, p, mmis, nc = init_factorization(m, self.n_components)

        n_blocks = n_blocks
        p_block = int(p / n_blocks)
        tolerance = self.tol
        log_iter = self.verbose
        if regularization is None:
            nmf_sparse_level = 0
        else:
            if regularization == "components":
                nmf_sparse_level = sparsity
            elif regularization == "transformation":
                nmf_sparse_level = -sparsity
            else:
                nmf_sparse_level = 0
        ntf_unimodal = self.unimodal
        ntf_smooth = self.smooth
        ntf_left_components = self.apply_left
        ntf_right_components = self.apply_right
        ntf_block_components = self.apply_block
        if self.random_state is not None:
            random_seed = self.random_state
            np.random.seed(random_seed)

        my_status_box = StatusBoxTqdm(verbose=log_iter)

        if (w is None) & (h is None) & (q is None):
            mt0, mw0, mb0, add_message, err_message, cancel_pressed = ntf_init(
                m=m,
                mmis=mmis,
                mt_nmf=np.array([]),
                mw_nmf=np.array([]),
                nc=nc,
                tolerance=tolerance,
                log_iter=log_iter,
                ntf_unimodal=ntf_unimodal,
                ntf_left_components=ntf_left_components,
                ntf_right_components=ntf_right_components,
                ntf_block_components=ntf_block_components,
                n_blocks=n_blocks,
                init_type=self.init_type,
                my_status_box=my_status_box,
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

            if q is None:
                mb0 = np.ones((n_blocks, nc))
            else:
                mb0 = np.copy(q)

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

        max_iterations = self.max_iter

        if n_bootstrap is None:
            nmf_robust_n_runs = 0
        else:
            nmf_robust_n_runs = n_bootstrap

        if nmf_robust_n_runs <= 1:
            nmf_algo = 5
        else:
            nmf_algo = 6

        if self.leverage == "standard":
            nmf_calculate_leverage = 1
            nmf_use_robust_leverage = 0
        elif self.leverage == "robust":
            nmf_calculate_leverage = 1
            nmf_use_robust_leverage = 1
        else:
            nmf_calculate_leverage = 0
            nmf_use_robust_leverage = 0

        if self.random_state is not None:
            random_seed = self.random_state
            np.random.seed(random_seed)

        if update_w:
            nmf_fix_user_lhe = 0
        else:
            nmf_fix_user_lhe = 1

        if update_h:
            nmf_fix_user_rhe = 0
        else:
            nmf_fix_user_rhe = 1

        if update_q:
            nmf_fix_user_bhe = 0
        else:
            nmf_fix_user_bhe = 1

        mt_conv, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed = r_ntf_solve(
            m=m,
            mmis=mmis,
            mt0=mt0,
            mw0=mw0,
            mb0=mb0,
            nc=nc,
            tolerance=tolerance,
            log_iter=log_iter,
            max_iterations=max_iterations,
            nmf_fix_user_lhe=nmf_fix_user_lhe,
            nmf_fix_user_rhe=nmf_fix_user_rhe,
            nmf_fix_user_bhe=nmf_fix_user_bhe,
            nmf_algo=nmf_algo,
            nmf_robust_n_runs=nmf_robust_n_runs,
            nmf_calculate_leverage=nmf_calculate_leverage,
            nmf_use_robust_leverage=nmf_use_robust_leverage,
            nmf_sparse_level=nmf_sparse_level,
            ntf_unimodal=ntf_unimodal,
            ntf_smooth=ntf_smooth,
            ntf_left_components=ntf_left_components,
            ntf_right_components=ntf_right_components,
            ntf_block_components=ntf_block_components,
            n_blocks=n_blocks,
            nmf_priors=np.array([]),
            my_status_box=my_status_box,
        )

        volume = nmf_det(mt, mw, 1)

        for message in add_message:
            logger.info(message)

        my_status_box.close()

        estimator = {}
        if nmf_robust_n_runs <= 1:
            estimator.update([("W", mt), ("H", mw), ("Q", mb), ("volume", volume), ("diff", diff)])
        else:
            estimator.update(
                [("W", mt), ("H", mw), ("Q", mb), ("volume", volume), ("WB", mt_pct), ("HB", mw_pct), ("diff", diff)]
            )

        return estimator
