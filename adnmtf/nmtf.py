""" Classes accessing Non-negative matrix and tensor factorization functions
"""

# Author: Paul Fogel

# License: MIT
# Dec 28, '19
import numpy as np
import logging

from .estimator import Estimator
from .nmtf_base import init_factorization, nmf_init, r_ntf_solve, ntf_init
from .nmtf_utils import nmf_det, StatusBoxTqdm

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
        >>> from adnmtf import NMF
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
    ) -> Estimator:
        """To implement in daughter class"""
        pass

    @staticmethod
    def predict(estimator, blocks=None, cluster_by_stability=False, custom_order=False):
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------
        estimator: `nmtf.estimator.Estimator`
            Modified in place
        blocks: array-like, shape(n_blocks), default None
            Size of each block (if any) in ordered heatmap.
        cluster_by_stability: boolean, default False
             Use stability instead of leverage to assign samples/features to clusters
        custom_order:  boolean, default False
             if False samples/features with highest leverage or stability appear on top of each cluster
             if True within cluster ordering is modified to suggest a continuum  between adjacent clusters

        Example
        -------
        >>> from adnmtf import NMF
        >>> myNMFmodel = NMF(n_components=4)
        >>> m = ...  # matrix to be factorized
        >>> myestimator = myNMFmodel.fit_transform(m)
        >>> myNMFmodel.predict(myestimator)
        """
        estimator.predict(blocks, cluster_by_stability, custom_order)

    @staticmethod
    def permutation_test_score(estimator, y, n_permutations=100):
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------
        estimator: `nmtf.estimator.Estimator`
            Modified in place
        y:  array-like, group to be predicted
        n_permutations:  integer, default: 100

        Example
        -------
        >>> from adnmtf import NMF
        >>> myNMFmodel = NMF(n_components=4)
        >>> m = ...  # matrix to be factorized
        >>> myestimator = myNMFmodel.fit_transform(m)
        >>> sample_group = ...  # the group each sample is associated with
        >>> myNMFmodel.permutation_test_score(myestimator, sample_group, n_permutations=100)
        """
        estimator.permutation_test_score(y, n_permutations)


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
    ) -> Estimator:
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
        `nmtf.estimator.Estimator`

        Example
        -------
        >>> from adnmtf import NMF
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

        nmf_algo = "non-robust"
        log_iter = self.verbose
        my_status_box = StatusBoxTqdm(verbose=log_iter)
        tolerance = self.tol
        if w is None and h is None:
            mt, mw = nmf_init(m, mmis, np.array([]), np.array([]), nc)
        elif h is None:
            mw = np.ones((p, nc))
            mt = w.copy()
        else:
            mt = np.ones((n, nc))
            mw = h.copy()

            # TODO (pcotte): this is not pytested
            # TODO (pcotte): might be optimised, maybe ?
            for k in range(0, nc):
                mt[:, k] = mt[:, k] / np.linalg.norm(mt[:, k])
                mw[:, k] = mw[:, k] / np.linalg.norm(mw[:, k])

        if n_bootstrap is None:
            nmf_robust_n_runs = 0
        else:
            nmf_robust_n_runs = n_bootstrap

        if nmf_robust_n_runs > 1:
            nmf_algo = "robust"

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

        _, mt, mw, mb, mt_pct, mw_pct, diff, add_message, err_message, cancel_pressed = r_ntf_solve(
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
            nmf_algo=nmf_algo,
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
        if nmf_fix_user_lhe == 0 and nmf_fix_user_rhe == 0:
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
        # TODO (pcotte): might be optimised, maybe ?
        for k in range(0, nc):
            max_col = np.max(mt[:, k])
            if max_col > 0:
                mt[:, k] /= max_col
                mw[:, k] *= mev[k] * max_col
                mev[k] = 1
            else:
                mev[k] = 0

        if nmf_algo == "non-robust":
            estimator = Estimator(w=mt, h=mw, volume=volume, diff=diff, leverage=self.leverage, verbose=self.verbose)
        else:
            estimator = Estimator(
                w=mt,
                h=mw,
                volume=volume,
                wb=mt_pct,
                hb=mw_pct,
                diff=diff,
                leverage=self.leverage,
                verbose=self.verbose
            )

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
        >>> from adnmtf import NTF
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
    ) -> Estimator:
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
        `nmtf.estimator.Estimator`

        Example
        -------
        >>> from adnmtf import NTF
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
            # TODO (pcotte): might be optimised, maybe ?
            for k in range(0, nc):
                for i_block in range(0, n_blocks):
                    mfit[:, i_block * p_block: (i_block + 1) * p_block] += (
                            mb0[i_block, k] * np.reshape(mt0[:, k], (n, 1)) @ np.reshape(mw0[:, k], (1, p_block))
                    )

            scale_ratio = (np.linalg.norm(mfit) / np.linalg.norm(m)) ** (1 / 3)
            # TODO (pcotte): might be optimised, maybe ?
            for k in range(0, nc):
                mt0[:, k] /= scale_ratio
                mw0[:, k] /= scale_ratio
                mb0[:, k] /= scale_ratio

            mfit = np.zeros((n, p))
            # TODO (pcotte): might be optimised, maybe ?
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
            nmf_algo = "non-robust"
        else:
            nmf_algo = "robust"

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

        if nmf_robust_n_runs <= 1:
            estimator = Estimator(
                w=mt,
                h=mw,
                q=mb,
                volume=volume,
                diff=diff,
                leverage=self.leverage,
                verbose=self.verbose
            )
        else:
            estimator = Estimator(
                w=mt,
                h=mw,
                q=mb,
                volume=volume,
                wb=mt_pct,
                hb=mw_pct,
                diff=diff,
                leverage=self.leverage,
                verbose=self.verbose
            )

        return estimator
