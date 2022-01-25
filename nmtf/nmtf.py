""" Classes accessing Non-negative matrix and tensor factorization functions
"""

# Author: Paul Fogel

# License: MIT
# Dec 28, '19

from .nmtf_base import (
    non_negative_factorization,
    nmf_predict,
    nmf_permutation_test_score,
    non_negative_tensor_factorization,
)


class NMF:

    def __init__(self, n_components=None, tol=1e-6, max_iter=150, leverage="standard", random_state=None, verbose=0):
        """Initialize NMF model

        Parameters
        ----------
        n_components : integer
            Number of components, if n_components is not set : n_components = min(n_samples, n_features)
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
        self, x, w=None, h=None, update_w=True, update_h=True, n_bootstrap=None, regularization=None, sparsity=0
    ) -> dict:
        """Compute Non-negative Matrix Factorization (NMF)

        Find two non-negative matrices (W, H) such as x = W @ H.T + Error.
        This factorization can be used for example for
        dimensionality reduction, source separation or topic extraction.

        The objective function is minimized with an alternating minimization of W
        and H.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Constant matrix.
        w : array-like, shape (n_samples, n_components)
            prior W
        h : array-like, shape (n_features, n_components)
            prior H
        update_w : boolean, default: True
            Update or keep W fixed
        update_h : boolean, default: True
            Update or keep H fixed
        n_bootstrap : integer, default: 0
            Number of bootstrap runs.
        regularization :  None | 'components' | 'transformation'
            Select whether the regularization affects the components (H), the
            transformation (W) or none of them.
        sparsity : float, default: 0
            Sparsity target with 0 <= sparsity < 1 representing either:
            - the % rows in W or H set to 0 (when use_hals = False)
            - the mean % rows per column in W or H set to 0 (when use_hals = True)
            sparsity == 1: adaptive sparsity through hard thresholding and hhi

        Returns
        -------
        dict: Estimator (dictionary) with following entries
            W : array-like, shape (n_samples, n_components)
                Solution to the non-negative least squares problem.
            H : array-like, shape (n_components, n_features)
                Solution to the non-negative least squares problem.
            volume : scalar, volume occupied by W and H
            WB : array-like, shape (n_samples, n_components)
                A sample is clustered in cluster k if its leverage on component k is higher than on any other
                components. During each run of the bootstrap, samples are re-clustered.
                Each row of WB contains the frequencies of the n_components clusters following the bootstrap.
                    Only if n_bootstrap > 0.
            HB : array-like, shape (n_components, n_features)
                A feature is clustered in cluster k if its leverage on component k is higher than on any other
                components. During each run of the bootstrap, features are re-clustered.
                Each row of HB contains the frequencies of the n_components clusters following the bootstrap.
                    Only if n_bootstrap > 0.
            B : array-like, shape (n_observations, n_components) or (n_features, n_components)
                Only if active convex variant, H = B.T @ X or W = X @ B
            diff : scalar, objective minimum achieved

        Example
        -------
        >>> from nmtf import NMF
        >>> myNMFmodel = NMF(n_components=4)

        >>>m = ...  # matrix to be factorized
        >>> estimator = myNMFmodel.fit_transform(m)

        References
        ----------

        P. Fogel, D.M. Hawkins, C. Beecher, G. Luta, S. S. Young (2013). A Tale of Two Matrix Factorizations.
        The American Statistician, Vol. 67, Issue 4.

        C. H.Q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
        IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1
        """
        return non_negative_factorization(
            x,
            W=w,
            H=h,
            n_components=self.n_components,
            update_W=update_w,
            update_H=update_h,
            n_bootstrap=n_bootstrap,
            tol=self.tol,
            max_iter=self.max_iter,
            regularization=regularization,
            sparsity=sparsity,
            leverage=self.leverage,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def predict(self, estimator, blocks=None, cluster_by_stability=False, custom_order=False) -> dict:
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------
        estimator : tuplet as returned by fit_transform
        blocks : array-like, shape(n_blocks), default None
            Size of each block (if any) in ordered heatmap.
        cluster_by_stability : boolean, default False
             Use stability instead of leverage to assign samples/features to clusters
        custom_order :  boolean, default False
             if False samples/features with highest leverage or stability appear on top of each cluster
             if True within cluster ordering is modified to suggest a continuum  between adjacent clusters

        Returns
        -------
        dict: Completed estimator with following entries:
            WL : array-like, shape (n_samples, n_components)
                 Sample leverage on each component
            HL : array-like, shape (n_features, n_components)
                 Feature leverage on each component
            QL : array-like, shape (n_blocks, n_components)
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
            QC : vector-like, shape (size(blocks))
                 Block assigned cluster (NTF only)

        Example
        -------
        >>> from nmtf import NMF
        >>> myNMFmodel = NMF(n_components=4)
        >>> m = ...  # matrix to be factorized
        >>> myestimator = myNMFmodel.fit_transform(m)
        >>> myestimator = myNMFmodel.predict(myestimator)
        """
        return nmf_predict(
            estimator,
            blocks=blocks,
            leverage=self.leverage,
            cluster_by_stability=cluster_by_stability,
            custom_order=custom_order,
            verbose=self.verbose,
        )

    def permutation_test_score(self, estimator, y, n_permutations=100) -> dict:
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------
        estimator : tuplet as returned by fit_transform
        y :  array-like, group to be predicted
        n_permutations :  integer, default: 100

        Returns
        -------
        dict: Completed estimator with following entries:
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

        Example
        -------
        >>> from nmtf import NMF
        >>> myNMFmodel = NMF(n_components=4)
        >>> m = ...  # matrix to be factorized
        >>> myestimator = myNMFmodel.fit_transform(m)
        >>> sample_group = ...  # the group each sample is associated with
        >>> myestimator = myNMFmodel.permutation_test_score(myestimator, sample_group, n_permutations=100)
        """
        return nmf_permutation_test_score(estimator, y, n_permutations=n_permutations, verbose=self.verbose)


class NTF:

    def __init__(
        self,
        n_components=None,
        unimodal=False,
        smooth=False,
        apply_left=False,
        apply_right=False,
        apply_block=False,
        tol=1e-6,
        max_iter=150,
        leverage="standard",
        random_state=None,
        init_type=1,
        verbose=0,
    ):
        """Initialize NTF model

        Parameters
        ----------
        n_components : integer
            Number of components, if n_components is not set : n_components = min(n_samples, n_features)
        unimodal : Boolean, default: False
        smooth : Boolean, default: False
        apply_left : Boolean, default: False
        apply_right : Boolean, default: False
        apply_block : Boolean, default: False
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
        init_type : integer, default 1
            init_type = 1 : NMF initialization applied on the reshaped matrix [vectorized (1st & 2nd dim) x 3rd dim]
            init_type = 2 : NMF initialization applied on the reshaped matrix [1st dim x vectorized (2nd & 3rd dim)]
        verbose : integer, default: 0
            The verbosity level (0/1).


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
        self.n_components = n_components
        self.unimodal = unimodal
        self.smooth = smooth
        self.apply_left = apply_left
        self.apply_right = apply_right
        self.apply_block = apply_block
        self.tol = tol
        self.max_iter = max_iter
        self.leverage = leverage
        self.random_state = random_state
        self.init_type = init_type
        self.verbose = verbose

    def fit_transform(
        self,
        x,
        n_blocks,
        n_bootstrap=None,
        regularization=None,
        sparsity=0,
        w=None,
        h=None,
        q=None,
        update_w=True,
        update_h=True,
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
        x : array-like, shape (n_samples, n_features x n_blocks)
            Constant matrix.
            X is a tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd
            dimensions.
        n_blocks : integer
            Number of blocks defining the 3rd dimension of the tensor
        n_bootstrap : integer
            Number of bootstrap runs
        regularization :  None | 'components' | 'transformation'
            Select whether the regularization affects the components (H), the
            transformation (W) or none of them.
        sparsity : float, default: 0
            Sparsity target with 0 <= sparsity < 1 representing either:
            - the % rows in W or H set to 0 (when use_hals = False)
            - the mean % rows per column in W or H set to 0 (when use_hals = True)
            sparsity == 1: adaptive sparsity through hard thresholding and hhi
        w : array-like, shape (n_samples, n_components)
            Prior W
        h : array-like, shape (n_features, n_components)
            Prior H
        q : array-like, shape (n_blocks, n_components)
            Prior Q
        update_w : boolean, default: True
            Update or keep W fixed
        update_h : boolean, default: True
            Update or keep H fixed
        update_q : boolean, default: True
            Update or keep Q fixed

        Returns
        -------
        dict: Estimator with following entries
            W : array-like, shape (n_samples, n_components)
                Solution to the non-negative least squares problem.
            H : array-like, shape (n_features, n_components)
                Solution to the non-negative least squares problem.
            Q : array-like, shape (n_blocks, n_components)
                Solution to the non-negative least squares problem.
            volume : scalar, volume occupied by W and H
            WB : array-like, shape (n_samples, n_components)
                Percent consistently clustered rows for each component.
                only if n_bootstrap > 0.
            HB : array-like, shape (n_features, n_components)
                Percent consistently clustered columns for each component.
                only if n_bootstrap > 0.
            diff : scalar, objective minimum achieved

        Example
        -------
        >>> from nmtf import NTF
        >>> myNTFmodel = NTF(n_components=4)
        >>> t = ...  # tensor with 5 blocks to be factorized
        >>> estimator = myNTFmodel.fit_transform(t, 5)

        Reference
        ---------

        A. Cichocki, P.H.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor
        factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.
        """
        return non_negative_tensor_factorization(
            x,
            n_blocks,
            w=w,
            h=h,
            q=q,
            n_components=self.n_components,
            update_w=update_w,
            update_h=update_h,
            update_q=update_q,
            regularization=regularization,
            sparsity=sparsity,
            unimodal=self.unimodal,
            smooth=self.smooth,
            apply_left=self.apply_left,
            apply_right=self.apply_right,
            apply_block=self.apply_block,
            n_bootstrap=n_bootstrap,
            tol=self.tol,
            max_iter=self.max_iter,
            leverage=self.leverage,
            random_state=self.random_state,
            init_type=self.init_type,
            verbose=self.verbose,
        )

    def predict(self, estimator, blocks=None, cluster_by_stability=False, custom_order=False) -> dict:
        """See `nmtf.nmtf.NMF.predict`"""
        return nmf_predict(
            estimator,
            blocks=blocks,
            leverage=self.leverage,
            cluster_by_stability=cluster_by_stability,
            custom_order=custom_order,
            verbose=self.verbose,
        )

    @staticmethod
    def permutation_test_score(estimator, y, n_permutations=100):
        """See `nmtf.nmtf.NMF.permutation_test_score`"""
        return nmf_permutation_test_score(estimator, y, n_permutations=n_permutations)
