""" Interface to Non-negative matrix and tensor factorization core module

"""

# Author: Paul Fogel

# License: MIT
# Dec 28, '19
# import development source
from .irmfpro import non_negative_factorization, nmf_predict, nmf_permutation_test_score, \
    non_negative_tensor_factorization

# import from package
# from irmfpro.irmfpro import *

# from ..base import BaseEstimator, TransformerMixin


class NMF:
    def __init__(
        self,
        n_components=None,
        beta_loss="frobenius",
        tol=1e-6,
        max_iter=150,
        max_iter_mult=20,
        the_leverage="standard",
        convex=None,
        kernel="linear",
        random_state=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.max_iter_mult = max_iter_mult
        self.leverage = the_leverage
        self.convex = convex
        self.kernel = kernel
        self.random_state = random_state
        self.verbose = verbose

    """Initialize NMF model

    Parameters
    ----------
    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    n_update_W : integer
        Estimate last n_update_W components from initial guesses.
        If n_update_W is not set : n_update_W = n_components.

    n_update_H : integer
        Estimate last n_update_H components from initial guesses.
        If n_update_H is not set : n_update_H = n_components.

    beta_loss : string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler'}.
        Beta divergence to be minimized, measuring the distance between x
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss == 'kullback-leibler', the input
        matrix x cannot contain zeros.

    tol : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations.

    max_iter_mult : integer, default: 20
        Maximum number of iterations in multiplicative warm-up to projected gradient (beta_loss = 'frobenius' only).

    the_leverage :  None | 'standard' | 'robust', default 'standard'
        Calculate the_leverage of w and h rows on each component.

    convex :  None | 'components' | 'transformation', default None
        Apply convex constraint on w or h.

    kernel :  'linear', 'quadratic', 'radial', default 'linear'
        Can be set if convex = 'transformation'.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : integer, default: 0
        The verbosity level (0/1).


    Returns
    -------
    NMF model model

    Examples
    --------

    >>> import numpy as np

    >>> x = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])

    >>> from sklearn.decomposition import non_negative_factorization

    >>> w, h, n_iter = non_negative_factorization(x, n_components=2, \

        random_state=0)


    References
    ----------

    Fogel

    Lin
    """

    def fit_transform(
        self,
        x,
        w=None,
        h=None,
        update_w=True,
        update_h=True,
        n_bootstrap=None,
        regularization=None,
        sparsity=None,
        skewness=False,
        null_priors=False,
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

                update_w : boolean, default: True
                    Update or keep w fixed

                update_h : boolean, default: True
                    Update or keep h fixed

                n_bootstrap : integer, default: 0
                    Number of bootstrap runs.

                regularization :  None | 'components' | 'transformation'
                    Select whether the regularization affects the components (h), the
                    transformation (w) or none of them.

                sparsity : double, default: 0.
                    Sparsity target with 0 <= sparsity <= 1 representing the % rows in w or h set to 0.

                skewness : boolean, default False
                    When solving mixture problems, columns of x at the extremities of the convex hull will be given
                    largest weights.
                    The column weight is a function of the skewness and its sign.
                    The expected sign of the skewness is based on the skewness of w components, as returned by the first
                    pass
                    of a 2-steps convex NMF. Thus, during the first pass, skewness must be set to False.
                    Can be set only if convex = 'transformation' and prior w and h have been defined.

                null_priors : boolean, default False
                    Cells of h with prior cells = 0 will not be updated.
                    Can be set only if prior h has been defined.

                Returns
                -------

                Estimator (dictionary) with following entries

                w : array-like, shape (n_samples, n_components)
                    Solution to the non-negative least squares problem.

                h : array-like, shape (n_components, n_features)
                    Solution to the non-negative least squares problem.

                v : scalar, volume occupied by w and h

                WB : array-like, shape (n_samples, n_components)
                    Percent consistently clustered rows for each component.
                    only if n_bootstrap > 0.

                HB : array-like, shape (n_components, n_features)
                    Percent consistently clustered columns for each component.
                    only if n_bootstrap > 0.

                b : array-like, shape (n_observations, n_components) or (n_features, n_components)
                    only if active convex variant, h = b.T @ x or w = x @ b


                Examples
                --------

                >>> import numpy as np
                >>> x = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
                >>> from sklearn.decomposition import non_negative_factorization
                >>> w, h, n_iter = non_negative_factorization(x, n_components=2, \

                    random_state=0)


                References
                ----------

                Fogel

                Lin
                """
        return non_negative_factorization(
            x,
            w=w,
            h=h,
            n_components=self.n_components,
            update_w=update_w,
            update_h=update_h,
            beta_loss=self.beta_loss,
            n_bootstrap=n_bootstrap,
            tol=self.tol,
            max_iter=self.max_iter,
            max_iter_mult=self.max_iter_mult,
            regularization=regularization,
            sparsity=sparsity,
            the_leverage=self.leverage,
            convex=self.convex,
            kernel=self.kernel,
            skewness=skewness,
            null_priors=null_priors,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def predict(self, estimator, blocks=None, cluster_by_stability=False, custom_order=False):
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------

        estimator : tuplet as returned by fit_transform

        blocks : array-like, shape(n_blocks), default None
            Size of each block (if any) in ordered heatmap.

        cluster_by_stability : boolean, default False
             Use stability instead of the_leverage to assign samples/features to clusters

        custom_order :  boolean, default False
             if False samples/features with highest the_leverage or stability appear on top of each cluster
             if True within cluster ordering is modified to suggest a continuum  between adjacent clusters

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

        return nmf_predict(
            estimator,
            blocks=blocks,
            the_leverage=self.leverage,
            cluster_by_stability=cluster_by_stability,
            custom_order=custom_order,
            verbose=self.verbose,
        )

    def permutation_test_score(self, estimator, y, n_permutations=100):
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------

        estimator : tuplet as returned by fit_transform

        y :  array-like, group to be predicted

        n_permutations :  integer, default: 100

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

        return nmf_permutation_test_score(estimator, y, n_permutations=n_permutations, verbose=self.verbose)


class NTF:
    def __init__(
        self,
        n_components=None,
        fast_hals=True,
        n_iter_hals=2,
        n_shift=0,
        unimodal=False,
        smooth=False,
        apply_left=False,
        apply_right=False,
        apply_block=False,
        tol=1e-6,
        max_iter=150,
        the_leverage="standard",
        random_state=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.fast_hals = fast_hals
        self.n_iter_hals = n_iter_hals
        self.n_shift = n_shift
        self.unimodal = unimodal
        self.smooth = smooth
        self.apply_left = apply_left
        self.apply_right = apply_right
        self.apply_block = apply_block
        self.tol = tol
        self.max_iter = max_iter
        self.leverage = the_leverage
        self.random_state = random_state
        self.verbose = verbose

    """Initialize NTF model

    Parameters
    ----------
    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

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
    NTF model

    Examples
    --------

    >>> import numpy as np

    >>> x = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])

    >>> from sklearn.decomposition import non_negative_factorization

    >>> w, h, n_iter = non_negative_factorization(x, n_components=2, \

        random_state=0)


    References
    ----------

    Fogel

    Lin
    """

    def fit_transform(
        self, x, n_blocks, n_bootstrap=None, w=None, h=None, f=None, update_w=True, update_h=True, update_f=True
    ):
        # noinspection PyShadowingNames
        """Compute Non-negative Matrix Factorization (NMF)

                Find three non-negative matrices (w, h, f) such as x = w @@ h @@ f + Error (@@ = tensor product).
                This factorization can be used for example for
                dimensionality reduction, source separation or topic extraction.

                The objective function is minimized with an alternating minimization of w
                and h.

                Parameters
                ----------

                x : array-like, shape (n_samples, n_features x n_blocks) Constant matrix. x is a tensor with shape (
                n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd dimensions.

                n_blocks : integer

                w : array-like, shape (n_samples, n_components)
                    prior w

                h : array-like, shape (n_features, n_components)
                    prior h

                f : array-like, shape (n_blocks, n_components)
                    prior h

                update_w : boolean, default: True
                    Update or keep w fixed

                update_h : boolean, default: True
                    Update or keep h fixed

                update_f : boolean, default: True
                    Update or keep f fixed

                n_bootstrap : Number of bootstrap runs

                Returns
                -------

                Estimator (dictionary) with following entries

                w : array-like, shape (n_samples, n_components)
                    Solution to the non-negative least squares problem.

                h : array-like, shape (n_features, n_components)
                    Solution to the non-negative least squares problem.

                f : array-like, shape (n_blocks, n_components)
                    Solution to the non-negative least squares problem.

                E : array-like, shape (n_samples, n_features x n_blocks) E is the residual tensor with shape (
                n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd dimensions.

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
                >>> w, h, n_iter = non_negative_factorization(x, n_components=2, \

                    random_state=0)


                References
                ----------

                Fogel

                Lin
                """

        return non_negative_tensor_factorization(
            x,
            n_blocks,
            w=w,
            h=h,
            f=f,
            n_components=self.n_components,
            update_w=update_w,
            update_h=update_h,
            update_f=update_f,
            fast_hals=self.fast_hals,
            n_iter_hals=self.n_iter_hals,
            n_shift=self.n_shift,
            unimodal=self.unimodal,
            smooth=self.smooth,
            apply_left=self.apply_left,
            apply_right=self.apply_right,
            apply_block=self.apply_block,
            n_bootstrap=n_bootstrap,
            tol=self.tol,
            max_iter=self.max_iter,
            the_leverage=self.leverage,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def predict(self, estimator, blocks=None, cluster_by_stability=False, custom_order=False):
        """See function description in class NMF

        """

        return nmf_predict(
            estimator,
            blocks=blocks,
            the_leverage=self.leverage,
            cluster_by_stability=cluster_by_stability,
            custom_order=custom_order,
            verbose=self.verbose,
        )

    @staticmethod
    def permutation_test_score(estimator, y, n_permutations=100):
        """See function description in class NMF

        """

        return nmf_permutation_test_score(estimator, y, n_permutations=n_permutations)
