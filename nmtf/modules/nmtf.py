""" Classes accessing Non-negative matrix and tensor factorization functions

"""

# Author: Paul Fogel

# License: MIT
# Dec 28, '19

from .nmtf_base import *

class NMF:
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
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss == 'kullback-leibler', the input
        matrix X cannot contain zeros.
    
    use_hals : boolean
        True -> HALS algorithm (note that convex & kullback-leibler loss options are not supported)
        False-> Projected gradiant

    tol : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations.

    max_iter_mult : integer, default: 20
        Maximum number of iterations in multiplicative warm-up to projected gradient (beta_loss = 'frobenius' only).

    leverage :  None | 'standard' | 'robust', default 'standard'
        Calculate leverage of W and H rows on each component.

    convex :  None | 'components' | 'transformation', default None
        Apply convex constraint on W or H.

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
    NMF model

    Example
    -------
    >>> from nmtf import *
    >>> myNMFmodel = NMF(n_components=4)

    References
    ----------
        
        P. Fogel, D.M. Hawkins, C. Beecher, G. Luta, S. S. Young (2013). A Tale of Two Matrix Factorizations.
        The American Statistician, Vol. 67, Issue 4.

        C. H.Q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
        IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1

    """

    def __init__(self, n_components=None,
                       beta_loss='frobenius',
                       use_hals = False,
                       tol=1e-6,
                       max_iter=150, max_iter_mult=20,
                       leverage='standard',
                       convex=None, kernel='linear',
                       random_state=None,
                       verbose=0):
        self.n_components = n_components
        self.beta_loss = beta_loss
        self.use_hals = use_hals
        self.tol = tol
        self.max_iter = max_iter
        self.max_iter_mult = max_iter_mult
        self.leverage = leverage
        self.convex = convex
        self.kernel = kernel
        self.random_state = random_state
        self.verbose = verbose

    def fit_transform(self, X, W=None, H=None,
                               update_W=True,
                               update_H=True,
                               n_bootstrap=None,
                               regularization=None, sparsity=0,
                               skewness=False,
                               null_priors=False):

        """Compute Non-negative Matrix Factorization (NMF)

        Find two non-negative matrices (W, H) such as x = W @ H.T + Error.
        This factorization can be used for example for
        dimensionality reduction, source separation or topic extraction.

        The objective function is minimized with an alternating minimization of W
        and H.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Constant matrix.

        W : array-like, shape (n_samples, n_components)
            prior W
            If n_update_W == 0 , it is used as a constant, to solve for H only.

        H : array-like, shape (n_features, n_components)
            prior H
            If n_update_H = 0 , it is used as a constant, to solve for W only.

        update_W : boolean, default: True
            Update or keep W fixed

        update_H : boolean, default: True
            Update or keep H fixed

        n_bootstrap : integer, default: 0
            Number of bootstrap runs.

        regularization :  None | 'components' | 'transformation'
            Select whether the regularization affects the components (H), the
            transformation (W) or none of them.

        sparsity : float, default: 0
            Sparsity target with 0 <= sparsity <= 1 representing either:
            - the % rows in W or H set to 0 (when use_hals = False)
            - the mean % rows per column in W or H set to 0 (when use_hals = True)

        skewness : boolean, default False
            When solving mixture problems, columns of X at the extremities of the convex hull will be given largest weights.
            The column weight is a function of the skewness and its sign.
            The expected sign of the skewness is based on the skewness of W components, as returned by the first pass
            of a 2-steps convex NMF. Thus, during the first pass, skewness must be set to False.
            Can be set only if convex = 'transformation' and prior W and H have been defined.

        null_priors : boolean, default False
            Cells of H with prior cells = 0 will not be updated.
            Can be set only if prior H has been defined.

        Returns
        -------

        Estimator (dictionary) with following entries

        W : array-like, shape (n_samples, n_components)
            Solution to the non-negative least squares problem.

        H : array-like, shape (n_components, n_features)
            Solution to the non-negative least squares problem.

        volume : scalar, volume occupied by W and H

        WB : array-like, shape (n_samples, n_components)
            A sample is clustered in cluster k if its leverage on component k is higher than on any other components.
            During each run of the bootstrap, samples are re-clustered.
            Each row of WB contains the frequencies of the n_components clusters following the bootstrap.
                Only if n_bootstrap > 0.

        HB : array-like, shape (n_components, n_features)
            A feature is clustered in cluster k if its leverage on component k is higher than on any other components.
            During each run of the bootstrap, features are re-clustered.
            Each row of HB contains the frequencies of the n_components clusters following the bootstrap.
                Only if n_bootstrap > 0.

        B : array-like, shape (n_observations, n_components) or (n_features, n_components)
            Only if active convex variant, H = B.T @ X or W = X @ B

        diff : scalar, objective minimum achieved

        Example
        -------
        >>> from nmtf import *
        >>> myMMFmodel = NMF(n_components=4)

        >>> # M: matrix to be factorized
        >>> estimator = myNMFmodel.fit_transform(M)

        References
        ----------
        
        P. Fogel, D.M. Hawkins, C. Beecher, G. Luta, S. S. Young (2013). A Tale of Two Matrix Factorizations.
        The American Statistician, Vol. 67, Issue 4.

        C. H.Q. Ding et al (2010) Convex and Semi-Nonnegative Matrix Factorizations
        IEEE Transactions on Pattern Analysis and Machine Intelligence Vol: 32 Issue: 1

        """
        return non_negative_factorization(X, W=W, H=H, n_components=self.n_components,
                                            update_W=update_W,
                                            update_H=update_H,
                                            beta_loss=self.beta_loss,
                                            use_hals=self.use_hals,
                                            n_bootstrap=n_bootstrap,
                                            tol=self.tol,
                                            max_iter=self.max_iter, max_iter_mult=self.max_iter_mult,
                                            regularization=regularization, sparsity=sparsity,
                                            leverage = self.leverage,
                                            convex=self.convex, kernel=self.kernel,
                                            skewness=skewness,
                                            null_priors=null_priors,
                                            random_state=self.random_state,
                                            verbose=self.verbose)

    def predict(self, estimator, blocks=None, cluster_by_stability=False, custom_order=False):

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

        Completed estimator with following entries:
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
        >>> from nmtf import *
        >>> myMMFmodel = NMF(n_components=4)
        >>> # M: matrix to be factorized
        >>> estimator = myNMFmodel.fit_transform(M)
        >>> estimator = myNTFmodel.predict(estimator)

        """

        return nmf_predict(estimator, blocks=blocks, leverage=self.leverage, cluster_by_stability=cluster_by_stability,
                           custom_order=custom_order, verbose=self.verbose)


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

        Example
        -------
        >>> from nmtf import *
        >>> myMMFmodel = NMF(n_components=4)
        >>> # M: matrix to be factorized
        >>> estimator = myNMFmodel.fit_transform(M)
        >>> # sampleGroup: the group each sample is associated with
        >>> estimator = myNMFmodel.permutation_test_score(estimator, RowGroups, n_permutations=100)

        """

        return nmf_permutation_test_score(estimator, y, n_permutations=n_permutations, verbose=self.verbose)

class NTF:
    """Initialize NTF model

    Parameters
    ----------
    n_components : integer
        Number of components, if n_components is not set : n_components = min(n_samples, n_features)

    fast_hals : boolean, default: False
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
    >>> from nmtf import *
    >>> myNTFmodel = NTF(n_components=4)

    Reference
    ---------
    A. Cichocki, P.H.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

    """

    def __init__(self, n_components=None,
                       fast_hals=False, n_iter_hals=2, n_shift=0,
                       unimodal=False, smooth=False,
                       apply_left=False, apply_right=False, apply_block=False,
                       tol=1e-6,
                       max_iter=150,
                       leverage='standard',
                       random_state=None,
                       init_type=1,
                       verbose=0):
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
        self.leverage = leverage
        self.random_state = random_state
        self.init_type = init_type
        self.verbose = verbose

    def fit_transform(self, X, n_blocks, n_bootstrap=None, 
                            regularization=None, sparsity=0, 
                            W=None, H=None, Q=None, 
                            update_W=True, update_H=True, update_Q=True):

        """Compute Non-negative Tensor Factorization (NTF)

        Find three non-negative matrices (W, H, Q) such as x = W @@ H @@ Q + Error (@@ = tensor product).
        This factorization can be used for example for
        dimensionality reduction, source separation or topic extraction.

        The objective function is minimized with an alternating minimization of W
        and H.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features x n_blocks)
            Constant matrix.
            X is a tensor with shape (n_samples, n_features, n_blocks), however unfolded along 2nd and 3rd dimensions.

        n_blocks : integer, number of blocks defining the 3rd dimension of the tensor

        n_bootstrap : Number of bootstrap runs

        regularization :  None | 'components' | 'transformation'
            Select whether the regularization affects the components (H), the
            transformation (W) or none of them.

        sparsity : float, default: 0
            Sparsity target with 0 <= sparsity <= 1 representing the mean % rows per column in W or H set to 0
.
        W : array-like, shape (n_samples, n_components)
            prior W

        H : array-like, shape (n_features, n_components)
            prior H

        Q : array-like, shape (n_blocks, n_components)
            prior Q

        update_W : boolean, default: True
            Update or keep W fixed

        update_H : boolean, default: True
            Update or keep H fixed

        update_Q : boolean, default: True
            Update or keep Q fixed
        
        Returns
        -------

        Estimator (dictionary) with following entries

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
        >>> from nmtf import *
        >>> myNTFmodel = NTF(n_components=4)
        >>> # M: tensor with 5 blocks to be factorized
        >>> estimator = myNTFmodel.fit_transform(M, 5)
        
        Reference
        ---------

        A. Cichocki, P.H.A.N. Anh-Huym, Fast local algorithms for large scale nonnegative matrix and tensor factorizations,
        IEICE Trans. Fundam. Electron. Commun. Comput. Sci. 92 (3) (2009) 708–721.

        """

        return non_negative_tensor_factorization(X, n_blocks, W=W, H=H, Q=Q, n_components=self.n_components,
                                                update_W=update_W,
                                                update_H=update_H,
                                                update_Q=update_Q,
                                                fast_hals=self.fast_hals, n_iter_hals=self.n_iter_hals, n_shift=self.n_shift, 
                                                regularization=regularization, sparsity=sparsity, unimodal=self.unimodal, smooth=self.smooth,
                                                apply_left=self.apply_left,
                                                apply_right=self.apply_right,
                                                apply_block=self.apply_block,
                                                n_bootstrap=n_bootstrap,
                                                tol=self.tol,
                                                max_iter=self.max_iter,
                                                leverage=self.leverage,
                                                random_state=self.random_state,
                                                init_type=self.init_type,
                                                verbose=self.verbose)

    def predict(self, estimator, blocks=None, cluster_by_stability=False, custom_order=False):

        """See function description in class NMF

        """

        return nmf_predict(estimator, blocks=blocks, leverage=self.leverage, cluster_by_stability=cluster_by_stability,
                           custom_order=custom_order, verbose=self.verbose)

    def permutation_test_score(self, estimator, y, n_permutations=100):

        """See function description in class NMF

        """

        return nmf_permutation_test_score(estimator, y, n_permutations=n_permutations)

