import logging

import numpy as np

from .nmtf_utils import StatusBoxTqdm, build_clusters, global_sign

logger = logging.getLogger(__name__)


# TODO (pcotte): typing

class Estimator:
    """
    Estimator object. Created by `nmtf.nmtf.NMTF.fit_transform`, and updated by `nmtf.nmtf.NMTF.predict` and
    `nmtf.nmtf.NMTF.fit_transform.permutation_test_score`. The same Estimator class is used both by NMF and NTF, the
    difference will be that some attributes will be None in the case of NMF.

    Attributes
    -------
    w: array-like, shape (n_samples, n_components)
        Solution to the non-negative least squares problem.
    h: array-like, shape (n_components, n_features)
        Solution to the non-negative least squares problem.
    q: array-like, shape (n_blocks, n_components)
        Only for NTF. Solution to the non-negative least squares problem.
    volume: scalar, volume occupied by W and H
    wb: array-like, shape (n_samples, n_components)
        A sample is clustered in cluster k if its leverage on component k is higher than on any other
        components. During each run of the bootstrap, samples are re-clustered.
        Each row of WB contains the frequencies of the n_components clusters following the bootstrap.
        Only if n_bootstrap > 0.
    hb: array-like, shape (n_components, n_features)
        A feature is clustered in cluster k if its leverage on component k is higher than on any other
        components. During each run of the bootstrap, features are re-clustered.
        Each row of HB contains the frequencies of the n_components clusters following the bootstrap.
        Only if n_bootstrap > 0.
    b: array-like, shape (n_observations, n_components) or (n_features, n_components)
        Only for NMF and only if active convex variant, H = B.T @ X or W = X @ B
    diff: scalar, objective minimum achieved
    wl: array-like, shape (n_samples, n_components)\n
        Set by `nmtf.estimator.Estimator.predict`. Sample leverage on each component\n
    hl: array-like, shape (n_features, n_components)\n
        Set by `nmtf.estimator.Estimator.predict`. Feature leverage on each component\n
    ql: array-like, shape (n_blocks, n_components)\n
        Set by `nmtf.estimator.Estimator.predict`. Block leverage on each component (NTF only)\n
    wr: vector-like, shape (n_samples)\n
        Set by `nmtf.estimator.Estimator.predict`. Ranked sample indexes (by cluster and leverage or stability)
        Used to produce ordered heatmaps\n
    hr: vector-like, shape (n_features)\n
        Set by `nmtf.estimator.Estimator.predict`. Ranked feature indexes (by cluster and leverage or stability)
        Used to produce ordered heatmaps\n
    wn: vector-like, shape (n_components)\n
        Set by `nmtf.estimator.Estimator.predict`. Sample cluster bounds in ordered heatmap\n
    hn: vector-like, shape (n_components)\n
        Set by `nmtf.estimator.Estimator.predict`. Feature cluster bounds in ordered heatmap\n
    wc: vector-like, shape (n_samples)\n
        Set by `nmtf.estimator.Estimator.predict`. Sample assigned cluster\n
    hc: vector-like, shape (n_features)\n
        Set by `nmtf.estimator.Estimator.predict`. Feature assigned cluster\n
    qc: vector-like, shape (size(blocks))\n
        Set by `nmtf.estimator.Estimator.predict`. Block assigned cluster (NTF only)
    score: float
        Set by `nmtf.estimator.Estimator.permutation_test_score`. The true score without permuting targets.
    pvalue: float
        Set by `nmtf.estimator.Estimator.permutation_test_score`. The p-value, which approximates the probability that
        the score would be obtained by chance.
    cs: array-like, shape(n_components)
        Set by `nmtf.estimator.Estimator.permutation_test_score`. The size of each cluster
    cp: array-like, shape(n_components)
        Set by `nmtf.estimator.Estimator.permutation_test_score`. The pvalue of the most significant group within each
        cluster
    cg: array-like, shape(n_components)
        Set by `nmtf.estimator.Estimator.permutation_test_score`. The index of the most significant group within each
        cluster
    cn: array-like, shape(n_components, n_groups)
        Set by `nmtf.estimator.Estimator.permutation_test_score`. The size of each group within each cluster
    """
    def __init__(self, w, h, volume, diff, leverage, verbose, wb=None, hb=None, b=None, q=None):
        self.w = w
        self.h = h
        self.volume = volume
        self.wb = wb
        self.hb = hb
        self.diff = diff
        self.leverage = leverage
        self.verbose = verbose
        self.b = b
        self.q = q
        self.wl = None
        self.hl = None
        self.wr = None
        self.hr = None
        self.wn = None
        self.hn = None
        self.wc = None
        self.hc = None
        self.ql = None
        self.qc = None
        self.score = None
        self.pvalue = None
        self.cs = None
        self.cp = None
        self.cg = None
        self.cn = None
        
        if q is None:
            self.kind = "nmf"
        else:
            self.kind = "ntf"
        
        if b is not None and self.kind == "ntf":
            raise ValueError("NTF estimator can not have 'b' attribute")

    def predict(self, blocks=None, cluster_by_stability=False, custom_order=False):
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------
        blocks: array-like, shape(n_blocks), default None
            Size of each block (if any) in ordered heatmap.
        cluster_by_stability: boolean, default False
             Use stability instead of leverage to assign samples/features to clusters
        custom_order:  boolean, default False
             if False samples/features with highest leverage or stability appear on top of each cluster
             if True within cluster ordering is modified to suggest a continuum  between adjacent clusters
        """
        mt = self.w
        mw = self.h
        if self.kind == "ntf":
            # X is a 3D tensor, in unfolded form of a 2D array
            # horizontal concatenation of blocks of equal size.
            mb = self.q
            nmf_algo = "ntf"
            n_blocks = mb.shape[0]
            blk_size = mw.shape[0] * np.ones(n_blocks)
        else:
            mb = np.array([])
            nmf_algo = "nmf"
            if blocks is None:
                n_blocks = 1
                blk_size = np.array([mw.shape[0]])
            else:
                n_blocks = blocks.shape[0]
                blk_size = blocks

        if self.wb is not None:
            mt_pct = self.wb
        else:
            mt_pct = None

        if self.hb is not None:
            mw_pct = self.hb
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
        self.update(
            wl=mtn,
            hl=mwn,
            wr=r_ct,
            hr=r_cw,
            wn=n_ct,
            hn=n_cw,
            wc=row_clust,
            hc=col_clust,
            ql=mbn,
            qc=block_clust
        )

    # TODO (pcotte): this function is not called by any pytest. Make a pytest calling it.
    def permutation_test_score(self, y, n_permutations=100):
        """Derives from factorization result ordered sample and feature indexes for future use in ordered heatmaps

        Parameters
        ----------
        y:  array-like, group to be predicted
        n_permutations:  integer, default: 100
        """
        mt = self.w
        r_ct = self.wr
        n_ct = self.wn
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

        self.update(
            score=prun,
            pvalue=pglob,
            cs=cluster_size,
            cp=cluster_prob,
            cg=cluster_group,
            cn=cluster_ngroup,
        )

    def update(self, **kwargs):
        """Updates this estimator's attributes according to given keyword arguments. Only attributes already defined
        in the estimator can be updated this way."""
        for item in kwargs:
            if not hasattr(self, item):
                raise ValueError(f"Can not update attribute '{item}'")
            setattr(self, item, kwargs.get(item))
