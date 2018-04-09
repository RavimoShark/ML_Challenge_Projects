"""Classes and utilities for flexcrf linear chain data structures."""

# Author: Slim ESSID <slim.essid@telecom-paristech.fr>
# License: LGPL v3

import itertools as it
import numpy as np
import pandas as pd
from .base import _BaseDataFrame

# TODO: move functions from feature_extraction.py here and clean them up


# -- some utilities ----------------------------------------------------------

def to_int(ch):
    if ch.isdigit() or (ch.startswith('-') and ch[1:].isdigit()):
        return int(ch)
    else:
        return ch


def remap_labels(s):
    """
    if s like '2:1_2', return (2, '1_2')
    otherwise return (s, s)
    """
    special_y_flag = ':' in s if isinstance(s, str) else False
    if special_y_flag:  # '2:1_2'-like spec detected
        s_prt, s_drv = s.split(':')
        return to_int(s_prt), to_int(s_drv)
    else:
        return s, s


def ensure_list(s):
    """return s if s is a list and a list [s] if s is a string"""

    if isinstance(s, str):
        tmp, s = s, []
        s.insert(0, tmp)
    else:
        return s


def _primitive_obs_tuples(g_xy_desc, ND='ND'):
    """Prepare tuples to define multiindex pandas structure for g_xy values"""

    feat_info = {feat: {'type': typ, 'dim': dim, 'labels': y}
                 for typ, feat, dim, y in g_xy_desc}

    g_xy_tuples = []
    for _, feat, dim, y in g_xy_desc:
        if len(y) == 0:
            continue
        g_xy_tuples += list(it.product([ND], list(y), [feat], range(dim)))
        # (y_{i-1}=ND,  y_i, <feature name>, <feature coef index>)

    # y_{i-1}=ND means y_{i-1} ignored to indicate observation functions
    # (as opposed to transition functions); NOTE THAT nan can't be used here,
    # or it will cause problems with pandas indexing methods, e.g. when
    # calling df.loc[] to set values.

    return g_xy_tuples, feat_info


def _primitive_transition_tuples(h_xyy_desc):
    """Prepare tuples to define flexcrf multiindex for h_xyy values"""

    feat_info = {feat: {'type': typ, 'dim': dim, 'labels': y}
                 for typ, feat, dim, y in h_xyy_desc}

    h_xyy_tuples = []
    for _, feat, dim, yy in h_xyy_desc:
        if len(yy) == 0:
            continue  # type t1 functions, merely indicators not to stored here
        for s, q in yy:
            h_xyy_tuples += list(it.product([s], [q], [feat], range(dim)))
            # (y_{i-1}, y_i, <feature name>, <feature coef index>)

    return h_xyy_tuples, feat_info


def _derived_obs_tuples(g_xy_desc, f_xy_desc=None, label_set=None,
                        feat_ind0=0, ND='ND'):
    """
    Prepare tuples to define flexcrf multiindex for f_xy values based on
    f_xy_desc. If the latter is None a defaut specification will be created
    that requires
    i) label_set to be not None and
    ii) NO tied/restricted labels (e.g. '1_2') were defined in g_xy_desc,
    as such labels require that a f_xy_desc be specified.
    """

    if f_xy_desc is None:  # derived feat. spec not given, create default spec
        if label_set is None:
            raise ValueError("label_set needs to be given when f_xy_desc"
                             " is None")
        f_xy_desc = {feat:
                     {y: y for y in label_set}
                     if len(labels) == 0  # o3 type
                     or labels == {ND}  # o2 type
                     else
                     {y: y for y in labels}
                     for _, feat, _, labels in g_xy_desc}

    f_xy_tuples = []
    for _, feat, dim, y in g_xy_desc:
        for y_ in f_xy_desc[feat]:
            s_ = f_xy_desc[feat][y_]
            if np.iterable(s_):
                s_ = ensure_list(s_)
                for s in s_:
                    s_prt, s_drv = remap_labels(s)
                    if y == {ND}:
                        s_prt = ND
                    f_xy_tuples += list(it.product([ND], [y_], [ND], [s_prt],
                                        [ND], [s_drv], [feat], range(dim)))
            else:
                if y == {ND}:
                    s_prt = ND
                else:
                    s_prt = s_
                f_xy_tuples += list(it.product([ND], [y_], [ND], [s_prt],
                                               [ND], [s_],
                                               [feat], range(dim)))
    n_fxy_maps = len(f_xy_tuples)

    # add a unique index number for each feat function (relating to a unique
    # ('y_drv', 'feat_grp', 'ingrp_ind') triplet
    triplets = set([(arg[5], arg[6], arg[7]) for arg in f_xy_tuples])
    # these are all possible ('y_drv', 'feat_grp', 'ingrp_ind') triplets.
    feat_indices_ = {k: feat_ind+feat_ind0
                     for feat_ind, k in enumerate(triplets)}
    for i, (_, y_obs, _, y_prt, _, y_drv,
            feat_grp, ingrp_ind) in enumerate(f_xy_tuples):
            f_xy_tuples[i] += tuple(
                [feat_indices_[y_drv, feat_grp, ingrp_ind]])
    n_f_xy_feat = len(triplets)

    return f_xy_tuples, n_f_xy_feat, n_fxy_maps


def _derived_transition_tuples(h_xyy_desc, t_xyy_desc=None, label_set=None,
                               feat_ind0=0, ND='ND'):
    """
    Prepare tuples to define multiindex pandas structure for t_xyy values
    based on t_xyy_desc. If the latter is None a defaut specification will be
    created that requires
    i) label_set to be not None and
    ii) NO tied/restricted labels (e.g. '1_2') were defined in h_xyy_desc,
    as such labels require that a t_xyy_desc be specified.
    """
    if t_xyy_desc is None:  # derived feat. spec not given, create default spec
        if label_set is None:
            raise ValueError("label_set needs to be given when t_xyy_desc"
                             " is None")
        t_xyy_desc = {feat:
                      {(y1, y2): (y1, y2)
                       for (y1, y2) in it.product(label_set, label_set)
                       }
                      if len(label_tr) == 0  # e.g. t1 type
                      or label_tr == {ND}  # e.g. t3 type
                      else
                      {(y1, y2): (y1, y2) for (y1, y2) in label_tr}
                      for _, feat, _, label_tr in h_xyy_desc}

    t_xyy_tuples = []
    for _, feat, dim, yy in h_xyy_desc:
        for yy_ in t_xyy_desc[feat]:
            ss_ = t_xyy_desc[feat][yy_]
            if len(yy) == 0:  # 't1' feature function
                t_xyy_tuples += list(it.product([yy_[0]], [yy_[1]], [ND], [ND],
                                                [ss_[0]], [ss_[1]], [feat],
                                                range(dim)))
            else:
                ss_ = ensure_list(ss_)
                for s1, s2 in ss_:
                    s1_prt, s1_drv = remap_labels(s1)
                    s2_prt, s2_drv = remap_labels(s2)
                    t_xyy_tuples += list(it.product(
                                         [yy_[0]], [yy_[1]],
                                         [s1_prt], [s2_prt],
                                         [s1_drv], [s2_drv],
                                         [feat], range(dim)))
    n_txyy_maps = len(t_xyy_tuples)

    # add a unique index number for each feat function
    triplets = set([(arg[4], arg[5], arg[6], arg[7]) for arg in t_xyy_tuples])
    feat_indices_ = {k: feat_ind+feat_ind0
                     for feat_ind, k in enumerate(triplets)}
    for i, label_names in enumerate(t_xyy_tuples):
            t_xyy_tuples[i] += tuple(
                [feat_indices_[label_names[4:]]])
    n_t_xyy_feat = len(triplets)

    return t_xyy_tuples, n_t_xyy_feat, n_txyy_maps


# -- class definitions -------------------------------------------------------

class FeatFunIndex(object):
    """
    flexcrf MultiIndex class for bigram-based linear chain data (wrapping
    pandas MultiIndex).

    Parameters
    ----------
    label_set : list or set of int
        List of unique output label values

    g_xy_desc : list of 4-element tuples (str, str, int, set)
        Structure describing observation feature groups to be stored in the
        MultiIndex, following (<feat_type>, <feat_group_name>,
                               <nb_coefs_per_group>,
                               <labels_for_which_grp_is_defined>).
    f_xy_desc : dict (otional)
        Strucuture describing how to derive features from primitive ones.

    h_xyy_desc : list of 4-element tuples (str, str, int, set)
        Structure describing observation feature groups to be stored in the
        MultiIndex, following (<feat_type>, <feat_group_name>,
                               <nb_coefs_per_group>,
                               <labels_for_which_grp_is_defined>).

    t_xyy_desc : dict (otional)
        Strucuture describing how to derive features from primitive ones.

    Attributes
    ----------
    ND : str or int
        Value used to indicate "Non Defined" (e.g. y1=ND for obs feat function)

    feat_info : dict
        Information about the features (type, dim and labels)

    n_feat : int
        Total number of columns

    n_feat_fun : int
        Total number of feature functions (= number of model parameters)

    """
    # TODO: allow for using either only obs funs or only transition funs

    def __init__(self, g_xy_desc, h_xyy_desc, label_set, ND='ND',
                 f_xy_desc=None, t_xyy_desc=None):

        self.ND = ND
        self.label_set = label_set

        g_xy_tuples, feat_info = _primitive_obs_tuples(g_xy_desc, ND=ND)
        n_g_xy_feat = len(g_xy_tuples)
        h_xyy_tuples, feat_info_ = _primitive_transition_tuples(h_xyy_desc)
        n_h_xyy_feat = len(h_xyy_tuples)

        gh_index_tuples = g_xy_tuples + h_xyy_tuples
        # VERY IMPORTANT FOR INDEXING TO WORK PROPERLY: make sure all levels of
        # the multiple index object are sorted lexicographically (see
        # http://pandas.pydata.org/pandas-docs/stable/advanced.html#the-need-for-sortedness-with-multiindex)
        # hence:
        gh_index_tuples.sort()

        if f_xy_desc is None and t_xyy_desc is None:  # do derived features
            self.feat_type = 'primitive'
            # pandas multiple index for feature functions
            self.index = pd.MultiIndex.from_tuples(gh_index_tuples,
                                                   names=['y1', 'y2',
                                                          'feat_grp',
                                                          'ingrp_ind'])
            self.n_feat = n_h_xyy_feat + n_g_xy_feat
            self.n_feat_fun = self.n_feat
        else:
            self.feat_type = 'derived'

            if f_xy_desc is None:  # derived feat. spec not given, use default
                pass
                # TODO: make sure no tied/restricted labels (e.g. '1_2') were
                # defined in g_xy_desc, as such labels require that a f_xy_desc
                # be specified.
            f_xy_tuples, n_f_xy_feat, n_fxy_maps = \
                _derived_obs_tuples(g_xy_desc, f_xy_desc,
                                    label_set=label_set, ND=ND)
            self.f_xy_tuples = f_xy_tuples

            if t_xyy_desc is None:  # derived feat. spec not given, use default
                pass
                # TODO: make sure no tied/restricted labels (e.g. '1_2') were
                # defined in g_xy_desc, as such labels require that a f_xy_desc
                # be specified.
            t_xyy_tuples, n_t_xyy_feat, n_txyy_maps = \
                _derived_transition_tuples(h_xyy_desc, t_xyy_desc,
                                           label_set=label_set,
                                           feat_ind0=n_f_xy_feat, ND=ND)
            self.t_xyy_tuples = t_xyy_tuples

            ft_index_tuples = f_xy_tuples + t_xyy_tuples
            ft_index_tuples.sort()  # VERY IMPORTANT

            # pandas multiple index for feature functions
            self.index = \
                pd.MultiIndex.from_tuples(ft_index_tuples,
                                          names=['y1_obs', 'y2_obs',
                                                 'y1_prt', 'y2_prt',
                                                 'y1_drv', 'y2_drv',
                                                 'feat_grp', 'ingrp_ind',
                                                 'feat_ind'])
            self.n_feat = n_fxy_maps + n_txyy_maps
            self.n_feat_fun = n_f_xy_feat + n_t_xyy_feat
            # this is the final number of feat functions
            # used in the model, not to be confused with n_feat_maps counting
            # all mappings of primitive features into derived features,
            # typically 1:1_2 and 2:1_2 contribute 2 elements to n_feat_maps
            # but only one element to n_feat_fun, since the two mappings yield
            # one derived feature function linked to y = '1_2'.

            feat_info.update(feat_info_)  # merge the two feat_info dicts
            self.feat_info = feat_info


# -----------------------------------------------------------------------------

class LinearChainData(_BaseDataFrame):
    """
    flexcrf data class for bigram-based linear chain data.

    Parameters
    ----------
    mindex : pandas.core.index.MultiIndex
        A pandas MultiIndex object defining the structure of the data columns

    n_obs : int
        Number of observations in the sequence. If None, an empty DataFrame
        will be created.

    index : list or ndarray, shape (n_obs,)
        Time position index.

    data : ndarray, shape (n_obs, n_feat)
        Array containing the data to fill the flexcrf data frame. If None
        an empty DataFrame will be created (by filling it with NaNs).

    is_sparse : boolean, default=False
        Wether to use a SparseDataFrame.

    Attributes
    ----------
    ND : str or int
        Value used to indicate "Non Defined" (e.g. y1=ND for obs feat function)

    n_feat : int
        Total number of columns

    n_feat_fun : int
        Total number of feature functions (= number of model parameters)

    n_obs : int
        Total number of observations.

    shape : tuple
        Data frame shape.

    feat_type : str
        'derived' or 'primitive'.

    label_set : list
        List of output label values.

    """
    # TODO: make sure when f_xy is created that on first row (t=0) all values
    # are 0 for y1!='ND'

    def __init__(self, mindex=None, data=None, n_obs=None,
                 index=None, is_sparse=False):

        _BaseDataFrame.__init__(self, mindex=mindex, data=data,
                                n_obs=n_obs, index=index, is_sparse=is_sparse)

        self.label_set = list(self.data.columns.levels[1].values)

        if self.feat_type == 'derived':  # derived features data frame
            self.n_feat_fun = np.unique(
                self.data.columns.levels[8].values).shape[0]

    def feat_indices(self):
        """
        Returns the feature indices as they appear through the columns of the
        current data frame.
        """

        if self.feat_type == 'derived':  # only for derived features
            return list(self.data.columns.labels[8].values())

    def y1_values(self):
        """Returns the set of all unique values for y1 (or y1_obs)"""

        vals = set(self.data.columns.levels[0].tolist())
        vals.remove(self.ND)

        return list(vals)

    def y2_values(self):
        """Returns the set of all unique values for y2 (or y2_obs)"""

        vals = set(self.data.columns.levels[1].tolist())

        return list(vals)

    def select(self, rows=slice(None), y1=slice(None), y2=slice(None),
               y1_prt=slice(None), y2_prt=slice(None), y1_drv=slice(None),
               y2_drv=slice(None), feat=slice(None), g_ind=slice(None),
               feat_ind=slice(None), arr_out=False):
        """
        Selects a slice from the flexcrf data frame matching the values of
        the chosen index keys.

        Parameters
        ----------

        y1 :

        y2 :

        y1_prt :
        ...

        """
        if self.feat_type == 'primitive':
            data = self.data.loc[
                rows,
                (y1, y2, feat, g_ind)
                ]
        elif self.feat_type == 'derived':
            data = self.data.loc[
                rows,
                (y1, y2, y1_prt, y2_prt, y1_drv, y2_drv,
                 feat, g_ind, feat_ind)
                ]
        else:
            raise ValueError('Size of MultiIndex can only be 4 or 9, %d'
                             ' detected' % len(self.data.columns.names))
        if arr_out:
            return data.values
        else:
            return data

    def iselect(self, rows=0, y1=slice(None), y2=slice(None),
                y1_prt=slice(None), y2_prt=slice(None), y1_drv=slice(None),
                y2_drv=slice(None), feat=slice(None), g_ind=slice(None),
                feat_ind=slice(None)):
        """
        Returns indices of columns matching the values of the chosen index keys

        Parameters
        ----------

        y1 :

        y2 :

        y1_prt :
        ...

        """
        if self.feat_type == 'derived':
            return self.data.loc[
                rows,
                (y1, y2, y1_prt, y2_prt, y1_drv, y2_drv,
                 feat, g_ind, feat_ind)
                ].index.labels[8]
            # TODO: there should be a better way of doing this, check pandas!!!
        else:
            raise ValueError('Size of MultiIndex for derived feat can only  be'
                             ' 9, %d detected' % len(self.data.columns.names))

    def set(self, data, rows=slice(None), y1=slice(None), y2=slice(None),
            y1_prt=slice(None), y2_prt=slice(None), y1_drv=slice(None),
            y2_drv=slice(None), feat=slice(None), g_ind=slice(None),
            feat_ind=slice(None)):
        """
        Fills the flexcrf data frame across the slices matching the values of
        the chosen index keys.

        Parameters
        ----------
        data :

        y1 :

        y2 :

        y1_prt :
        ...

        """
        if self.feat_type == 'primitive':
            self.data.loc[rows, (y1, y2, feat, g_ind)] = data
        elif self.feat_type == 'derived':
            self.data.loc[
                rows,
                (y1, y2, y1_prt, y2_prt, y1_drv, y2_drv,
                 feat, g_ind, feat_ind)
                ] = data
        else:
            raise ValueError('Size of MultiIndex can only be 4 or 9, %d'
                             ' detected' % len(self.data.columns.names))
