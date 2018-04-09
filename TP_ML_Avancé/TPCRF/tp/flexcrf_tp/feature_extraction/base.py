"""Base class for flexcrf data structures, based on padas data frames"""

# Author: Slim ESSID <slim.essid@telecom-paristech.fr>
# License: LGPL v3

import numpy as np
import pandas as pd


# some utility functions

def _frame_with_type_from_store(data_frame, df_type):
    """Instantiate a new data frame of type df_type."""

    from .linear_chain import LinearChainData

    if df_type == 'LinearChainData':
        d_frame = LinearChainData(data=data_frame)
    else:
        raise ValueError("Unknown flexcrf data frame type")

    return d_frame


# -- flexcrf data classes definitions -----------------------------------------
class _BaseDataFrame(object):
    """
    Base class for flexcrf data frames, based on pandas data frames with
    a multiindex as columns and time across rows.

    Parameters
    ----------
    mindex : pandas.core.index.MultiIndex or None
        A pandas MultiIndex object defining the structure of the data columns.
        If None then a pandas.DataFrame is expected as an input for data.

    data : ndarray or pandas.DataFrame, shape (n_obs, n_feat)
        Array containing the data to fill the flexcrf data frame or
        pandas.DataFrame to be cast to a flexcrf DataFrame.
        If None an empty DataFrame will be created (by filling it with NaNs).

    n_obs : int
        Number of observations in the sequence. If None, an empty DataFrame
        will be created.

    index : list or ndarray, shape (n_obs,)
        Time position index.

    is_sparse : boolean, default=False
        Wether to use a SparseDataFrame.

    Attributes
    ----------
    ND : str or int
        Value used to indicate "Non Defined" (e.g. y1=ND for obs feat function)

    n_feat : int
        Total number of columns.

    n_obs : int
        Total number of observations.

    shape : tuple
        Data frame shape.

    feat_type : str
        'derived' or 'primitive'.

    """

    def __init__(self, mindex=None, data=None, n_obs=None,
                 index=None, is_sparse=False):

        # index label value for Not Defined cases (e.g $y_{i-1}$ for obs feats)
        self.ND = -1

        if mindex is None:
            if data is None or isinstance(data, np.ndarray):
                raise ValueError("Missing a MultiIndex")

        if data is None:
            if n_obs is None:
                data = np.empty((1, len(mindex)))
            else:
                data = np.empty((n_obs, len(mindex)))

        if index is None:
            index = range(data.shape[0])

        if is_sparse:
                self.data = pd.SparseDataFrame(data, columns=mindex,
                                               index=index)
        else:
            if isinstance(data, np.ndarray):
                self.data = pd.DataFrame(data, columns=mindex, index=index)
            else:
                self.data = data

        self.shape = self.data.shape
        self.n_obs = data.shape[0]
        self.n_feat = self.data.shape[1]

        if len(self.data.columns.levels) == 9:  # derived features data frame
            self.feat_type = 'derived'
        else:
            self.feat_type = 'primitive'

    def get_feat_names(self):
        if len(self.data.columns.levels) == 9:  # derived features data frame
            feat_name_level = 7
        else:
            feat_name_level = 2

        return list(self.data.columns.levels[feat_name_level].values)


class DataStore(object):
    """
    Base class for flexcrd data stores, based on pandas.io.pytables.HDFStore.

    Parameters
    ----------

    file_path : str
        Full path to hdf5 file (a pandas.io.pytables.HDFStore) containing
        the data.

    access_mode : str
        'r' (default), 'w', 'a' or 'r+' as for pandas.HDFStore.

    Attributes
    ----------

    store : pandas.io.pytables.HDFStore
        Store to hold the data arranged following flexcrf conventions.

    """

    def __init__(self, file_path, access_mode='r'):
        pd.set_option('io.hdf.default_format', 'table')  # to append data
        self.access_mode = access_mode
        self.store = pd.HDFStore(file_path, access_mode)
        self.file_path = file_path
        if access_mode == 'w':
            self._n_seq = 0
            self._n_feat_fun = 0
            self._df_type = None
        else:
            self._n_seq = len(self.store.root.__members__)
            self._n_feat_fun = self.store.root._v_attrs.N_FEAT_FUN
            self._df_type = self.store.root._v_attrs.DF_TYPE

    def close(self):
        self.store.close()

    def _get_n_seq(self):
        """Get number of training data sequences inside the store"""
        return self._n_seq

    def _set_n_seq(self):
        raise RuntimeError("n_seq cannot be modified")

    n_seq = property(_get_n_seq, _set_n_seq)

    def _get_n_feat_fun(self):
        """Get number of training data sequences inside the store"""
        return self._n_feat_fun

    def _set_n_feat_fun(self):
        raise RuntimeError("n_feat_fun cannot be modified")

    n_feat_fun = property(_get_n_feat_fun, _set_n_feat_fun)

    def get_seq_names(self):
        """Return the list of names of all sequences inside the store"""
        return self.store.root.__members__

    def get_seq_descriptor_types(self, seq_name):
        """Return descriptors available for the sequence, e.g. 'features',
           'potentials' or 'labels'
        """
        return self.store.root._f_get_child(seq_name).__members__

    def has_features(self):
        """True if store contains /features/"""
        fst_seq_name = self.get_seq_names()[0]
        if 'features' in self.get_seq_descriptor_types(fst_seq_name):
            return True
        else:
            return False

    def has_potentials(self):
        """True if store contains /potentials/"""
        fst_seq_name = self.get_seq_names()[0]
        if 'potentials' in self.get_seq_descriptor_types(fst_seq_name):
            return True
        else:
            return False

    def get_seq_info(self, seq_name):
        """ !! NOT FINISHED !!
        Return information about sequence named `seq_name`.

        Parameters
        ----------
        seq_name : str
            Name of the sequence to be returned

        Returns
        -------
        info : dict
            Dictionnary with keys `has_features`, `has_potentials`,
            `n_feat_fun`, `n_obs`

        """

        info = {'has_features': False, 'has_potentials': False,
                'n_feat_fun': None, 'n_obs': None}

        if self.has_features():
            info['has_features'] = True

        if self.has_potentials():
            info['has_potentials'] = True

        return info

    def get_sequence(self, seq_name, with_feat=True, with_psi=False):
        """
        Return the data of sequence named `seq_name`.

        Parameters
        ----------
        seq_name : str
            Name of the sequence to be returned

        with_feat : bool
            If True (default), return feature function values.

        with_psi : bool
            If True (default is False), return potential values.

        Returns
        -------
        f_xy : LinearChainData or another _BaseDataFrame child
            A flexcrf data object containing the feature function values.

        psi_xy : _BaseDataFrame or one of its children
            A flexcrf data object containing the user-defined potential values.

        y : ndarray (n_obs,)
            Labels for each observation in the current sequence.
        """

        y = np.squeeze(self.store.get(seq_name+'/labels').values)
        out = []
        if with_feat:
            f_xy = self.store.get(seq_name+'/features')
            f_xy = _frame_with_type_from_store(f_xy, self._df_type)
            out.append(f_xy)
        if with_psi:
            psi_xy = self.store.get(seq_name+'/potentials')
            psi_xy = _frame_with_type_from_store(psi_xy, self._df_type)
            out.append(psi_xy)

        out.append(y)
        return out

    def append_sequence(self, seq_name, y, f_xy=None, psi_xy=None):
        """
        Append data of sequence named `seq_name` to the store.

        Parameters
        ----------
        seq_name : str
            Name of the sequence.

        f_xy : _BaseDataFrame or one of its children
            A flexcrf data object containing the feature function values.

        psi_xy : _BaseDataFrame or one of its children
            A flexcrf data object containing the user-defined potential values.

        y : pd.DataFrame (n_obs,)
            Labels for each observation in the current sequence.

        """

        self.store.append(seq_name+'/labels/', pd.DataFrame(y))
        if f_xy is not None:
            self.store.append(seq_name+'/features/', f_xy.data)
            # note that f_xy.data and not f_xy is saved, as only the former
            # is supported by pandas.io.pytables.HDFStore.
        if psi_xy is not None:
            self.store.append(seq_name+'/potentials/', psi_xy.data)

        self._n_seq = len(self.store.root.__members__)

        if self._n_seq == 1:  # first sequence added
            if f_xy is not None:
                d_type = type(f_xy).__name__
                if f_xy.feat_type == 'derived':
                    self._n_feat_fun = f_xy.n_feat_fun
                else:
                    self._n_feat_fun = -1  # not informative for primitive feat

            elif psi_xy is not None:
                d_type = type(psi_xy).__name__

            else:
                raise ValueError("Both f_xy and psi_xy are None")

            self._df_type = d_type
            self.store.root._v_attrs.DF_TYPE = d_type
            self.store.root._v_attrs.N_OBS = len(y)
            self.store.root._v_attrs.N_FEAT_FUN = self._n_feat_fun


if __name__ == '__main__':  # just for testing purposes
    pass
