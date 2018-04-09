"""Linear chain CRF models, specification, inference and learning."""

# Author: Slim ESSID <slim.essid@telecom-paristech.fr>
# License: LGPL v3

import itertools as it
import warnings

# import pandas as pd
import numpy as np

from ..feature_extraction.linear_chain import LinearChainData
from ..utils import logsumexp


# -- Helper functions for linear chain CRF inference and learning -------------

def _feat_fun_values(f_xy, y, return_dframe=False, with_f_x_sum=True):
    """
    Computes feature function values for given y sequence.

    Parameters
    ----------
    f_xy : LinearChainData
        A flexcrf data frame with a multiindex indicating feature values w.r.t
        all possible (y_{i-1}, y_i, x) configs, following flexcrf conventions.

    y : ndarray, shape (n_obs,)
        Defines the sequence of outputs.

    return_dframe: bool, default=True
        Whether to return an ndarray (default) or a pandas dataframe.

    with_f_x_sum: bool, default=True
        Whether to compute and return f_x_sum.

    Returns
    -------
    f_x : ndarray or pandas dataframe, shape (n_obs, n_feat_fun).
        Feature function values for given y sequence.

    f_x_sum : ndarray, shape (n_feat_fun,)
        Sum (across time) of feature function values for current y sequence;
        returned only if `return_dframe=False` and `with_f_x_sum=True`.

    """

    label_set = np.unique(y)
    n_obs, n_feat_maps = f_xy.shape
    ND = f_xy.ND
    if return_dframe:
        f_x = LinearChainData(f_xy.data.columns,
                              data=np.zeros((n_obs, n_feat_maps)))
    else:
        f_x = np.zeros((n_obs, f_xy.n_feat_fun))

    #  -- Label-dependent observation features f(y_i, x, i) -------------------
    # for a given label s, get obs. feats in parallel for more efficiency
    for s in label_set:
        s_position_mask = (y == s)
        if return_dframe:
            f_x.set(
                f_xy.select(s_position_mask, ND, s),
                s_position_mask, ND, s
                )
            # y_{i-1}=ND means ignoring y_{i-1}
        else:
            feat_indices = f_xy.iselect(y1=ND, y2=s)
            f_x[np.ix_(s_position_mask, feat_indices)] = \
                f_xy.select(s_position_mask, ND, s, arr_out=True)

    #  -- Transition features f(y_{i-1}, y_i, x, i) ---------------------------
    transitions = y[:-1].astype(np.complex)
    # complex type used to represent y_{i-1} in real part and y_i in imaginary
    transitions.imag = y[1:]
    transition_set = np.unique(transitions)
    # for a given label-pair t, get the feats in parallel for efficiency
    for t in transition_set:
        t_position_mask = np.r_[False, transitions == t]
        # False added at first position since first pair in `transitions`
        # is (y0, y1)
        if return_dframe:
            f_x.set(
                f_xy.select(t_position_mask, int(t.real), int(t.imag)),
                t_position_mask, int(t.real), int(t.imag)
                )
        else:
            feat_indices = f_xy.iselect(y1=int(t.real), y2=int(t.imag))
            f_x[np.ix_(t_position_mask, feat_indices)] = \
                f_xy.select(t_position_mask, int(t.real), int(t.imag))

    if not return_dframe:
        if with_f_x_sum:
            f_x_sum = f_x.sum(axis=0)
            return f_x, f_x_sum
        else:
            return f_x


def _potential_values(psi_xy, y, return_dframe=True):
    """
    Computes potential values for given y sequence.

    Parameters
    ----------
    psi_xy : LinearChainData
        A flexcrf data frame with a multiindex indicating feature values w.r.t
        all possible (y_{i-1}, y_i, x) configs, following flexcrf conventions.

    y : ndarray, shape (n_obs,)
        Defines the sequence of outputs.

    return_dframe: bool, default=True
        Whether to return an ndarray (default) or a pandas dataframe.

    Returns
    -------
    psi_x : ndarray or pandas dataframe, shape (n_obs, n_potential_fun).
         User-defined potential values for given y sequence.

    """

    return _feat_fun_values(psi_xy, y, return_dframe=return_dframe,
                            with_f_x_sum=False)


def _compute_all_potentials(f_xy=None, theta=None, psi_xy=None,
                            log_version=True):
    """
    Computes local potentials, (or log-potentials) `m_xy` based on feature
    function `f_xy` and/or user-defined potentials `psi_xy`.

    Parameters
    ----------
    f_xy : LinearChainData
        A flexcrf data frame with a multiindex indicating feature values w.r.t
        all possible (y_{i-1}, y_i, x) configs, following flexcrf conventions.

    theta : ndarray, shape (n_feat_fun,)
        Parameters of the exponential potentials.

    psi_xy : LinearChainData
        A flexcrf data frame with a multiindex indicating user-defined
        potentials w.r.t all possible (y_{i-1}, y_i, x) configs, following
        flexcrf conventions.

    log_version : boolean, default=True
        A flag indicating wether log-potentials should be computed.

    Returns
    -------
    m_xy : ndarray, shape (n_obs, n_labels, n_labels)
        Values of potentials (or log-potentials) $M_i(y_{i-1}, y_i, x)$
        computed based on feature functions f_xy and/or user-defined potentials
        `psi_xy`. At t=0, m_xy[0, 0, :] contains values of $M_1(y_0, y_1)$ with
        $y_0$ the fixed initial state.

    f_m_xy : ndarray, shape (n_feat_fun, n_obs, n_labels, n_labels)
        Values of products (or sums) of feat fun values and potentials
        (or log-potentials, respectively), i.e.:
        $f_k(y_{i-1}, y_i, x) M_i(y_{i-1}, y_i, x)$.

    TODO: use sparse ndarrays for f_m_xy and m_xy
    """

    if f_xy is not None:
        ND = f_xy.ND
        #label_set = f_xy.label_set
        n_labels = len(f_xy.label_set)
        n_obs = f_xy.n_obs
        n_feat_fun = theta.shape[0]

        f_m_xy = np.zeros((n_feat_fun, n_obs, n_labels, n_labels))
        # align theta coefficients with feat funs in f_xy according to feat_ind
        theta = theta[f_xy.feat_indices()]
        theta_f_xy = LinearChainData(f_xy.data.columns,
                                     data=theta*f_xy.data.values)
        y1_vals = theta_f_xy.y1_values()
        y2_vals = theta_f_xy.y2_values()
    elif psi_xy is not None:
        ND = psi_xy.ND
        #label_set = psi_xy.label_set
        n_labels = len(psi_xy.label_set)
        n_obs = psi_xy.n_obs
        n_feat_fun = psi_xy.n_feat_fun
        y1_vals = psi_xy.y1_values()
        y2_vals = psi_xy.y2_values()

        f_m_xy = None

    else:
        raise ValueError("Found no features nor potentials")

    m_xy = np.zeros((n_obs, n_labels, n_labels))

    # TODO: paralellise the following (using a single loop with it.product()
    # instead of a double loop makes it efficient...)
    for i, j in it.product(range(len(y1_vals)), range(len(y2_vals))):

        y1, y2 = y1_vals[i], y2_vals[j]

        if f_xy is not None:
            # -- Observation functions ---
            m_xy[:, i, j] += np.sum(
                theta_f_xy.select(y1=ND, y2=y2, arr_out=True), axis=1)

            o_feat_indices = theta_f_xy.iselect(y1=ND, y2=y2)
            # inefficent without parallisation, since called many times with
            # the same y2.

            # -- Transition functions ---
            m_xy[:, i, j] += np.sum(
                theta_f_xy.select(y1=y1, y2=y2, arr_out=True), axis=1)
            t_feat_indices = theta_f_xy.iselect(y1=y1, y2=y2)

            # -- Product of f_xy and m_xy (in log), needed for gradient comp
            f_m_xy[o_feat_indices, :, i, j] = m_xy[:, i, j] + np.log(
                f_xy.select(y1=ND, y2=y2, arr_out=True).T)
            f_m_xy[t_feat_indices, :, i, j] = m_xy[:, i, j] + np.log(
                f_xy.select(y1=y1, y2=y2, arr_out=True).T)

        if psi_xy is not None:
            # -- Observation functions ---
            m_xy[:, i, j] += np.sum(np.log(
                psi_xy.select(y1=ND, y2=y2, arr_out=True), axis=1))
            # -- Transition functions ---
            m_xy[:, i, j] += np.sum(np.log(
                psi_xy.select(y1=y1, y2=y2, arr_out=True), axis=1))

    # m_xy[0, 0, :] already contains values of $M_1(y_0, y_1)$ at t=0;
    # as done in first loop above. Now set the remaining rows to 0:
    m_xy[0, 1:, :] = 0

    if not log_version:
        m_xy = np.exp(m_xy)
        if f_xy is not None:
            f_m_xy = np.exp(f_m_xy)

    return m_xy, f_m_xy


def _forward_score(m_xy, n=None, log_version=True):
    """
    Computes forward score for input potentials m_xy upto position n.

    Parameters
    ----------
    m_xy : ndarray, shape (n_obs, n_labels, n_labels)
        Values of potentials (or log-potentials) $M_i(y_{i-1}, y_i, x)$
        computed based on feature functions f_xy and/or user-defined potentials
        `psi_xy`. At t=0, m_xy[0, 0, :] contains values of $M_1(y_0, y_1)$ with
        $y_0$ the fixed initial state.

    n : integer, default=None
        Time position up to which to compute forward score; if not specified
        (default) the score is computed for the whole sequence.

    log_version : boolean, default=True
        A flag indicating wether log-alpha scores should be computed to avoid
        numerical overflow.

    Returns
    -------
    alpha : ndarray, shape (n+1, n_labels)
        Forward score values, i.e. $\alpha_m(y_m)$ or $\log(\alpha_m(y_m))$,
        for $m \in \{0, ..., n\}$ and $y_m$ in label set, with $\alpha_0=1$
        (or 0 with log-version).

    TODO: Cythonise this function for more efficiency.
    """

    if n is None:
        n = m_xy.shape[0]

    n_labels = m_xy.shape[2]
    alpha = np.zeros((n+1, n_labels))
    alpha[1, :] = m_xy[0, 0, :]

    if log_version:  # to avoid numerical overflow
        for m in xrange(2, n+1):
            alpha[m, :] = logsumexp(m_xy[m-1, :, :].T + alpha[m-1, :], axis=1)
    else:
        for m in xrange(2, n+1):
            alpha[m, :] = np.sum(m_xy[m-1, :, :].T * alpha[m-1, :], axis=1)
        alpha[0, :] = 1

    return alpha


def _backward_score(m_xy, n0=None, log_version=True):
    """
    Computes backward score for input potentials m_xy back to position n.

    Parameters
    ----------
    m_xy : ndarray, shape (n_obs, n_labels, n_labels)
        Values of potentials (or log-potentials) $M_i(y_{i-1}, y_i, x)$
        computed based on feature functions f_xy and/or user-defined potentials
        `psi_xy`. At t=0, m_xy[0, 0, :] contains values of $M_1(y_0, y_1)$ with
        $y_0$ the fixed initial state.

    n0 : integer, default=None
        Time position back to which to compute backward score; if not specified
        (default) the score is computed for the whole sequence, back to $n0=1$.

    log_version : boolean, default=True
        A flag indicating wether log-beta scores should be computed to avoid
        numerical overflow.

    Returns
    -------
    beta : ndarray, shape (n+1, n_labels)
        backward score value, i.e. $\log(\beta_m(y_m))$,
        for $m \in \{1, ..., n\}$. Also, if n0 = 1, $\log(Z(x, \theta))$ is
        computed and stored at beta[0, :].

    TODO: Cythonise this function for more efficiency.
    """

    if n0 is None:
        n0 = 1

    n, n_labels = m_xy.shape[0], m_xy.shape[2]
    n_obs = n-n0+1
    beta = np.empty((n_obs+1, n_labels))
    m_ = n-1
    if log_version:  # to avoid numerical overflow
        beta[n_obs, :] = 0
        for m in xrange(n_obs-1, n0-1, -1):
            m_ -= 1
            beta[m, :] = logsumexp(m_xy[m_+1, :, :] + beta[m+1, :], axis=1)
        if n0 == 1:
            # put $\log(\sum_{y_1} M_1(y_0, y_1)\times\beta_1(y_1))$,
            # i.e. $\log(Z(x, \theta))$, at beta[0, :]
            beta[0, :] = logsumexp(m_xy[0, 0, :] + beta[1, :])
    else:
        beta[n_obs, :] = 1
        for m in xrange(n_obs-1, n0, -1):
            m_ -= 1
            beta[m, :] = np.sum(m_xy[m_+1, :, :] * beta[m+1, :], axis=1)
        if n0 == 1:
            # add $\sum_{y_1} M_1(y_0, y_1)\times\beta_1(y_1)$,
            # i.e. $\log(Z(x, \theta))$, at beta[0, :]
            beta[0, :] = np.sum(m_xy[0, 0, :]*beta[1, :])
    return beta


def _partition_fun_value(alpha, beta=None, log_version=True, tolerance=1e-6):
    """
    Computes partition function $Z(x, \theta)$ value for current $x$ and
    $\theta$, or its log version, based on previously computed `alpha` scores.
    If 'beta' scores are also passed, use them to compute the same score and
    compare both versions raising warnings if they are different.

    Parameters
    ----------
    alpha : ndarray, shape (n+1, n_labels)
        Forward score values, i.e. $\alpha_m(y_m)$ or $\log(\alpha_m(y_m))$ if
        `log_version=True`, for $m \in \{0, ..., n\}$ and $y_m$ in label set.

    beta : ndarray, shape (n+1, n_labels)
        Backward score values, i.e. $\beta_m(y_m)$ or $\log(\beta_m(y_m))$ if
        `log_version=True`, for $m \in \{1, ..., n\}$ and $y_m$ in label set.

    log_version : boolean, default=True
        A flag indicating wether $log(Z(x, \theta))$ should be computed to
        avoid numerical overflow.

    tolerance : float
        To be used to compare forward and backward versions of the computation.

    Returns
    -------
    z_x : float
        Partition function $Z(x, \theta)$ value for current $x$ and
        $\theta$ or its log if `log_version=True`
    """

    if log_version:
        z_x = logsumexp(alpha[-1:, :], axis=1)[0]
        # a_max = np.max(alpha[-1:, :])
        # z_x = a_max + np.log(np.sum(np.exp(alpha[-1:, :]-a_max), axis=1))[0]
    else:
        z_x = np.sum(alpha[-1:, :], axis=1)

    if beta is not None:
        z_x_ = beta[0, 0]  # this is already computed in _backward_score()
        if (log_version and abs(z_x-z_x_) > tolerance) or \
                (not log_version and abs(z_x/z_x_-1) > tolerance):
            warnings.warn("Partition function value obtained with forward \
                and backward score are different")
        return z_x, z_x_
    else:
        return z_x


def _posterior_score(f_x=None, theta=None, psi_x=None, z_x=None,
                     log_version=True):
        """
        Computes posterior score $p(y|x,\theta)$ for a particular $y$ output
        configuration.

        Given precomputed potentials `psi_x` and/or
        model parameters `theta` and feature function values `f_x` for the
        actual y configuration, compute the posterior score $p(y|x,\theta)$.

        Parameters
        ----------
        f_x : ndarray, shape (n_obs, n_feat_fun)
            Defining output-specific feature function values as output
            by `_feat_fun_values()`.

        theta : ndarray, shape (n_feat_fun,)
            Parameters of the exponential potentials.

        psi_x : ndarray, shape (n_obs, n_potential_fun)
            Values of user-defined fixed potentials for considered $y$
            values, as output by `_potential_values()`.

        z_x : float, default=None
            The log of the parition function value to be used to normalise the
            score so as to compute the posterior probability. If None (default)
            the score will not be normalised (useful?).

        Returns
        -------
        p_score : float
            Posterior score, or log-score if `log_version=True`, which is
            either the posterior probability or total potential value,
            depending on `posterior_prob` value.
        """

        if (f_x, theta, psi_x) == (None, None, None):
            raise ValueError("Got no features nor potentials: nothing to "
                             "compute here.")
        if f_x is None:
            warnings.warn("Got no features: only potentials will be used to "
                          "compute posterior score.")
        if theta is None:
            warnings.warn("Got no parameters: only potentials will be used to "
                          "compute posterior score.")

        if log_version:
            if (f_x, theta) == (None, None):
                p_score = np.sum(np.log(psi_x.ravel()))
            else:
                p_score = theta.dot(f_x.sum(axis=0))
                if psi_x is not None:
                    p_score += np.sum(np.log(psi_x.ravel()))
            if z_x is None:
                return p_score
            else:
                return p_score-z_x
        else:
            if (f_x, theta) == (None, None):
                p_score = np.prod(psi_x.ravel())
            else:
                p_score = np.exp(theta.dot(f_x.sum(axis=0)))
                if psi_x is not None:
                    p_score *= np.prod(psi_x.ravel())
            if z_x is None:
                return p_score
            else:
                return p_score/np.exp(z_x)
