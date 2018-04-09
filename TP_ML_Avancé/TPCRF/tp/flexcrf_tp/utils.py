import numpy as np


def logsumexp(arr, axis=0):
    """
    Computes logsumexp of arr, along the chosen axis, assuming arr is in
    the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.utils.extmath import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """

    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out

if __name__ == '__main__':
    bb = np.random.rand(100, 50)
    cc_ = np.log(np.sum(np.exp(bb), axis=1))
    cc = logsumexp(bb, axis=1)
    print(np.count_nonzero(abs(cc - cc_)>1e-10))

    cc_ = np.log(np.sum(np.exp(bb), axis=0))
    cc = logsumexp(bb, axis=0)
    print(np.count_nonzero(abs(cc - cc_)>1e-10))

    bb = np.random.rand(100, 50, 20, 10)
    cc_ = np.log(np.sum(np.exp(bb), axis=3))
    cc = logsumexp(bb, axis=3)
    print(np.count_nonzero(abs(cc - cc_)>1e-10))
