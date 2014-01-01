#!/usr/bin/env python
"""
Relative filter module

"""
__author__ = 'Alisue (lambdalisue@hashnote.net)'
import numpy as np


def relative(dataset, ori=0, column=1, fail_silently=True):
    """
    Convert dataset to relative value from the value of :attr:`ori`

    Parameters
    ----------
    dataset : list of numpy array list
        A list of numpy array list
    ori : integer or numpy array, optional
        A relative original data index or numpy array
    column : integer, optional
        An index of base column to calculate the relative value
    fail_silently : boolean
        If `True`, do not raise exception if no data exists

    Returns
    -------
    ndarray
        A list of numpy array list

    Examples
    --------
    >>> import numpy as np
    >>> from maidenhair.filters.relative import relative
    >>> dataset = []
    >>> dataset.append([np.array([0, 1, 2]), np.array([3, 4, 5])])
    >>> dataset.append([np.array([0, 1, 2]), np.array([3, 5, 7])])
    >>> dataset.append([np.array([0, 1, 2]), np.array([100, 103, 106])])
    >>> expected = [
    ...     [np.array([0, 1, 2]), np.array([0, 50, 100])],
    ...     [np.array([0, 1, 2]), np.array([0, 100, 200])],
    ...     [np.array([0, 1, 2]), np.array([4850, 5000, 5150])],
    ... ]
    >>> proceed = relative(dataset)
    >>> np.array_equal(proceed, expected)
    True

    """
    try:
        if isinstance(ori, int):
            # relative from the [ori]th array
            ori = dataset[ori][column]
        # calculate min/max difference
        if ori.ndim > 1 and len(ori[0]) > 1:
            ori = np.average(ori, axis=1)
        orimin = np.min(ori)
        orimax = np.max(ori)
        oridiff = orimax - orimin
        # convert
        for data in dataset:
            data[column] /= oridiff / 100.0
        # baseline
        baseline = ori[0]
        for data in dataset:
            data[column] -= baseline
        return dataset
    except IndexError, e:
        if fail_silently:
            # fail silently
            return dataset
        raise e

if __name__ == '__main__':
    import doctest; doctest.testmod()
