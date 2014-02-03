#!/usr/bin/env nosetests -v
# coding=utf-8
"""
"""
__author__ = 'Alisue <lambdalisue@hashnote.net>'
import numpy as np
from nose.tools import *
from maidenhair.filters.baseline import baseline


def test_baseline_default():
    X = np.array([
        [ 99, 100, 101],
        [100, 101, 102],
        [101, 102, 103],
    ])
    Y = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    dataset = [
        [X, Y],
        [X, Y],
        [X, Y],
    ]

    eX = X
    eY = np.array([
        [0, 0, 0],
        [3, 3, 3],
        [6, 6, 6],
    ])
    expected = [
        [eX, eY],
        [eX, eY],
        [eX, eY],
    ]

    proceed = baseline(dataset)
    np.testing.assert_array_equal(proceed, expected)
