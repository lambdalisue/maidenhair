#!/usr/bin/env nosetests -v
# coding=utf-8
import numpy as np
from nose.tools import *
from mock import MagicMock as Mock

from maidenhair.classification.classify import classify_dataset
from maidenhair.classification.classify import default_classify_function


def create_test_dataset():
    X = np.array([0, 1, 2, 3, 4, 5])
    Y = np.array([10, 11, 12, 13, 14, 15])
    dataset = [
        ['./test/dataset_type01_00.000.000.txt', X, Y],
        ['./test/dataset_type01_00.001.000.txt', X, Y],
        ['./test/dataset_type01_00.002.000.txt', X, Y],
        ['./test/dataset_type01_01.000.000.txt', X, Y],
        ['./test/dataset_type01_01.001.000.txt', X, Y],
        ['./test/dataset_type01_01.002.000.txt', X, Y],
        ['./test/dataset_type02_00.000.000.txt', X, Y],
        ['./test/dataset_type02_00.001.000.txt', X, Y],
        ['./test/dataset_type02_00.002.000.txt', X, Y],
        ['./test/dataset_type02_01.000.000.txt', X, Y],
        ['./test/dataset_type02_01.001.000.txt', X, Y],
        ['./test/dataset_type02_01.002.000.txt', X, Y],
    ]
    return dataset


def test_classify_dataset_default():
    dataset = create_test_dataset()
    nameset = [x[0] for x in dataset]
    collection = classify_dataset(dataset,
                                  fn=default_classify_function)

    # return collection should be an instance of dictionary
    ok_(isinstance(collection, dict))

    # default fn will classify dataset with filename before last '_'
    # character
    eq_(collection.keys(), [
        './test/dataset_type01.txt',
        './test/dataset_type02.txt',
    ])

    # there should be 6 dataset
    for name, dataset in collection.items():
        eq_(len(dataset), 6)

    # there should be 3 columns in each data
    for name, dataset in collection.items():
        for i in range(0, 6):
            eq_(len(dataset[i]), 3)

    # there should be 6 rows in each axis
    for (name, dataset) in collection.items():
        for i in range(0, 6):
            eq_(len(dataset[i][1]), 6)
            eq_(len(dataset[i][2]), 6)

    # there should be 1 columns in each axis
    for (name, dataset) in collection.items():
        for i in range(0, 6):
            ok_(isinstance(dataset[i][1][0], (int, float, np.float64)))
            ok_(isinstance(dataset[i][2][0], (int, float, np.float64)))


def test_classify_dataset_fn_called():
    mock_function = Mock(return_value='')
    dataset = create_test_dataset()
    dataset = classify_dataset(dataset,
                               fn=mock_function)
    ok_(mock_function.called)
