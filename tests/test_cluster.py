#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the methods in _cluster.py"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from shutil import copy

from ananke._database import TimeSeriesData
from ananke._cluster import (calculate_slopes, generate_STS_distance_matrix,
                             sts_matrix_generator, cluster_dbscan, zscore,
                             run_cluster)

#def calculate_slopes(matrix, time_points, mask)
def test_calculate_slopes_mask():
    mat = np.matrix([[1, 5, 7, 0, 12, 14],
                     [12, 5, 7, 18, 0, 0],
                     [6, 2, 9, 17, 0, 5]])
    tp = np.array([0,5,7,0,5,7])
    mask = np.array(['1', '1', '1', '2', '2', '2'])
    res = calculate_slopes(mat, tp, mask)
    true_res = np.matrix([[0.8, 1, 2.4, 1],
                          [-1.4, 1, -3.6, 0],
                          [-0.8, 3.5, -3.4, 2.5]])
    assert (res - true_res).sum() <= 10e-10

def test_calculate_slopes_no_mask():
    mat = np.matrix([[1, 5, 7, 0, 12, 14],
                     [12, 5, 7, 18, 0, 0],
                     [6, 2, 9, 17, 0, 5]])
    tp = np.array([0, 5, 7, 11, 15, 19])
    mask = np.array(['1', '1', '1', '1', '1', '1'])
    res = calculate_slopes(mat, tp, mask)
    true_res = np.matrix([[0.8, 1, -1.75, 3, 0.5],
                          [-1.4, 1, 2.75, -4.5, 0],
                          [-0.8, 3.5, 2, -4.25, 1.25]])
    assert (res - true_res).sum() <= 10e-10

#def generate_STS_distance_matrix(slope_matrix, nthreads)
#def test_generate_STS_distance_matrix():
#    pass

#def sts_matrix_generator(ind, slope_matrix)
#def test_sts_matrix_generator():
#    pass

#def cluster_dbscan(dist_matrix, eps=1, distance_measure="sts")
#def test_cluster_dbscan():
#    pass

#def zscore(x)
def test_zscore():
    data = [15.0, 12.0, 17.0, 67.0, 0.0, 14.0, 75.0]
    z = zscore(data)
    true_z = [-0.4953833 , -0.60488908, -0.42237944,  1.40271692, 
              -1.04291221, -0.53188522,  1.69473233]
    # The true values here are rounded, the result from zscore is more
    # precise, so we check that the sum of the elementwise differences is
    # arbitrarily small
    assert (z - np.array(true_z)).sum() < 10e-10

def test_zscore_zero_std():
    data = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    z = zscore(data)
    true_z = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert (z == np.array(true_z)).all()

#Main method
#def run_cluster
#def test_run_cluster():
#    pass
