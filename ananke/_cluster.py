import sys
import argparse
import multiprocessing
from functools import partial
from math import sqrt

import h5py as h5
import numpy as np
from scipy.sparse import vstack, coo_matrix
from sklearn.cluster import DBSCAN

from ._database import TimeSeriesData

#  Calculate the slopes, mx
def calculate_slopes(matrix, time_points, mask):
    """Calculates the slopes (first order difference) of the time-series 
    matrix. If there are multiple time-series in the matrix, the mask array
    defines the borders of these time-series. The inter-time-series slopes
    are discarded as they are meaningless to the downstream distance measures.

    Parameters
    ----------
    matrix: np.matrix
        The time-series matrix with samples/time-points as columns, sequences/
        time-series as rows, and sequence count as the entries.
    time_points: list or np.array
        The time-points. Should be the same length as matrix.shape[1].
    mask: list or np.array
        A list or arbitrary types where each unique value represents a
        time-series.

    Returns
    -------
    slope_matrix: np.matrix
        A matrix of size ngenes by nsamples - 1.
    """
    time_points = np.array(time_points)
    border = []
    for i in range(len(mask) - 1):
        border.append(mask[i] == mask[i + 1])
    border = np.array(border)
    time_difference = time_points[1:] - time_points[0:len(time_points) - 1]
    time_difference = time_difference[border]
    if (min(time_difference) <= 0):
        raise ValueError("Minimum time difference is less than or equal to" \
                         " zero (may be caused by two consecutive samples" \
                         " with identical time points)")
    slope_matrix = matrix[:, 1:] - matrix[:, 0:matrix.shape[1] - 1]
    slope_matrix = slope_matrix[:, border]
    slope_matrix = slope_matrix / time_difference
    return slope_matrix

#  Handles multithreading of STS distance matrix calculation
def generate_STS_distance_matrix(slope_matrix, nthreads=4):
    """Takes in the slope matrix and returns the distance matrix. Uses parallel
    processing.

    Parameters
    ----------
    slope_matrix: np.matrix
        Matrix of the time-series slopes, produced by calculate_slopes()
    nthreads: int
        Number of threads to use (default 4).

    Returns
    -------
    sts_dist_matrix: np.matrix
        Pair-wise STS distance matrix, size ngenes x ngenes.
    """
    sts_dist_matrix = np.zeros(shape = (slope_matrix.shape[0],
                                        slope_matrix.shape[0]),
                                        dtype='float64')
    nrows = slope_matrix.shape[0]
    p = multiprocessing.Pool(nthreads)
    partial_sts_matrix_generator = partial(sts_matrix_generator, 
                                           slope_matrix = slope_matrix)
    print("Beginning parallel calculations on %d threads" % nthreads)
    count = 1
    for result in p.imap_unordered(partial_sts_matrix_generator, 
                                   range(0, nrows - 1), 1000):
        ind = result[0]
        dists = result[1].flatten()
        sts_dist_matrix[ind, ind:] = dists
        sts_dist_matrix[ind:, ind] = dists
        count += 1
        sys.stdout.write("\r%d/%d" % (count, nrows))
        sys.stdout.flush()
    p.close()
    p.join()
    return sts_dist_matrix

#  Calculates short time-series distance
#  somewhat efficiently
def sts_matrix_generator(ind, slope_matrix):
    """Work-horse function. Computes the short time-series (STS) distance for
    an index, ind of the slope matrix.

    Parameters
    ----------
    ind: int
        The index of the slope matrix that is being computed.
    slope_matrix: np.matrix
        The slope matrix.

    Returns
    -------
        (ind, dists): ind is the index and dists is a np.matrix containing the
                      STS distances
    """
    mx = slope_matrix[ind, :]
    mv = slope_matrix[ind:, :]
    mx_rep = np.vstack((mx,)*mv.shape[0])
    diff = mx_rep - mv
    diff = np.square(diff)
    sts_squared = diff.sum(axis=1)
    dists = np.sqrt(sts_squared)
    return (ind, dists)

#  DBSCAN from scikit learn     
def cluster_dbscan(matrix, distance_measure="sts", eps=1):
    """Clusters the distance matrix for a given epsilon value, if distance
    measure is sts. Other distance measures are: [‘cityblock’, ‘cosine’, 
    ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’, ‘braycurtis’, ‘canberra’, 
    ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, 
    ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
    ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]

    Parameters
    ----------
    matrix: np.matrix
        The input matrix. If distance measure is sts, this should be the sts
        distance matrix. If other distance, this should be the time-series
        matrix of size ngenes x nsamples.
    distance_measure: str
        The distance measure, default is sts, short time-series distance.
        Any distance measure available in scikit-learn is available here.
        Note: multiple time-series is NOT supported for distances other than    
        "sts".

    Returns
    -------
    cluster_labels: list of int
        A list of size ngenes that defines cluster membership.
    """
    if (distance_measure == "sts"):
        dbs = DBSCAN(eps=eps, metric='precomputed', min_samples=2)
    else:
        dbs = DBSCAN(eps=eps, metric=distance_measure, min_samples=2)
    cluster_labels = dbs.fit_predict(matrix)
    return cluster_labels

def zscore(x):
    """Computes the Z-score of a vector x. Removes the mean and divides by the
    standard deviation. Has a failback if std is 0 to return all zeroes.

    Parameters
    ----------
    x: list of int
        Input time-series

    Returns
    -------
    z: list of float
        Z-score normalized time-series
    """
    mean = np.mean(x)
    sd = np.std(x)
    if sd == 0:
        z = np.zeros_like(x)
    else:
        z = (x - mean)/sd
    return z

#  Main method
def run_cluster(timeseriesdata_path, num_cores, distance_measure = "sts",
                param_min = 0.01, param_max = 1000, param_step = 0.01):
    """For a given Ananke data file, clusters using DBSCAN using the pairwise
    distance measure distance_measure (default short time-series, "sts").
    Clusters over a range of DBSCAN epsilon values, defined by param_min,
    param_max, and param_step. Clusters are written to the Ananke file.

    Parameters
    ----------
    timeseriesdata_path: str
        Path to the Ananke data file
    num_cores: int
        Number of threads for STS distance computation.
    distance_measure: str
        Distance measure to use. Can be "sts" for short-time series (default) 
        or any distance measure available in scikit-learn.
    param_min: float
        Minimum epsilon value.
    param_max: float
        Maximum epsilon value.
    param_step: float
        Step size between param_min and param_max.
    """
    print("Loading time-series database file")
    timeseriesdb = TimeSeriesData(timeseriesdata_path)
    print("Importing time-series matrix")
    matrix = timeseriesdb.get_sparse_matrix()
    time_points = timeseriesdb.get_array_by_chunks("samples/time")
    mask = timeseriesdb.get_mask()
    matrix = matrix.todense()
    nrows = matrix.shape[0]
    if nrows <= 1:
        raise ValueError("Time-series matrix contains no information. " \
                         "Was all of your data filtered out?")
    #Normalize the matrix for sequence depth then into Z-scores
    print("Normalizing matrix")
    matrix = matrix / matrix.sum(0)
    #Normalizing is an issue if you have a column that was completely filtered
    #Set these columns back to 0
    matrix[np.invert(np.isfinite(matrix))] = 0
    #Standardize to Z-scores
    norm_matrix = matrix
    for mask_val in np.unique(mask):
        y = norm_matrix[:, np.where(mask == mask_val)[0]]
        y = np.apply_along_axis(zscore, 1, y)
        norm_matrix[:, np.where(mask == mask_val)[0]] = y
        del y
    if (distance_measure == "sts"):
        print("Calculating slopes")
        slope_matrix = calculate_slopes(norm_matrix, time_points, mask)
        print("Generating STS distance matrix")
        sts_dist_matrix = generate_STS_distance_matrix(slope_matrix, num_cores)
        del slope_matrix
        del norm_matrix
    max_nclusters = 0    
    max_eps = 0
    prev_nclusters = 0
    break_out = False
    parameter_range = np.arange(param_min, param_max, param_step)
    actual_parameters = []
    cluster_label_matrix = np.empty(shape = (nrows, len(parameter_range)), 
                                             dtype=int)
    for ind, eps in enumerate(parameter_range):
        actual_parameters.append(eps)
        if (distance_measure == "sts"):
            cluster_labels = cluster_dbscan(sts_dist_matrix, "sts", eps)
        else:
            cluster_labels = cluster_dbscan(norm_matrix, distance_measure, eps)
        nclusters = len(list(np.unique(cluster_labels)))
        cluster_label_matrix[:, ind] = cluster_labels
        if nclusters > 1:
            break_out = True
        if (prev_nclusters != nclusters):
            print(str(eps) + ": " + str(nclusters))
        if (prev_nclusters == 1) & (nclusters == 1) & break_out:
          param_max = eps
          break
        else:
          prev_nclusters = nclusters
    #Print out the clusters with their sequence IDs
    timeseriesdb.h5_table["genes/clusters"].resize((nrows,
                                                    len(actual_parameters)))
    for i in range(0, cluster_label_matrix.shape[0]):
        encoded_labels = [ str(x).encode() for x \
                in cluster_label_matrix[i, 0:len(actual_parameters)] ]
        timeseriesdb.h5_table["genes/clusters"][i, :] = encoded_labels
    cluster_attrs = timeseriesdb.h5_table["genes/clusters"].attrs
    cluster_attrs.create("param_min", param_min)
    cluster_attrs.create("param_max", param_max)
    cluster_attrs.create("param_step", param_step)
