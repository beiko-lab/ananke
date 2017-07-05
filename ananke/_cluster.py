import sys
import argparse
import multiprocessing
from functools import partial
from math import sqrt

import h5py as h5
import numpy as np
from scipy.sparse import vstack, coo_matrix
from scipy.stats.mstats import gmean
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

def normalize_simple(matrix, mask):
    """Normalizes a matrix by columns, and then by rows. With multiple
    time-series, the data are normalized to the within-series total, not the
    entire data set total.

    Parameters
    ----------
    matrix: np.matrix
        Time-series matrix of abundance counts. Rows are sequences, columns
        are samples/time-points.
    mask: list or np.array
        List of objects with length matching the number of timepoints, where
        unique values delineate multiple time-series. If there is only one
        time-series in the data set, it's a list of identical objects.

    Returns
    -------
    normal_matrix: np.matrix
        Matrix where the columns (within-sample) have been converted to 
        proportions, then the rows are normalized to sum to 1.
    """
    normal_matrix = matrix / matrix.sum(0)
    normal_matrix[np.invert(np.isfinite(normal_matrix))] = 0
    for mask_val in np.unique(mask):
        y = normal_matrix[:, np.where(mask == mask_val)[0]]
        y = np.apply_along_axis(zscore, 1, y)
        normal_matrix[:, np.where(mask == mask_val)[0]] = y
        del y
    return normal_matrix

def normalize_clr(matrix, delta = 0.65, threshold = 0.5):
    """Normalizes a matrix by centre log ratio transform with zeros imputed
    by the count zero multiplicative method from the zCompositions package
    by Javier Palarea-Albaladejo and Josep Antoni Martin-Fernandez. Uses two
    parameters, delta and threshold, identically to the zCompositions
    implementation. This scheme is the same as used by the CoDaSeq R package.

    Parameters
    ----------
    matrix: np.matrix
        Time-series matrix of abundance counts. Rows are sequences, columns
        are samples/time-points.
    delta: float
        Fraction of the upper threshold used to impute zeros (default=0.65)
    threshold: float
        For a vector of counts, factor applied to the quotient 1 over the 
        number of trials (sum of the counts) used to produce an upper limit 
        for replacing zero counts by the CZM method (default=0.5).

    Returns
    -------
    normal_matrix: np.matrix
        Matrix where the columns (within-sample) have been converted to centre 
        log ratio transformed values to control for within-sample
        compositionality, and the rows are brought onto the same scale by
        computing the Z-score of each element in the time-series.
    """

    #Zero imputation with count zero multiplicative
    # This algorithm was originally written with samples as rows
    # so we need the transpose
    X = matrix.T
    #N = nsamples
    N = X.shape[0]
    #D = nsequences
    D = X.shape[1]
    #Column means without 0's included
    n = np.apply_along_axis(lambda x: x[np.nonzero(x)].sum(), 1, X)
    #Replacement matrix
    replace = delta*np.ones((D,N))*(threshold/n)
    replace = replace.T
    #Normalize by columns, using only nonzero values
    X2 = np.apply_along_axis(lambda x: x/(x[np.nonzero(x)].sum()), 1, X)
    colmins = np.apply_along_axis(lambda x: x[np.nonzero(x)].min(), 0, X2)
    corrected = 0
    for idx, row in enumerate(X2):
        zero_indices = np.where(row == 0)[1]
        nonzero_indices = np.where(row != 0)[1]
        X2[idx, zero_indices] = replace[idx, zero_indices]
        over_min = np.where(X2[idx, zero_indices] > colmins[zero_indices])
        if len(over_min[0]) > 0:
            corrected += len(over_min[0])
            X2[idx, over_min[1]] = delta*colmins[over_min[1]]
        X2[idx, nonzero_indices] = (1-X2[idx, zero_indices].sum()) * \
                                   X2[idx, nonzero_indices]
    normal_matrix = X2.T
    # Do the CLR transform
    normal_matrix = normal_matrix/gmean(normal_matrix)
    # Normalize within time-series to remove scaling factors
    normal_matrix = np.apply_along_axis(zscore, 1, normal_matrix)
    return normal_matrix

#  Main method
def run_cluster(timeseriesdata_path, num_cores, distance_measure = "sts",
                param_min = 0.01, param_max = 1, param_step = 0.01,
                clr = False):
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
    clr: boolean
        Indicates whether to normalize with simple proportions (False, default)
        or centred log ratio with zero imputation by count zero multiplicative
        method (True).
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
    if clr:
        # Normalize using the centred log ratio after zero imputation with
        # count zero multiplicative (CZM) method
        normal_matrix = normalize_clr(matrix)
    else:
        #Normalize the matrix for sequence depth then into Z-scores
        print("Normalizing samples by simple division")
        normal_matrix = normalize_simple(matrix, mask)
    
    if (distance_measure == "sts"):
        print("Calculating slopes")
        slope_matrix = calculate_slopes(normal_matrix, time_points, mask)
        print("Generating STS distance matrix")
        sts_dist_matrix = generate_STS_distance_matrix(slope_matrix, num_cores)
        max_dist = sts_dist_matrix.max()
        sts_dist_matrix /= max_dist
        del slope_matrix
        del normal_matrix
    
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
            cluster_labels = cluster_dbscan(normal_matrix, 
                                            distance_measure, 
                                            eps)
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
