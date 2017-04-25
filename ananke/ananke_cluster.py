import h5py as h5
import numpy as np
from scipy.sparse import vstack, coo_matrix
import sys
from math import sqrt
from sklearn.cluster import DBSCAN
import argparse
import multiprocessing
from functools import partial

from .ananke_database import TimeSeriesData

#Input:
#  - Tab-separated table file with sequence hash as rownames, time points as column names, sequence abundances as entries
#  - Absolute sequence abundance minimum threshold
#  - Number of threads
#  - Possible pre-computed distance matrix

#Output:
#  - Cluster label file with clustering parameter as column names, hash as row names, temporal cluster number as entries
#  - Distance matrix (for fast reclustering)


#  Calculate the slopes, mx
def calculate_slopes(matrix, time_points, mask):
    time_points = np.array(time_points)
    border = []
    for i in range(len(mask)-1):
        border.append(mask[i] == mask[i+1])
    border = np.array(border)
    time_difference = time_points[1:] - time_points[0:len(time_points)-1]
    time_difference = time_difference[border]
    if (min(time_difference) <= 0):
        raise ValueError("Minimum time difference is less than or equal to zero (may be caused by two consecutive samples with identical time points)")
    slope_matrix = matrix[:, 1:]-matrix[:, 0:matrix.shape[1]-1]
    slope_matrix = slope_matrix[:,border]
    slope_matrix = slope_matrix / time_difference
    return slope_matrix

#  Handles multithreading of STS distance matrix calculation
def generate_STS_distance_matrix(slope_matrix, nthreads=4):
    sts_dist_matrix = np.zeros(shape=(slope_matrix.shape[0],slope_matrix.shape[0]),dtype='float64')
    nrows = slope_matrix.shape[0]
    p = multiprocessing.Pool(nthreads)
    partial_sts_matrix_generator = partial(sts_matrix_generator, slope_matrix = slope_matrix)
    print("Beginning parallel calculations on %d threads"%nthreads)
    count = 1
    for result in p.imap_unordered(partial_sts_matrix_generator, range(0, nrows-1), 1000):
        ind = result[0]
        dists = result[1].flatten()
        sts_dist_matrix[ind,ind:] = dists
        sts_dist_matrix[ind:,ind] = dists
        count += 1
        sys.stdout.write("\r%d/%d" % (count, nrows))
        sys.stdout.flush()
    p.close()
    p.join()
    return sts_dist_matrix

#  Calculates short time-series distance
#  somewhat efficiently
def sts_matrix_generator(ind, slope_matrix):
    mx = slope_matrix[ind, :]
    mv = slope_matrix[ind:, :]
    mx_rep = np.vstack((mx,)*mv.shape[0])
    diff = mx_rep - mv
    diff = np.square(diff)
    sts_squared = diff.sum(axis=1)
    dists = np.sqrt(sts_squared)
    return (ind, dists)

#  DBSCAN from scikit learn     
def cluster_dbscan(dist_matrix, eps=1, distance_measure="sts"):
    if (distance_measure == "sts"):
        dbs = DBSCAN(eps=eps, metric='precomputed', min_samples=2)
    else:
        dbs = DBSCAN(eps=eps, metric=distance_measure, min_samples=2)
    cluster_labels = dbs.fit_predict(dist_matrix)
    return cluster_labels

def zscore(x):
    mean = np.mean(x)
    sd = np.std(x)
    if sd == 0:
        z = np.zeros_like(x)
    else:
        z = (x-mean)/sd
    return z

#  Main method
def run_cluster(timeseriesdata_path, num_cores, param_min=0.01, param_max=1000, param_step=0.01, distance_measure="sts"):
    print("Loading time-series database file")
    timeseriesdb = TimeSeriesData(timeseriesdata_path)
    print("Importing time-series matrix")
    matrix = timeseriesdb.get_sparse_matrix()
    time_points = timeseriesdb.get_time_points()
    mask = timeseriesdb.get_mask()
    matrix = matrix.todense()
    nrows = matrix.shape[0]
    if nrows <= 1:
        raise ValueError("Time-series matrix contains no information. Was all of your data filtered out?")
    #Normalize the matrix for sequence depth then into Z-scores
    print("Normalizing matrix")
    matrix = matrix/matrix.sum(0)
    #Normalizing is an issue if you have a column that was completely filtered
    #Set these columns back to 0
    matrix[np.invert(np.isfinite(matrix))] = 0
    #Standardize to Z-scores
    norm_matrix = matrix
    for mask_val in np.unique(mask):
        y = norm_matrix[:,np.where(mask==mask_val)[0]]
        y = np.apply_along_axis(zscore,1,y)
        norm_matrix[:,np.where(mask==mask_val)[0]] = y
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
    #TODO: Make it so that it doesn't record until the first param where nclusters >1
    parameter_range = np.arange(param_min, param_max, param_step)
    actual_parameters = []
    cluster_label_matrix = np.empty(shape=(nrows,len(parameter_range)), dtype=int)
    for ind, eps in enumerate(parameter_range):
        actual_parameters.append(eps)
        if (distance_measure == "sts"):
            cluster_labels = cluster_dbscan(sts_dist_matrix, eps)
        else:
            cluster_labels = cluster_dbscan(norm_matrix, eps, distance_measure)
        nclusters = len(list(np.unique(cluster_labels)))
        cluster_label_matrix[:,ind] = cluster_labels
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
    timeseriesdb.h5_table["genes/clusters"].resize((nrows,len(actual_parameters)))
    for i in range(0,cluster_label_matrix.shape[0]):
        timeseriesdb.h5_table["genes/clusters"][i,:] = cluster_label_matrix[i,0:len(actual_parameters)]
    timeseriesdb.h5_table["genes/clusters"].attrs.create("param_min", param_min)
    timeseriesdb.h5_table["genes/clusters"].attrs.create("param_max", param_max)
    timeseriesdb.h5_table["genes/clusters"].attrs.create("param_step", param_step)
