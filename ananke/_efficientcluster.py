import os
import sys
from random import randrange
from multiprocessing import Pool
from functools import partial
from bisect import insort

import h5py as h5
import numpy as np
from fastdtw import fastdtw

from ._database import TimeSeriesData
from ._cluster import normalize_simple, zscore

# Creates, updates, and retrieves distances from the disk
class OutOfCoreDistances:
    def __init__(self, h5_path, nrows):
        try:
            os.remove(h5_path)
            print("Found existing ananke_distances.h5, deleting")
        except OSError:
            pass
       # Create the H5 file with the correct dimensions
        self.h5_file = h5.File(h5_path)
        self.h5_file.create_dataset("distances", (nrows, nrows),
                                    dtype=np.float16, fillvalue=100)

    def set_distance(self, i, j, distance):
       # Insert the distance into the h5 file
       # Possibly save a bunch up and write in a batch?
       # Only fill the upper triangle
       min_index = min(i,j)
       max_index = max(i,j)
       self.h5_file["distances"][min_index, max_index] = distance

    def set_distances(self, i, distances):
       self.h5_file["distances"][i, i+1:] = distances.tolist()

    def get_distance(self, i, j):
        # Get the position from the disk
       min_index = min(i,j)
       max_index = max(i,j)
       return self.h5_file["distances"][min_index, max_index]

# Class that holds the information from a time series for DBSCAN clustering
class TimeSeries:
    def __init__(self, seqid, ts_data, abundance, index, time_point_slopes):
        self.name = seqid
        self.abundance = abundance
        self.data = ts_data
        self.slopes = (ts_data[1:] - ts_data[0:-1]) / time_point_slopes
        # Set to the noise bin, -1, by default.
        # Overwrite when proven otherwise.
        self.cluster_id = -1
        # Initialize as not a core point
        # Not core and self.n_neighbourhood > 0 implies a border point
        self.core = False
        # Track where we've been so we don't walk in circles
        self.visited = False
        # TimeSeries' integer index in the timeseries_list
        # Since we insertion sort, we don't know where this is until it's done
        # So this gets filled in manually later
        # It's used to make sure we only check TimeSeries against less abundant
        # neighbours, making sure we don't double our work
        self.index = None
        # Non-abundance sorted order (i.e., original order in H5 file)
        self.original_index = index

    def __eq__(self, other):
        return self.name == other.name

    def __neq__(self, other):
        return not(self.__eq__)

    def __lt__(self, other):
        return self.abundance < other.abundance

    def __le__(self, other):
        return self.abundance <= other.abundance

    def __gt__(self, other):
        return self.abundance > other.abundance

    def __ge__(self, other):
        return self.abundance >= other.abundance

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return "Name: %s, Abundance: %d, Core: %r" % (self.name,
                                                      self.abundance,
                                                      self.core)

    def is_core(self):
        return self.core

    def reset(self):
        # Maintain the distances, but reset the other values
        self.n_neighbourhood = 0
        self.cluster_id = -1
        self.core = False
        self.visited = False

# Computes the distance and then adds it to both TimeSeries structures
# time_series1 and time_series2 are TimeSeries structures
def compute_distance(time_series1, time_series2, 
                     distance_measure='sts'):
    if distance_measure == 'sts':
        # Take the difference of the slopes, square and sum
        # The square distance is OK here, we don't need to sqrt it
        distance = np.sum(np.subtract(time_series1.slopes, 
                                      time_series2.slopes)**2)
    elif distance_measure == 'dtw':
        distance, path = fastdtw(time_series1.data, time_series2.data)
    return distance


def sample_distances(timeseries_list, distance_measure):
    # Sample some distances to guess the max epsilon, used for scaling
    # In the RAM-hungry version, we know the max up-front and can scale
    # perfectly, but in this case we have to guess and then check
    nrows = len(timeseries_list)
    max_dist = 0
    # Sample twice as many distances as we have unique genes
    # TODO: Validate that this is a good enough amount of sampling
    n_iters = int(2*nrows)
    for i in range(0, n_iters):
        x = randrange(0, nrows, 1)
        y = randrange(0, nrows, 1)
        ts1 = timeseries_list[x]
        ts2 = timeseries_list[y]
        distance = compute_distance(ts1, ts2, 
                                    distance_measure = distance_measure)
        if distance > max_dist:
            max_dist = distance

    print("After %d samples of the distances, the max distance was %f" \
          % (n_iters, max_dist))

    return max_dist

# Cluster around a given seed, finding its neighbours, its neighbours neighbours,
# and so on, until no more new neighbours are found. This is the core clustering
# function, and is a re-organization of the DBSCAN algorithm intended to be
# more memory efficient
def cluster_around_seed(ts_seed, timeseries_list, timeseriesdb, 
                        epsilon, distance_measure, 
                        min_pts, cluster_id, num_threads):
    cluster_attrs = timeseriesdb.h5_table["genes/clusters"].attrs
    max_dist = cluster_attrs["epsilon_factor"]

    # Store found neighbours within epsilon
    queue = set() 
    # Initialize the neighbour compute queue with the abundant seed 
    queue.add(ts_seed)

    n_neighbours = 0

    # Grab a pool of threads
    p = Pool(num_threads)

    while queue:
        ts1 = queue.pop()
        neighbourhood = []
        if ts1.visited:
            continue
        else:
            ts1.visited = True
        # If a time series has been "visited", that means it has been
        # checked against all other time series, and since the distances
        # are symmetric, we can exclude them here to avoid double computing
        timeseries_subset = [x for x in timeseries_list if not x.visited]
        p_distance_func = partial(compute_distance, 
                                  time_series2 = ts1, 
                                  distance_measure = distance_measure)
        results = p.map(p_distance_func, timeseries_subset)
        for ts2, distance in zip(timeseries_subset, results):
            distance = distance / max_dist
            if distance <= epsilon:
                neighbourhood.append(ts2)
        if len(neighbourhood) >= min_pts:
            for neighbour in neighbourhood:
                if (not neighbour.visited) & (neighbour not in queue):
                    queue.add(neighbour)
                    n_neighbours += 1
                    neighbour.cluster_id = cluster_id
                    neighbour.queued = True
                
        # Update status and reset appropriate counters
        print("Found %d neighbours" %
              (n_neighbours, ), end='\r')
    indices = [x.original_index for x in [ts1] + neighbourhood]
    # Cluster is done, so write to the file
    timeseriesdb.insert_cluster(indices, cluster_id, epsilon)

def load_data(timeseriesdata_path):
    print("Loading time-series database file")
    timeseriesdb = TimeSeriesData(timeseriesdata_path)

    print("Importing time-series matrix")
    matrix = timeseriesdb.get_sparse_matrix()
    time_points = timeseriesdb.get_array_by_chunks("samples/time")
    time_point_slopes = time_points[1:] - time_points[0:-1]
    names = timeseriesdb.get_array_by_chunks("genes/sequenceids")
    mask = timeseriesdb.get_mask()
    nrows = matrix.shape[0]
    abundances = matrix.sum(1)
    # Normalize by simple scaling
    matrix = normalize_simple(matrix, mask)

    if nrows <= 1:
        raise ValueError("Time-series matrix contains no information. " \
                         "Was all of your data filtered out?")

    return timeseriesdb, matrix, nrows, abundances, time_point_slopes


   

def auto_cluster(timeseriesdata_path, min_pts = 2,
                      distance_measure = 'sts', param_min = 0.01,
                      param_max = 0.6, param_step = 0.01, n_precompute = 100,
                      num_threads = 1, store_ooc = False):

    ######## Setup ########
    timeseriesdb, matrix, nrows, 
    abundances, time_point_slopes = load_data(timeseriesdata_path)


    print("Initializing TimeSeries objects")
    # Start with largest epsilon and decrease
    epsilon = param_max
    timeseries_list = []
    # Initialize all time-series objects from the data matrix
    for i in range(0, nrows):
        # Make a new TimeSeries object
        ts = TimeSeries(names[i],
                        np.squeeze(np.asarray(matrix[i,].T)),
                        abundances[i],
                        i,
                        time_point_slopes)
        insort(timeseries_list, ts)
        print("%d time series initialized and sorted" % (i,), end='\r')
    print("\nRemoving original matrix from memory")
    # Remove the matrix from memory, since all the data is stored elsewhere
    del matrix

    # Set the insertion-sorted list indices somewhere that we can retrieve them
    # in constant time. Used for eliminating double-distance-calculations,
    # since the distance is symmetric (dist(A,B)=dist(B,A))
    for i in range(0, len(timeseries_list)):
        timeseries_list[i].index = i

    print("Computing STS distances and clustering neighbourhoods for " \
          "epsilon=%f" % (epsilon,))

    #ooc_distances = OutOfCoreDistances("ananke_distances.h5", nrows)

    max_dist = sample_distances(timeseries_list, distance_measure)

    # Store the max_dist as epsilon_factor, because this is an estimate
    # and affects the scaling of epsilon to a domain of [0,1] and therefore
    # affects clustering consistency, so if we need to pick clustering back up
    # we must store this value to re-normalize against
    cluster_attrs = timeseriesdb.h5_table["genes/clusters"].attrs
    cluster_attrs.create("epsilon_factor", max_dist)
    cluster_attrs.create("param_min", param_min)
    cluster_attrs.create("param_max", param_max)
    cluster_attrs.create("param_step", param_step)

    # Pre-processing is complete, onto the distance calculations

    ######## CLUSTERING ########
    n_clusts = 0
    n_processed = 0

    # Take the top n_precompute most abundant sequences
    abundant_sequences = timeseries_list[-n_precompute:]

    # While there are unclustered sequences in our list
    while abundant_sequences:
        ts = abundant_sequences.pop()
        try:
            while ts.visited:
                ts = abundant_sequences.pop()
        except IndexError:
            break
        n_clusts += 1
        print("\nComputing cluster #%s" % (n_clusts,))
        cluster_around_seed(ts, timeseries_list, timeseriesdb,
                            epsilon, distance_measure,
                            min_pts, n_clusts, num_threads)
    print("\nPre-computation of most abundant sequences complete")
