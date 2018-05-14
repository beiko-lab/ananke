import os
import sys
from random import randrange
from multiprocessing import Pool
from functools import partial
from bisect import insort

import h5py as h5
import numpy as np
from fastdtw import fastdtw
from sklearn.preprocessing import normalize

from ._database import TimeSeriesData
from ._cluster import normalize_simple, zscore
from ._ddtw import DDTW

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
    def __init__(self, seqid, ts_data, time_point_deltas, abundance=None, index=None):
        self.name = seqid
        self.abundance = abundance
        self.data = ts_data
        self.slopes = (ts_data[1:] - ts_data[0:-1]) / time_point_deltas
        # Set to the noise bin, -1, by default.
        # Overwrite when proven otherwise.
        self.cluster_id = -2
        # Initialize as not a core point
        # Not core and self.n_neighbourhood > 0 implies a border point
        self.core = False
        # Track where we've been so we don't walk in circles
        self.visited = False
        self.queued = False
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
        self.cluster_id = -2
        self.core = False
        self.visited = False
        self.queued = False

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
    elif distance_measure == 'ddtw':
        distance, path = DDTW(time_series1.data, time_series2.data)
        distance = distance[-1, -1]
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

class TimeSeriesIterator(object):
    def __init__(self, timeseriesdata, time_point_deltas):
        self.timeseriesdata = timeseriesdata
        self.time_point_deltas = time_point_deltas
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.timeseriesdata._h5t["data/timeseries/matrix"].shape[0]:
            raise StopIteration
        ts_data = self.timeseriesdata._h5t["data/timeseries/matrix"][self.i,:]
        i = self.i
        self.i += 1
        return TimeSeries(str(i), ts_data/sum(ts_data), self.time_point_deltas)

#Search through each of the timeseries in the list and find the neighbours within
# epsilon, for a given epsilon
# Return a list of (eps, cluster_id) pairs, denoting the nearest cluster at each epsilon level
#TODO: Figure out how to properly handle multiple series and replicates
#For now, it just takes a single timeseries at data/timeseries/*
#TODO: multithread it
def get_nearest_cluster(timeseriesdata, seed_data, distance_measure = 'sts', n_threads=1):
    cluster_attrs = timeseriesdata._h5t["timeseries/clusters"].attrs
    max_dist = cluster_attrs["epsilon_factor"]
    target = "data/timeseries/"
    time_points = timeseriesdata.get_array_by_chunks(target + "time")
    time_points = np.array([ int(x) for x in time_points ])
    time_point_deltas = time_points[1:] - time_points[0:-1]
    ts1 = TimeSeries("seed", np.array(seed_data)/sum(seed_data), time_point_deltas)
    p_dist_func = partial(compute_distance, 
                          time_series2 = ts1,
                          distance_measure = distance_measure)
    min_distance = 1000
    time_series_iterator = TimeSeriesIterator(timeseriesdata, time_point_deltas)
    p = Pool(n_threads)
    results = p.map(p_dist_func, time_series_iterator, chunksize = 1000)
    for i, distance in enumerate(results):
    #for i, timeseries in enumerate(time_series_iterator):
    #    distance = p_dist_func(timeseries)
        distance = distance / max_dist
        if distance < min_distance:
            min_distance = distance
            nearest_neighbour_index = i
    p.close() 
    return zip(timeseriesdata.get_epsilon_range(), 
            timeseriesdata._h5t["timeseries/clusters"]
                    [nearest_neighbour_index])

# Cluster around a given seed, finding its neighbours, its neighbours neighbours,
# and so on, until no more new neighbours are found. This is the core clustering
# function, and is a re-organization of the DBSCAN algorithm intended to be
# more memory efficient
def cluster_around_seed(ts_seed, timeseries_list, timeseriesdata, 
                        epsilon, distance_measure, 
                        min_pts, cluster_id, n_threads):
    
    cluster_attrs = timeseriesdata._h5t["timeseries/clusters"].attrs
    max_dist = cluster_attrs["epsilon_factor"]
    epsilon_index = timeseriesdata.get_epsilon_index(epsilon)
    epsilon_range = timeseriesdata.get_epsilon_range()
    # Store found neighbours within epsilon
    queue = set()
    # Initialize the neighbour compute queue with the abundant seed 
    queue.add(ts_seed)

    n_neighbours = 0
    cluster = []
    # Grab a pool of threads
    p = Pool(n_threads)

    while queue:
        ts1 = queue.pop()
        neighbourhood = []
        cluster.append(ts1)
        if ts1.visited:
            continue
        else:
            ts1.visited = True
        # If a time series has been "visited", that means it has been
        # checked against all other time series, and since the distances
        # are symmetric, we can exclude them here to avoid double computing
        timeseries_subset = [x for x in timeseries_list if not (x.visited | x.queued)]
        if epsilon_index < len(epsilon_range) - 1:
            # If a previous clustering exists, exploit it to limit this search
            upstairs_neighbour_ids = timeseriesdata.get_ts_neighbours(ts1.original_index, epsilon_range[epsilon_index + 1])
            timeseries_subset = [x for x in timeseries_subset if x.original_index in upstairs_neighbour_ids]
            
        p_distance_func = partial(compute_distance, 
                                  time_series2 = ts1, 
                                  distance_measure = distance_measure)
        results = p.map(p_distance_func, timeseries_subset, chunksize=1000)
        for ts2, distance in zip(timeseries_subset, results):
            distance = distance / max_dist
            if distance <= epsilon:
                neighbourhood.append(ts2)
        #If the point has more than min_pts neigh ours within epsilon
        #then it is a core point and its neighbourhood is in the cluster
        #Since we register every core point's neighbours, this is sufficient
        if len(neighbourhood) >= min_pts:
            for neighbour in neighbourhood:
                    queue.add(neighbour)
                    n_neighbours += 1
                    neighbour.cluster_id = cluster_id
                    cluster.append(neighbour)
                    neighbour.queued = True
               
        # Update status and reset appropriate counters
        print("Found %d neighbours" %
              (n_neighbours, ), end='\r')
        if len(cluster) == len(timeseries_list):
            break
        
    p.close()
    indices = [x.original_index for x in cluster]
    # Cluster is done, so write to the file
    timeseriesdata.insert_cluster(indices, cluster_id, epsilon)

def load_data_for_clustering(timeseriesdata, series=None):
    if series is None: 
        target = "data/timeseries/"
    else:
        target = "data/%s/" % (series,)
    print("Importing time-series matrix")
    matrix = timeseriesdata._h5t[target + "matrix"]
    time_points = timeseriesdata.get_array_by_chunks(target + "time")
    time_points = np.array([ int(x) for x in time_points ])
    time_point_deltas = time_points[1:] - time_points[0:-1]
    names = timeseriesdata.get_array_by_chunks("timeseries/ids")
    nrows = matrix.shape[0]
    ncols = matrix.shape[1]
    abundances = np.zeros(nrows)
    for i in range(ncols):
        abundances += matrix[:,i]
    # Normalize by simple scaling
    print("Normalizing with sklearn")
    #matrix = normalize(normalize(matrix, 'l1', 1, copy=True), 'l1', 0, copy=False)
    matrix = normalize(matrix, 'l1', 1, copy=False)
    if nrows <= 1:
        raise ValueError("Time-series matrix contains no information. " \
                         "Was all of your data filtered out?")

    return matrix, names, abundances, time_point_deltas

def auto_cluster(timeseriesdata, min_pts = 2,
                      distance_measure = 'sts', param_min = 0.01,
                      param_max = 0.6, param_step = 0.01, n_precompute = 100,
                      n_threads = 1, store_ooc = False):

    ######## Setup ########
    matrix, names, abundances, time_point_deltas = load_data_for_clustering(timeseriesdata)
    nrows = matrix.shape[0]
    print("Initializing TimeSeries objects")
    
    timeseries_list = []
    # Initialize all time-series objects from the data matrix
    for i in range(0, nrows):
        # Make a new TimeSeries object
        ts = TimeSeries(names[i],
                        np.squeeze(np.asarray(matrix[i,].T)),
                        time_point_deltas,
                        abundances[i],
                        i)
        insort(timeseries_list, ts)
        print("%d time series initialized and sorted" % (i+1,), end='\r')
    print("\nRemoving original matrix from memory")
    # Remove the matrix from memory, since all the data is stored elsewhere
    del matrix

    # Set the insertion-sorted list indices somewhere that we can retrieve them
    # in constant time. Used for eliminating double-distance-calculations,
    # since the distance is symmetric (dist(A,B)=dist(B,A))
    for i in range(0, len(timeseries_list)):
        timeseries_list[i].index = i

    #ooc_distances = OutOfCoreDistances("ananke_distances.h5", nrows)

    max_dist = sample_distances(timeseries_list, distance_measure)

    # Store the max_dist as epsilon_factor, because this is an estimate
    # and affects the scaling of epsilon to a domain of [0,1] and therefore
    # affects clustering consistency, so if we need to pick clustering back up
    # we must store this value to re-normalize against
    cluster_attrs = timeseriesdata.initialize_for_clustering(param_min, param_max, param_step)
    cluster_attrs.create("epsilon_factor", max_dist)
    # Start with largest epsilon and decrease
    epsilon_range = np.arange(param_min, param_max, param_step)

    for epsilon in epsilon_range[::-1]:
    # Pre-processing is complete, onto the distance calculations
        print("Computing for %f" % (epsilon,))
        ######## CLUSTERING ########
        n_clusts = 0
        epsilon_index = timeseriesdata.get_epsilon_index(epsilon)
        print(epsilon_index)
        # Take the top n_precompute most abundant sequences
        abundant_sequences = timeseries_list[-n_precompute:]

        # While there are unclustered sequences in our list
        while abundant_sequences:
            ts = abundant_sequences.pop()
            try:
                #Cluster_id != -2 means it is not clustered
                while timeseriesdata.get_cluster_by_index(ts.original_index, epsilon) != -2:
                    ts = abundant_sequences.pop()
            except IndexError:
                break
            n_clusts += 1
            #print("\nComputing cluster #%s" % (n_clusts,))
            cluster_around_seed(ts, timeseries_list, timeseriesdata,
                                epsilon, distance_measure,
                                min_pts, n_clusts, n_threads)
        for ts in timeseries_list:
            ts.reset()
    print("\nPre-computation of most abundant sequences complete")
