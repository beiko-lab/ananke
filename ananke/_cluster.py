from ananke._distances import distance_function
from ananke._database import AnankeDB
from ananke._DBloomSCAN import DBloomSCAN, ExternalHashBloom
from bitarray import bitarray
import numpy as np
from zlib import compress

def find_nearest_timeseries(anankedb, query_data, tsdist, n_threads = 1, n_chunks = 100, in_memory=True):
    if in_memory:
        data_matrix = np.empty(anankedb._h5t["data/timeseries/matrix"].shape)
        anankedb._h5t["data/timeseries/matrix"].read_direct(data_matrix)
        #If we can do this in RAM, we can efficiently pre-transform the matrix
        #to reduce the heft of the pair-wise comparisons as much as possible
        data_matrix = tsdist.transform_matrix(data_matrix)
    else:
        data_matrix = anankedb._h5t["data/timeseries/matrix"]
    query_data = query_data/sum(query_data)
    min_distance = np.inf
    min_index = None
    # Break the data_matrix into chunks in case the source is on disk
    def chunks(N, nb):
        step = N / nb
        return [(round(step*i), round(step*(i+1))) for i in range(nb)]
    for i, j in chunks(data_matrix.shape[0], n_chunks):
        data = data_matrix[i:j, :]
        for k in range(i, j):
            if in_memory:
                row = data[k-i, :]
            #If we're doing this on-disk, we couldn't do the matrix
            #transform, so this has to be done here
            else:
                row = tsdist.transform_row(data[k-i, :])
            distance = tsdist.distance(query_data, row/sum(row))
            if distance < min_distance:
                min_distance = distance
                min_index = k
    return min_index

def save_blooms(anankedb, dbloomscan):
    bfs = anankedb._h5t.require_group("bloomfilters")
    attrs = bfs.attrs
    for bloom in dbloomscan.bloom_garden.blooms:
        barray = np.void(compress(dbloomscan.bloom_garden.blooms[bloom].bitarray.tobytes()))
        attrs[str(bloom)] = barray

def load_blooms(anankedb, distance_measure="sts", data_in_memory=True):
    if distance_measure == "sts":
        distance_function = sts_distance
    elif distance_measure == "dtw":
        distance_function = dtw_distance
    elif distance_measure == "ddtw":
        distance_function = ddtw_distance
    else:
        raise ValueError("Unknown distance measure '%s'." % (distance_measure,))
    if "bloomfilters" not in anankedb._h5t:
        raise IndexError("No bloom filters to load.")
    capacity = anankedb._h5t["timeseries/ids"].shape[0]
    bfs = anankedb._h5t["bloomfilters"]
    attrs = bfs.attrs
    data_fetcher = data_fetcher_factory(anankedb, data_in_memory)
    dist_range = anankedb._h5["bloomfilters/dist_range"][:]
    dbloomscan = DBloomSCAN(capacity, distance_function,
                            data_fetcher, dist_range)
    for bloom_array in attrs:
        new_bloom = ExternalHashBloom(capacity)
        new_bloom.bitarray = bitarray(bloom_array)
        bloom_garden.blooms[bloom] = new_bloom
    return dbloomscan

def data_fetcher_factory(anankedb, tsdist, in_memory=True):
    # Hopefully once the data_fetcher is returned, the garbage collector
    # lets this hang around...
    if in_memory:
        data_matrix = np.empty(anankedb._h5t["data/timeseries/matrix"].shape)
        anankedb._h5t["data/timeseries/matrix"].read_direct(data_matrix)
        data_matrix = tsdist.transform_matrix(data_matrix)
        def data_fetcher(index):
            return data_matrix[index,:]/sum(data_matrix[index,:])
    else:
        data_matrix = anankedb._h5t["data/timeseries/matrix"]
        def data_fetcher(index):
            data = tsdist.transform_row(data_matrix[index,:])
            return data/sum(data)
    return data_fetcher

def precompute_distances(anankedb, distance_measure, dist_range, in_memory=True):
    capacity = anankedb.nts
    time_points = anankedb.get_timepoints()
    dist = distance_function(distance_measure, time_points=time_points)
    data_fetcher = data_fetcher_factory(anankedb, dist, in_memory) 
    dbloomscan = DBloomSCAN(capacity, dist.distance,
                            data_fetcher, dist_range)
    dbloomscan.compute_distances()
    save_blooms(anankedb, dbloomscan)
    return dbloomscan
