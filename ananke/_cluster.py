from ananke._distances import distance_function
from ananke._database import AnankeDB
from ananke._DBloomSCAN import DBloomSCAN, ExternalHashBloom
from bitarray import bitarray
import numpy as np
from zlib import compress, decompress

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

def bitarray_to_uint8array(barray):
    chunks = np.arange(len(barray), 0, -8)
    uint8array=np.zeros(len(chunks), dtype='uint8')
    for i, u in enumerate(chunks):
        l = max(0,u-8)
        chunk = barray[l:u]
        chunk = chunk[::-1]
        uint8array[i] = int(chunk.to01(), 2)
    return uint8array

def uint8array_to_bitarray(uint8array, length):
    bits = list(np.unpackbits(uint8array)[length - 1::-1])
    return bitarray(bits)

def save_blooms(anankedb, dbloomscan):
    bfs = anankedb._h5t.require_group("bloomfilters")
    attrs = bfs.attrs
    attrs["max_dist"] = dbloomscan.max_dist
    for bloom_id in dbloomscan.bloom_garden.blooms:
        bloom = dbloomscan.bloom_garden.blooms[bloom_id]
        uint8array = bitarray_to_uint8array(bloom.bitarray)
        bloom_size = len(bloom.bitarray)
        if str(bloom_id) in bfs:
            bfs[str(bloom_id)].write_direct(uint8array)
            attrs[str(bloom_id)] = bloom_size
        else:
            bfs.create_dataset(str(bloom_id), 
                               (len(uint8array),), 
                               dtype="uint8",
                               data = uint8array)
            attrs[str(bloom_id)] = bloom_size
    #Clean up any extra data sets
    for bloom_id in bfs:
        #If it isn't from this import
        if float(bloom_id) not in dbloomscan.bloom_garden.blooms.keys():
            del bfs[bloom_id]
        
def load_blooms(anankedb, distance_measure="sts", in_memory=True):
    time_points = anankedb.get_timepoints()
    dist = distance_function(distance_measure, time_points=time_points)
    if "bloomfilters" not in anankedb._h5t:
        raise IndexError("No bloom filters to load.")
    bfs = anankedb._h5t["bloomfilters"]
    attrs = bfs.attrs
    data_fetcher = data_fetcher_factory(anankedb, dist, in_memory)
    dist_range = sorted([float(bloom_id) for bloom_id in anankedb._h5t["bloomfilters"]])
    dbloomscan = DBloomSCAN(anankedb.nts, dist.distance,
                            data_fetcher, dist_range, max_dist = attrs["max_dist"])
    for bloom_id in bfs:
        bloom_size = attrs[bloom_id]
        new_bloom = ExternalHashBloom(20*anankedb.nts)
        uint8array = bfs[bloom_id][:]
        barray = uint8array_to_bitarray(uint8array, bloom_size)
        new_bloom.bitarray = barray
        dbloomscan.bloom_garden.blooms[float(bloom_id)] = new_bloom
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
