import pybloom
from pybloom.utils import range_fn
import numpy as np
from random import randrange
from bitarray import bitarray
from zlib import compress
import xxhash
from struct import pack, unpack

#### REFACTOR ####

# This has to be defined here for multiprocessing to give access to
# the data matrix to worker processes, currently unused until I figure out
# how to get it to work

global_data_matrix = np.empty(shape=(1,1))

class DBloomSCAN(object):
    def __init__(self, n_objects, distance_computer, data_fetcher, 
                 dist_range=np.arange(0.0001,0.201,0.0001), max_dist=None):
        self.n_objects = n_objects
        self.compute_distance = distance_computer
        self.fetch_data = data_fetcher
        self.dist_range = np.array([round(x, 5) for x in dist_range])
        self.bloom_garden = BloomGarden(self.dist_range, 20 * self.n_objects)
        if max_dist is None:
            self.max_dist = self._sample_distances()
        else:
            self.max_dist = max_dist
        self._computed = 0

    def set_data_source(self, data):
        global global_data_matrix 
        global_data_matrix = data
            
    def add_distance(self, data):
        i, j, distance = data
        #If the distance is bigger than our largest range, it isn't important, discard ASAP
        if distance > self.dist_range[-1]:
            return
        def bloom_condition(bloom):
            return bloom >= distance
        #Given a real distance d, if d <= name, that means that the objects are closer than the threshold,
        #meaning they are neighbours at that distance or less
        pruned_blooms = self.bloom_garden.add((min(i,j),max(i,j)), bloom_condition)
        # If we had to prune a full filter, we kick it out of our dist range
        if pruned_blooms == 0:
            pruned_blooms = None
        else:
            pruned_blooms = -pruned_blooms
        self.dist_range = self.dist_range[:pruned_blooms]
        self._computed += 1

    #Take in a tuple of data series and computes the distance
    def _compute_distance_wrapper(self, indices):
        i, j = indices
        data1 = self.fetch_data(i)
        data2 = self.fetch_data(j)
        distance = self.compute_distance(data1, data2)
        distance = distance / self.max_dist
        data = (i, j, distance)
        self.add_distance(data)

    def _consume_distance_wrapper(self, queue):
        while True:
            data = queue.get()
            self.add_distance(data)
            #if self._computed % 10000 == 0:
            percent = 100*float(self._computed)/(self.n_objects*(self.n_objects-1)/2)
            print("%0.2f%%" % (percent,), end='\r')
                
    def _sample_distances(self):
        # Sample some distances to guess the max epsilon, used for scaling
        # In the RAM-hungry version, we know the max up-front and can scale
        # perfectly, but in this case we have to guess and then check
        nrows = self.n_objects
        max_dist = 0
        # Sample twice as many distances as we have unique genes
        # TODO: Validate that this is a good enough amount of sampling
        n_iters = 2*int(nrows)
        for i in range(0, n_iters):
            x = randrange(0, nrows)
            y = randrange(0, nrows)
            if x == y:
                i -= 1
                break
            distance = self.compute_distance(self.fetch_data(x), 
                                             self.fetch_data(y))
            if distance > max_dist:
                max_dist = distance

        return max_dist

    def are_neighbours(self, i, j, distance, validate = True):
        if distance not in self.bloom_garden.blooms:
            raise ValueError("Distance not in computed range")
        bf_result = (min(i,j), max(i,j)) in self.bloom_garden.blooms[distance]
        #If the bloom filter says no, they are definitely not neighbours within this distance
        if not bf_result:
            return False
        else:
            if validate:
                #Compute the actual distance
                verified_distance = self.compute_distance(self.fetch_data(i), 
                                                          self.fetch_data(j))
                verified_distance = verified_distance / self.max_dist
                if verified_distance >= distance:
                    return False
                else:
                    return True
            else:
                # Return it as a positive, and put it on the caller to double check
                return True
            
    def DBSCAN(self, epsilon, min_pts = 2, expand_around=None):
        if epsilon not in list(self.dist_range):
            dist_range = self.dist_range
            delta = dist_range - epsilon
            old_epsilon = epsilon
            epsilon = self.dist_range[np.argmin(abs(delta))]
            print("Bloom filter does not exist for this epsilon value, %f. Using the closest precomputed value, %f." % (old_epsilon,epsilon))
        cluster_number = 0
        clustered = set()
        cluster_assignments = {}
        if expand_around is not None:
            index_queue = [ expand_around ]
        else:
            index_queue = range(0, self.n_objects)
        for i in index_queue:
            if i in clustered:
                continue
            cluster_number += 1
            cluster_assignments[cluster_number] = [i]
            cluster_queue = [i]
            clustered.add(i)
            while cluster_queue:
                k = cluster_queue.pop()
                neighbourhood = []
                for j in range(0, self.n_objects):
                    if (j != k) & (j not in clustered) & \
                       (self.are_neighbours(k, j, epsilon, validate=True)):
                        neighbourhood.append(j)
                # min_pts neighbourhood size includes the point itself, so we account for that here
                # This means k is a core point
                if len(neighbourhood) >= min_pts - 1:
                    cluster_queue.extend(neighbourhood)
                    #if it is in range of a core point, it's in the cluster
                    cluster_assignments[cluster_number].extend(neighbourhood)
                    clustered.update(neighbourhood)
        return cluster_assignments
            
    def compute_distances(self, n_threads = 1):
        # Serial solution
        if n_threads == 1:
            c = 0
            for i in range(0, self.n_objects):
                data1 =  self.fetch_data(i)
                #TODO: Chunk this range and fetch chunks in case the source is on disk
                for j in range(i+1, self.n_objects):
                    data2 = self.fetch_data(j)
                    result = self.compute_distance(data1, data2)
                    self.add_distance((i, j, result))
                    c += 1
                    if c % 10000 == 0:
                        percent = 100*float(c)/(self.n_objects*(self.n_objects-1)/2)
                        print("%0.2f%%" % (percent,), end='\r')
                percent = 100*float(c)/(self.n_objects*(self.n_objects-1)/2)
                print("%0.2f%%" % (percent,), end='\r')
        else:
            raise NotImplementedError

class ExternalHashBloom(pybloom.BloomFilter):
    def __init__(self, capacity, error_rate=0.001):
        super().__init__(capacity, error_rate)
        self.make_hashes = make_hashfuncs(self.num_slices, self.bits_per_slice)

    #Overwrite the existing add function, but remove the hash check
    def add_hashes(self, hashes, skip_check = False):
        bitarray = self.bitarray
        bits_per_slice = self.bits_per_slice
        found_all_bits = True
        if self.count > self.capacity:
            raise IndexError("BloomFilter is at capacity")
        offset = 0
        for k in hashes:
            if not skip_check and found_all_bits and not bitarray[offset + k]:
                found_all_bits = False
            self.bitarray[offset + k] = True
            offset += bits_per_slice

        if skip_check:
            self.count += 1
            return False
        elif not found_all_bits:
            self.count += 1
            return False
        else:
            return True 

class BloomGarden(object):
    def __init__(self, filter_names, capacity):
        self.blooms = {}
        for name in filter_names:
            self.blooms[name] = ExternalHashBloom(capacity)

    def add(self, key, name_condition):
        prune_list = []
        pruned_blooms = 0
        hashes = None
        for name in self.blooms:
            if not hashes:
                hashes = self.blooms[name].make_hashes(key)
                #This is a generator, so we need to coerce it
                #to something static or the first insert depletes it
                hashes = list(hashes)
            if name_condition(name):
                try:
                    #We can skip the check because each pair we check is unique
                    self.blooms[name].add_hashes(hashes, skip_check=True)
                except IndexError:
                    #print("Bloom filter '%s' hit capacity, closing" % (str(name),))
                    prune_list.append(name)
                    pruned_blooms += 1
        for bloom in prune_list:
            del self.blooms[bloom]
        if not self.blooms:
            raise IndexError("All bloom filters closed. Try using a smaller minimum epsilon value.")
        return pruned_blooms

# This is taken from pybloom, but modified to use xxhash.xxh64()
def make_hashfuncs(num_slices, num_bits):
    if num_bits >= (1 << 31):
        fmt_code, chunk_size = 'Q', 8
    elif num_bits >= (1 << 15):
        fmt_code, chunk_size = 'I', 4
    else:
        fmt_code, chunk_size = 'H', 2
    total_hash_bits = 8 * num_slices * chunk_size
    hashfn = xxhash.xxh64
    fmt = fmt_code * (hashfn().digest_size // chunk_size)
    num_salts, extra = divmod(num_slices, len(fmt))
    if extra:
        num_salts += 1
    salts = tuple(hashfn(hashfn(pack('I', i)).digest()) for i in range_fn(num_salts))
    def _make_hashfuncs(key):
        if isinstance(key, str):
            key = key.encode('utf-8')
        else:
            key = str(key).encode('utf-8')
        i = 0
        for salt in salts:
            h = salt.copy()
            h.update(key)
            for uint in unpack(fmt, h.digest()):
                yield uint % num_bits
                i += 1
                if i >= num_slices:
                    return
    return _make_hashfuncs


#MOVE THIS TO be transparent within the Ananke() class

#Currently overwrites blooms
#TODO: warn before overwriting?
def save_blooms_to_ananke(bloom_garden, timeseriesdata):
    bfs = timeseriesdata._h5t.require_group("bloomfilters")
    attrs = bfs.attrs
    for bloom in bloom_garden.blooms:
        barray = np.void(bloom_garden.blooms[bloom].bitarray.tobytes())
        attrs[str(bloom)] = barray

def load_blooms_from_ananke(timeseriesdata, bloom_garden):
    if "bloomfilters" not in timeseriesdata._h5t:
        raise IndexError("No bloom filters to load.")
    capacity = timeseriesdata._h5t["timeseries/ids"].shape[0]
    bfs = timeseriesdata._h55["bloomfilters"]
    attrs = bfs.attrs
    for bloom_array in attrs:
        new_bloom = ExternalHashBloom(capacity)
        new_bloom.bitarray = bitarray(bloom_array)
        bloom_garden.blooms[bloom] = new_bloom
