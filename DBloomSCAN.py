
# coding: utf-8

# In[39]:


import pybloom
import numpy as np
from fastdtw import fastdtw
from ananke._database_rework import TimeSeriesData
from ananke._ddtw import DDTW
from random import randrange
from itertools import combinations
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial
from queue import Queue
from bitarray import bitarray
from zlib import compress

# In[42]:

#This is simply a subclass of the bloom filter that lets you use
#pre-computed hashes and add those, which avoids the expense of recomputing
#the hash function on the same data for each bloom in the garden
class ExternalHashBloom(pybloom.BloomFilter):
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

#This class will contain a bunch of bloom filters, but only computes
#the hash once before inserting, as needed, into the filters
#Can we make a bunch of subclassed Pybloom filters, steal one make_hashes function,
#and then overwrite the Pybloom add function
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
                    self.blooms[name].add_hashes(hashes)
                    #self.blooms[name].add(key)
                except IndexError:
                    print("Bloom filter '%s' hit capacity, closing" % (str(name),))
                    prune_list.append(name)
                    pruned_blooms += 1
        for bloom in prune_list:
            del self.blooms[bloom]
        if not self.blooms:
            raise IndexError("All bloom filters closed. Try using smaller epsilon value.")
        return pruned_blooms



class BloomDistance(object):
    def __init__(self, n_objects, distance_computer, data_fetcher, dist_min=0.0001, dist_max=0.201, dist_step=0.0001):
        self.n_objects = n_objects
        self.compute_distance = distance_computer
        self.fetch_data = data_fetcher
        self.dist_range = [round(x, 5) for x in np.arange(dist_min, dist_max, dist_step)]
        self.bloom_garden = BloomGarden(self.dist_range, 100 * self.n_objects)
        self.false_positives = 0
        self.true_positives = 0
        self.max_dist = self._sample_distances()
            
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
    
    #Take in a tuple of data series and computes the distance
    def _compute_distance_wrapper(self, data):
        i, j, data1, data2 = data
        distance = self.compute_distance(data1, data2)
        distance = distance / self.max_dist
        return i, j, distance
                
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

        print("After %d samples of the distances, the max distance was %f" % (n_iters, max_dist))
        
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
                    #This is a false positive, and they aren't neighbours
                    self.false_positives += 1
                    return False
                else:
                    self.true_positives += 1
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
        c = 0
        for i in range(0, self.n_objects):
            data1 =  self.fetch_data(i)
            if n_threads == 1:
                for j in range(i+1, self.n_objects):
                    data2 = self.fetch_data(j)
                    result = self._compute_distance_wrapper((i, j, data1, data2))
                    self.add_distance(result)
                    c += 1
                    if c % 10000 == 0:
                        percent = 100*float(c)/(self.n_objects*(self.n_objects-1)/2)
                        print("%0.2f%%" % (percent,), end='\r')
                percent = 100*float(c)/(self.n_objects*(self.n_objects-1)/2)
                print("%0.2f%%" % (percent,), end='\r')
            else:
                #TODO: On my laptop, this is incredibly slow, because the bloom filter hashes are slower than the distance calculations,
                #which causes a back-up in the producer-consumer model and bogs the whole thing down
                #I may have fixed this with the BloomGarden addition, but need to test
                n_objects = self.n_objects
                fetch_data = self.fetch_data
                class Pairs:
                    def __init__(self):
                        self.n_objects = n_objects
                        self.fetch_data = fetch_data
                    def __iter__(self):
                        self.j = i
                        return self
                    def __next__(self):
                        self.j += 1
                        if self.j > self.n_objects:
                            raise StopIteration
                        return (i, self.j, data1, self.fetch_data(self.j))
                j = range(i+1, self.n_objects)
                p = Pool(n_threads)
                results = p.imap(self._compute_distance_wrapper, Pairs(), chunksize=100)
                for result in results:
                    self.add_distance(result)
                    c+=1
                    if c % 10000 == 0:
                        percent = 100*float(c)/(self.n_objects*(self.n_objects-1)/2)
                        print("%0.2f%%" % (percent,), end='\r')
                p.close()
                p.join()
        print("\nDone!")

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
