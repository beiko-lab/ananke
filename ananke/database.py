import h5py as h5
import numpy as np
import pandas as pd

import gzip
from collections import defaultdict, Counter
from hashlib import md5
from typing import List
from q2_extractor.Extractor import q2Extractor
from bitarray import bitarray

import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
from plotly.graph_objs import Figure, Layout, Scatter
import colorlover as cl

from ananke._distances import distance_function
from ananke.DBloomSCAN import DBloomSCAN, ExternalHashBloom

from .__init__ import __version__


# Utility functions

def version_greater_than(origin_version, version):
    """Compares version numbers

    Parameters
    ----------
    origin_version: str
        origin file version string, triple (major, minor, release), in self._h5t.attrs["origin_version"]
    version: str
        version string, triple (major, minor, release)

    Returns
    -------
    boolean
        True if origin_version is greater than provided string
    """
    major, minor, release = origin_version.decode("ASCII").split(".")
    comp_major, comp_minor, comp_release = version.split(".")
    if int(major) > int(comp_major):
        return True
    elif int(major) == int(comp_major):
        if int(minor) > int(comp_minor):
            return True
        elif int(minor) == int(comp_minor):
            if int(release) > int(comp_release):
                return True
    return False

# Main Database class

class Ananke(object):

    def __init__(self, h5_filepath):
        """Constructor for Ananke object. Creates an empty file with
        appropriate schema in place, if file did not exist. Otherwise,
        validates and links to file on disk.

        Parameters
        ----------
        h5_file_path: str
            filepath to Ananke HDF5 file (if doesn't exist, will make file at 
            this location)

        Returns
        -------
        self: TimeSeriesData object
        """
        h5t = h5.File(h5_filepath, 'a')
        self._h5t = h5t
        
        # Keeps track of whether the file is initialized
        self.initialized = False
        # Number of time series in the data set
        self.nts = 0
        self.featureids = []
        self.schema = {}
        # Create the required datasets (initialize empty)
        
        if "origin_version" in self._h5t.attrs:
            origin_version = self._h5t.attrs["origin_version"]
            if not version_greater_than(origin_version, "0.5.0"):
                raise ImplementationError("Ananke version 0.X files are not compatible with Ananke 1.0. " \
                                          "Please re-input data with the current Ananke version.")
            else:
                #Any existing-file-loading steps should be done here; none right now
                if len(self._h5t["data"]) > 0:
                    self.initialized = True
                self.nts = self._h5t["features/ids"].shape[0]
                self.featureids = self._h5t["features/ids"][:]
                self.schema = self._resolve_schema()
                self._make_feature_index()
                return
        else:
            # Create a new file if origin_version doesn't exist
            self._h5t.attrs.create("origin_version", str(__version__),
                                   dtype=h5.special_dtype(vlen=bytes))
             
            self._h5t.create_group("data")

            # Create genes group
            self._h5t.create_group("features")
            #IDs are hash values
            self._h5t["features"].create_dataset("ids", shape=(0,), 
                                   dtype=h5.special_dtype(vlen=bytes), 
                                   maxshape=(None,))

    def __del__(self):
        self._h5t.close()

    def __str__(self):
        info = "Origin version: %s" % \
               (self._h5t.attrs["origin_version"].decode(),) + "\n"
        if not self.initialized:
            info += "This file is uninitialized. Initialize with `ananke add-metadata`.\n"
        else:
            for address in self._h5t["data"]:
                series, replicate = address.split("__")
                group = self._h5t["data/" + address]
                info += "Series %s " % (series,)
                info += "Replicate %s " % (replicate,)
                info += "Num. of Time Points: %d\n" % (group["matrix"].shape[1])
        info += "Num. of Features: %d\n" % (self.nts,)
        return info

    #Schema of the format {series_name: {replicate_id1: {sample_id:timestamp, ... },
    #                                     ... },
    #                      ... }
    def initialize(self, schema):
        if self.initialized:
            print("File already initialized, skipping. Delete and recreate or choose a new filename to initialize to.")
            return
        for series in schema.keys():
            for replicate in schema[series].keys():
                address = series + "__" + replicate
                samples = schema[series][replicate]
                sample_ids = list(schema[series][replicate].keys())
                timestamps = list(schema[series][replicate].values())
                timestamps = [float(x) for x in timestamps]
                if (len(sample_ids) != len(np.unique(timestamps))):
                    raise ValueError("Time-stamps not unique (duplicate "
                            "time-stamps detected), please resolve conflicts.")
                timestamps, sample_ids= zip(*sorted(zip(timestamps, sample_ids)))
                # TODO: Check for duplicate sample_ids, and duplicate timestamps
                self._h5t.create_group("data/" + address)
                self._h5t.create_dataset("data/" + address + "/matrix",
                                        dtype=np.int32,
                                        shape=(0,len(timestamps)), 
                                        maxshape=(None, len(timestamps)),
                                        compression="gzip",  # I wonder what the effect of the compression is on speed and file size... TODO: look into this
                                        fillvalue=0)
                self._h5t.create_dataset("data/" + address + "/timestamps",
                                 dtype=h5.special_dtype(vlen=bytes),
                                 data=[str(x).encode() for x in timestamps])
                self._h5t.create_dataset("data/" + address + "/sampleids",
                                 dtype=h5.special_dtype(vlen=bytes),
                                 data=[str(x).encode() for x in sample_ids])
        self.initialized = True
        self.nts = self._h5t["features/ids"].shape[0]
        self.featureids = self._h5t["features/ids"][:]
        self._make_feature_index()
        self.schema = self._resolve_schema()

    def add_feature(self, feature_id):
        if not self.initialized:
            raise ValueError("Cannot add feature to an uninitialized data set")
        else:
            if isinstance(feature_id, str):
                feature_id = feature_id.encode()
            self._h5t["features/ids"].resize(size=self.nts+1, axis=0)
            for address in self._h5t["data"]:
                dat = self._h5t["data/" + address + "/matrix"]
                dat.resize(size=self.nts+1, axis=0)
            self.nts += 1
            #0-based indexing
            self._h5t["features/ids"][self.nts-1] = feature_id
            self.featureids = np.append(self.featureids, feature_id)
            #TODO Only update the index instead of rebuilding
            self._make_feature_index()

    def add_features(self, feature_id_list):
        if not self.initialized:
            raise ValueError("Cannot add feature to an uninitialized data set")
        else:
            n_add = 0
            feature_id_list = [x.encode() if type(x) == str else x for x in feature_id_list]
            feature_id_list = np.setdiff1d(feature_id_list, self.featureids)
            n_add = len(feature_id_list)
            for address in self._h5t["data"]:
                dat = self._h5t["data/" + address + "/matrix"]
                dat.resize(size=self.nts+n_add, axis=0)
            self._h5t["features/ids"].resize(size=self.nts+n_add, axis=0)
            self._h5t["features/ids"][self.nts:self.nts+n_add] = feature_id_list
            self.nts += n_add
            self.featureids = np.append(self.featureids, feature_id_list)
            self._make_feature_index()
            
    def __index__(self, a):
        # Keep a RAM-based index of the feature names, this is usually OK
        try:
            return self.feature_index[a]
        except:
            try:
                return self.feature_index[a.encode()]
            except:
                return None

    #TODO: Multi-selection series and replicate SET
    def __setitem__(self, key, value):
        replicate = None
        sample_id = None
        if len(key) == 4:
            featureid, series, replicate, sample_id = key
        elif len(key) == 3:
            featureid, series, replicate = key
        elif len(key) == 2:
            featureid, series = key
            replicates = self.get_replicates(series)
            if len(replicates) > 1:
                raise IndexError("More than one replicate, replicate must be " \
                                 "explicitly specified. Options are: %s" % \
                                                   (", ".join(replicates)))
            else:
                replicate = replicates[0]
        index = self.__index__(str(featureid).encode())
        if index is None:
            # Add a new feature to all timeseries
            self.add_feature(str(featureid).encode())
            index = self.nts - 1
        address = series + "__" + replicate
        if sample_id is not None:
            sample_index = self._get_sample_index(sample_id)
            self._h5t["data/" + address + "/matrix"][index, sample_index] = value
        self._h5t["data/" + address + "/matrix"][index, :] = np.array(value)

    #TODO: Multi-selection series and replicate GET
    def __getitem__(self, key):
        #If you supply the replicate as the third value,
        #then return that specific replicate
        #else, return the replicates merged
        #TODO: pass along different merge function through this GET
        replicate = None
        sample_id = None
        if len(key) == 4:
            featureid, series, replicate, sample_id = key
        elif len(key) == 3:
            featureid, series, replicate = key
        elif len(key) == 2:
            featureid, series = key
            replicate = None
        if type(featureid) == int:
            index = featureid
        else:
            index = self.__index__(str(featureid).encode())
        if (replicate is not None) and (sample_id is not None):
            sample_index = self._get_sample_index(sample_id)
            address = series + "__" + replicate
            row = self._h5t["data/" + address + "/matrix"][index, :]
            return row[sample_index]
        if replicate is not None:
            address = series + "__" + replicate
            return self._h5t["data/" + address + "/matrix"][index, :]
        else:
            return self.get_merged_replicates(featureid, series)

    def _resolve_schema(self):
        schema = {}
        for address in self._h5t["data/"]:
            series, replicate = address.split("__")
            if series not in schema:
                schema[series] = {}
            if replicate not in schema[series]:
                schema[series][replicate] = {}
            schema[series][replicate].update(dict(zip(self._h5t["data/" + 
                                                       address + 
                                                       "/sampleids"][:],
                                                      self._h5t["data/" + 
                                                       address + 
                                                       "/timestamps"][:])))
        return schema

    def _make_feature_index(self):
        self.feature_index = dict([(y,x) for x, y in enumerate(self.featureids)])

    def _get_sample_index(self, sample_id):
        series, replicate = self.get_sample_address(sample_id)
        address = series + "__" + replicate
        return np.argwhere(self._h5t["data/" + address + "/sampleids"][:] == str(sample_id).encode())[0][0]

    def get_replicates(self, series):
        return list(self.schema[series].keys())
       
    def get_series(self):
        return list(self.schema.keys())

    def get_sample_address(self, sample_id):
        sample_id = str(sample_id).encode()
        for series in self.schema:
            for replicate in self.schema[series]:
                if sample_id in self.schema[series][replicate]:
                    return (series, replicate)
        return None

    def get_timepoints(self, series):
        #Grab the first replicate's points, they must all have
        #the same time points
        return np.array(
                list(
                 list(self.schema[series].values())[0].values()),
                dtype=np.float)

    def get_merged_replicates(self, feature_id, series,
                              merge_func=lambda x: np.mean(x, axis=0)):
        replicates = self.get_replicates(series)
        reps = []
        for replicate in replicates:
            rep = self.__getitem__((feature_id, 
                                    series, 
                                    replicate))
            reps.append(rep)
        reps = np.array(reps)
        return merge_func(reps)

############## PLOTTING FUNCTIONS ################

    def plot_feature_clusters(self, dbs, featureid, title_index = {}, max_members=100):
        if isinstance(featureid, int):
            index = featureid
        else:
            index = self.__index__(featureid)
      
        return [self.plot_feature_at_distance(dbs, index, i, 
                                              title_index, max_members) 
                for i in dbs.dist_range]

    def plot_feature_at_distance(self, dbs, featureid, epsilon, title_index={}, max_members=100, plot_agg=False):
        if isinstance(featureid, int):
            index = featureid
        else:
            index = self.__index__(featureid)
        try:
            clusts = dbs.DBSCAN(epsilon=epsilon, expand_around=index, warn=False,
                                max_members=max_members)
        except:
            return None
        cluster = clusts[1]
        if len(cluster) > max_members:
            print("Too big, exiting")
            return None
        if len(cluster) == 1:
           colour_mapper = {cluster[0]: cl.scales['3']['qual']['Paired'][0]}
        elif len(cluster) == 2:
           colour_mapper = {cluster[0]: cl.scales['3']['qual']['Paired'][0], 
                            cluster[1]: cl.scales['3']['qual']['Paired'][1]}
        elif (len(cluster) > 2) & (len(cluster) <= 11):
           colour_mapper = dict([(y,cl.scales[str(len(cluster))]['qual']['Paired'][x]) for x,y in enumerate(cluster)])
        else:
           colour_mapper = None
        series_trace = []
        for series in self.get_series():
            data = []
            if plot_agg:
                agg = None
            for i in cluster:
                try:
                    tax_name = title_index[self.featureids[i].decode()]
                    tax_name = ";".join(tax_name.split(";")[-2:])
                except:
                    tax_name = self.featureids[i].decode()
                dat = self.get_merged_replicates(i, series)
                show_legend = True if series == self.get_series()[0] else False
                colour = colour_mapper[i] if colour_mapper is not None else None 
                if not plot_agg:
                    data.append(Scatter(y=dat/sum(dat),
                            x=self.get_timepoints(series),
                            name=str(i)+": "+tax_name, line={"color":colour}, 
                            showlegend=show_legend))
                else:
                    if agg is None:
                        agg = dat/sum(dat)
                    else:
                        agg += dat/sum(dat)
            if plot_agg:
                agg = agg/len(cluster)
                data.append(Scatter(y=agg,
                                    x=self.get_timepoints(series),
                                    name="Mean of cluster around %d" % (index,)))
            series_trace.append(data)
        fig = tools.make_subplots(rows=len(self.get_series()), cols=1, subplot_titles=self.get_series(), print_grid=False)
        fig['layout']['legend']['x'] = 1.02
        fig['layout']['legend']['y'] = 1
        for j, series_data in enumerate(series_trace):
            for data in series_data:
                fig.append_trace(data, row=j+1, col=1)
        fig['layout'].update(height=300*len(self.get_series()), width=900, title="Feature %s" % (str(featureid),))
        return fig

############## FILTERING FUNCTIONS ###############

    def filter_features(self, threshold=5, chunk_size=1e5):
        rejected_ids = []
        nnz_summary = {}
        for address in self._h5t["data/"]:
            m=self._h5t["data/" + address + "/matrix"].shape[1]
            n=self.nts
            nnz_summary[address] = np.zeros(shape=(n,))
            temp_mat = np.zeros(shape=(n,m), dtype=np.int32)
            self._h5t["data/" + address + "/matrix"].read_direct(temp_mat)
            for r_index in range(temp_mat.shape[0]):
                #Take the nnz elements for each series/replicate
                nnz=np.count_nonzero(temp_mat[r_index,:])
                nnz_summary[address][r_index] = nnz
        nnz_total = np.sum([nnz_summary[address] for address in nnz_summary], axis=0)
        rejected_ids = np.where(nnz_total < threshold)[0]
        print("Removing %d features, %d remain" % (len(rejected_ids), self.nts-len(rejected_ids)))
        self.delete_features(rejected_ids)

    def delete_features(self, feature_indexes):
        for address in self._h5t["data/"]:
            data_matrix = self._h5t["data/" + address + "/matrix"]
            keep_indexes = np.setdiff1d(np.arange(0, self.nts), feature_indexes, assume_unique=True)
            if "temp_matrix" in self._h5t:
                del self._h5t["temp_matrix"]
            temp = self._h5t.create_dataset("temp_matrix",
                                           (len(keep_indexes),
                                           data_matrix.shape[1]))
            ar = np.empty(shape=temp.shape)
            temp_mat = np.empty(shape=data_matrix.shape)
            data_matrix.read_direct(temp_mat)
            for i, j in enumerate(keep_indexes):
                self._h5t["temp_matrix"].write_direct(temp_mat, np.s_[j,:], np.s_[i,:])
            del self._h5t["data/" + address + "/matrix"]
            self._h5t["data/" + address + "/matrix"] = self._h5t["temp_matrix"]
            del self._h5t["temp_matrix"]
        self.nts = self.nts - len(feature_indexes)
        self.featureids = self.featureids[keep_indexes]
        self._make_feature_index()
        #reshape features/ids
        self._h5t["features/ids"].resize(self.featureids.shape)
        self._h5t["features/ids"].write_direct(self.featureids)

############## CLUSTERING FUNCTIONS ##############

    def _make_dbloomscan(self, distance_measure,
                               distance_range=np.arange(0.1,1,0.1),
                               norm_ord=None,
                               matrix_merge_func=lambda x: np.mean(x, axis=2),
                               max_dist=None):
        # First, set the data source in memory, make a distance
        # function for each time series, and pre-process the data
        # as necessary
        data_matrices = {}
        #Load the data matrices into memory, collapsing the replicates 
        for series in self.schema.keys():
            time_points = self.get_timepoints(series)
            dist = distance_function(distance_measure, time_points=time_points)
            data_matrices[series] = {}
            data_matrices[series]["distance"] = dist
            nreps = len(self.schema[series].keys())
            first_rep = list(self.schema[series].keys())[0]
            shape = self._h5t["data/" + series + \
                              "__" + first_rep + "/matrix"].shape
            data_matrix = np.empty((shape[0], shape[1]))
            # Some transformations shrink the results matrix, which is what
            # we are merging, so we need to know what size it will be
            result_matrix = dist.transform_matrix(data_matrix)
            data_matrix = np.empty((result_matrix.shape[0],
                                    result_matrix.shape[1],
                                    nreps))
            i = 0
            for rep in self.schema[series].keys():
                address = "data/" + series + "__" + \
                          rep + "/matrix"
                temp = np.empty(self._h5t[address].shape)
                self._h5t[address].read_direct(temp)
                data_matrix[:,:,i] = dist.transform_matrix(temp)
                data_matrix[:,:,i] = data_matrix[:,:,i]/np.linalg.norm(data_matrix[:,:,i], axis=1, ord=norm_ord)[:, np.newaxis]
                i+=1
            data_matrix = matrix_merge_func(data_matrix)
            data_matrices[series]["matrix"] = data_matrix
            
        def data_fetcher(index=None, custom_query=None):
            if custom_query:
                data = [data_matrices[series]["distance"].transform_row(query) / np.linalg.norm(query, axis=0, ord=norm_ord)\
                        for series, query in zip(self.get_series(), custom_query)]
            else:
                data = [data_matrices[series]["matrix"][index,:] \
                        for series in data_matrices]
            return data

        def data_computer(ts_list1, ts_list2):
            dists = []
            tsdists = [data_matrices[series]["distance"] for series in data_matrices]
            for tsdist, ts1, ts2 in zip(tsdists, ts_list1, ts_list2):
                dists.append(tsdist.distance(ts1, ts2))
            return np.sum(dists)

        dbloomscan = DBloomSCAN(self.nts, data_computer,
                                data_fetcher, distance_range,
                                max_dist)
        return dbloomscan


    def get_nearest_timeseries(self, query, distance_measure="sts"):
        #Each series and replicate must have its own tsdist
        dbs = self._make_dbloomscan(distance_measure)
        min_dist = np.inf
        closest_idx = None
        for i in range(0, self.nts):
            dist = dbs.compute_distance(dbs.fetch_data(custom_query=query), 
                                        dbs.fetch_data(i))
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx
        

    def precompute_distances(self, distance_measure="sts",
                             distance_range=np.arange(0.1,1,0.1),
                             norm_ord=None,
                             matrix_merge_func=lambda x: np.mean(x, axis=2),
                             save_results = True):
        dbs = self._make_dbloomscan(distance_measure, distance_range, norm_ord,
                                    matrix_merge_func)
        dbs.compute_distances()
        return dbs

    def save_cluster_result(self, dbs, dist_name):
        bfs = self._h5t.require_group("bloomfilters/%s" % (dist_name))
        attrs = bfs.attrs
        for bloom in dbs.bloom_garden.blooms:
            barray = dbs.bloom_garden.blooms[bloom].bitarray
            bools = np.frombuffer(barray.unpack(), dtype=np.bool)
            packed = np.packbits(bools)
            attrs["bitarray_length"] = len(barray)
            attrs["max_dist"] = dbs.max_dist
            ds = bfs.require_dataset("bloom_%s" % (str(bloom),),
                                    (len(packed),),
                                    dtype='uint8')
            ds.write_direct(packed)
  
    def load_cluster_result(self, distance_name):
        if "bloomfilters" not in self._h5t:
            raise IndexError("No bloom filters to load.")
        if distance_name not in self._h5t["bloomfilters"]:
            raise IndexError("No bloom filter for this distance.")
        capacity = self._h5t["features/ids"].shape[0]
        bfs = self._h5t["bloomfilters/%s" % (distance_name,)]
        attrs = bfs.attrs
        barray_len = attrs["bitarray_length"]
        max_dist = attrs["max_dist"]
        distance_range = [float(x.split("_")[-1]) for x in bfs]
        dbs = self._make_dbloomscan(distance_name, distance_range, 
                                    max_dist=max_dist)
        for bloom in bfs:
            bloom_distance = float(bloom.split("_")[-1])
            bloom_array = np.ndarray(shape=bfs[bloom].shape, dtype=np.uint8)
            bfs[bloom].read_direct(bloom_array)
            bloom_array = np.unpackbits(bloom_array)
            new_bloom = ExternalHashBloom(200*capacity)
            new_bloom.bitarray = bitarray(bloom_array[0:barray_len].tolist(), endian="little")
            dbs.bloom_garden.blooms[bloom_distance] = new_bloom
        return dbs
################# IMPORT FUNCTIONS ##########################

    def import_from_qiime(self, artifact):
        #Import from a QIIME OTU table artifact
        table = q2Extractor(artifact).extract_data()
        self.add_features(table.index)
        for series in self.schema:
            for rep in self.schema[series]:
                series_names = self._h5t["data/" + series + \
                                         "__" + rep + "/sampleids"]
                series_names = [name.decode() for name in series_names]
                if not np.all([name in table.columns.values for name in series_names]):
                    missing_columns = ",".join([name for name in series_names if name not in table.columns.values])
                    raise ValueError("Columns missing from this table! Cannot import. Missing: %s" % (missing_columns,))
                else:
                    print("Importing series %s, replicate %s" % (series, rep))
                    series_table = table[series_names]
                    for idx in series_table.index:
                        self._h5t["data/" + series + 
                              "__" + rep + "/matrix"][self.__index__(idx),:] = \
                                                           series_table.loc[idx]

    def import_from_fasta(self, fasta_file, unique_fasta=None, min_count=2):
        i=0
        if unique_fasta:
            unique_fasta = open(unique_fasta, 'w')
        if ".gz" in fasta_file:
            seqfile = gzip.open(fasta_file, 'r')
        else:
            seqfile = open(fasta_file, 'r')
        seq_hashes = Counter()
        for line in seqfile:
            if isinstance(line, bytes): line = line.decode()
            sample_id = line.split("_")[-2][1:]
            line = seqfile.readline()
            if isinstance(line, bytes): line = line.decode()
            seq_hash = md5(line.encode()).hexdigest().encode()
            if seq_hash not in seq_hashes:
                if unique_fasta:
                    unique_fasta.write(">%s\n" % (seq_hash,))
                    unique_fasta.write(line)
            seq_hashes[seq_hash] += 1
            i+=1
            if i%100000 == 0:
                print("Reading in sequences: %d" % (i,), end="\r")
        add_ids = [name for name, count in seq_hashes.items() if count >= min_count] 
        print("\nAdding %d unique sequences to file with a minimum count of %d" % (len(add_ids), min_count))
        self.add_features(add_ids)
        add_ids = set(add_ids)
        del seq_hashes
        i = 0
        skipped = 0
        count_cache = {}
        seqfile.seek(0)
        for line in seqfile:
            if isinstance(line, bytes):
                line = line.decode()
            sample_id = line.split("_")[-2][1:]
            line = seqfile.readline()
            if isinstance(line, str): line = line.encode()
            seq_hash = md5(line).hexdigest().encode()
            try:
                series, replicate = self.get_sample_address(sample_id)
            except:
                i += 1
                skipped += 1
                continue
            if seq_hash not in add_ids:
                i += 1
                skipped += 1
                continue
            address = str(series) + "__" + str(replicate)
            sample_index = self._get_sample_index(sample_id)
            full_address = address + "__" + str(sample_index)
            feature_index = self.__index__(seq_hash)
            if full_address not in count_cache:
                count_cache[full_address] = np.zeros(shape=(self.nts,))
            count_cache[full_address][feature_index] += 1
            i+=1
            if i%100000 == 0:
                print("Tallying sequences across timepoints: %d" % (i,), end="\r")
        print("Tallying sequences across timepoints: %d" % (i,), end="\r")
        for full_address in count_cache:
            series, replicate, s_id = full_address.split("__")
            s_id = int(s_id)
            ar = count_cache[full_address]
            print("Adding %d counts to %s" % (int(sum(ar)), full_address))
            self._h5t["data/" + series + "__" + replicate + "/matrix"].write_direct(ar, np.s_[:], np.s_[:, s_id])
