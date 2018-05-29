import h5py as h5
import numpy as np
import pandas as pd
from functools import lru_cache
import arrow

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

class AnankeDB(object):
    """Object that contains the Ananke analysis and facilitates interacting with
       an Ananke .h5 file
    """

    def __init__(self, h5_filepath):
        """Constructor for TimeSeriesData object. Creates an empty file with
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
        # Create the required datasets (initialize empty)
        
        if "origin_version" in self._h5t.attrs:
            origin_version = self._h5t.attrs["origin_version"]
            if not version_greater_than(origin_version, "0.3.0"):
                raise ImplementationError("Ananke version 0.X files are not compatible with Ananke 1.0. " \
                                          "Please re-input data with the current Ananke version.")
            else:
                #Any existing-file-loading steps should be done here; none right now
                if len(self._h5t["data"]) > 0:
                    self.initialized = True
                self.nts = self._h5t["timeseries/ids"].shape[0]
                return
        else:
            # Create a new file if origin_version doesn't exist
            self._h5t.attrs.create("origin_version", str(__version__),
                                   dtype=h5.special_dtype(vlen=bytes))
             
            self._h5t.create_group("data")

            # Create genes group
            self._h5t.create_group("timeseries")
            #IDs are hash values
            self._h5t["timeseries"].create_dataset("ids", shape=(0,), 
                                   dtype=h5.special_dtype(vlen=bytes), 
                                   maxshape=(None,))
            self._h5t["timeseries"].create_dataset("clusters", shape=(0,0), 
                                   dtype=np.int16, 
                                   maxshape=(None,None), fillvalue=-2)
            self._h5t["timeseries"].create_dataset("taxonomy", shape=(0,), 
                                   dtype=h5.special_dtype(vlen=bytes), 
                                   maxshape=(None,))
            self._h5t["timeseries"].create_dataset("altclusters", shape=(0,), 
                                   dtype=h5.special_dtype(vlen=bytes),
                                   maxshape=(None,))

    def __del__(self):
        """Destructor for Ananke object

        Closes the connection to the HDF5 file
        """
        if self._h5t:
            self._h5t.close()

    def __str__(self):
        info = "Origin version: %s" % \
               (self._h5t.attrs["origin_version"].decode(),) + "\n"
        if not self.initialized:
            info += "This file is uninitialized. Initialize with `ananke add-metadata`.\n"
        else:
            for group_name in self._h5t["data"]:
                group = self._h5t["data/" + group_name]
                if "series" in group.attrs:
                    info += "Series %s " % (group.attrs["series"].decode(),)
                if "replicate" in group.attrs:
                    info += "Replicate %s " % (group.attrs["replicate"].decode(),)
                info += "Num. of Time Points: %d\n" % (group["matrix"].shape[1])
            info += "Num. of Time Series: %d" % (self._h5t["timeseries/ids"].shape[0],)
        return info

    def create_series(self, name, time, sample_names, replicate_id = None):
        """ Creates a new series in the h5 file. If a replicate_id is supplied,
            the name of the series will be name_replicateid. All series with
            the same name but different replicate_ids will be aggregated
            prior to clustering.
        """
        #TODO: Update to have nrows to match the numbers of timeseries in the file already
        if replicate_id is not None:
            name = name + "_" + str(replicate_id)
            self._h5t["data/" + name].attrs.create("replicate", stsr(replicate_id).encode(),
                                                   dtype=h5.special_dtype(vlen=bytes))
        self._h5t.create_group("data/" + name)
        self._h5t.create_dataset("data/" + name + "/matrix",
                                 dtype=np.int32,
                                 shape=(0,len(time)), 
                                 maxshape=(None, len(time)),
                                 compression="gzip",  # I wonder what the effect of the compression is on speed and file size... TODO: look into this
                                 fillvalue=0)
        self._h5t.create_dataset("data/" + name + "/time",
                                 dtype=h5.special_dtype(vlen=bytes),
                                 data=[str(x).encode() for x in time])
        self._h5t.create_dataset("data/" + name + "/names",
                                 dtype=h5.special_dtype(vlen=bytes),
                                 data=[str(x).encode() for x in sample_names])
        self._h5t["data/" + name].attrs.create("series", name.encode(),
                                               dtype=h5.special_dtype(vlen=bytes))

        # Update the replicate/series schema
        self._resolve_structure()

    def add_timeseries_ids(self, names):
        """ Adds empty timeseries to each of the data sets in the file.
            Once added through this function, timeseries can be set by
            id or index.
        """
        stored_names = self._h5t["timeseries/ids"][:]
        #stored_names = self.get_array_by_chunks("timeseries/ids")
        # Don't re-add duplicates
        new_names = [ x for x in names if x.encode() not in stored_names ]
        nts = self._h5t["timeseries/ids"].shape[0]
        self._resize_data(nts + len(new_names))
        self._h5t["timeseries/ids"][nts:] = [ x.encode() for x in new_names ]
        self.nts += len(names)
        return self.nts

    def get_timeseries_data(self, name_or_index, data, series=None, replicate=None):
        if (series is not None) & (replicate is not None):
            target = "%s_%s" % (series, replicate)
        elif (series is not None):
            target = series
        elif (replicate is not None):
            target = replicate
        else:
            target = "timeseries"

        if target not in self._h5t["data"]:
            raise IndexError("Series %s not found in data." % (target,))

        if type(name_or_index) is int:
            index = name_or_index
        else:
            index = self.get_timeseries_index(name_or_index)
            if index == -1:
                raise IndexError("Name %s not found in timeseries ids." % (name_or_index,))

        return self._h5t["data/%s/matrix" % (target,)][index, :]

    def get_timepoints(self, series=None, replicate=None):
        if (series == None) & (replicate == None):
            target = "timeseries"
        elif replicate:
            target = "%s_%s" % (series, replicate)
        elif series:
            target = series
        return [int(x) for x in self._h5t["data/%s/time" % (target,)][:]]

    def set_timeseries_data(self, name_or_index, data, series=None, replicate=None, action='add'):
        """ Add timeseries to a data set
        name is the feature/timeseries name (e.g., hash of the sequence)
        index is the index in the timeseries order where the data should be inserted
        data is the timeseries data, length of ntimepoints for the target
        target is the dataset (series + replicate)
        action: add or replace, if timeseries already exists
        """
        if (series is not None) & (replicate is not None):
            target = "%s_%s" % (series, replicate)
        elif (series is not None):
            target = series
        elif (replicate is not None):
            target =  replicate
        else:
            target = "timeseries"

        data_addr = "data/%s/matrix" % (target,)

        if type(name_or_index) is int:
            index = name_or_index
        else:
            index = self.get_timeseries_index(name_or_index)
            if index == -1:
                raise IndexError("Name %s not found in timeseries ids." % (name_or_index,))

        if (len(data) != self._h5t[data_addr].shape[1]):
            raise ValueError("Input data length %d does not match matrix rows, %d" 
                              % (len(data), self._h5t[data_addr].shape[1]))

        if action == 'add':
            self._h5t[data_addr][index, :] += data
        elif action == 'replace':
            self._h5t[data_addr][index, :] = data
        else:
            raise ValueError("Unknown action '%s', valid options: add, replace" % (action,))

    def _resolve_structure(self):
        #Resolve the structure of the file, specifically: return the series:replicate scheme
        # This helps us to aggregate the replicates and link the series as appropriate
        structure = {}
        for dataset_name in self._h5t["data"]:
            attrs = self._h5t["data/%s" % (dataset_name,)].attrs
            if "replicate" in attrs:
                replicate = attrs["replicate"]
            else:
                replicate = None
            if "series" in attrs:
                series = attrs["series"]
            else:
                series = dataset_name
            if replicate is not None:
                if series not in structure:
                    structure[series] = ["%s_%s" % (series, replicate)]
                else:
                    structure[series].append("%s_%s" % (series, replicate))
            else:
                structure[series] = [series]
        self.structure = structure

    @lru_cache(maxsize=1028)
    def get_timeseries_index(self, name):
        """ Gets the index of a timeseries by its name/identifier. Getting
            a timeseries by index is faster than getting it by its identifier,
            so get by index whenever possible.

            Calls to this method are cached to minimize repeated lookup times.

            Returns -1 if the name is not found in the index list.
        """
        res = np.where(self._h5t["timeseries/ids"][:] == name.encode())
        if len(res[0]) > 0:
            return res[0][0]
        else:
            return -1

    @lru_cache(maxsize=256)
    def get_sample_index(self, name, replicate_id = None, return_dataset = False):
        """ Gets the index of a sample by its name/identifier. If you have replicates
            in your data set, the replicate ID must be provided.

            If return_dataset is True, it will return a tuple including the name of the
            dataset as the first item and the index as the second. If return_dataset is 
            false, it will only return the index.
        """
        if replicate_id is not None:
            name = name + "_" + str(replicate_id)
        for dataset in self._h5t["data"]:
            names = self._h5t["data/" + dataset + "/names"][:]
            res = np.where(names == name.encode())
            if len(res[0]) > 0:
                if return_dataset:
                    return (str(dataset), res[0][0])
                else:
                    return res[0][0]
        #If not found, give -1
        if return_dataset:
            return (None, -1)
        else:
            return -1

    def filter_data(self, filter_method = 'min_sample_presence', threshold = 2):
        #TODO: Update this to go through each of the datasets rather than just the single time-series default
        #TODO: Make it remove the corresponding ids and other timeseries metadata as needed
        matrix = self._h5t["data/timeseries/matrix"]
        #Chunks for smoother HDF5 reading
        def chunks(N, nb):
            step = N / nb
            return [(round(step*i), round(step*(i+1))) for i in range(nb)]
        nrows, ncols = matrix.shape
        if filter_method is "min_sample_presence":
            def filter_function(row):
                return np.count_nonzero(row) < threshold
        elif filter_method is "min_sample_proportion":
            def filter_function(row):
                return np.count_nonzero(row)/len(row) < threshold
        else:
            raise ValueError("Unknown filter method '%s'." % (filter_method,))
        cursor = 0
        #Grab big chunks for efficiency
        for i, j in chunks(nrows, 10000):
            rows = matrix[i:j,:]
            for k in range(i,j):
                if not filter_function(rows[k-i,:]):
                    if k != cursor:
                        matrix[cursor, :] = rows[k-i,:]
                        cursor += 1
        matrix.resize(size=(cursor - 1, ncols))

    def _resize_data(self, nts):
        """Resizes the arrays in the HDF5 data file to have nts timeseries

        Parameters
        ----------
        nts: int (optional)
            number of time-series in the data set
        """
        for dataset_name in self._h5t["data"]:
            dataset = self._h5t["data/" + dataset_name + "/matrix"]
            dataset.resize((nts, dataset.shape[1]))
        for dataset_name in self._h5t["timeseries"]:
            shape = list(self._h5t["timeseries/" + dataset_name].shape)
            shape[0] = nts
            self._h5t["timeseries/" + dataset_name].resize(shape)

# Vestigial Stuff that may need to be ported into AnankeDB

class TimeSeriesData(object):
    """Class that represents an HDF5 data file on disk that contains Ananke
       time-series information, including the time-series for each sequence
       (as a sparse matrix), the sequence/gene metadata, and the 
       sample/timepoint metadata.

       Contains methods for manipulating and validating Ananke data files.
    """

    def add_taxonomy_data(self, input_filename):
        """Takes in a tab-separated file where the first column is the sequence
        hash, and the second column is the (semicolon-delimited) taxonomic
        classification, and stores it in the HDF5 file. The QIIME script 
        assign_taxonomy.py output format is compatible with this method. Use
        the unique sequences file from the 'tabulate' step for the
        classification.
    
        Parameters
        ----------
        input_filename: str
            location of the taxonomic classification file
        """
        sequence_to_tax = {}
        with open(input_filename, 'r') as in_file:
            for line in in_file:
                line = line.split("\t")
                sequence_to_tax[line[0].encode()] = line[1].strip()
        tax_list = np.empty(shape=self._h5t["timeseries/ids"].shape, 
                            dtype=object)
        for i, sequence_id in enumerate(self._h5t["timeseries/ids"]):
            try:
                tax_list[i] = sequence_to_tax[sequence_id]
            except KeyError:
                tax_list[i] = "NF"
        self.insert_array_by_chunks('timeseries/taxonomy', tax_list)
            

    def add_sequencecluster_data(self, input_filename):
        """Takes in the location of a tab-separated file where the first column
        is the OTU ID and the remaining columns are the sequence IDs (i.e., 
        hashes). This is the same format as QIIME outputs from pick_otus.py.
    
        Parameters
        ----------
        input_filename: str
            location of the sequence cluster file
        """
        sequence_to_cluster = {}
        with open(input_filename, 'r') as in_file:
            for line in in_file:
                line = line.split("\t")
                cluster_num = line[0]
                line = line[1:]
                for sequence in line:
                    sequence = sequence.strip().encode()
                    sequence_to_cluster[sequence] = cluster_num
        cluster_list = np.empty(shape=self._h5t["timeseries/ids"].shape,
                                dtype=object)
        for i, sequence_id in enumerate(self._h5t["timeseries/ids"][:]):
            try:
                cluster_list[i] = sequence_to_cluster[sequence_id]
            except KeyError:
                cluster_list[i] = "NF"
        self.insert_array_by_chunks('timeseries/altclusters', cluster_list)

    def insert_array_by_chunks(self, target, array, 
                               transform_func = lambda x: str(x).encode(),
                               chunk_size = 1000):
        """Inserts a list/array into the HDF5 file at the given data set
        location. Do this in chunks to prevent excessive memory usage by h5py.
    
        Parameters
        ----------
        target: str
            HDF5 data set location (e.g., "samples/time")
        array: list
            list of values to be inserted into HDF5 file
        transform_func: function
            function that coerces the values in array to the desired data type
            for the given HDF5 target data set. Default is to coerce to
            bitstrings.
        chunk_size: int
            the number of items from array to insert at one time. Default is
            1000.
        """
        n = self._h5t[target].shape[0]
        chunks = np.append(np.arange(0, n, chunk_size), n)
        for i in range(len(chunks)-1):
            #  Don't overrun the source array
            #  and only fill the front of the target
            if (chunks[i+1] >= len(array)):
                self._h5t[target][chunks[i]:len(array)] = \
                     [ transform_func(x) for x in array[chunks[i]:len(array)] ]
                break
            else:
                self._h5t[target][chunks[i]:chunks[i+1]] = \
                    [ transform_func(x) for x in array[chunks[i]:chunks[i+1]] ]
    
    def get_array_by_chunks(self, target, start=None, end=None, chunk_size = 1000):
        """Fetches an array from the HDF5 file in chunk_size chunks. This helps
        prevent memory usage issues with HDF5.
    
        Parameters
        ----------
        target: str
            HDF5 data set location (e.g., "samples/time")
        chunk_size: int
            the number of items from array to fetch at one time. Default is 
            1000.
    
        Returns
        -------
        arr: list
            the complete array from target location in HDF5 file
        """
        if (start is not None) & (end is not None):
            arr_size = end - start
        else:
            arr_size = self._h5t[target].shape[0]
            start = 0
            end = arr_size + 1
        arr = np.empty(arr_size,
                       dtype=self._h5t[target].dtype)
        chunks = list(range(start, end, chunk_size))
        if chunks[-1] != arr.shape[0]:
            chunks = chunks + [arr.shape[0]]
        for i,j in zip(chunks[0:-1], chunks[1:]):
            arr[i:j] = self._h5t[target][i:j]
        return arr
