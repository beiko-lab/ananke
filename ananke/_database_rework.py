import h5py as h5
import numpy as np
import pandas as pd
from functools import lru_cache

import arrow

from .__init__ import __version__


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
 

class TimeSeriesData(object):
    """Class that represents an HDF5 data file on disk that contains Ananke
       time-series information, including the time-series for each sequence
       (as a sparse matrix), the sequence/gene metadata, and the 
       sample/timepoint metadata.

       Contains methods for manipulating and validating Ananke data files.
    """

    def __init__(self, h5_file_path):
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
        # Create the new file, if required
        h5t = h5.File(h5_file_path)
        self._h5t = h5t
        # Create the required datasets (initialize empty)
        
        if "origin_version" in self._h5t.attrs:
            origin_version = self._h5t.attrs["origin_version"]
            if not version_greater_than(origin_version, "0.3.0"):
                raise ImplementationError("Ananke version 0.X files are not compatible with Ananke 1.0 at this time")
            else:
                #Any existing-file-loading steps should be done here; none right now
                pass
        else:
            # Create a new file if origin_version doesn't exist
            self._h5t.attrs.create("origin_version", str(__version__),
                                   dtype=h5.special_dtype(vlen=bytes))

             
            print("Creating required data sets in new HDF5 file at %s" % 
                                                            (h5_file_path, ))
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
        """Destructor for TimeSeriesData object

        Closes the connection to the HDF5 file
        """
        #if self._h5t:
        #    self._h5t.close()
        pass

    def __str__(self):
        info = "Origin version: %s" % \
               (self._h5t.attrs["origin_version"].decode(),) + "\n"
        n_datasets = len(self._h5t["data"])
        if (n_datasets == 0):
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

    def create_series(self, name, time, sample_names):
        self._h5t.create_group("data/" + name)
        self._h5t.create_dataset("data/" + name + "/matrix",
                                 dtype=np.int32,
                                 shape=(0,len(time)), 
                                 maxshape=(None, len(time)),
                                 compression="gzip",
                                 fillvalue=0)
        self._h5t.create_dataset("data/" + name + "/time",
                                 dtype=h5.special_dtype(vlen=bytes),
                                 data=[str(x).encode() for x in time])
        self._h5t.create_dataset("data/" + name + "/names",
                                 dtype=h5.special_dtype(vlen=bytes),
                                 data=[str(x).encode() for x in sample_names])

    def register_timeseries(self, names):
        nts = self._h5t["timeseries/ids"].shape[0]
        print(self._h5t["timeseries/ids"].shape)
        self.resize_data(nts + len(names))
        self._h5t["timeseries/ids"][nts:] = [ x.encode() for x in names ]
        return nts + len(names)

    @lru_cache(maxsize=256)
    def get_sample_index(self, name, return_dataset = False):
        # Get the column index from a sample name
        for dataset in self._h5t["data"]:
            names = self._h5t["data/" + dataset + "/names"][:]
            res = np.where(names == name.encode())
            if len(res[0]) > 0:
                if return_dataset:
                    return (str(dataset), res[0][0])
                else:
                    return res[0][0]
        #If not found, give -1
        return (None, -1)

    @lru_cache(maxsize=1028)
    def get_timeseries_index(self, name):
        res = np.where(self._h5t["timeseries/ids"][:] == name.encode())
        if len(res[0]) > 0:
            return res[0][0]
        else:
            return -1

    def set_timeseries_data(self, data, series=None, replicate=None, name=None, index=None, action='add'):
        #Add timeseries to a data set
        #name is the feature/timeseries name (e.g., hash of the sequence)
        #index is the index in the timeseries order where the data should be inserted
        #data is the timeseries data, length of ntimepoints for the target
        #target is the dataset (series + replicate)
        #action: add or replace, if timeseries already exists
        if (series is not None) & (replicate is not None):
            target = "data/%s_%s/matrix" % (series, replicate)
        elif (series is not None):
            target = "data/%s/matrix" % (series,)
        elif (replicate is not None):
            target = "data/%s/matrix" % (replicate,)
        else:
            target = "data/timeseries/matrix"

        if (len(data) != self._h5t[target].shape[1]):
            raise ValueError("Input data length %d does not match matrix rows, %d" 
                              % (len(data), self._h5t[target].shape[1]))
        if (name is None) and (index is None):
            raise ValueError("Must supply one of 'name' or 'index' to insert into data matrix.")
        if index is None:
            index = self.get_timeseries_index(name)
        if (index == -1) | (index >= self._h5t["timeseries/ids"].shape[0]):
            raise ValueError("Time series not in data, must be registered.")
        if action == 'add':
            self._h5t[target][index, :] += data
        elif action == 'replace':
            self._h5t[target][index, :] = data
        else:
            raise ValueError("Unknown action '%s', valid options: add, replace" % (action,))

    # TODO: Update to accept lists of length nseries for timepoints
    def initialize_by_shape(self, timepoints=np.arange(10), nseries = 1, nreplicates = 1):
        names = []
        for i in np.arange(nseries):
            for j in np.arange(nreplicates):
                if nseries > 1:
                    series_name = "timeseries%d" % (i,)
                    series_rep_name = series_name + "_%d" % (j,)
                else:
                    series_name = "timeseries"
                    series_rep_name = series_name
                names.append(series_rep_name)
                self.create_series(series_rep_name, timepoints, 
                                   [ series_rep_name + "_S%d" % (j,) for j in np.arange(len(timepoints)) ]
                                  )
                if nseries > 1:
                    self._h5t["data/" + series_rep_name].attrs.create("series", series_name.encode(),
                                                           dtype=h5.special_dtype(vlen=bytes))
                if nreplicates > 1:
                    self._h5t["data/" + series_rep_name].attrs.create("replicate", str(j).encode(),
                                                       dtype=h5.special_dtype(vlen=bytes))
        return names

    def initialize_from_metadata(self, metadata_path, name_col, 
                                 time_col, time_format="X",
                                 replicate_col=None, series_col=None):
        """Take in a path, metadata_path, read file into a pandas dataframe.
        Validate that time_name and time_mask (if applicable) are present.

        Parameters
        ----------
        metadata_path: str
            filepath to metadata file
        name_col: str
            name of column in metadata file that contains the sample names
        time_col: str
            name of column in metadata file that contains time points
        time_format: str
            format of the timestamp, e.g. "MM/DD/YYYY". If left blank, then
            it will treat times as integers.
        replicate_col: str
            name of column in metadata file that contains the replicate category 
            that denotes the replicate the sample belongs to
        series_col: str
            name of column in metadata file that contains the series category
            that denotes the time series the sample belongs to

        Returns
        -------
        mm: pandas.DataFrame
            Pandas dataframe containing sample metadata
        """

        mm = pd.read_csv(metadata_path, sep="\t", header=0)
       
        if time_col not in mm:
            raise KeyError("Specified time point column name (%s) is not found " \
                           "in metadata file." % (time_col,))

        if name_col not in mm:
            raise KeyError("Specific name column (%s) is not found in " \
                           "metadata file." % (name_col,))

        parsed_times = [arrow.get(str(x), time_format) for x in mm[time_col]]
        time_offsets = [(x - min(parsed_times)).total_seconds() for x in parsed_times]
        mm["_offsets"] = time_offsets

        if replicate_col is not None:
            if replicate_col not in mm:
                raise KeyError("Specified replicate column name (%s) is not " \
                               "found in metadata file." % (replicate_col,))
        if series_col is not None:
            if series_col not in mm:
                raise KeyError("Specified series column name (%s) is not " \
                               "found in metadata file." % (series_col,))

        sort_order = [x for x in [series_col, replicate_col, "_offsets"] \
                      if x is not None]
        mm = mm.sort_values(by=sort_order)

        if len(sort_order) == 3:
            #We have the trifecta: replicates, multi timeseries, and the offsets
            for series in mm[series_col].unique():
                series_subset = mm[mm[series_col == series]]
                for replicate in series_subset[replicate_col].unique():
                    time = series_subset[
                           series_subset[replicate_col] == replicate, time_col]
                    name = str(series) + "_" + str(replicate)
                    sample_names = series_subset[name_col]
                    self.create_series(name, time, sample_names)
                                             
                    self._h5t["data/" + name].attrs.create("replicate", replicate.encode(),
                                                           dtype=h5.special_dtype(vlen=bytes))
                    self._h5t["data/" + name].attrs.create("series", series.encode(),
                                                           dtype=h5.special_dtype(vlen=bytes))

        elif len(sort_order) == 2:
            #We only have two to deal with either series or replicate
            for rep_or_series in mm[sort_order[0]].unique():
                subset = mm[mm[sort_order[0]] == rep_or_series]
                time = np.array(subset[time_col], dtype=str)
                name = str(rep_or_series)
                sample_names = subset[name_col]
                self.create_series(name, time, sample_names)

                if replicate_col is not None:
                    att_str = "replicate"
                else:
                    att_str = "series"

                self._h5t["data/" + name].attrs.create(att_str, name.encode(),
                                          dtype=h5.special_dtype(vlen=bytes))
        else:
            #We only have one to deal with
            time = mm[time_col]
            sample_names = mm[name_col]
            self.create_series("timeseries", time, sample_names)


    def resize_data(self, nts):
        """Resizes the arrays in the HDF5 data file. If any size parameter
        is not provided, it is inferred from the previous data shape.

        Parameters
        ----------
        nts: int (optional)
            number of time-series in the data set
        nsamples: int (optional)
            number of samples/time-points in the data set
        """
        for dataset_name in self._h5t["data"]:
            dataset = self._h5t["data/" + dataset_name + "/matrix"]
            dataset.resize((nts, dataset.shape[1]))
        for dataset_name in self._h5t["timeseries"]:
            shape = list(self._h5t["timeseries/" + dataset_name].shape)
            shape[0] = nts
            self._h5t["timeseries/" + dataset_name].resize(shape)
        
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

    def get_epsilon_index(self, epsilon):
        cluster_attrs = self._h5t["timeseries/clusters"].attrs
        p_min = cluster_attrs["param_min"]
        p_max = cluster_attrs["param_max"]
        p_step = cluster_attrs["param_step"]
        p_index = int(round((epsilon - p_min) / p_step))
        return p_index 

    def insert_cluster(self, indices, cluster_num, epsilon):
        p_index = self.get_epsilon_index(epsilon)
        indices = np.sort(np.unique(indices))
        self._h5t["timeseries/clusters"][indices, p_index] = str(cluster_num)

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
       
#    def filter_data(self, outfile, filter_type='presence', threshold=0.1):
#        """Creates a new HDF5 file containing a subset of the data that match
#        the filter criterion. This is used to reduce large data sets so that
#        fewer time series are included, which can help with computation 
#        efficiency downstream.
#    
#        Parameters:
#        -----------
#        outfile: str
#            location of new HDF5 data file. This will be created.
#        filter_type: str
#            one of "proportion", "abundance", or "presence". This affects the
#            interpretation of the threshold parameter. For "proportion", a 
#            sequence must contain threshold % of the data. For "abundance", a
#            sequence must have been recorded at least threshold times. For 
#            "presence", a sequence must have been present (>0 abundance) in at 
#            least threshold % of time points.
#        threshold: float
#            the cut-off for the filter, the meaning of this changes depending
#            on filter_type.
#        """
#        pass
