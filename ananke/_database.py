import warnings
from os import getcwd

import h5py as h5
import numpy as np
from scipy.sparse import csr_matrix
from .__init__ import __version__

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
        h5_table = h5.File(h5_file_path, 'a')
        self.h5_table = h5_table
        # Create the required datasets (initialize empty)
        if set(self.h5_table.keys()) != {"genes", "timeseries", "samples"}:
            self.filled_data = False
            print("Creating required data sets in new HDF5 file at %s" % 
                                                            (h5_file_path, ))
            h5_table.create_group("timeseries")
            h5_table["timeseries"].require_dataset("data", shape=(1,), 
                                                    dtype=np.dtype('int32'), 
                                                    maxshape=(None,))
            h5_table["timeseries"].require_dataset("indices", shape=(1,), 
                                                   dtype=np.dtype('int32'), 
                                                   maxshape=(None,))
            h5_table["timeseries"].require_dataset("indptr", shape=(1,), 
                                                   dtype=np.dtype('int32'), 
                                                   maxshape=(None,))

            # Keeps track of which version created the file
            h5_table.attrs.create("origin_version", str(__version__), 
                                  dtype=h5.special_dtype(vlen=bytes))

            # Create genes group
            h5_table.create_group("genes")
            h5_table["genes"].require_dataset("sequences", shape=(1,), 
                              dtype=h5.special_dtype(vlen=bytes), 
                              maxshape=(None,), exact=False)
            h5_table["genes"].require_dataset("sequenceids", shape=(1,), 
                              dtype=h5.special_dtype(vlen=bytes), 
                              maxshape=(None,), exact=False)
            h5_table["genes"].require_dataset("clusters", shape=(1,1), 
                              dtype=h5.special_dtype(vlen=bytes), 
                              maxshape=(None,None), exact=False, fillvalue=-2)
            h5_table["genes"].require_dataset("taxonomy", shape=(1,), 
                              dtype=h5.special_dtype(vlen=bytes), 
                              maxshape=(None,), exact=False)
            h5_table["genes"].require_dataset("sequenceclusters", shape=(1,), 
                              dtype=h5.special_dtype(vlen=bytes), 
                              maxshape=(None,), exact=False)

            #Fill some arrays with ghost values so rhdf5 doesn't segfault
            self.fill_array("genes/taxonomy", b"NF")
            self.fill_array("genes/sequenceclusters", b"NF")
            
            h5_table.create_group("samples")
            h5_table["samples"].require_dataset("names", shape=(1,), 
                                dtype=h5.special_dtype(vlen=bytes), 
                                maxshape=(None,), exact=False)
            h5_table["samples"].require_dataset("time", shape=(1,), 
                                dtype=np.dtype('int32'), 
                                maxshape=(None,), exact=False)
            h5_table["samples"].require_dataset("metadata", shape=(1,), 
                                dtype=h5.special_dtype(vlen=bytes), 
                                maxshape=(None,), exact=False)
            h5_table["samples"].require_dataset("mask", shape=(1,), 
                                dtype=h5.special_dtype(vlen=bytes), 
                                maxshape=(None,), exact=False)
        else:
            self.filled_data = True
        #These indices help us populate the timeseries sparse matrix
        # used by the add_timeseries_data method so that data can be added
        # piece-wise and never has to be stored fully in RAM
        self._ts_data_index = 0
        self._ts_indptr_index = 0

    def __del__(self):
        """Destructor for TimeSeriesData object

        Closes the connection to the HDF5 file
        """
        #if self.h5_table:
        #    self.h5_table.close()
        pass

    def version_greater_than(self, version):
        """Compares version numbers

        Parameters
        ----------
        version: str
            version string, triple (major, minor, release)

        Returns
        -------
        boolean
            True if TimeSeriesData object is greater than provided string
        """
        if "origin_version" not in self.h5_table.attrs:
            return False
        origin_version = self.h5_table.attrs["origin_version"]
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
        
    def resize_data(self, ngenes=None, nsamples=None, nobs=None):
        """Resizes the arrays in the HDF5 data file. If any size parameter
        is not provided, it is inferred from the previous data shape.

        Parameters
        ----------
        ngenes: int (optional)
            number of genes/time-series in the data set
        nsamples: int (optional)
            number of samples/time-points in the data set
        nobs: int (optional)
            number of non-zero observations in the data set
        """
        #If any of the parameters are not set, try to infer them
        if (ngenes == None):
            ngenes = self.h5_table["timeseries/indptr"].shape[0] - 1
        if (nsamples == None):
            nsamples = self.h5_table["samples/time"].shape[0]
        if (nobs == None):
            nobs = self.h5_table["timeseries/data"].shape[0]
        #Note: if this shrinks the data, it will truncate it
        self.h5_table["timeseries/data"].resize((nobs,))
        self.h5_table["timeseries/indices"].resize((nobs, ))
        self.h5_table["timeseries/indptr"].resize((ngenes + 1, ))
        self.h5_table["genes/sequences"].resize((ngenes, ))
        self.h5_table["genes/sequenceids"].resize((ngenes, ))
        self.h5_table["genes/clusters"].resize((ngenes, 20))
        self.h5_table["genes/taxonomy"].resize((ngenes, ))
        self.h5_table["genes/sequenceclusters"].resize((ngenes, ))
        self.h5_table["samples/names"].resize((nsamples, ))
        self.h5_table["samples/time"].resize((nsamples, ))
        self.h5_table["samples/metadata"].resize((nsamples, ))
        if self.version_greater_than("0.1.0"):
            self.h5_table["samples/mask"].resize((nsamples, ))
        self.fill_array("genes/taxonomy", b"NF")
        self.fill_array("genes/sequenceclusters", b"NF")

    def fill_array(self, target, value, chunk_size = 1000):
        """Fill the target HDF5 array with a single value. Useful for 
        initializing an array, since the rhdf5 package tends to segfault if you
        load an uninitialized data set.

        Parameters
        ----------
        target: str
            the location of the HDF5 array, e.g., "samples/time"
        value: any
            the value to fill the array with
        chunk_size: int
            the number of items to insert at a time. This only needs to be
            increased for very large data sets.
        """
        n = self.h5_table[target].shape[0]
        chunks = np.append(np.arange(0, n, chunk_size), n)
        for i in range(len(chunks)-1):
            self.h5_table[target][chunks[i]:chunks[i+1]] = (
                                            [value]*(chunks[i+1] - chunks[i]) )

    def add_timeseries_data(self, data, indices, indptr, sequences):
        """Adds, in chunks, the time-series data in compressed sparse row (csr)
        format. This can be invoked multiple times as the data are tabulated,
        so that the entire csr matrix never has to be in RAM all at once. The
        matrix is inserted row-wise.
    
        Parameters
        ----------
        data: list of numeric
            a list of the non-zero observations in the csr matrix
        indices: list of int
            a list of the column/sample indexes that correspond to the entries 
            in the 'data' array
        indptr: list of int
            a list of the positions in the data and indices lists that
            indicates the start of each row
        sequences: list of str
            a list of the sequence IDs (i.e., hashes) that correspond to each
            row
        """
        #TODO: Check that incoming data size are compatible with
        # HDF5 dataset size
        if not self.filled_data:
            self._ts_data_index = 0
            self._ts_indptr_index = 0
            self.filled_data = True
            data_size = self.h5_table["timeseries/data"].shape[0]
            self.h5_table["timeseries/indptr"][-1] = data_size
        self.h5_table["timeseries/data"][
                      self._ts_data_index:self._ts_data_index + len(data)
                     ] = data
        self.h5_table["timeseries/indices"][
                      self._ts_data_index:self._ts_data_index + len(indices)
                     ] = indices
        self.h5_table["timeseries/indptr"][
                      self._ts_indptr_index:self._ts_indptr_index + len(indptr)
                     ] = indptr
        b_sequences = [str(x).encode() for x in sequences]
        self.h5_table["genes/sequenceids"][
                      self._ts_indptr_index:self._ts_indptr_index + 
                      len(sequences) ] = b_sequences
        self._ts_data_index += len(data)
        self._ts_indptr_index += len(indptr)

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
        tax_list = np.empty(shape=self.h5_table["genes/sequenceids"].shape, 
                            dtype=object)
        for i, sequence_id in enumerate(self.h5_table["genes/sequenceids"]):
            try:
                tax_list[i] = sequence_to_tax[sequence_id]
            except KeyError:
                tax_list[i] = "NF"
        self.insert_array_by_chunks('genes/taxonomy', tax_list)
            

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
        cluster_list = np.empty(shape=self.h5_table["genes/sequenceids"].shape,
                                dtype=object)
        for i, sequence_id in enumerate(self.h5_table["genes/sequenceids"][:]):
            try:
                cluster_list[i] = sequence_to_cluster[sequence_id]
            except KeyError:
                cluster_list[i] = "NF"
        self.insert_array_by_chunks('genes/sequenceclusters', cluster_list)

    def get_epsilon_index(self, epsilon):
        cluster_attrs = self.h5_table["genes/clusters"].attrs
        p_min = cluster_attrs["param_min"]
        p_max = cluster_attrs["param_max"]
        p_step = cluster_attrs["param_step"]
        p_index = int(round((epsilon - p_min) / p_step))
        return p_index 

    def insert_cluster(self, indices, cluster_num, epsilon):
        p_index = self.get_epsilon_index(epsilon)
        indices = np.sort(np.unique(indices))
        self.h5_table["genes/clusters"][indices, p_index] = str(cluster_num)

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
        n = self.h5_table[target].shape[0]
        chunks = np.append(np.arange(0, n, chunk_size), n)
        for i in range(len(chunks)-1):
            #  Don't overrun the source array
            #  and only fill the front of the target
            if (chunks[i+1] >= len(array)):
                self.h5_table[target][chunks[i]:len(array)] = \
                     [ transform_func(x) for x in array[chunks[i]:len(array)] ]
                break
            else:
                self.h5_table[target][chunks[i]:chunks[i+1]] = \
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
            arr_size = self.h5_table[target].shape[0]
            start = 0
            end = arr_size + 1
        arr = np.empty(arr_size,
                       dtype=self.h5_table[target].dtype)
        chunks = list(range(start, end, chunk_size))
        if chunks[-1] != arr.shape[0]:
            chunks = chunks + [arr.shape[0]]
        for i,j in zip(chunks[0:-1], chunks[1:]):
            arr[i:j] = self.h5_table[target][i:j]
        return arr
        
    def get_sparse_matrix(self, chunk_size = 1000):
        """Fetches the time-series data matrix in compressed sparse row (csr)
        format. Does this in chunks to prevent memory usage issues.
    
        Parameters
        ----------
        chunk_size: int
            the number of items to fetch at one time. Default is 1000.
    
        Returns
        -------
        scipy.sparse.csr_matrix
            csr matrix object containing sequences/time-series as rows, samples
            /time-points as columns
        """
        data = np.empty(self.h5_table["timeseries/data"].shape)
        indices = np.empty(self.h5_table["timeseries/indices"].shape)
        indptr = np.empty(self.h5_table["timeseries/indptr"].shape)       
        chunks = list(range(0, data.shape[0], chunk_size))
        if chunks[-1] != data.shape[0]:
            chunks = chunks + [data.shape[0]]
        for i,j in zip(chunks[0:-1], chunks[1:]):
            self.h5_table["timeseries/data"].read_direct(data, np.s_[i:j],
                                                               np.s_[i:j])       
        chunks = list(range(0, indices.shape[0], chunk_size))
        if chunks[-1] != indices.shape[0]:
            chunks = chunks + [indices.shape[0]]
        for i,j in zip(chunks[0:-1], chunks[1:]):
            self.h5_table["timeseries/indices"].read_direct(indices,
                                                            np.s_[i:j],
                                                            np.s_[i:j])       
        chunks = list(range(0, indptr.shape[0], chunk_size))
        if chunks[-1] != indptr.shape[0]:
            chunks = chunks + [indptr.shape[0]]
        for i,j in zip(chunks[0:-1], chunks[1:]):
            self.h5_table["timeseries/indptr"].read_direct(indptr,
                                                           np.s_[i:j],
                                                           np.s_[i:j])
        return csr_matrix((data, indices, indptr))
        
    def get_mask(self):
        """Fetches the mask values that indicate which time-series a sample
        belongs to, if there are multiple time-series present.
    
        Returns
        -------
        list of x: list of values with a unique value for each time-series
        """
        if self.version_greater_than("0.1.0"):
            return self.h5_table["samples/mask"][:]
        else:
            #We didn't support multi time-series, so return a dummy mask
            return [1]*len(self.h5_table["samples/names"])

    def get_cluster_labels(self, i):
        """Fetches the time-series clusters at a given epsilon parameter index
    
        Parameters
        ----------
        i: int
            index of interest for the desired clustering parameter epsilon
    
        Returns
        -------
        list of int: list of time-series cluster IDs
        """
        return self.h5_table["genes/clusters"][:,i]

    def filter_data(self, outfile, filter_type='presence', threshold=0.1):
        """Creates a new HDF5 file containing a subset of the data that match
        the filter criterion. This is used to reduce large data sets so that
        fewer time series are included, which can help with computation 
        efficiency downstream.
    
        Parameters:
        -----------
        outfile: str
            location of new HDF5 data file. This will be created.
        filter_type: str
            one of "proportion", "abundance", or "presence". This affects the
            interpretation of the threshold parameter. For "proportion", a 
            sequence must contain threshold % of the data. For "abundance", a
            sequence must have been recorded at least threshold times. For 
            "presence", a sequence must have been present (>0 abundance) in at 
            least threshold % of time points.
        threshold: float
            the cut-off for the filter, the meaning of this changes depending
            on filter_type.
        """
        #New data structures
        filtered_data = []
        filtered_indices = []
        #Starts off with 0 as the starting index
        filtered_indptr = [0]
        filtered_sequences = []
        row_data = None

        print("Loading unfiltered data...")
        #Get the existing data matrix
        sparse_matrix = self.get_sparse_matrix()
        nrows = sparse_matrix.shape[0]
        #Coerce to float to ensure division doesn't implicitly round
        ncols = float(sparse_matrix.shape[1])
        threshold = float(threshold)

        #Pre-compute this value and store it
        if filter_type == 'proportion':
            total_sum = float(sparse_matrix.sum())

        print("Filtering genes by method '%s' using threshold: %f" % \
                                                  (filter_type, threshold))
        for row_index in range(0, nrows):
            add_data = False
            row_data = sparse_matrix[row_index,:]
            if filter_type == 'proportion':
                if row_data.sum()/total_sum > threshold:
                    add_data = True
            elif filter_type == 'abundance':
                if row_data.sum() >= threshold:
                    add_data = True
            elif filter_type == 'presence':
                presence_proportion = row_data.nnz/ncols
                if presence_proportion >= threshold:
                    add_data = True

            if add_data:
                filtered_sequences.append(
                                 self.h5_table["genes/sequenceids"][row_index]
                                         )
                filtered_data.extend(row_data.data)
                filtered_indptr.append(len(filtered_data))
                filtered_indices.extend(row_data.indices)

        # Raise error is there's no data left
        if len(filtered_data) == 0:
            raise ValueError("Warning: All data filtered. " \
                             "Consider relaxing filter criterion.")

        print("Writing to output file %s" % (outfile,))
        filtered_tsdb = TimeSeriesData(outfile)
        ngenes = len(filtered_indptr) - 1
        nsamples = ncols
        nobs = len(filtered_data)
        sample_name_array = self.get_array_by_chunks("samples/names")
        time_points = self.get_array_by_chunks("samples/time")
        mask = self.get_mask()

        # Check for columns with zero counts
        # This can be done by checking the indices array
        col_range = np.arange(0, ncols)
        missing_indices = np.setdiff1d(col_range, filtered_indices)
        missing_indices = [ int(x) for x in missing_indices ]
        nsamples = nsamples - len(missing_indices)
        if len(missing_indices) > 0:
            missing_samples = sample_name_array[missing_indices]
            warnings.warn("Samples missing after filtering: %s" % 
                          str(missing_samples))
            # Invert the missing indices to get the kept indices
            keep_cols = ~np.in1d(col_range, missing_indices)
            sample_name_array = sample_name_array[keep_cols]
            time_points = time_points[keep_cols]
            mask = mask[keep_cols]

        filtered_tsdb.resize_data(ngenes, nsamples, nobs)
        filtered_tsdb.insert_array_by_chunks("samples/names", 
                                             sample_name_array,
                                             transform_func = lambda x: x)
        
        filtered_tsdb.insert_array_by_chunks("samples/time",
                                             time_points,
                                             transform_func = float)
        
        filtered_tsdb.insert_array_by_chunks("samples/mask",
                                             mask,
                                             transform_func = lambda x: x)
        
        # We only need to modify the indices array to remove empty cols
        # Reduce the indexes higher than the missing ones by 1
        filtered_indices = np.array(filtered_indices)
        for ind in missing_indices:
            filtered_indices[np.where(filtered_indices > ind)] = \
            filtered_indices[np.where(filtered_indices > ind)] - 1
    
        # Put into HDF5 file by chunks for memory reasons
        filtered_tsdb.insert_array_by_chunks("timeseries/data",
                                             filtered_data,
                                             transform_func = np.int32)
        filtered_tsdb.insert_array_by_chunks("timeseries/indptr",
                                             filtered_indptr,
                                             transform_func = np.int32)
        filtered_tsdb.insert_array_by_chunks("timeseries/indices",
                                             filtered_indices,
                                             transform_func = np.int32)
        filtered_tsdb.insert_array_by_chunks("genes/sequenceids",
                                             filtered_sequences,
                                             transform_func = lambda x: x)
