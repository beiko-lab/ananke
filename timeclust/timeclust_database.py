import h5py as h5
import numpy as np
from scipy.sparse import csr_matrix
from os import getcwd

class TimeSeriesData(object):
    def __init__(self, h5_file):
        #Create the new file
        h5_table = h5.File(h5_file, 'a')
        #Open or create the required groups:
        #Create the required datasets (initialize empty)
        if "timeseries" not in h5_table:
            h5_table.create_group("timeseries")
            h5_table["timeseries"].require_dataset("data", shape=(1,), dtype=np.dtype('int32'), maxshape=(None,))
            h5_table["timeseries"].require_dataset("indices", shape=(1,), dtype=np.dtype('int32'), maxshape=(None,))
            h5_table["timeseries"].require_dataset("indptr", shape=(1,), dtype=np.dtype('int32'), maxshape=(None,))
            self.filled_data = False
        else:
            #TODO: More sophisticated check of what's populated
            self.filled_data = True
        if "genes" not in h5_table:
            h5_table.create_group("genes")
            h5_table["genes"].require_dataset("sequences", shape=(1,), dtype=h5.special_dtype(vlen=bytes), maxshape=(None,), exact=False)
            h5_table["genes"].require_dataset("sequenceids", shape=(1,), dtype=h5.special_dtype(vlen=bytes), maxshape=(None,), exact=False)
            h5_table["genes"].require_dataset("clusters", shape=(1,1), dtype=h5.special_dtype(vlen=bytes), maxshape=(None,None), exact=False)
            h5_table["genes"].require_dataset("taxonomy", shape=(1,), dtype=h5.special_dtype(vlen=bytes), maxshape=(None,), exact=False)
            h5_table["genes"].require_dataset("sequenceclusters", shape=(1,), dtype=h5.special_dtype(vlen=bytes), maxshape=(None,), exact=False)
        if "samples" not in h5_table:
            h5_table.create_group("samples")
            h5_table["samples"].require_dataset("names", shape=(1,), dtype=h5.special_dtype(vlen=bytes), maxshape=(None,), exact=False)
            h5_table["samples"].require_dataset("time", shape=(1,), dtype=np.dtype('int32'), maxshape=(None,), exact=False)
            h5_table["samples"].require_dataset("metadata", shape=(1,), dtype=h5.special_dtype(vlen=bytes), maxshape=(None,), exact=False)
        self.h5_table = h5_table
        #These indices help us populate the timeseries sparse matrix
        self.ts_data_index = 0
        self.ts_indptr_index = 0

    def __del__(self):
        print("Closing file")
        self.h5_table.close()

    def resize_data(self, ngenes=None, nsamples=None, nobs=None):
        #If any of the parameters are not set, try to infer them
        if (ngenes == None):
            ngenes = self.h5_table["timeseries/indptr"].shape[0]-1
        if (nsamples == None):
            nsamples = self.h5_table["samples/time"].shape[0]
        if (nobs == None):
            nobs = self.h5_table["timeseries/data"].shape[0]
        #Note: if this shrinks the data, it will truncate it
        self.h5_table["timeseries/data"].resize((nobs,))
        self.h5_table["timeseries/indices"].resize((nobs,))
        self.h5_table["timeseries/indptr"].resize((ngenes+1,))
        self.h5_table["genes/sequences"].resize((ngenes,))
        self.h5_table["genes/sequenceids"].resize((ngenes,))
        self.h5_table["genes/clusters"].resize((ngenes,20))
        self.h5_table["genes/taxonomy"].resize((ngenes,))
        self.h5_table["genes/sequenceclusters"].resize((ngenes,))
        self.h5_table["samples/names"].resize((nsamples,))
        self.h5_table["samples/time"].resize((nsamples,))
        self.h5_table["samples/metadata"].resize((nsamples,))
        #Fill some arrays with ghost values so rhdf5 doesn't segfault
        self.fill_array("genes/taxonomy","None")
        self.fill_array("genes/sequenceclusters","None")

    def fill_array(self, target, value):
        n = self.h5_table[target].shape[0]
        chunks = np.append(np.arange(0,n,1000), n)
        for i in range(len(chunks)-1):
            self.h5_table[target][chunks[i]:chunks[i+1]] = [value]*(chunks[i+1]-chunks[i])

    def add_names(self, names_list):
        self.h5_table["samples/names"][:] = names_list

    def add_timepoints(self, timepoints_list):
        self.h5_table["samples/time"][:] = timepoints_list

    def add_timeseries_data(self, data, indices, indptr, sequences):
        if not self.filled_data:
            self.ts_data_index = 0
            self.ts_indptr_index = 0
            self.filled_data = True
            self.h5_table["timeseries/indptr"][-1] = self.h5_table["timeseries/data"].shape[0]
        self.h5_table["timeseries/data"][self.ts_data_index:self.ts_data_index+len(data)] = data
        self.h5_table["timeseries/indices"][self.ts_data_index:self.ts_data_index+len(indices)] = indices
        self.h5_table["timeseries/indptr"][self.ts_indptr_index:self.ts_indptr_index+len(indptr)] = indptr
        self.h5_table["genes/sequenceids"][self.ts_indptr_index:self.ts_indptr_index+len(sequences)] = sequences
        self.ts_data_index += len(data)
        self.ts_indptr_index += len(indptr)

    def add_taxonomy_data(self, input_filename):
        #Input: tab-separated file, first column is sequence hash (ie, unique sequence identifier
        #       second column is the taxonomic classification, third column is posterior probability (ignored))
        sequence_to_tax = {}
        with open(input_filename, 'r') as in_file:
            for line in in_file:
                line = line.split("\t")
                sequence_to_tax[line[0]] = line[1]
        tax_list = np.empty(shape=self.h5_table["genes/sequenceids"].shape, dtype=object)
        for i, sequence_id in enumerate(self.h5_table["genes/sequenceids"]):
            try:
                tax_list[i] = sequence_to_tax[sequence_id]
            except KeyError:
                tax_list[i] = "NF"
        print tax_list[0:10]
        self.insert_array_by_chunks('genes/taxonomy', tax_list)
            

    def add_sequencecluster_data(self, input_filename):
        #Input: seq_otus.txt filename (tab-separated file, first column is OTU name, remaining columns are sequences)
        #Processing: Must list OTU names in the same order as genes/sequenceids
        #Output: fill genes/sequenceclusters with this list
        sequence_to_cluster = {}
        with open(input_filename, 'r') as in_file:
            for line in in_file:
                line = line.split("\t")
                cluster_num = line[0]
                line = line[1:]
                for sequence in line:
                    sequence_to_cluster[sequence] = cluster_num
        cluster_list = np.empty(shape=self.h5_table["genes/sequenceids"].shape, dtype=object)
        for i, sequence_id in enumerate(self.h5_table["genes/sequenceids"]):
            try:
                cluster_list[i] = sequence_to_cluster[sequence_id]
            except KeyError:
                cluster_list[i] = "NF"
        self.insert_array_by_chunks('genes/sequenceclusters', cluster_list)

    def insert_array_by_chunks(self, target, array):
        n = self.h5_table[target].shape[0]
        chunks = np.append(np.arange(0,n,1000), n)
        for i in range(len(chunks)-1):
            self.h5_table[target][chunks[i]:chunks[i+1]] = array[chunks[i]:chunks[i+1]]
            
    def get_sparse_matrix(self, chunk_size=1000):
        print("intializing empty arrays")
        data = np.empty(self.h5_table["timeseries/data"].shape)
        indices = np.empty(self.h5_table["timeseries/indices"].shape)
        indptr = np.empty(self.h5_table["timeseries/indptr"].shape)
        
        chunks = range(0, data.shape[0], chunk_size)
        if chunks[-1] != data.shape[0]:
            chunks = chunks + [data.shape[0]]
        for i,j in zip(chunks[0:-1], chunks[1:]):
            self.h5_table["timeseries/data"].read_direct(data,np.s_[i:j],np.s_[i:j])
        print("Data array read!")
        
        chunks = range(0, indices.shape[0], chunk_size)
        if chunks[-1] != indices.shape[0]:
            chunks = chunks + [indices.shape[0]]
        for i,j in zip(chunks[0:-1], chunks[1:]):
            self.h5_table["timeseries/indices"].read_direct(indices,np.s_[i:j],np.s_[i:j])
        print("indices read!")
        
        chunks = range(0, indptr.shape[0], chunk_size)
        if chunks[-1] != indptr.shape[0]:
            chunks = chunks + [indptr.shape[0]]
        for i,j in zip(chunks[0:-1], chunks[1:]):
            self.h5_table["timeseries/indptr"].read_direct(indptr,np.s_[i:j],np.s_[i:j])
        
        print("arrays loaded from HDF5 file")
        return csr_matrix((data, indices, indptr))
        
    def get_time_points(self):
        return self.h5_table["samples/time"]

    def get_cluster_labels(self, i):
        return self.h5_table["genes/clusters"][:,i]

    def filter_data(self, outfile, threshold=0, filter_type='presence'):
        #Options for filter:
        #proportion: if gene accounts for less than threshold % of the data, filter it
        #abundance: if gene was recorded less than threshold times, filter it
        #presence: if gene was recorded in fewer than threshold % of time points, filter it
        
        #New data structures
        filtered_data = []
        filtered_indices = []
        filtered_indptr = [0]
        filtered_sequences = []
        row_data = None
        sparse_matrix = self.get_sparse_matrix()
        nrows = sparse_matrix.shape[0]
        ncols = float(sparse_matrix.shape[1])
        threshold = float(threshold)
        if (filter_type == 'proportion'):
            total_sum = float(sparse_matrix.sum())
        for row_index in range(0,nrows):
            add_data = False
            row_data = sparse_matrix[row_index,:]
            if filter_type == 'proportion':
                if row_data.sum()/total_sum > threshold:
                    add_data = True
            elif filter_type == 'abundance':
                if row_data.sum() > threshold:
                    add_data = True
            elif filter_type == 'presence':
                presence_proportion = row_data.nnz/ncols
                if presence_proportion >= threshold:
                    add_data = True
            if add_data:
                filtered_sequences.append(self.h5_table["genes/sequenceids"][row_index])
                filtered_data.extend(row_data.data)
                filtered_indptr.append(len(filtered_data))
                filtered_indices.extend(row_data.indices)
        if len(filtered_data) == 0:
            raise ValueError, "All data filtered. Consider relaxing filter criterion."
        else:
            new_timeseriesdb = TimeSeriesData(outfile)
            ngenes = len(filtered_indptr)-1
            nsamples = ncols
            nobs = len(filtered_data)
            new_timeseriesdb.resize_data(ngenes, nsamples, nobs)
            sample_name_array = self.h5_table["samples/names"]
            new_timeseriesdb.add_names(sample_name_array)
            time_points = self.h5_table["samples/time"][:]
            new_timeseriesdb.add_timepoints(time_points)
            #Put into HDF5 file by chunks because otherwise memory usage balloons
            new_timeseriesdb.insert_array_by_chunks("timeseries/data", filtered_data)
            new_timeseriesdb.insert_array_by_chunks("timeseries/indptr", filtered_indptr)
            new_timeseriesdb.insert_array_by_chunks("timeseries/indices", filtered_indices)
            new_timeseriesdb.insert_array_by_chunks("genes/sequenceids", filtered_sequences)
