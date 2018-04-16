#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides functionality for tabulating sequence files (FASTA) into
an Ananke HDF5 file.
"""
import sys
import warnings
import hashlib
from functools import lru_cache

import pandas as pd
import numpy as np
from collections import Counter
import arrow

from ._database_rework import TimeSeriesData

class CountAccumulator(object):
    """Class that holds counts for arbitrary datasets until a certain number
       have been obtained, then pushes them to their correct location in the
       HDF5 file. Allows the HDF5 file to be incrementally built up. Will add
       counts to an existing file if initialized using a file with data."""

    def __init__(self, timeseriesdb, push_at = 5e6,
                 item_hash = lambda x: hashlib.md5(x).hexdigest()):
        self.db = timeseriesdb
        self.id_indexes = dict(zip(timeseriesdb._h5t["timeseries/ids"][:], 
                               np.arange(0, timeseriesdb._h5t["timeseries/ids"].shape[0])))
        self.limit = push_at
        # Holds the counts in a dictionary in {"dataset_name": { column_index: Counter((hash, count)) }} format
        self.counts = {}
        # Holds the sequence hashes (in order)
        self.pending_ids = []
        self.pending_rows = []
        self.pending_columns = []
        self.pending_counts = []
        # Count records
        self.count_total = 0
        # This hash is how we will store the IDs
        self.item_hash = item_hash
        # Set a warnings filter for the count function
        warnings.simplefilter("once")

    @lru_cache(maxsize=100)
    def find_id(self, Id, full_check = True):
        if full_check:
            res = np.where(self.id_indexes.keys() == Id)
            if len(res[0]) >= 1:
                return res[0][0]
        try:
            idx = self.pending_ids.index(Id)
            return len(self.id_indexes.keys()) + idx
        except:
            return -1

    # Adds an item to the bloom filter, increments the value
    def count(self, item, sample_name, increment = 1):
        ih = self.item_hash(item.encode())
        ih = str(ih).encode()
        dataset_name, column_index = self.db.get_sample_index(sample_name, return_dataset=True)
        if column_index == -1:
            warnings.warn("Sample %s not found in any data set. Skipping all entries from this sample." % (sample_name,))
            return
        if dataset_name not in self.counts:
            self.counts[dataset_name] = {}
        if column_index not in self.counts[dataset_name]:
            self.counts[dataset_name][column_index] = Counter()
        self.counts[dataset_name][column_index][ih] += increment
        self.count_total += increment
        if self.count_total >= self.limit:
            self.push()
            self.count_total = 0
        
    # Writes the accumulated counts 
    def push(self, chunksize = 10):
        print("Pushing 100k sequences")
        #row_chunk = []
        #count_chunk = []
        for dataset_name, columns in self.counts.items():
            dataset = self.db._h5t["data/" + dataset_name + "/matrix"]
            for column_index, counter in columns.items():
                print("Sorting hashes")
                for item_hash, count in counter.items():
                    try:
                        row_index = self.id_indexes[item_hash]
                    except KeyError:
                        row_index = -1
                    if row_index == -1:
                        row_index = len(self.id_indexes.keys()) + len(self.pending_ids)
                        self.pending_ids.append(item_hash)
                        self.id_indexes.update([(item_hash, row_index)])
                    if row_index < dataset.shape[0]:
                        #row_chunk.append(row_index)
                        #count_chunk.append(count_chunk)
                        #if len(row_chunk) >= chunksize:
                            #r_s, n_s = map(list, zip(*sorted(zip(row_chunk, count_chunk))))
                        dataset[row_index, column_index] = count
                            #row_chunk = []
                            #count_chunk = []
                    else:
                        self.pending_rows.append(row_index)
                        self.pending_columns.append(column_index)
                        self.pending_counts.append(count)
                #if len(row_chunk) > 0:
                #    r_s, n_s = map(list, zip(*sorted(zip(row_chunk, count_chunk))))
                #    dataset[r_s, column_index] = n_s
                #    row_chunk = []
                #    count_chunk = []
            print("Resizing data for new features")
            self.db.resize_data(len(self.id_indexes.keys()) + len(self.pending_ids))
            print("Inserting pending feature IDs")
            self.db._h5t["timeseries/ids"][len(self.id_indexes.keys()):] = self.pending_ids
            self.id_indexes.update(zip(self.pending_ids, self.pending_rows))
            if len(self.pending_rows) > 0:
                print("Inserting feature counts")
                r_s, c_s, n_s = map(list, zip(*sorted(zip(self.pending_rows,
                                                          self.pending_columns,
                                                          self.pending_counts))))
                print("Sorted")
                r_s = np.array(r_s)
                c_s = np.array(c_s)
                n_s = np.array(n_s)
                for value in np.unique(c_s):
                    print("Inserting a column")
                    rows = r_s[np.where(c_s == value)]
                    counts = n_s[np.where(c_s == value)]
                    N = max(round(len(rows) / chunksize), 1)
                    row_chunks = np.array_split(rows, N)
                    count_chunks = np.array_split(counts, N)
                    for row_chunk, count_chunk in zip(row_chunks, count_chunks):
                        dataset[row_chunk, value] += count_chunk
                self.pending_ids = []
                self.pending_rows = []
                self.pending_columns = []
                self.pending_counts = []

def fasta_to_ananke(seqf, timeseriesdb, size_labels=False):
    """Count the unique sequences in a FASTA file, tabulating by sample.

    Parameters
    ----------
    seqf: file
        input FASTA sequence file (not wrapped, two lines per record)
    size_labels: boolean
        true if FASTA file is already compressed to unique sequences (and 
        contains USEARCH-style size annotations in the label, i.e., 
        >SEQUENCEID;size=####;

    """

    the_count = CountAccumulator(timeseriesdb)

    for line in seqf:
        assert line[0] == ">", "Label line began with %s, not >. Is " \
          "your FASTA file one-line-per-sequence?" % (line[0],)
        #Assume first line is header
        sample_name = line.split("_")[0][1:]
        sequence = seqf.readline().strip()
        assert sequence[0] != ">", "Expected sequence, got label. Is \
          your FASTA file one-line-per-sequence?"
        if size_labels:
            if "=" not in line:
                raise ValueError("FASTA size labels specified but not found.")
            size = line.strip().split(";")[-1].split("=")[-1]
            the_count.count(sequence, sample_name, increment = int(size))
        else:
            the_count.count(sequence, sample_name)
    the_count.push()
    seqf.close()

#TODO: Rewrite 
def dada2_to_ananke(table_path, metadata_path, time_name, timeseriesdata_path,
                    outseq_path, time_mask=None):
    """Converts a DADA2 table from dada2::makeSequenceTable to an Ananke HDF5
    file. Table must have sequences as rows, samples/time-points as columns
    (i.e., output from DADA2 should be transposed). Should be exported using
    `write.table(t(seqtab), table_path)` from R.

    Parameters
    ----------
    table_path: str
        Path to the csv table output from DADA2.
    metadata_path: str
        Path to the tab-separated metadata file.
    time_name: str
        Name of the column that contains the time points as integers, offset
        from zero.
    timerseriesdata_path: str
        Path to the new output Ananke HDF5 file.
    outseq_path: str
        Path to the new output unique sequence FASTA file.
    time_mask: str (optional)
        Name of the column that contains the time-series masking information.
    """
    # Grab the metadata from the file
    metadata_mapping = read_metadata(metadata_path, time_name, time_mask)
    if time_mask is not None:
        metadata_mapping = metadata_mapping.sort_values([time_mask, 
                                                         time_name])
    else:
        metadata_mapping = metadata_mapping.sort_values([time_name])

    # Now open the sequence file
    # Input format assumptions:
        #- sequences and headers take 1 line each (i.e., not wrapped FASTA)
        #- no blank lines

    # Open files for reading and writing
    outseqf = open(outseq_path, 'w')
    timeseriesdb = TimeSeriesData(timeseriesdata_path)

    # Open table file, read it with pandas
    seqtab = pd.read_csv(table_path, sep=" ")

    print("Writing table to file")

    # Get the shape of the data
    sample_name_array = np.array(metadata_mapping["#SampleID"])
    ngenes = seqtab.shape[0]
    nsamples = len(sample_name_array)

    # Pare down the sequence tab to include only the necessary samples
    # sorted in order of mask then time points
    seqtab = seqtab.loc[:, sample_name_array]
    # Sparse-ify it
    csr_seqtab = csr_matrix(seqtab)

    nobs = csr_seqtab.nnz

    # Resize the Ananke TimeSeriesData object
    timeseriesdb.resize_data(ngenes, nsamples, nobs)
    timeseriesdb.insert_array_by_chunks("samples/names", sample_name_array)
    timeseriesdb.insert_array_by_chunks("samples/time",
                                        metadata_mapping[time_name],
                                        transform_func = float)

    if time_mask is not None:
        timeseriesdb.insert_array_by_chunks("samples/mask",
                                            metadata_mapping[time_mask])
    else:
        #Set a dummy mask
        timeseriesdb.insert_array_by_chunks("samples/mask",
                                            [1]*len(sample_name_array))

    seqhashes = [ hash_sequence(x) for x in seqtab.index ]

    # Export sequences to FASTA
    for i in range(0, ngenes):
        total = seqtab.iloc[i].sum()
        if total > 0:
            seqhash = seqhashes[i]
            outseqf.write(">%s;size=%d;\n" % (seqhash, total))
            outseqf.write(seqtab.index[i].strip() + "\n")
    
    timeseriesdb.insert_array_by_chunks("genes/sequenceids", seqhashes)

    timeseriesdb.insert_array_by_chunks("timeseries/data",
                                        csr_seqtab.data,
                                        int)
    timeseriesdb.insert_array_by_chunks("timeseries/indptr",
                                        csr_seqtab.indptr,
                                        int)
    timeseriesdb.insert_array_by_chunks("timeseries/indices",
                                        csr_seqtab.indices,
                                        int)

    print("Done writing to %s" % timeseriesdata_path)

def csv_to_ananke(csv_path, metadata_path, time_name, timeseriesdata_path,
                  time_mask=None):
    """Imports a CSV file and metadata file into an Ananke HDF5 file.
    The CSV file must have the time-series as rows and the samples/
    time points as columns. The first line must be a header that contains
    the sample names.

    Parameters
    ----------
    csv_path: Location of the 
    metadata_path: Location of the metadata mapping file.
    time_name: Column name containing the time points in the metadata file.
    timeseriesdata_path: Output Ananke .h5 file path.
    time_mask: Column name containing the time mask indicator in the metadata
               file.
    """
    # Pull in the metadata reading
    metadata_mapping = read_metadata(metadata_path, time_name, time_mask)
    if time_mask is not None:
        metadata_mapping = metadata_mapping.sort_values([time_mask,
                                                         time_name])   
    else:                                                                       
        metadata_mapping = metadata_mapping.sort_values([time_name])

    csv_table = pd.read_csv(csv_path, header=0, 
                            index_col=0, sep="\t")

    # Construct the Ananke object
    timeseriesdb = TimeSeriesData(timeseriesdata_path)
    # Sort columns by time point

    sample_name_array = np.array(metadata_mapping["#SampleID"])
    ngenes = csv_table.shape[0]
    nsamples = len(sample_name_array)

    # Pare down the sequence tab to include only the necessary samples
    # sorted in order of mask then time points
    csv_table = csv_table.loc[:, sample_name_array]

    #Get row (i.e., time series) names
    seqids = csv_table.index

    #Convert the table to a sparse matrix for storage in CSR format
    csr_mat = csr_matrix(csv_table.to_sparse().to_coo())

    nobs = csr_mat.nnz

    timeseriesdb.resize_data(ngenes, nsamples, nobs)
    timeseriesdb.insert_array_by_chunks("samples/names", sample_name_array)
    timeseriesdb.insert_array_by_chunks("samples/time",
                                        metadata_mapping[time_name],
                                        transform_func = float)

    if time_mask is not None:
        timeseriesdb.insert_array_by_chunks("samples/mask",
                                            metadata_mapping[time_mask])
    else:
        #Set a dummy mask
        timeseriesdb.insert_array_by_chunks("samples/mask",
                                            [1]*len(sample_name_array))

    timeseriesdb.insert_array_by_chunks("genes/sequenceids", seqids)

    timeseriesdb.insert_array_by_chunks("timeseries/data",
                                        csr_mat.data,
                                        int)
    timeseriesdb.insert_array_by_chunks("timeseries/indptr",
                                        csr_mat.indptr,
                                        int) 
    timeseriesdb.insert_array_by_chunks("timeseries/indices",
                                        csr_mat.indices,
                                        int)

    print("Done writing to %s" % timeseriesdata_path)
