#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the methods in _database.py"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from shutil import copy

from ananke._database import TimeSeriesData


#Test constructor of HDF5 Ananke file
# TimeSeriesData(self, h5_file)

def timeseriesdata_constructor_new_file(temp_dir):
    """Tests the TimeSeriesData class constructor when the file does not
    exist. Tests that a new file is created, that all expected data sets
    are present, and that the flag that indicates that the file is empty
    is set to False.
    """
    tsd = TimeSeriesData(temp_dir + "/new_ananke.h5")
    tsd_file = Path(temp_dir + "/new_ananke.h5")
    #Check that the file was really created
    assert tsd_file.is_file()
    #Check that the data sets have been created
    assert set(tsd.h5_table.keys()) == {"genes", "timeseries", "samples"}
    assert set(tsd.h5_table["timeseries"].keys()) == {"data", "indices", 
                                                      "indptr"}
    assert set(tsd.h5_table["genes"].keys()) == {"sequences", "sequenceids",
                                                 "clusters", "taxonomy",
                                                 "sequenceclusters"}
    assert set(tsd.h5_table["samples"].keys()) == {"names", "time",
                                                   "metadata", "mask"}
    #Check that the empty file flag is set
    assert tsd.filled_data == False
    

def timeseriesdata_constructor_existing_file():
    """Tests the TimeSeriesData class constructor when the file exists. Tests
    that the data are as expected, and that the filled data flag is set."""
    tsd = TimeSeriesData("./tests/data/small.h5")
    assert tsd.filled_data
    #Validate that the time-series matrix data are correct
    dat = np.matrix([[0, 2, 0, 0],
                     [2, 1, 3, 4],
                     [1, 1, 1, 1],
                     [0, 0, 1, 0]])
    assert (dat == tsd.get_sparse_matrix().todense()).all()
    assert (tsd.get_time_points() == [0, 4, 10, 12]).all()
    sequence_ids = np.array([b'9247ec5fd33e99387d41e8fc0d7ee278',
                             b'53f74905f7826fee79dd09b0c12caddf',
                             b'8829fefe91ead511a30df118060b1030',
                             b'7f97b4991342bf629966aeac0708c94f'])
    assert (sequence_ids == tsd.get_array_by_chunks("genes/sequenceids")).all()
    sample_names = np.array([b'72c', b'29', b'qpa', b'15'])
    assert (sample_names == tsd.get_array_by_chunks("samples/names")).all()

# __del__(self)
#Test if HDF5 file closed
def timeseriesdata_destructor():
    """Tests that the destructor closes the table connection properly"""
    tsd = TimeSeriesData("./tests/data/small.h5")
    h5_table = tsd.h5_table
    #h5py returns True if open
    assert h5_table
    del tsd
    #h5py returns False if closed
    assert not h5_table

# version_greater_than(self, version)
def test_version_greater_than():
    """Tests the version comparison function"""
    tsd = TimeSeriesData("./tests/data/small.h5")
    #Current version: 0.2.1
    assert tsd.version_greater_than("0.0.1")
    assert tsd.version_greater_than("0.1.9")
    assert not tsd.version_greater_than("0.3.0")
    assert not tsd.version_greater_than("0.2.9")

# resize_data(self, ngenes=None, nsamples=None, nobs=None)
# Validate resizing works as expected, checking data shapes for all arrays
def test_resize(temp_dir):
    """Tests the resize function, ensuring that all data sets are the correct
    shape after resizing, and that the taxonomy and sequenceclusters arrays
    are filled with dummy values (needed to prevent rhdf5 segfault)"""
    copy("./tests/data/small.h5", temp_dir + "/resize.h5") 
    tsd = TimeSeriesData(temp_dir + "/resize.h5")
    h5tab = tsd.h5_table
    ngenes = 574
    nobs = 1645
    nsamples = 25
    tsd.resize_data(ngenes, nsamples, nobs)
    assert h5tab["timeseries/data"].shape == (nobs,)
    assert h5tab["timeseries/indices"].shape == (nobs,)
    assert h5tab["timeseries/indptr"].shape == (ngenes + 1,)
    assert h5tab["genes/sequences"].shape == (ngenes,)
    assert h5tab["genes/sequenceids"].shape == (ngenes,)
    assert h5tab["genes/taxonomy"].shape == (ngenes,)
    assert h5tab["genes/sequenceclusters"].shape == (ngenes,)
    #Has a dummy second dimension set to 20, clustering resizes this
    assert h5tab["genes/clusters"].shape == (ngenes, 20)
    assert h5tab["samples/names"].shape == (nsamples,)
    assert h5tab["samples/time"].shape == (nsamples,)
    assert h5tab["samples/metadata"].shape == (nsamples,)
    assert h5tab["samples/mask"].shape == (nsamples,)
    #Check that the auto-fill is completed
    assert (h5tab["genes/taxonomy"][:] == [b"NF"]*ngenes).all()
    assert (h5tab["genes/sequenceclusters"][:] == [b"NF"]*ngenes).all()

# fill_array(self, target, value)
# Validate that an entire array is filled with a given value
def test_fill_array(temp_dir):
    """Tests that the target array is filled with a tiled version of the 
    provided value."""
    copy("./tests/data/small.h5", temp_dir + "/fill.h5")
    tsd = TimeSeriesData(temp_dir + "/fill.h5")
    tsd.fill_array("samples/metadata", b"17")
    nsamples = tsd.h5_table["samples/metadata"].shape[0]
    assert (tsd.h5_table["samples/metadata"][:] == [b"17"]*nsamples).all()
    tsd.fill_array("genes/sequences", b"x37cny", chunk_size = 25)
    ngenes = tsd.h5_table["genes/sequences"].shape[0]
    assert (tsd.h5_table["genes/sequences"][:] == [b"x37cny"]*ngenes).all()

# add_timeseries_data(self, data, indices, indptr, sequences)
def test_add_timeseries_data(temp_dir):
    """Tests that time-series data are added to the file properly, and that
    the data can be added piece-wise and are put in the correct location."""
    #Create a new, empty file
    tsd = TimeSeriesData(temp_dir + "/add.h5")
    #Set test data dimensions
    ngenes = 10
    nobs = 23
    nsamples = 5
    #Resize it
    tsd.resize_data(ngenes, nsamples, nobs)
    #Arbitrary sequences IDs
    sequences = ["abc","def","ghi","jkl","mno","pqr","stu","vwx","yz","123"]
    #First batch of sparse matrix
    data1 = [7, 9, 6, 6, 6, 7, 6, 8, 8]
    indices1 = [0, 2, 4, 0, 1, 2, 4, 0, 4]
    indptr1 = [0,  3,  7]
    tsd.add_timeseries_data(data1, indices1, indptr1, sequences[0:3])
    
    #Second batch of sparse matrix
    data2 = [7, 6, 7, 8, 7, 9, 9, 7, 9, 6, 8, 6, 9, 8]
    indices2 = [4, 3, 4, 2, 1, 0, 2, 3, 1, 3, 0, 1, 2, 3]
    indptr2 = [9, 10, 12, 13, 14, 17, 19, 23]
    tsd.add_timeseries_data(data2, indices2, indptr2, sequences[3:])

    #Ground truth
    validate = np.array([[7, 0, 9, 0, 6],
                         [6, 6, 7, 0, 6],
                         [8, 0, 0, 0, 8],
                         [0, 0, 0, 0, 7],
                         [0, 0, 0, 6, 7],
                         [0, 0, 8, 0, 0],
                         [0, 7, 0, 0, 0],
                         [9, 0, 9, 7, 0],
                         [0, 9, 0, 6, 0],
                         [8, 6, 9, 8, 0]])

    assert (tsd.get_sparse_matrix().todense() == validate).all()

def test_add_taxonomy_data(temp_dir):
    """Tests that taxonomy data is added in the correct order from the source
    file and imports into the data file"""
    copy("./tests/data/small.h5", temp_dir + "/taxa.h5")
    tsd = TimeSeriesData(temp_dir + "/taxa.h5")
    tsd.add_taxonomy_data("./tests/data/taxonomy.txt")
    tax = tsd.get_array_by_chunks("genes/taxonomy")
    seqids = tsd.get_array_by_chunks("genes/sequenceids")
    #Grab the classifications from the file
    tax_dict = {}
    with open("./tests/data/taxonomy.txt", "r") as in_file:
        for line in in_file:
            line = line.split("\t")
            tax_dict[line[0].encode()] = line[1].strip()
    #Get the ground truth in the same order as in the data file
    ground_tax = [ tax_dict[x].encode() for x in seqids ]
    ground_tax = np.array(ground_tax)
    assert (tax == ground_tax).all()

def test_add_sequencecluster_data(temp_dir):
    """Tests that sequence-identity cluster data is added in the correct order
    from the source file and imports into the data file"""
    copy("./tests/data/small.h5", temp_dir + "/sequencecluster.h5")
    tsd  = TimeSeriesData(temp_dir + "/sequencecluster.h5")
    tsd.add_sequencecluster_data("./tests/data/seqclusters.txt")
    seqclusts = tsd.get_array_by_chunks("genes/sequenceclusters")
    seqids = tsd.get_array_by_chunks("genes/sequenceids")
    #Grab the sequence clusters from the file
    clust_dict = {}
    with open("./tests/data/seqclusters.txt", "r") as in_file:
        for line in in_file:
            line = line.split("\t")
            clust = line[0]
            for seqid in line[1:]:
                seqid = seqid.strip().encode()
                clust_dict[seqid] = clust
    ground_clust = [ clust_dict[x].encode() for x in seqids ]
    ground_clust = np.array(ground_clust)
    assert (seqclusts == ground_clust).all()

def test_insert_array_by_chunks(temp_dir):
    """Tests that an array is fully inserted into the HDF5 file"""
    copy("./tests/data/small.h5", temp_dir + "/arrayfill.h5")
    tsd = TimeSeriesData(temp_dir + "/arrayfill.h5")
    insert = [5, 16, 12, 8]
    tsd.insert_array_by_chunks("samples/time", insert,
                               transform_func = int)
    #Access table directly so as not to fail if get_array fails
    assert (np.array(insert) == tsd.h5_table["samples/time"][:]).all()
    #Test a small chunking example
    insert2 = ["abc", "qqq", "wtf", "hmm"]
    binsert2 = [ x.encode() for x in insert2 ]
    tsd.insert_array_by_chunks("genes/sequenceids", insert2, chunk_size = 2)
    assert (np.array(binsert2) == tsd.h5_table["genes/sequenceids"][:]).all()

def test_get_array_by_chunks(temp_dir):
    """Tests that an array can be retrieved in chunks from the HDF5 file"""
    copy("./tests/data/small.h5", temp_dir + "/arrayget.h5")
    tsd = TimeSeriesData(temp_dir + "/arrayget.h5")
    data1 = np.array([ 0,  4, 10, 12])
    assert (tsd.get_array_by_chunks("samples/time") == data1).all()
    data2 = np.array([b'9247ec5fd33e99387d41e8fc0d7ee278',
                      b'53f74905f7826fee79dd09b0c12caddf',
                      b'8829fefe91ead511a30df118060b1030',
                      b'7f97b4991342bf629966aeac0708c94f'])
    seqids = tsd.get_array_by_chunks("genes/sequenceids", chunk_size = 2)
    assert (data2 == seqids).all()

def test_get_sparse_matrix():
    tsd = TimeSeriesData("./tests/data/small.h5")
    dat = np.matrix([[0, 2, 0, 0],
                     [2, 1, 3, 4],
                     [1, 1, 1, 1],
                     [0, 0, 1, 0]])
    sm = tsd.get_sparse_matrix()
    assert (sm == dat).all()

def test_get_mask():
    """Tests fetching a mask, whether the samples/mask array exists or not"""
    #Array exists
    tsd = TimeSeriesData("./tests/data/small.h5")
    mask = tsd.get_mask()
    assert (mask == np.array([b'1', b'1', b'1', b'1'])).all()
    #Array does not exist
    tsd2 = TimeSeriesData("./tests/data/old_version.h5")
    mask = tsd.get_mask()
    assert (mask == np.array([b'1', b'1', b'1', b'1'])).all()
    

#Test to be written after clustering refactor
def test_get_cluster_labels():
    pass

def test_filter_data_abundance(temp_dir):
    """Tests a filtering of the data using an abundance cut-off, with no
    samples missing after filtering."""
    tsd = TimeSeriesData("./tests/data/small.h5")
    outfile = temp_dir + "/abundance_filter.h5"
    tsd.filter_data(outfile, "abundance", 2)
    filtered_tsd = TimeSeriesData(outfile)
    sm = filtered_tsd.get_sparse_matrix()
    dat = np.matrix([[0, 2, 0, 0],
                     [2, 1, 3, 4],
                     [1, 1, 1, 1]])
    seqids = np.array([b'9247ec5fd33e99387d41e8fc0d7ee278',
                       b'53f74905f7826fee79dd09b0c12caddf',
                       b'8829fefe91ead511a30df118060b1030'])
    assert (sm.todense() == dat).all()
    assert (filtered_tsd.h5_table["genes/sequenceids"][:] == seqids).all()

#Test the case where some columns (i.e., time points) have no counts
# and therefore should be removed
def test_filter_data_zero_cols(temp_dir):
    """Tests that columns are correctly remove when they are empty"""
    tsd = TimeSeriesData(temp_dir + "/filter_cols.h5")
    data = [1, 3, 5, 1, 1, 2, 5, 6, 5, 6, 2, 8]
    indptr = [0,  3,  4,  8, 12]
    indices = [1, 3, 4, 2, 0, 1, 3, 4, 0, 1, 3, 4]
    sequences = ["qqq", "abc", "zzz", "hi"]
    tsd.resize_data(len(indptr) - 1, len(indptr), len(data))
    tsd.add_timeseries_data(data, indices, indptr, sequences)
    tsd.insert_array_by_chunks("samples/names", ["a", "b", "c", "d", "e"])
    tsd.insert_array_by_chunks("samples/time", [0, 10, 15, 30, 35],
                               transform_func = int)
    tsd.insert_array_by_chunks("samples/mask", ["1", "1", "1", "1", "1"])
    outfile = temp_dir + "/filtered_zeros.h5"
    with pytest.warns(UserWarning):
        tsd.filter_data(outfile, "abundance", 2)
    filtered_tsd = TimeSeriesData(temp_dir + "/filtered_zeros.h5")
    assert (filtered_tsd.h5_table["samples/names"][:] == \
            np.array([b'a', b'b', b'd', b'e'])).all()
    assert (filtered_tsd.h5_table["samples/mask"][:] == [b'1']*4).all()
