#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the methods in _tabulate.py"""

import pandas as pd
import numpy as np
import pytest

from ananke._tabulate import (fasta_to_ananke, tabulate, 
                              read_metadata, write_csr)
from ananke._database import TimeSeriesData

#Testing: read_metadata(metadata_path, time_name, time_mask)

def compare_fasta_unordered(f1, f2):
    """Return True if the FASTA files f1 and f2 have the same records, but
    ignore the order of the records. Labels and sequences must be identical"""
    f1 = open(f1)
    f2 = open(f2)
    f1_set = set()
    f2_set = set()

    while True:
        label = f1.readline()
        if not label:
            break
        seq = f1.readline()
        f1_set.update((label + seq,))
    while True:
        label = f2.readline()
        if not label:
            break
        seq = f2.readline()
        f2_set.update((label + seq,))
    f1.close()
    f2.close()
    return f1_set == f2_set

def test_read_metadata_with_mask():
    """Test reading in metadata with a time_mask column specified

    Tests for correct data type, all expected columns exist, and
    correct data sorting (by time_mask then time_point)"""

    metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt", 
                                     "time_points", "time_mask")
    assert type(metadata_mapping) is pd.core.frame.DataFrame
    assert "#SampleID" in metadata_mapping
    assert "time_mask" in metadata_mapping
    assert "time_points" in metadata_mapping
    assert list(metadata_mapping["time_points"]) == [0,4,10,12,0,4,10,12]
    assert list(metadata_mapping["time_mask"]) == [0,0,0,0,1,1,1,1]


def test_read_metadata_without_mask():
    """Tests reading in metadata with a time_mask column unspecified

    Tests for correct data type, all expected columns exist, and
    correct data sorting (by time_points)"""

    metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt",
                                     "time_points", None)
    assert type(metadata_mapping) is pd.core.frame.DataFrame
    assert "#SampleID" in metadata_mapping
    assert "time_points" in metadata_mapping
    assert list(metadata_mapping["time_points"]) == [0,0,4,4,10,10,12,12]


def test_read_metadata_no_header():
    """Test that a KeyError is raised when no header exists in metadata file"""

    with pytest.raises(KeyError):
        metadata_mapping = read_metadata("./tests/data/" \
                                     "metadata_mapping_noheader.txt",
                                     "time_points", None)


def test_read_metadata_no_hash():
    """Test that a KeyError is raised when file does not start with a hash"""
    with pytest.raises(KeyError):
        metadata_mapping = read_metadata("./tests/data/" \
                                         "metadata_mapping_nohash.txt",
                                         "time_points", None)


def test_read_metadata_missing_column():
    """Test that a KeyError is raised when specifying a column that is not
    present, both for time_points and time_mask"""

    with pytest.raises(KeyError):
        metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt",
                                         "something_else", None)
    with pytest.raises(KeyError):
        metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt",
                                         "time_points", "something_else")


# Required tests TBI:
# - multiple identical time points in a series
# - duplicate sample names

# Testing tabulate(seqf, metadata_mapping, size_labels)

def test_tabulate_no_size_labels():
    """Test that a FASTA file with no size labels is properly tabulated"""
    
    seqf = open("./tests/data/sequences_no_size.fasta")
    metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt",
                                     "time_points", "time_mask")
    seqcount = tabulate(seqf, metadata_mapping, False)
    true_data = {'ATGCTATCGATCGATGCATCGATCGATGCTATCGTACGTGGCA':
                 {'15':4, '72c':2, 'qpa':3, '29':1},
                 'TTGCTATCGATCGATGCATCGATCGATGCTATCGTACGTGGCA':
                 {'15':1, '72c':1, 'qpa':1, '29':1},
                 'TTGCTATCGATCGATGCATCGATCGATGCTATCGTACGTGGCT':
                 {'qpa':1},
                 'GATGTCGTAGCTGTAGCTAGCTAGCTAGCTGCTG':
                 {'29':2}}
    seqf.close()
    assert seqcount == true_data

def test_tabulate_size_labels():
    """Test that a FASTA file with size labels is properly tabulated"""
    
    seqf = open("./tests/data/sequences_size.fasta")
    metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt",
                                     "time_points", "time_mask")
    seqcount = tabulate(seqf, metadata_mapping, True)
    true_data = {'ATGCTATCGATCGATGCATCGATCGATGCTATCGTACGTGGCA':
                 {'15':4, '72c':2, 'qpa':3, '29':1},
                 'TTGCTATCGATCGATGCATCGATCGATGCTATCGTACGTGGCA':
                 {'15':1, '72c':1, 'qpa':1, '29':1},
                 'TTGCTATCGATCGATGCATCGATCGATGCTATCGTACGTGGCT':
                 {'qpa':1},
                 'GATGTCGTAGCTGTAGCTAGCTAGCTAGCTGCTG':
                 {'29':2}}
    seqf.close()
    assert seqcount == true_data

def test_tabulate_wrapped_FASTA():
    """Test that a the proper exception is raised if a wrapped FASTA is
    provided. Input FASTA must be one line for label, one line for sequence"""
    seqf = open("./tests/data/sequences_wrapped.fasta")
    metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt",
                                     "time_points", "time_mask")
    with pytest.raises(AssertionError):
        seqcount = tabulate(seqf, metadata_mapping, False)
    seqf.close()


def test_tabulate_skipped_sample_warning():
    """Test that a warning is provided if sequences in the FASTA file are not
    used because there is no corresponding sample in the metadata mapping"""
    seqf = open("./tests/data/sequences_extra_sample.fasta")
    metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt",
                                     "time_points", "time_mask")
    with pytest.warns(UserWarning):
        seqcount = tabulate(seqf, metadata_mapping, False)
    seqf.close()

def test_tabulate_unordered():
    """Test that a FASTA file with sequences presented in a random order
    (i.e., sequences next to one another don't necessarily belong to the same
    sample/time point) is properly tabulated"""
    
    seqf = open("./tests/data/sequences_unordered.fasta")
    metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt",
                                     "time_points", "time_mask")
    seqcount = tabulate(seqf, metadata_mapping, False)
    true_data = {'ATGCTATCGATCGATGCATCGATCGATGCTATCGTACGTGGCA':
                 {'15':4, '72c':2, 'qpa':3, '29':1},
                 'TTGCTATCGATCGATGCATCGATCGATGCTATCGTACGTGGCA':
                 {'15':1, '72c':1, 'qpa':1, '29':1},
                 'TTGCTATCGATCGATGCATCGATCGATGCTATCGTACGTGGCT':
                 {'qpa':1},
                 'GATGTCGTAGCTGTAGCTAGCTAGCTAGCTGCTG':
                 {'29':2}}
    seqf.close()
    assert seqcount == true_data

# Testing write_csr(timeseriesdb, seqcount, outseqf, sample_name_array)

def test_write_csr(temp_dir):
    """Test of the write_csr function, and verify it produces a proper FASTA
    file output, and Ananke TimeSeriesData object.
    Validate the CSR data in the TimeSeriesData object."""
    
    seqf = open("./tests/data/sequences_no_size.fasta")
    outseqf = open(temp_dir + "/seq.unique.fasta", 'w')
    timeseriesdb = TimeSeriesData(temp_dir + "/ananke.h5")

    metadata_mapping = read_metadata("./tests/data/metadata_mapping.txt",
                                     "time_points", "time_mask")
    seqcount = tabulate(seqf, metadata_mapping, False)
    sample_name_array = np.array(metadata_mapping["#SampleID"])

    ngenes = len(seqcount)
    nsamples = len(sample_name_array)
    nobs = 0

    for sequence, abundance_dict in seqcount.items():
        nobs += len(abundance_dict)

    timeseriesdb.resize_data(ngenes, nsamples, nobs)
    timeseriesdb.insert_array_by_chunks("samples/names", sample_name_array)
    timeseriesdb.insert_array_by_chunks("samples/time", 
                                        metadata_mapping["time_points"],
                                        transform_func = float)

    timeseriesdb.insert_array_by_chunks("samples/mask", 
                                        metadata_mapping["time_mask"])

    unique_indices = write_csr(timeseriesdb, seqcount,
                               outseqf, sample_name_array)

    seqf.close()
    outseqf.close()
    # Check the table in the TimeSeriesData object
    true_data = [[2.0, 1.0, 3.0, 4.0],
                 [1.0, 1.0, 1.0, 1.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 2.0, 0.0, 0.0]]

    generated_data = timeseriesdb.get_sparse_matrix().todense()
    # Note: the inner lists have a specific ordering (temporal)
    #       but the order of those lists (i.e., the order of the sequences
    #       is not important
    for data in true_data:
        assert data in generated_data

    del timeseriesdb

    # Check the output FASTA file
    assert compare_fasta_unordered(temp_dir + "/seq.unique.fasta", 
                       "tests/data/seq.unique.fasta")

# Testing fasta_to_ananke(sequence_path, metadata_path, time_name, \                  
#              timeseriesdata_path, outseq_path, time_mask=None, \               
#              size_labels = False)

def test_fasta_to_ananke_no_mask():
    """Test the entire pipeline, from sequence data to Ananke object

    Test the sample_names and time_points array in Ananke object
    Test that the dummy mask is properly applied"""
    pass


def test_fasta_to_ananke_mask():
    """Test the entire pipeline, from sequence data to Ananke object

    Test the sample_names and time_points arrays in Ananke object
    Test that the time series mask is properly applied"""
    pass
