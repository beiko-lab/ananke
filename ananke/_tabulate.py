#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides functionality for tabulating sequence files (FASTA) into
an Ananke HDF5 file.
"""
import sys
import warnings
import hashlib

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix

from ._database import TimeSeriesData

#  TODO: - Add an observer to the timeseriesdb class for progress

def read_metadata(metadata_path, time_name, time_mask):
    """Take in a path, metadata_path, read file into a pandas dataframe.
    Validate that time_name and time_mask (if applicable) are present.

    Parameters
    ----------
    metadata_path: str
        filepath to metadata file
    time_name: str
        name of column in metadata file that contains time points as integers 
        relative to the starting time point
    time_mask: str
        name of column in metadata file that contains the masking category 
        that delineates multiple time series

    Returns
    -------
    metadata_mapping: pandas.DataFrame
        Pandas dataframe containing sample metadata
    """
    metadata_mapping = pd.read_csv(metadata_path, sep="\t", header=0)
    try:
        time_points = np.sort(metadata_mapping[time_name].unique())
    except:
        raise KeyError("Specified time point column name (%s) is not found " \
                       "in metadata file." % (time_name,))

    # Check if #SampleID is a column (required for QIIME metadata format)
    if "#SampleID" not in metadata_mapping:
        raise KeyError("Metadata mapping file does not start with #SampleID.")

    if time_mask is not None:
        if time_mask not in metadata_mapping:
            raise KeyError("Specified time mask column name (%s) is not " \
                           "found in metadata file." % (time_mask,))
        else:
            #Get the values sorted by mask first, then time points
            metadata_mapping = metadata_mapping.sort_values(by=[time_mask,
                                                                time_name])
    else:
        metadata_mapping = metadata_mapping.sort_values(by=time_name)

    return metadata_mapping


def tabulate(seqf, metadata_mapping, size_labels):
    """Count the unique sequences in a FASTA file, tabulating by sample.

    Parameters
    ----------
    seqf: file
        input FASTA sequence file (not wrapped, two lines per record)
    metadata_mapping: pandas.DataFrame
        metadata table contained in Pandas DataFrame
    size_labels: boolean
        true if FASTA file is already compressed to unique sequences (and contains USEARCH-style size
        annotations in the label, i.e., >SEQUENCEID;size=####

    Returns
    -------
    seqcount: dict {int:{str:int}}
        dict of dicts, with first key as the DNA sequence hash, second key
        as the sample name, and final int value as the sequence count within 
        that sample
    """

    i = 0
    samplenames = set(list(metadata_mapping["#SampleID"]))
    sample_name_array = np.array(metadata_mapping["#SampleID"])
    seqcount = defaultdict(lambda: defaultdict(int))
    prev_sample_name = ""

    #Keep track of skipped sequences
    skipped = 0
    skipped_samples = set()

    for line in seqf:
        assert line[0] == ">", "Label line began with %s, not >. Is " \
          "your FASTA file one-line-per-sequence?" % (line[0],)
        #Assume first line is header
        sample_name = line.split("_")[0][1:]
        if sample_name != prev_sample_name:
            if sample_name in samplenames:
                prev_sample_name = sample_name
            else:
                #Skip the next sequence
                # Readline if py3, next if py2
                if sys.version_info[0] >= 3:
                    seqf.readline()
                else:
                    seqf.next()
                i += 1
                skipped += 1
                skipped_samples.add(sample_name)
                continue
        if sys.version_info[0] >= 3:
            sequence = seqf.readline().strip()
        else:
            sequence = seqf.next().strip()
        assert sequence[0] != ">", "Expected sequence, got label. Is \
          your FASTA file one-line-per-sequence?"
        if size_labels:
            if (";" not in line) | ("=" not in line):
                raise ValueError("FASTA size labels specified but not found.")
            size = line.strip().split(";")[-1].split("=")[-1]
            seqcount[sequence][sample_name] += int(size)
        else:
            seqcount[sequence][sample_name] += 1
        i+=1
        #This needs to be replaced by something better
        if (i%10000 == 0):
            sys.stdout.write("\r%d" % i)
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    if (skipped > 0):
        warnings.warn("Skipped %d sequences (no match to sample name" \
          "in metadata file). Sample names: %s" % (skipped, \
          str(list(skipped_samples))))

    return seqcount


def write_csr(timeseriesdb, seqcount, outseqf, sample_name_array):
    """Convert the seqcount dict structure to a compressed sparse row matrix,
    then write it to an Ananke TimeSeriesData object and a FASTA file, in
    chunks.

    Parameters
    ----------
    timeseriesdb: ananke._database.TimeSeriesData
        TimeSeriesData object that encapsulates a .h5 file
    seqcount: dict {int:{str:int}}
        dict of dicts output from tabulate function
    outseqf: file
        a file object pointing to a FASTA file that will contain unique
        sequences and their size data
    sample_name_array: numpy.array
        an array containing the sample names, taken from the metadata mapping

    Returns
    -------
    unique_indices: list
        contains all of the sample names included in the Ananke data file
    """
    # Set stage for a CSR sparse matrix
    data = []
    indptr = []
    indices = []
    # Use to ensure all indices exist at the end
    unique_indices = []
    hashseq_list = []
    j = 0

    # From the seqcount dict containing the time-series for each hash
    # build up a CSR matrix structure
    # At the same time, print each unique sequence and its size to a FASTA
    # file that can be used for clustering and taxonomic classification
    for sequence, abundance_dict in seqcount.items():
        md5hash = hashlib.md5()
        md5hash.update(sequence.encode())
        hashseq = md5hash.hexdigest()
        hashseq_list.append(hashseq)
        abundance_list = []
        rowsum = 0
        indptr.append(j)
        for col, sample_name in enumerate(sample_name_array):
            if sample_name in abundance_dict.keys():
                abundance = abundance_dict[sample_name]
                rowsum += abundance
                data.append(abundance)
                indices.append(col)
                if (col not in unique_indices):
                    unique_indices.append(col)
                j += 1
        #Write the unique sequence file
        outseqf.write(">%s;size=%d;\n" % (hashseq,rowsum))
        outseqf.write("%s\n" % sequence)
        #Don't let this get too large before dumping it to disk
        if (len(indptr) >= 500):
            timeseriesdb.add_timeseries_data(data, indices, \
                                             indptr, hashseq_list)
            # Clear out the existing data
            hashseq_list = []
            data = []
            indices = []
            indptr = []
    # Add any data that's left        
    timeseriesdb.add_timeseries_data(data, indices, indptr, hashseq_list)

    return unique_indices

def hash_sequence(sequence):
    md5hash = hashlib.md5()
    md5hash.update(sequence.encode())
    return md5hash.hexdigest()


def fasta_to_ananke(sequence_path, metadata_path, time_name, \
              timeseriesdata_path, outseq_path, time_mask=None, \
              size_labels = False):
    """Count the unique sequences in a FASTA file, tabulating by time points.

    Save the results to an Ananke HDF5 file.
    """

    # Grab the metadata from the file
    metadata_mapping = read_metadata(metadata_path, time_name, time_mask)

    # Now open the sequence file
    # Input format assumptions: 
        #- sequences and headers take 1 line each (i.e., not wrapped FASTA)
        #- no blank lines
    
    # Open files for reading and writing
    seqf = open(sequence_path,'r')
    outseqf = open(outseq_path, 'w')
    timeseriesdb = TimeSeriesData(timeseriesdata_path)
    
    # Iterate through the input FASTA file, tabulating unique sequences
    seqcount = tabulate(seqf, metadata_mapping, size_labels)

    print("Writing table to file")

    # Get the shape of the data
    sample_name_array = np.array(metadata_mapping["#SampleID"])
    ngenes = len(seqcount)
    nsamples = len(sample_name_array)
    nobs = 0

    for sequence, abundance_dict in seqcount.items():
        nobs += len(abundance_dict)

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

    unique_indices = write_csr(timeseriesdb, seqcount, \
                               outseqf, sample_name_array)

    print("Done writing to %s" % timeseriesdata_path)
    
    # Check consistency of the data, warn if samples are missing
    if (len(unique_indices) < nsamples):
        warnings.warn("Number of time-points retrieved from sequence " \
          "file is less than the number of samples in metadata file. " \
          "%d samples are missing. Consider removing extraneous samples " \
          "from the metadata file. You will want to run `ananke filter` to " \
          "remove empty samples."
          % (nsamples - len(unique_indices),))

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
            outseqf.write(">%s;size=%d\n" % (seqhash, total))
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
