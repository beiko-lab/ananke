#!/usr/bin/env python

from hashlib import md5
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from timeclust_database import TimeSeriesData

def aggregate(sequence_path, metadata_path, time_name, timeseriesdata_path, outseq_path):
    mm = pd.read_csv(metadata_path, sep="\t", header=0)
    #Get the time points (assumption: can be coerced to integers)
    try:
        time_points = np.sort(mm[time_name].unique())
    except:
        raise KeyError, "KeyError: Specified time point name is not found in metadata file"
    #Check if the group name exists
    #Now open the sequence file
    #Input format assumptions: - sequences and headers take 1 line each, no blank lines
    seqf = open(sequence_path,'r')
    timeseriesdb = TimeSeriesData(timeseriesdata_path)
    outseqf = open(outseq_path, 'w')
    i = 0
    samplenames = set(list(mm["#SampleID"]))
    sample_name_array = np.array(mm["#SampleID"])
    #Add metadata to HDF5 file
    seqcount = defaultdict(lambda: defaultdict(int))
    prev_sample_name = ""
    #Keep track of skipped sequences
    skipped = 0
    for line in seqf:
        #Assume first line is header
        sample_name = line.split("_")[0][1:]
        if sample_name != prev_sample_name:
            if sample_name in samplenames:
                prev_sample_name = sample_name
            else:
                #Skip the next sequence
                seqf.next()
                i += 1
                skipped += 1
                continue
        sequence = seqf.next().strip()
        seqcount[sequence][sample_name] += 1
        i+=1
        if (i%10000 == 0):
            print(i)
    if (skipped > 0):
        print("WARNING: Skipped %d sequences (no match to sample name in metadata file)" % (skipped,))
    print("Writing table to file")
    ngenes = len(seqcount)
    nsamples = len(sample_name_array)
    nobs = 0
    for sequence, abundance_dict in seqcount.iteritems():
        nobs += len(abundance_dict)
    timeseriesdb.resize_data(ngenes, nsamples, nobs)
    timeseriesdb.add_names(sample_name_array)
    timeseriesdb.add_timepoints(mm[time_name])
    data = []
    indptr = []
    indices = []
    #Use to ensure all indices exist at the end
    unique_indices = []
    hashseq_list = []
    j = 0
    for sequence, abundance_dict in seqcount.iteritems():
        hashseq = hash(sequence)
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
                
        outseqf.write(">%s;size=%d;\n" % (hashseq,rowsum))
        outseqf.write("%s\n" % sequence)

        if (len(indptr) >= 500):
            timeseriesdb.add_timeseries_data(data, indices, indptr, hashseq_list)
            hashseq_list = []
            data = []
            indices = []
            indptr = []
            
    timeseriesdb.add_timeseries_data(data, indices, indptr, hashseq_list)
    print("Done writing to %s"%timeseriesdata_path)
    #Check consistency of the data
    if (len(unique_indices) < nsamples):
        print("WARNING: Number of time-points retrieved from sequence file is less than the number of samples in metadata file. %d samples are missing. This is probably not what you want." % (nsamples - len(unique_col),))
