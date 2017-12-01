#!/usr/bin/env python
        
#Argument handler for temporal clustering scripts

import argparse

from ._tabulate import fasta_to_ananke, dada2_to_ananke
from ._cluster import run_cluster
from ._database import TimeSeriesData
from .ananke_simulate import create_simulation_data, score_simulation
from .ananke_stats import print_database_info
from .ananke_misc import translate_otus, rarefy_even

def main():
    #  Argument parsing
    parser = argparse.ArgumentParser(prog='main')
    subparsers = parser.add_subparsers(title='subcommands',description='the following subcommands are possible: tabulate, filter, cluster, add', dest='subparser_name')
    
    #  Tabulate script options
    tabulate_parser = subparsers.add_parser("tabulate")
    tabulate_parser.add_argument("-i", metavar="input", help="FASTA sequence file", required=True, type=str)
    tabulate_parser.add_argument("-m", metavar="mapping", help="Metadata mapping file", required=True, type=str)
    tabulate_parser.add_argument("-o", metavar="output", help="Output HDF5 data file", required=True, type=str)
    tabulate_parser.add_argument("-f", metavar="fasta", help="Output FASTA file", required=True, type=str)
    tabulate_parser.add_argument("-t", metavar="time_label", help="Column name for time points in metadata file", default="time_points", type=str)
    tabulate_parser.add_argument("--multi", help="Indicates the input data contains multiple time-series, requires column name of time-series mask variable", type=str, required=False)
    tabulate_parser.add_argument("--size_labels", help="Toggle if the number of occurrences of each sequence is in the FASTA label (in the format: '>SampleXYZ_50;size=100')", action="store_true")

    dada2_parser = subparsers.add_parser("import_from_dada2")
    dada2_parser.add_argument("-i", metavar="input", 
                              help="Input DADA2 table that has sequences as \
                                   rows and samples as columns",
                              required=True, type=str)
    dada2_parser.add_argument("-m", metavar="mapping",
                              help="Metadata mapping file",
                              required=True, type=str)
    dada2_parser.add_argument("-o", metavar="output",
                              help="Output HDF5 data file",
                              required=True, type=str)
    dada2_parser.add_argument("-f", metavar="fasta",
                              help="Output FASTA file",
                              required=True, type=str)
    dada2_parser.add_argument("-t", metavar="time_label",
                              help="Column name for time points in metadata \
                                    file",
                              required=True, type=str)
    dada2_parser.add_argument("--multi", help="Indicates the input data \
                                               contains multiple time-series, \
                                               requires column name of \
                                               time-series mask variable",
                              required=False, type=str)

    #  Filter script options
    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("-i", 
                               metavar="input", 
                               help="HDF5 data file", 
                               required=True, 
                               type=str)
    filter_parser.add_argument("-o", 
                               metavar="output", 
                               help="Filename for output filtered data file", 
                               required=True, 
                               type=str)
    filter_parser.add_argument("-t", metavar="threshold", help="Threshold for filtering criterion", required=True, type=str)
    filter_parser.add_argument("-f", metavar="filter", help="Filter type: proportion, abundance, presence", required=True, type=str, default='presence')

    #  Rarefy script options
    rarefy_parser = subparsers.add_parser("rarefy")
    rarefy_parser.add_argument("-i", metavar="input", help="HDF5 data file", required=True, type=str)

    #  Cluster script options
    cluster_parser = subparsers.add_parser("cluster")
    cluster_parser.add_argument("-i", metavar="input", help="HDF5 data file", required=True, type=str)
    cluster_parser.add_argument("-n", metavar="numthreads", help="Number of threads for computing distance matrix", required=True, type=str)
    cluster_parser.add_argument("-l", metavar="min_eps", help="Lower bound for epsilon for clustering step.", default=0.01, required=False, type=float)
    cluster_parser.add_argument("-u", metavar="max_eps", help="Upper bound for epsilon for clustering step.", default=10.0, required=False, type=float)
    cluster_parser.add_argument("-s", metavar="step_eps", help="Step size for epsilon for clustering step.", default=0.01, required=False, type=float)
    cluster_parser.add_argument("-d", metavar="dist_measure", help="Distance measure (default: sts, Options: sts, euclidean, cityblock, cosine, l1, l2, manhattan)", required=False, type=str, default="sts")
    cluster_parser.add_argument("--clr",
                                help="Normalize by centre log ratio transform" \
                                " to properly deal with compositionality of" \
                                " the data. If -c is not given, normalization" \
                                " is done using sample-wise proportions and " \
                                " time-series are transformed with Z-scores",
                                action="store_true")

    #  Status script options
    status_parser = subparsers.add_parser("info")
    status_parser.add_argument("-i", metavar="input", help="HDF5 data file", required=True, type=str)

    #  Add data scripts (taxonomy + sequence clusters)
    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("action", choices=["taxonomy", "sequence_clusters"], help="taxonomy: add taxonomic classifications into data file; sequence_clusters: add traditional sequence-based clustering labels into data file")
    add_parser.add_argument("-i", metavar="input", help="HDF5 data file", required=True, type=str)
    add_parser.add_argument("-d", metavar="data", help="Data file to add (taxonomic label file or identity cluster files)", required=True, type=str)
    simulation_parser = subparsers.add_parser("simulation")
    simulation_parser.add_argument("-d", metavar="database", help="HDF5 database filename (will be created)", required=True, type=str)
    simulation_parser.add_argument("-t", metavar="ntimepoints", help="Number of time-points in simulation", required=True, type=int)
    simulation_parser.add_argument("-r", metavar="nreps", help="Number of repetitions for each seed", required=True, type=int)
    score_parser = subparsers.add_parser("score_simulation")
    score_parser.add_argument("-d", metavar="database", help="HDF5 database filename (will be created)", required=True, type=str)

    #  Add misc utilities
    clusters_parser = subparsers.add_parser("translate_clusters")
    clusters_parser.add_argument("-i", metavar="input", help="Input FASTA sequence file", required=True, type=str)
    clusters_parser.add_argument("-c", metavar="clusters", help="Input cluster file (seq_otus.txt)", required=True, type=str)
    clusters_parser.add_argument("-o", metavar="output", help="Output cluster file", required=True, type=str)
    args = parser.parse_args()
    
    #Route to the proper routines
    if args.subparser_name == "tabulate":
        fasta_to_ananke(args.i, args.m, args.t, args.o, args.f, args.multi, args.size_labels)
    elif args.subparser_name == "import_from_dada2":
        dada2_to_ananke(args.i, args.m, args.t, args.o, args.f, args.multi)
    elif args.subparser_name == "filter":
        timeseriesdb = TimeSeriesData(args.i)
        timeseriesdb.filter_data(args.o, args.f, args.t)
    elif args.subparser_name == "cluster":
        run_cluster(args.i, int(args.n), args.d, args.l, 
                    args.u, args.s, args.clr)
    elif args.subparser_name == "add":
        if args.action == "taxonomy":
            timeseriesdb = TimeSeriesData(args.i)
            timeseriesdb.add_taxonomy_data(args.d)
        elif args.action == "sequence_clusters":
            timeseriesdb = TimeSeriesData(args.i)
            timeseriesdb.add_sequencecluster_data(args.d)
    elif args.subparser_name == "simulation":
        create_simulation_data(args.d, args.t, args.r)
    elif args.subparser_name == "score_simulation":
        score_simulation(args.d)
    elif args.subparser_name == "info":
        timeseriesdb = TimeSeriesData(args.i)
        print_database_info(timeseriesdb)
    elif args.subparser_name == "translate_clusters":
        translate_otus(args.i, args.c, args.o)
    elif args.subparser_name == "rarefy":
        timeseriesdb = TimeSeriesData(args.i)
        rarefy_even(timeseriesdb)

if __name__ == "__main__":
    main()
