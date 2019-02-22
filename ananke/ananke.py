#!/usr/bin/env python
        
#Argument handler for temporal clustering scripts

import argparse
import numpy as np
from ananke._database import Ananke
from ananke._simulation import simulate_and_import, score_simulation

def main():
    #  Argument parsing
    p = argparse.ArgumentParser(prog='main')
    sp = p.add_subparsers(title='subcommands',
                               description='The following subcommands \
                               are possible: initialize, info, import, filter, \
                               cluster', dest='action')
    
    #### Initialize
    ##### These subcommands are for creating a new (empty) AnankeDB file
    ##### from some input format. This builds the database with the appropriate
    ##### shapes and sample names.
    init_p = sp.add_parser("initialize")
    
    init_sp = init_p.add_subparsers(title="initialize", dest='initialize',
                                    description="Create a new Ananke database file by" \
                                                " specifying shape or a mapping file.")
    map_p = init_sp.add_parser("mapping")
    map_p.add_argument("-o", metavar="output",
                       help="Output HDF5 data file",
                       required=True,
                       type=str)
    map_p.add_argument("-m", metavar="mapping",
                       help="Metadata mapping file",
                       required=True,
                       type=str)
    map_p.add_argument("-s", metavar="sample_id_col",
                       help="Name of the column that contains the sample IDs",
                       required=True,
                       type=str)
    map_p.add_argument("-t", metavar="time_col",
                       help="Name of the column that contains the time points",
                       required=True,
                       type=str)
    map_p.add_argument("-n", metavar="series_name_col",
                       help="Name of the column that contains the series identifiers",
                       required=False,
                       type=str,
                       default=None)
    map_p.add_argument("-r", metavar="replicate_col",
                       help="Name of the column that contains the replicate identifiers",
                       required=False,
                       type=str,
                       default=None)
    map_p.add_argument("-f", metavar="time_format_str",
                       help="Format string for Arrow package to parse the time points",
                       required=False,
                       type=str,
                       default='X')

    shape_p = init_sp.add_parser("shape")
    shape_p.add_argument("-o", metavar="output",
                         help="Output HDF5 data file",
                         required=True,
                         type=str)
    shape_p.add_argument("-n", metavar="num_timepoints",
                         required=False,
                         type=int,
                         default=180)
    shape_p.add_argument("-t", metavar="timepoints",
                         help="Timepoints for the simulations",
                         nargs="*",
                         required=False,
                         default=None)
    shape_p.add_argument("-s", metavar="num_series",
                         help="Number of series to generate",
                         required=False,
                         type=int,
                         default=1)
    shape_p.add_argument("-r", metavar="num_replicates",
                         help="Number of replicates to generate (per series)",
                         required=False,
                         type=int,
                         default=1)

    #### Info
    ##### These subcommands provide information on an Ananke file or analysis
    ##### Currently, just takes in a file and spits out the __str__ representation
    ##### of the AnankeDB
    info_p = sp.add_parser("info")
    info_p.add_argument("-i", metavar="input",
                        help="Input .h5 file",
                        required=True,
                        type=str)
    
    ##### Import
    ###### These subcommands fill an empty AnankeDB file with counts from a
    ###### given data format
    import_p = sp.add_parser("import")
    import_sp = import_p.add_subparsers(title="import", dest="import_action",
                                        description="Import data into an initialized " \
                                                    "Ananke database file from fasta, " \
                                                    "or simulation.")

    simu_p = import_sp.add_parser("simulation")
    simu_p.add_argument("-i", metavar="input",
                        help="Input Ananke HDF5 file",
                        required=True,
                        type=str)
    simu_p.add_argument("-c", metavar="nclusts",
                        help="Number of clusters seeds to simulate (distinct time series patterns)",
                        default=50, type=int)
    simu_p.add_argument("-t", metavar="nts_per_clust",
                        help="Number noisy samples of each seed to simulate (distinct time series)",
                        default=10, type=int)
    simu_p.add_argument("-n", metavar="nsr",
                        help="Noise to signal ratio. Proportion of random noise time series, " \
                             "relative to amount of signal time series.",
                        default=1, type=float)
    simu_p.add_argument("-s", metavar="shift_amount",
                        help="Maximum amount that a simulated time series will be randomly shifted to " \
                             "simulate lagged processes",
                        default=0, type=int)
    simu_p.add_argument("-v", metavar="signal_variance",
                        help="Variance for the negative binomial sampling of the underlying process",
                        default=1.75, type=float)
    simu_p.add_argument("-o", metavar="output",
                        help="Prefix for output of ground truth signal data",
                        type=str,
                        default=None)
    
    filter_p = sp.add_parser("filter")
    filter_p.add_argument("-i", metavar="input",
                          help="Input Ananke HDF5 file",
                          required=True,
                          type=str)
    filter_p.add_argument("-m", metavar="method",
                          help="min_sample_presence requires that the sequence be observed in at least " \
                               "threshold samples. min_sample_proportion requires that the sequence be " \
                               "observed in at least threshold proportion of the samples.",
                          choices=["min_sample_presence", "min_sample_proportion"],
                          default="min_sample_presence", type=str)
    filter_p.add_argument("-t", metavar="threshold",
                          help="Filter threshold for the desired filter method.",
                          default=2,
                          type=float)

    comp_dist_p = sp.add_parser("compute_distances")
    comp_dist_p.add_argument("-i", metavar="input",
                           help="Input Ananke HDF5 file",
                           required=True,
                           type=str)
    comp_dist_p.add_argument("-m", metavar="dist_min",
                           help="Minimum distance measure for precomputation",
                           type=float,
                           default=0.001)
    comp_dist_p.add_argument("-M", metavar="dist_max",
                           help="Maximum distance measure for precomputation",
                           type=float,
                           default=0.15)
    comp_dist_p.add_argument("-s", metavar="dist_step",
                           help="Minimum distance measure for precomputation",
                           type=float,
                           default=0.005)
    comp_dist_p.add_argument("-d", metavar="distance_measure",
                           help="Distance measure for clustering",
                           choices=["sts","dtw","ddtw","euclidean"],
                           default="sts",
                           type=str)
    comp_dist_p.add_argument("--on-disk", action="store_false", dest="in_memory", 
                           help="If specified, the data will be pulled from disk. " \
                           "This will be slower, but reduces the memory footprint " \
                           "significantly.")
    comp_dist_p.add_argument("-z", metavar="simulation_signal",
                           help="Simulation signal used for scoring simulations.",
                           type=str)
    comp_dist_p.add_argument("-o", metavar="simulation_score_output",
                             help="File to append the simulation scores to",
                             type=str)

    cluster_p = sp.add_parser("cluster")
    cluster_p.add_argument("-i", metavar="input",
                           help="Input Ananke HDF5 file",
                           required=True,
                           type=str)
    cluster_p.add_argument("-e", metavar="epsilon",
                           help="Epsilon value for DBSCAN clustering. Will use nearest " \
                                 "value if supplied value was not precomputed.",
                           type=float,
                           default=0.01)
    cluster_p.add_argument("-m", metavar="min_pts",
                           help="DBSCAN min_pts parameter.",
                           default=2,
                           type=int)
    
    args = p.parse_args()
    

    if args.action == 'initialize':
        if args.initialize == 'mapping':
            adb = AnankeDB(args.o)
            adb.initialize_from_metadata()
        elif args.initialize == 'shape':
            adb = AnankeDB(args.o)
            initialize_by_shape(adb, args.n, args.t, args.s, args.r)
    elif args.action == 'info':
        adb = AnankeDB(args.i)
        print(adb)
    elif args.action == 'import':
        if args.import_action == 'simulation':
            adb = AnankeDB(args.i)
            true_data = simulate_and_import(adb, args.c, args.t, args.n, args.s, args.v)
            if args.o is not None:
                np.savetxt(args.o + "_signals.gz", true_data)
        elif args.import_action == 'fasta':
            raise NotImplementedError
    elif args.action == 'filter':
        adb = AnankeDB(args.i)
        adb.filter_data(args.m, args.t)
    elif args.action == 'compute_distances':
        adb = AnankeDB(args.i)
        dist_range = np.arange(args.m, args.M, args.s)
        dbs = precompute_distances(adb, args.d, dist_range, in_memory=args.in_memory)
        if args.z is not None:
            outfile = open(args.o, 'a')
            scores = score_simulation(adb, dbs, args.z, args.d, outfile)
            outfile.close()
    elif args.action == 'cluster':
        adb = AnankeDB(args.i)
        dbs = load_blooms(adb)
        print(dbs.DBSCAN(args.e, args.m))
    

if __name__ == "__main__":
    main()

