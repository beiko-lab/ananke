# *timeclust*: Clustering marker gene data by time-series patterns
*timeclust* is a tool to cluster your marker gene data (such as 16S) by time-series abundance patterns. This creates groups of sequences that have shown the same changes in response to the environment. This tool facilitates the generation of the clusters, and the companion [timeclust-ui](https://github.com/beiko-lab/timeclust-ui) application aids in the exploration of the results.

## Installation
The following Python packages are required:
- h5py >= 2.3.1
- numpy >= 1.6
- scipy >= 0.16.1
- scikit-learn >= 0.16
- pandas >= 0.17

To install *timeclust*, download the [source code](https://github.com/beiko-lab/timeclust/archive/master.zip), and in the extracted directory run:
```
python setup.py install
```
This can be run as root if you want to install *timeclust* globally.

## Running timeclust
### Required input data
*timeclust* requires a pretty minimal set of input data. It requires a FASTA sequence file and a metadata mapping file. The sequence file contains all sequences (pre-filtered for quality, if desired) with the sample ID in the FASTA header, followed by a sequence count. Sequences must be on a single line. For example:
```
>Sample1_0
ATGCGCATGCTATGCAT
>Sample1_1
ATCGAGCATCGATCGAC
>Sample2_0
AGCGATCGATCGATCGAT
...
```
If your FASTA file has sequences split into multiple lines, the clever guys who made [SWARM](https://github.com/torognes/swarm#linearization) have a little awk script to convert it to the required format:
```
awk 'NR==1 {print ; next} {printf /^>/ ? "\n"$0"\n" : $1} END {printf "\n"}' amplicons.fasta > amplicons_linearized.fasta
```

The metadata file is a tab-separated sheet with the first column label as "#SampleID" and one of the metadata columns must contain the time offset. The metadata file from a QIIME analysis is compatible:
```
#SampleID time_points
Sample1 0
Sample2 1
Sample3 5
Sample4 10
...
```
Note that you may only have one sample per time-point.

### Processing pipeline
There are several individual steps in the pipeline, listed below with the *timeclust* subcommand in brackets.

1. Tabulate unique sequences, creating a *m* x *n* time-series matrix, where *m* = number of unique sequences and *n* = number of time points (*tabulate*)
2. Filter out low information sequences (*filter*)
3. Cluster sequences using short time-series distance and DBSCAN (*cluster*)
4. (Optional) Add taxonomic information for the unique sequences (*add taxonomy*)
5. (Optional) Add sequence clustering (i.e., 97% OTU) information for the unique sequences (*add sequence_clusters*)

### Step 1: Tabulating sequences

*timeclust tabulate -i sequence_input -o hdf5_output -f unique_sequence_output -m mapping_file -t column_name*:
- -**i**: input FASTA file location
- -**o**: output HDF5 database file location
- -**f**: output unique sequence FASTA file location
- -**m**: input metadata file location (tab-separated)
- -**t**: name of column that describes time-point offset

Example:
```
timeclust tabulate -i sequences.fasta -o timeclust_db.h5 -f seq.unique.fasta -m sequence_metadata.tsv -t time_points
```

### Step 2: Filtering sequences

Sequences can be filtered in one of three ways:
- *proportion*: if a sequuence accounts for less than threshold % of the data, filter it
- *abundance*: if a sequence was seen less than threshold times in total, filter it
- *presence*: if a sequence was seen in fewer than threshold % of time-points, filter it

Due to the magnitude of many high-throughput sequencing data sets, it is a good idea to filter to less than 20,000 unique sequences. Since the software currently calculates a complete *m* x *m* distance matrix between sequences, this can take a lot of RAM for large *m*.

*timeclust filter -i hdf5_input -o hdf5_output -f filter_type -t filter_threshold*:
- -**i**: input HDF5 database file location
- -**o**: output HDF5 filtered database file location
- -**f**: filter type (one of *proportion*, *abundance*, or *presence* (default))
- -**t**: threshold value

Example:
```
#Filter all unique sequences with < 100 occurrences
timeclust filter -i timeclust_db.h5 -o timeclust_db_filtered.h5 -f abundance -t 100
```

### Step 3: Cluster sequences

Sequences are clustered by DBSCAN and results are placed into the existing HDF5 time-series database. Each data set is unique, and therefore will require clustering over a different range of *epsilon* parameters. A good bet is to calculate from 0.1 to 100 in steps of 0.1. You should find the number of clusters (which are printed to screen during computation) increases to a point, and begins decreasing. This should occur well before *eps*=100, but the clustering will halt once *eps* is large enough that everything is clustered into one group. Please note that small step sizes will take up a lot of disk space, as the results are all written to disk. 

*timeclust cluster -i hdf5_input -n num_threads -l eps_lower_limit -u eps_upper_limit -s eps_step_size*:
- -**i**: input HDF5 database file location
- -**n**: number of threads available for computing short time-series distances
- -**l**: *eps* clustering parameter minimum value (must be strictly greater than 0)
- -**u**: *eps* clustering parameter max value (will stop before this if data merges into a single cluster)
- -**s**: *eps* clustering parameter step-size

Example:
```
timeclust cluster -i timeclust_db_filtered.h5 -n 8 -l 0.1 -u 100 -s 0.1
```

### (Optional) Step 4 and 5: Add additional information to database

You can add taxonomic classification and sequence identity cluster information to the time-series database file, and these will appear in the [timeclust-ui](https://github.com/beiko-lab/timeclust-ui) interface for contrasting with the time-series clusters.

You **must** use the unique sequences file from the *tabulate* step as the basis for this information. The labels in this file will match up with the labels used in the time-series database, allowing the information to be properly merged. The unique sequence file has size information contained in the FASTA header to enable it to work with clustering software like USEARCH/UPARSE. This must be removed from the resulting taxonomy and OTU cluster files by using sed: `sed -i 's/;size=[0-9]*;//g' taxonomy.txt`

A more thorough example of how to do this is given [in the wiki](https://github.com/beiko-lab/timeclust/wiki/Generating-OTUs-and-taxonomy-for-import-into-timeclust).

*timeclust add taxonomy -i hdf5_input -d taxonomy_data*
*timeclust add sequence_clusters -i hdf5_input -d sequence_cluster_data*:
- -**i**: input HDF5 database file location
- -**d**: data file (e.g., tax_assignments.txt for taxonomy, or seq_otus.txt for sequence clusters)

Examples:
```
timeclust add taxonomy -i timeclust_db_filtered.h5 -d seq.unique_tax_assignments.txt
timeclust add sequence_clusters -i timeclust_db_filtered.h5 -d seq_otus.txt
```
