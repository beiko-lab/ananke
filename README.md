# timeclust: Clustering marker gene data by time-series patterns
timeclust is a tool to cluster your marker gene data (such as 16S) by time-series abundance patterns. This creates groups of sequences that have shown the same changes in response to the environment. This tool facilitates the generation of the clusters, and the companion [timeclust-ui](https://github.com/beiko-lab/timeclust-ui) application aids in the exploration of the results.

## Installation
The following Python packages are required:
- h5py >= 2.3.1
- numpy >= 1.6
- scipy >= 0.16.1
- scikit-learn >= 0.16
- pandas>=0.17

To install timeclust, download the [source code](https://github.com/beiko-lab/timeclust/archive/master.zip), and in the extracted directory run:
```
python setup.py install
```
This can be run as root if you want to install timeclust globally.
