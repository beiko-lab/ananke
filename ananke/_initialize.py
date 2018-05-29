import numpy as np
import random
from ._database import AnankeDB

# This file contains the initializers that shuttle data into Ananke files/objects

# TODO: Update to accept lists of length nseries for timepoints
def initialize_by_shape(anankedb, ntimepoints = 180, timepoints = None, nseries = 1, nreplicates = 1):
    names = []
    if timepoints is None:
        timepoints = np.array(np.cumsum([random.randint(1,15) for i in range(ntimepoints)]))
    for i in np.arange(nseries):
        for j in np.arange(nreplicates):
            if nseries > 1:
                series_name = "timeseries%d" % (i,)
                series_rep_name = series_name + "_%d" % (j,)
            else:
                series_name = "timeseries"
                series_rep_name = series_name
            names.append(series_rep_name)
        sample_names = [ series_rep_name + "_S%d" % (j,) for j in np.arange(len(timepoints)) ]
        anankedb.create_series(series_rep_name, 
                               timepoints,
                               sample_names) 
        if nseries > 1:
            attrs = anankedb._h5t["data/" + series_rep_name].attrs
            attrs.create("series", 
                         series_name.encode(),
                         dtype=h5.special_dtype(vlen=bytes))
        if nreplicates > 1:
            attrs = anankedb._h5t["data/" + series_rep_name]
            attrs.create("replicate", 
                         str(j).encode(),
                         dtype=h5.special_dtype(vlen=bytes))
    return names

def initialize_from_metadata(anankedb, metadata_path, name_col,
                 time_col, time_format="X",
                 replicate_col=None, series_col=None):
    """Take in a path, metadata_path, read file into a pandas dataframe.
    Validate that time_name and time_mask (if applicable) are present.

    Parameters
    ----------
    metadata_path: str
    filepath to metadata file
    name_col: str
    name of column in metadata file that contains the sample names
    time_col: str
    name of column in metadata file that contains time points
    time_format: str
    format of the timestamp, e.g. "MM/DD/YYYY". If left blank, then
    it will treat times as integers.
    replicate_col: str
    name of column in metadata file that contains the replicate category 
    that denotes the replicate the sample belongs to
    series_col: str
    name of column in metadata file that contains the series category
    that denotes the time series the sample belongs to

    Returns
    -------
    mm: pandas.DataFrame
    Pandas dataframe containing sample metadata
    """

    mm = pd.read_csv(metadata_path, sep="\t", header=0)

    if time_col not in mm:
        raise KeyError("Specified time point column name (%s) is not found " \
                       "in metadata file." % (time_col,))

    if name_col not in mm:
        raise KeyError("Specific name column (%s) is not found in " \
                       "metadata file." % (name_col,))

    parsed_times = [arrow.get(str(x), time_format) for x in mm[time_col]]
    time_offsets = [(x - min(parsed_times)).total_seconds() for x in parsed_times]
    mm["_offsets"] = time_offsets

    if replicate_col is not None:
        if replicate_col not in mm:
            raise KeyError("Specified replicate column name (%s) is not " \
                           "found in metadata file." % (replicate_col,))
    if series_col is not None:
        if series_col not in mm:
            raise KeyError("Specified series column name (%s) is not " \
                           "found in metadata file." % (series_col,))

    sort_order = [x for x in [series_col, replicate_col, "_offsets"] \
                  if x is not None]
    mm = mm.sort_values(by=sort_order)

    if len(sort_order) == 3:
        #We have the trifecta: replicates, multi timeseries, and the offsets
        for series in mm[series_col].unique():
            series_subset = mm[mm[series_col == series]]
            for replicate in series_subset[replicate_col].unique():
                time = series_subset[
                       series_subset[replicate_col] == replicate, time_col]
                name = str(series) + "_" + str(replicate)
                sample_names = series_subset[name_col]
                self.create_series(name, time, sample_names)
                attrs = self._h5t["data/" + name].attrs
                attrs.create("replicate", replicate.encode(),
                             dtype=h5.special_dtype(vlen=bytes))
                attrs.create("series", series.encode(),
                             dtype=h5.special_dtype(vlen=bytes))

    elif len(sort_order) == 2:
        #We only have two to deal with either series or replicate
        for rep_or_series in mm[sort_order[0]].unique():
            subset = mm[mm[sort_order[0]] == rep_or_series]
            time = np.array(subset[time_col], dtype=str)
            name = str(rep_or_series)
            sample_names = subset[name_col]
            self.create_series(name, time, sample_names)
        if replicate_col is not None:
            att_str = "replicate"
        else:
            att_str = "series"

            self._h5t["data/" + name].attrs.create(att_str, name.encode(),
                          dtype=h5.special_dtype(vlen=bytes))
    else:
        #We only have one to deal with
        time = mm[time_col]
        sample_names = mm[name_col]
        self.create_series("timeseries", time, sample_names)
