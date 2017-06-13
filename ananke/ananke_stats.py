import numpy as np

def print_database_info(tsdb):
    print("ananke database file information:")
    print("Time-series size: " + 
          str(tsdb.h5_table["timeseries/indptr"].shape[0] - 1) + 
          "x" + 
          str(len(tsdb.get_array_by_chunks("samples/time"))))
    print("Sparse time-series shape: ")
    print("  data:" + str(tsdb.h5_table["timeseries/data"].shape))
    print("  indptr:" + str(tsdb.h5_table["timeseries/indptr"].shape))
    print("  indices:" + str(tsdb.h5_table["timeseries/indices"].shape))
    
