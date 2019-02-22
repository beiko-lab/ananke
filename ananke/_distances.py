import numpy as np
### derivative dynamic time warping implementation of fastdtw
### this implementation is based on fastdtw (https://github.com/slaypni/fastdtw)

from fastdtw import fastdtw
import finitediff as fd

import matplotlib
import matplotlib.pyplot as plt

def distance_function(distance_measure, **kwargs):
    if distance_measure == "sts":
        return STS(kwargs["time_points"])
    elif distance_measure == "dtw":
        return DTW()
    elif distance_measure == "ddtw":
        return DDTW(kwargs["time_points"])
    elif distance_measure == "euclidean":
        return Euclidean()
    else:
        raise ValueError("Unrecognized distance measure '%s'." % (distance_measure,))

class TSDist():
    def __init__(self, aggregate_func=sum):
        self.aggregate_func = aggregate_func

# Core distance functions
# Each of these must take in 
class DTW(TSDist):
    def __init__(self):
        self.name = "DTW"

    def distance(self, data1, data2):
        distance, path = fastdtw(data1, data2)
        return distance

    def transform_matrix(self, matrix):
        return matrix

    def transform_row(self, row):
        return row

class DDTW(TSDist):
    def __init__(self, time_points):
        self.name = "DDTW"
        self.time_points = [int(x) for x in time_points]
        self.interp_points = [int(x) + 0.05 for x in time_points]

    def distance(self, derivative1, derivative2):
        # 2 = 2-norm = Euclidean distance
        distance, path = fastdtw(derivative1, derivative2)
        return distance

    def transform_row(self, row, nhead = 5, ntail = 5):
        return fd.interpolate_by_finite_diff(self.time_points, 
                                             row, self.interp_points,
                                             maxorder=1, ntail=ntail, 
                                             nhead=nhead)[:,1]

    def transform_matrix(self, matrix, nhead = 5, ntail = 5):
        return fd.interpolate_by_finite_diff(self.time_points, matrix, self.interp_points, 
                                             maxorder=1, ntail=ntail, nhead=nhead)[:,:,1].T

class STS(TSDist):
    def __init__(self, time_points):
        self.name = "STS"
        self.time_delta = np.array(time_points[1:]) - np.array(time_points[0:-1])
    
    def distance(self, slopes1, slopes2):
        distance = slopes1 - slopes2
        distance = np.square(distance)
        distance = np.sqrt(sum(distance))
        return distance
    
    def transform_matrix(self, matrix):
        slope_matrix = matrix[:, 1:] - matrix[:, 0:-1]
        slope_matrix = slope_matrix / self.time_delta
        return slope_matrix

    def transform_row(self, row):
        return (row[1:] - row[0:-1]) / self.time_delta


class Euclidean(TSDist):
    def __init__(self):
        self.name = "Euclidean"

    def distance(self, data1, data2):
        return np.sqrt(sum(np.square(data1-data2)))

    def transform_matrix(self, matrix):
        return matrix

    def transform_row(self, row):
        return row

def plot_raw_signals(signal_1, signal_2, title = 'raw_signals'):
    '''
    Arguments:
        signal_1 -- first time series, numpy array of shape ( n1,  )
        signal_2 -- second time series, numpy array of shape ( n2,  )
    Results:
          Figure 
    ''' 
    plt.plot(signal_1)
    plt.plot(signal_2)
    plt.grid()
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.show()

def plot_alignment_path(path, title = 'alignment_path'):
    '''
    Arguments:
          path -- aligned indices, list of indices
    Results:
        Figure
    '''
    plt.plot([index_pair[0] for index_pair in path], [index_pair[1] for index_pair in path])
    plt.grid()
    plt.title(title)
    plt.xlabel('signal 1')
    plt.ylabel('signal 2')
    plt.show()
    
def plot_aligned_signals(signal_1, signal_2, path, title = 'aligned_signals'):
    '''
    Arguments:
        signal_1 -- first time series, numpy array of shape ( n1,  )
        signal_2 -- second time series, numpy array of shape ( n2,  )
            path -- aligned indices, list of indices
    Results:
          Figure 
    ''' 
    plt.plot([signal_1[index_pair[0]] for index_pair in path])
    plt.plot([signal_2[index_pair[1]] for index_pair in path])
    plt.grid()
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('value')
    plt.show()
