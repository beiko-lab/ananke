import numpy as np
### derivative dynamic time warping implementation of fastdtw
### this implementation is based on fastdtw (https://github.com/slaypni/fastdtw)

from fastdtw import fastdtw

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

    def diff_dist(self, x, y):
        '''
        Computing drivative differences between dx and dy
        Arguments:
            x -- dx from signal 1, numpy array of shape ( 3,  )
            y -- dy from signal 2, numpy array of shape ( 3,  )
        Result:
              -- absolute difference of estimated derivatives of x, y
        '''
        dx = ((x[1] - x[0]) + (x[2]-x[0])/2)/2
        dy = ((y[1] - y[0]) + (y[2]-y[0])/2)/2
        return abs(dx-dy)

    def vector_dist(self, x, y):
        dx = ((x[1:-2] - x[0:-3]) + (x[2:-1]-x[0:-3])/2)/2
        dy = ((y[1] - y[0]) + (y[2]-y[0])/2)/2
        return np.abs(dx-np.repeat(dy, len(dx)))

    def distance(self, signal_1, signal_2):
        n_rows = signal_1.shape[0]-2
        n_cols = signal_2.shape[0]-2
        ddtw = np.empty((n_rows,n_cols))
        # Grab the base 3 pt finite difference
        ddtw[0,0] = self.diff_dist(signal_1[0:3], signal_2[0:3])
        vdist = self.vector_dist(signal_1, signal_2[0:3])
        for i in range(1, n_rows):
            ddtw[i,0] = ddtw[i-1,0] + vdist[i-1]
        vdist = self.vector_dist(signal_2, signal_1[0:3])
        for j in range(1, n_cols):
            ddtw[0,j] = ddtw[0,j-1] + vdist[j-1]
        for i in range(1, n_rows):
            vdist = self.vector_dist(signal_2, signal_1[i-1:i+2])
            for j in range(1, n_cols):
                temp = (ddtw[i-1,j-1], ddtw[i-1,j], ddtw[i,j-1])
                if (temp[0] <= temp[1]) & (temp[0] <= temp[2]):
                    best_idx = 0
                elif (temp[1] <= temp[2]) & (temp[1] <= temp[0]):
                    best_idx = 1
                else:
                    best_idx = 2
                ddtw[i,j] = vdist[j-1] + temp[best_idx]
        return ddtw[-1,-1]

    def transform_row(self, row):
        return row

    def transform_matrix(self, matrix):
        return matrix

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
