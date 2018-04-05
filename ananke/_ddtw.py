import numpy as np

def diff_dict(x,y):
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

def DDTW(signal_1, signal_2):
    '''
    Arguments:
        signal_1 -- first time series, numpy array of shape ( n1,  )
        signal_2 -- second time series, numpy array of shape ( n2,  )
    Results:
        ddtw -- distance matrix, numpy array of shape ( n1 - 2, n2 - 2 )
        ddtw_traceback -- traceback matrix, numpy array of shape ( n1 - 2, n2 - 2 )
    ''' 
    assert signal_1.shape[0] != 0 and signal_2.shape[0] != 0, '''Input signals must be a column vectors,
                                                                Please check the input signal dimension.'''
    assert signal_1.shape[0] >= 3 and signal_2.shape[0] >= 3, '''The length of your signal should be 
                                                                 greater than 3 to implement DDTW.'''
    n_rows = signal_1.shape[0]-2
    n_cols = signal_2.shape[0]-2
    ddtw = np.zeros((n_rows,n_cols))
    ddtw_traceback = np.zeros((n_rows,n_cols))
    ddtw[0,0] = diff_dist(signal_1[0:3], signal_2[0:3])
    for i in range(1, n_rows):
        ddtw[i,0] = ddtw[i-1,0] + diff_dist(signal_1[i-1:i+2], signal_2[0:3])
        ddtw_traceback[i,0] = 1
    for j in range(1, n_cols):
        ddtw[0,j] = ddtw[0,j-1] + diff_dist(signal_1[0:3], signal_2[j-1:j+2])
        ddtw_traceback[0,j] = 2
    for i in range(1, n_rows):
        for j in range(1, n_cols):
            temp = np.array([ddtw[i-1,j-1], ddtw[i-1,j], ddtw[i,j-1]])
            best_idx = np.argmin(temp)
            ddtw[i,j] = diff_dist(signal_1[i-1:i+2], signal_2[j-1:j+2]) + temp[best_idx]
            ddtw_traceback[i,j] = best_idx
    print(ddtw[-1,-1])
    return ddtw, ddtw_traceback
