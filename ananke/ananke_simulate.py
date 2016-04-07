import numpy as np
from scipy.sparse import csr_matrix
from ananke_database import TimeSeriesData
from sklearn import metrics

def generate_random_trials(template, ntimepoints, nreps):
    a = np.ndarray(shape=(nreps,ntimepoints))
    #First, we spin a bunch of variance on our basic binary template
    variance_template = [0]*ntimepoints
    for i in range(len(template)):
        #variance for 0's is 0.01, but for 1's it's 0.11
        variance_template[i]=np.random.normal(template[i],template[i]/10.0+0.01)
    #Next, our separate trials will be scaled, and have additional noise added
    for i in range(nreps):
        Y=[0]*ntimepoints
        #Scaling factor follows a Weibull distn, since we want the occasional
        #large value. This mimics what we see in real data (one abundant real
        #sequence and several less abundant echoes)
        #Require a minimum amount of info in a sequence
        while (sum(Y) < 10):
            mult = max(10,100*np.random.weibull(0.3))
            for j in range(len(Y)):
                Y[j]=int(mult*max(0,np.random.normal(variance_template[j],max(0.01,variance_template[j]/10.0+0.05))))
        a[i,:]=Y
    return a

def generate_drop(ntimepoints, nreps):
    #- Pick a random position between ntimepoints/5 and 4*ntimepoints/5
    #- Create a template such that it is 1 before/on that position, and
    #  0 afterwards
    event_pos = np.random.randint(int(ntimepoints/5.0),int(4*ntimepoints/5.0))
    event_template = [1]*event_pos + [0]*(ntimepoints-event_pos)
    return generate_random_trials(event_template, ntimepoints, nreps)

def generate_rise(ntimepoints, nreps):
    #- Pick a random position between ntimepoints/5 and 4*ntimepoints/5
    #- Create a template such that it is 0 before/on that position, and
    #  1 afterwards
    event_pos = np.random.randint(int(ntimepoints/5.0),int(4*ntimepoints/5.0))
    event_template = [0]*event_pos + [1]*(ntimepoints-event_pos)
    return generate_random_trials(event_template, ntimepoints, nreps)

def generate_normal(ntimepoints, nreps):
    #normal
    event_template = [1]*ntimepoints
    return generate_random_trials(event_template, ntimepoints, nreps)

def generate_noisy(ntimepoints, nreps):
    #noisy data
    event_template = [0.01]*ntimepoints
    return generate_random_trials(event_template, ntimepoints, nreps)

def generate_conditionally_rare(ntimepoints, nreps):
    #spike
    event_pos = np.random.randint(int(ntimepoints/5.0),int(4*ntimepoints/5.0))
    event_template = [0]*(event_pos-1) + [5] + [0]*(ntimepoints-event_pos)
    return generate_random_trials(event_template, ntimepoints, nreps)

def generate_seasonal(ntimepoints, nreps):
    #seasonal
    #Generate random period
    event_period = np.random.randint(3,max(4,ntimepoints/3))
    event_direction = np.random.randint(0,2)
    event_template = []
    while (len(event_template) < ntimepoints):
        event_template += [event_direction]*event_period + [1-event_direction]*event_period
    event_template = event_template[0:ntimepoints]
    return generate_random_trials(event_template, ntimepoints, nreps)

def create_simulation_data(h5_file, ntimepoints, nreps):
    print("Opening/creating database file")
    tsdatabase = TimeSeriesData(h5_file)
    print("Generating data")
    dense_matrix = np.ndarray(shape=(6*nreps, ntimepoints), dtype=np.float)
    dense_matrix[0:nreps,:] = generate_drop(ntimepoints, nreps)
    dense_matrix[nreps:2*nreps,:] = generate_rise(ntimepoints, nreps)
    dense_matrix[2*nreps:3*nreps,:] = generate_normal(ntimepoints, nreps)
    dense_matrix[3*nreps:4*nreps,:] = generate_noisy(ntimepoints, nreps)
    dense_matrix[4*nreps:5*nreps,:] = generate_conditionally_rare(ntimepoints, nreps)
    dense_matrix[5*nreps:6*nreps,:] = generate_seasonal(ntimepoints, nreps)
    print("Sparsifying matrix")
    sparse_matrix = csr_matrix(dense_matrix)
    print(sparse_matrix)
    nobs = len(sparse_matrix.data)
    print("Inserting data")
    tsdatabase.resize_data(6*nreps, ntimepoints, nobs)
    tsdatabase.insert_array_by_chunks('timeseries/data', sparse_matrix.data)
    tsdatabase.insert_array_by_chunks('timeseries/indptr', sparse_matrix.indptr)
    tsdatabase.insert_array_by_chunks('timeseries/indices', sparse_matrix.indices)
    tsdatabase.insert_array_by_chunks('genes/sequenceids', range(6*nreps))
    tsdatabase.insert_array_by_chunks('samples/time', range(ntimepoints))
    print("Done!")

def score_simulation(h5_file):
    print("Opening/creating database file")
    tsdatabase = TimeSeriesData(h5_file)
    nreps = (tsdatabase.h5_table["timeseries/indptr"].shape[0]-1)/6
    #Items belonging in the same cluster are next to one another
    true_labels = [0]*nreps+[1]*nreps+[2]*nreps+[3]*nreps+[4]*nreps+[5]*nreps
    #Order is: drop, rise, normal, noisy, conditionally rare, seasonal
    max_nmi = 0
    for i in range(tsdatabase.h5_table["genes/clusters"].shape[1]):
        pred_labels = tsdatabase.get_cluster_labels(i)
        nmi = metrics.adjusted_mutual_info_score(true_labels, pred_labels)
        if (nmi > max_nmi):
            max_nmi = nmi
    print("Maximum NMI of clusters is: %f" % (max_nmi,))
