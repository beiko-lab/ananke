from _ts_simulation import *
from _ddtw import *
from scipy.spatial.distance import squareform
import time
import matplotlib
import matplotlib.pyplot as plt
sim = gen_table(fl_sig=0,w_sig=6,
              fl_bg=-6,w_bg=6,
              bg_disp_mu=0,bg_disp_sigma=1,
              sig_disp_mu2=0,sig_disp_sigma2=1,
              n_clust=10,n_sig=10,n_tax_sig=1,n_bg=10,
              len_arima=100,len_ts=50,len_signal=30)
X = sim['signals']
print('generating distance matrix')
start = time.time()
d = np.zeros(int(X.shape[0]*(X.shape[0]-1)/2))
c = 0
for i in range(0,X.shape[0]):
    for j in range(i+1, X.shape[0]):
        ddtw, _ = DDTW(X[i,:], X[j,:])
        d[c] = ddtw[-1,-1]
        c += 1
distance_matrix = squareform(d)
end = time.time()
print('distance matrix generation time:', end-start)
distance_matrix = distance_matrix/np.max(distance_matrix)
plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')
plt.title('Vusualization of Distance Matrix generated using DDTW on simulated signals')
plt.savefig('demo_figure.png')