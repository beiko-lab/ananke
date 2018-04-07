from _ts_simulation import *
from _ddtw import *
from scipy.spatial.distance import squareform
import time
import matplotlib
import matplotlib.pyplot as plt
data = arima_seed_signal_generation(10, 100, 0, 6)
data = np.apply_along_axis(nbinom_rand, -1, data)
data = generate_signal(data, 50, 5, 0, 1, 0, 1)
X = data
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