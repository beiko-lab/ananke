import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
import numpy as np
from scipy.stats import nbinom
seeds = 3
np.random.seed(seeds)
def normab(x, a, b):
    # exp(a) is min mean for nb distribution
    # exp(b) is the max mean for nb distribution
    return (b-a)*(x - np.min(x))/(np.max(x) - np.min(x)) + a
def arima_seed_signal_generation(N, n_samples, a, b):
    # exp(a) is min mean for nb distribution
    # exp(b) is the max mean for nb distribution
    
    n = 0
    Y = np.zeros((N, n_samples))
    while( n < N ):
        flag = 0
        arparams = np.random.uniform(0.97,0.99,1)
        maparams = np.random.uniform(0,   0.99,1)
        y = arma_generate_sample(arparams, maparams, n_samples)
        for i in range(n):
            if np.corrcoef(Y[i,:], y)[0,1] > 0.1:
                flag = 1
                break
        if flag == 0:
            Y[n,:] = normab(y, a, b)
            n += 1
    return Y
def generate_signal(Y, dt, sc, mu_l, mu_h, sigma_l, sigma_h):
    N, n_samples = Y.shape
    data = np.zeros((sc*N, dt))
    for i in range(N):
        init = np.random.randint(int(dt/2), size=sc)
        for j in range(sc):
            mu = np.random.uniform(mu_l,mu_h,1)
            sigma = np.random.uniform(sigma_l,sigma_h,1)
            noise = np.random.normal(mu, sigma, dt)
            noise = normab(noise, 0, 1)
            noise = np.apply_along_axis(nbinom_rand, -1, noise)
            data[i*sc+j,:] = Y[i, init[j]:(init[j]+dt)] + noise
    return data
def nbinom_rand(mu):
    size = 1
    p = size/(size+mu)
    r = nbinom.rvs(1, p)
    return r