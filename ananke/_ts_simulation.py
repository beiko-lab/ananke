import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_generate_sample
import numpy as np
from scipy.stats import nbinom
seeds = 3
np.random.seed(seeds)
def normab(x, a, b):
    '''
    Normalize input signal x by a lower bound a and an upper bound b
    Arguments:
    x -- input singal, numpy array of shape ( n,  )
    a -- lower bound of the output signal, scalar
    b -- upper bound of the output signal, scalar
    Result:
      -- scaled and normalized output signal, numpy array of shape ( n,  ) 
    '''
    return (b-a)*(x - np.min(x))/(np.max(x) - np.min(x)) + a

def gen_arima(n, a, b):
    '''
    generate arima signal that scaled by a lower bound a and an upper bound b
    Arguments:
    n -- length of the arima signal, scalar
    a -- lower bound of the output signal, scalar
    b -- upper bound of the output signal, scalar
    Result:
      -- scaled and normalized arima signal, numpy array of shape ( n,  ) 
    '''
    
    arparams = np.random.uniform(0.97,0.99,1)
    maparams = np.random.uniform(0.00,0.99,1)
    
    y = None
    while y is None:
        try:
            y = arma_generate_sample(arparams, maparams, n)
        except:
            pass
    
    return normab(y, a, b)

def gen_table(fl_sig=0,w_sig=6,
              fl_bg=-6,w_bg=6,
              bg_disp_mu=0,bg_disp_sigma=1,
              sig_disp_mu2=0,sig_disp_sigma2=1,
              n_clust=10,n_sig=10,n_tax_sig=1,n_bg=700,
              len_arima=1000,len_ts=500,len_signal=300):
    '''
    generate simulated time series 
    Arguments:
    
    fl_sig -- lower bound of the arima signal, scalar
    w_sig -- upper bound of the arima signal, scalar
    fl_bg -- lower bound of the background noise seed, scalar
    w_bg -- upper bound of the background noise seed, scalar
    bg_disp_mu -- mu of dispersion for the background noise, scalar
    bg_disp_sigma -- sigma of dispersion for the background noise, scalar
    sig_disp_mu2 -- mu of dispersion for the noise of time series, scalar 
    sig_disp_sigma2 -- sigma of dispersion for the noise of time series, scalar 
    n_clust -- the number of clusters in time series, scalar
    n_sig -- the number of time series per cluster, scalar
    n_tax_sig -- the number of signal replicates per signal, scalar
    n_bg -- the number of background noise tiem series, scalar
    len_arima -- length of the arima signal, scalar (len_arima > len_ts > 2*len_signal)
    len_ts -- length of the time series, scalar
    len_signal -- the minimum length of overlaps among two signals from a cluster, scalar
    Result:
    out  -- scaled and normalized arima signal, a dictionary where
        out['table'] -- final simulated time series, numpy array of shape ( n_clust*n_sig*n_tax_sig+n_bg, len_ts )
    '''
    # optional input sig_disp_mu1=0,sig_disp_sigma1=0
    idx_signal = range((len_ts - len_signal//2),(len_ts + len_signal//2 - 1))
    min_ts = max(idx_signal) - len_ts + 1
    max_ts = min(idx_signal)
    mu_bg = normab(np.random.normal(0, 1, len_ts*n_bg),fl_bg,w_bg)
    bg_disp = np.random.normal(bg_disp_mu, bg_disp_sigma, len_ts*n_bg)
    bg_lambda = np.exp(mu_bg + bg_disp)
    
    background = np.reshape(np.random.poisson(bg_lambda,len(bg_lambda)),(len_ts,n_bg),order='F')
    
    dat = []
    for k in range(n_clust):

        cluster_out = {}

        timeseries_pure = gen_arima(len_arima,fl_sig,w_sig)
#         timeseries_noise1 = np.random.normal(sig_disp_mu1,sig_disp_sigma1,len(timeseries_pure))

        timeseries_lambda = []
        for i in range(len(timeseries_pure)):
            theta = np.exp(timeseries_pure[i])
#             theta = np.exp(timeseries_pure[i] + timeseries_noise1[i])
            if theta > np.exp(20):
                theta = np.exp(20)
            timeseries_lambda.append(theta)
        timeseries = np.random.poisson(timeseries_lambda,len(timeseries_lambda))

        timesteps = []
        for i in range(n_sig):
            ts_start = np.random.randint(min_ts, max_ts)
            timesteps.append(list(range(ts_start,ts_start+len_ts)))

        cluster = []
        pure_sig = []
        for i in range(n_sig):
            y_tmp = timeseries[timesteps[i]]
            cluster_tmp = []
            for mu in y_tmp:
                if sig_disp_sigma2 == 0:
                    theta = 0
                else:
                    theta = np.exp(np.random.normal(sig_disp_mu2,sig_disp_sigma2,1)[0])
                if theta > np.exp(20):
                    theta = np.exp(20)
                cluster_tmp.extend(mu + np.random.poisson(theta,n_tax_sig))
            cluster.append(cluster_tmp)
            pure_sig.append(y_tmp)
              
        cluster_out['timeseries_pure'] = pure_sig
        cluster_out['timeseries'] = timeseries
        cluster_out['cluster'] = cluster
        cluster_out['timesteps'] = timesteps

        dat.append(cluster_out)
        
    background = background.T      
    abund = np.vstack([d['cluster'] for d in dat])
    sig_pure = np.vstack([d['timeseries_pure'] for d in dat])
    final_table = np.concatenate((abund,background),axis=0)
    
    out = {}
    out['table'] = final_table
    out['signals'] = sig_pure
    out['background'] = background
    
    return out