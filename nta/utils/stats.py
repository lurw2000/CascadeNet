import numpy as np
import pandas as pd
# from hurst import compute_Hc
import scipy.signal
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.sparsefuncs import mean_variance_axis
from scipy.sparse import lil_matrix, csr_matrix, issparse
from tqdm import tqdm
import math
from scipy.stats import wasserstein_distance
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
from umap import UMAP
from collections import Counter
import nolds

# import nta.utils.stats as stats
from nta.utils.const import interarrival_eps as eps

five_tuple = ["srcip", "dstip", "srcport", "dstport", "proto"]

def time_unit_exp2time_point_count(total_duration, time_unit_exp):
    # This is more tricky than it seems due to floating point error
    # This is important because we want to keep the last interval even if it's truncated by total_duration
    # Error: num_t = int(np.round(total_duration / 10**time_unit_exp))
    #   c.t.e.g.: total_duration=2.342, time_unit_exp=-2 => num_t = 234 where it should be 235 to include the last 0.002s interval
    # Error: num_t = math.ceil(total_duration / 10**time_unit_exp)
    #   c.t.e.g.: total_duration=2.342, use time_point_count to find time_unit_exp
    #   let time_unit_exp = np.log10(total_duration / time_point_count)
    #   - time_point_count=199 => total_duration / 10**time_unit_exp = 199.00000000000003
    #   - time_point_count=200 => total_duration / 10**time_unit_exp = 199.99999999999997
    #   - time_point_count=199 => total_duration / 10**time_unit_exp = 201.00000000000003
    # Solution: We first use np.floor to remove the artifact of floating point arithmetic, then use np.ceil to round up
   return int(np.ceil(np.floor(total_duration / 10**time_unit_exp * 10) / 10))

def time_point_count2time_unit_exp(total_duration, time_point_count):
     return np.log10(total_duration / time_point_count) 

def pkt_count(df, time_unit_exp, total_duration, all_unit=False, verbose=True):
    """
    Convert a network trace into a time series of packet counts.
    The time series is aggregated by time unit, which is 10^time_unit_exp seconds.
    For example, if time_unit_exp = -3, then the time series measure the number of packets per 1e-3 seconds.

    :param df: Pandas dataframe of the network traffic
    :param time_unit_exp: Exponent of the time unit, int.
    :param all_unit: 
        if set true, return pkt counts of all time units; 
        otherwise, return pkt counts at least 1 and its correponding timestamps
        e.g.
        df = pd.DataFrame({"time": [0.010, 0.022, 0.035, 0.213]})
        pkt_count(df, -1, all_unit=False)   # [0, 0.2], [3, 1]
        pkt_count(df, -1, all_unit=False)   # [0, 0.1, 0.2], [3, 0, 1]
    :param verbose: Print the number of bars in the time series, bool.
    :returns:
      - unique: Time stamps of the time series, numpy array of float.
      - pkt_counts: Packet counts of the time series, numpy array of float.
    """
    T = np.array(df["time"]) * 1e7   # use int with modulo, avoid floating point modulo
    # T = T.astype(int)
    time_unit = 10**(7 + time_unit_exp)  # [1e6, 1e5, 1e4, 1e3] (/1e6 => [1e-1, 1e-2, 1e-3, 1e-4])
    TS = T - T%time_unit  # TS is T aggregated by time unit
    unique, pkt_counts = np.unique(TS, return_counts=True)

    if verbose:
        print("Time unit {:.1e} has {} bars".format(
            time_unit/1e7, len(unique)))
    
    if not all_unit:
        return unique/1e7, pkt_counts.astype(float)
    else:
        unique = unique/1e7
        ts = np.arange(0, total_duration+10**time_unit_exp, 10**time_unit_exp)
        all_pkt_counts = np.zeros(len(ts))
        print(unique.shape)
        print(total_duration)
        print(ts.shape)
    
        for i, t in enumerate(unique):
            ai = int(t*(10**(-time_unit_exp)))
            all_pkt_counts[ai] = pkt_counts[i]

        return ts, all_pkt_counts


def byte_count(df, time_unit_exp, total_duration, all_unit=False, verbose=True):
    """
    Similar to pkt_count(), but compute the number of bytes instead of the number of packets.
    """
    T = np.array(df["time"]) * (10**7)   # use int with modulo, avoid floating point modulo
    time_unit = 10**(7 + time_unit_exp)  # [1e6, 1e5, 1e4, 1e3] (/1e6 => [1e-1, 1e-2, 1e-3, 1e-4])
    TS = T - T%time_unit  # TS is T aggregated by time unit
    unique, pkt_counts = np.unique(TS, return_counts=True)

    byte_counts = np.zeros_like(pkt_counts)
    if "pkt_len" in df.columns:
        pkt_lens = df["pkt_len"].to_numpy()
    else:
        pkt_lens = df["pkt"].to_numpy()
    pkt_start = 0

    # for loop to compute byte_counts
    for i, pkt_count in enumerate(pkt_counts):
        byte_counts[i] = np.sum(pkt_lens[pkt_start:pkt_start+pkt_count])
        pkt_start += pkt_count 

    if verbose:
        print("Time unit {:.1e} has {} bars".format(
            time_unit/1e7, len(unique)))

    if not all_unit:
        return unique/1e7, byte_counts
    else:
        unique = unique/1e7
        ts = np.arange(0, total_duration+10**time_unit_exp, 10**time_unit_exp)
        all_byte_counts = np.zeros(len(ts))
        print(ts.shape) 
        for i, t in enumerate(unique):
            ai = int(t*(10**(-time_unit_exp)))
            all_byte_counts[ai] = byte_counts[i]

        return ts, all_byte_counts


def autocorr(data):
    """
    Compute the autocorrelation function of a time series.

    :param data: Time series, numpy array.
    :returns: Autocorrelation function, numpy array.
    """
    # compute the autocorrelation function using numpy.correlate()
    autocorr = scipy.signal.correlate(data, data, mode='full')

    # extract the positive lags of the autocorrelation function
    autocorr = autocorr[len(autocorr)//2:]

    # normalize the autocorrelation function
    autocorr = autocorr / autocorr[0]

    # ignore lag=0
    autocorr[0] = autocorr[1]

    # print the autocorrelation function array
    return autocorr


def cross_corr(data1, data2):
    """
    Compute the cross-correlation function of two time series.

    :param data1: Time series 1, numpy array.
    :param data2: Time series 2, numpy array.
    :returns: Correlation function, numpy array.
    """
    # compute the correlation function using numpy.correlate()
    ccorr = scipy.signal.correlate(data1, data2, mode='full')

    # extract the positive lags of the correlation function

    # normalize the correlation function
    ccorr = ccorr / np.max(np.abs(ccorr))

    # print the correlation function array
    return ccorr


# TODO: how to caluculate hurst exponent?
def hurst(data):
    """
    Compute the Hurst exponent of a time series.

    :param data: Time series, numpy array.
    :returns: Hurst exponent, float.
    """
    raise NotImplemented("Hurst exponent is not properly implemented yet.")
    # normalize the data
    data = (data - data.mean()) / data.std()

    # compute the Hurst exponent using the hurst package
    h, _, _ = compute_Hc(data, kind='change', simplified=True)


    # lags = range(2,100)

    # variancetau = []; tau = []

    # for lag in lags: 

    #     #  Write the different lags into a vector to compute a set of tau or lags
    #     tau.append(lag)

    #     # Compute the log returns on all days, then compute the variance on the difference in log returns
    #     # call this pp or the price difference
    #     pp = np.subtract(data[lag:], data[:-lag])
    #     variancetau.append(np.var(pp))

    # # we now have a set of tau or lags and a corresponding set of variances.
    # #print tau
    # #print variancetau

    # # plot the log of those variance against the log of tau and get the slope
    # m = np.polyfit(np.log10(tau),np.log10(variancetau),1)

    # h = m[0] / 2

    # print the Hurst exponent
    return h


def flow_pca(od_flows, n_components=2):
    U, Sigma, VT = randomized_svd(csr_matrix(od_flows), 
                                n_components=n_components,
                                n_iter=5,
                                random_state=None)
    if issparse(od_flows):
        od_flows = od_flows
    trunc_od_flows = U @ np.diag(Sigma) @ VT

    # compute explained variance
    explained_variance_ = exp_var = np.var(trunc_od_flows, axis=0)
    if issparse(trunc_od_flows):
        _, full_var = mean_variance_axis(od_flows, axis=0)
        full_var = full_var.sum()
    else:
        full_var = np.var(od_flows, axis=0).sum()
    explained_variance_ratio_ = exp_var / full_var

    return od_flows, trunc_od_flows, U, Sigma, VT, explained_variance_, explained_variance_ratio_


def flow_pca_from_dfg(dfg, gks, total_duration, time_unit_exp=-1, n_components=9, verbose=True):
    """
    Convert network traffic into a od_flow matrix (origin-destination flow) and use truncated SVD to reduce dimension

    We treat all records with the same 5-tuple as a flow.

    We construct a matrix X of shape (T, N) where T is #time intervals and N is #flows.

    X[:, j] is the ith flow (a univariate time series, e.g. packet counts across time of a particular 5-tuple)
    X[i, j] is a measurement in time series (e.g. packet count)

    Performing a SVD on X (treat each column as a vector, compute the top `n_components` eigenvectors).

    Using eigenvectors to represent X can only preserve limited amount of information. The percentage of information
        preserved can be calculated by summing the variance of each eigenvector (each eigenvector accounts for a portion
        of variance).
    
    Parameters
    ----------

    :param dfg: A pandas groupby object where each group is a flow.
    :param gks: The group keys of needed groups (e.g. group keys of flows with size <= 3).
    :param total_duration: The total duration of the traffic in seconds.
    :param time_unit_exp: The exponent of the time unit. For example, if time_unit_exp=-6, then the time unit is 1e-6 seconds.
    :param n_components: The number of components to keep.

    :returns:
        od_flows: The od_flow matrix, shape (T, N)
        trunc_od_flows: The truncated od_flow matrix, shape (T, n_components)
        U: The left singular vectors, shape (T, n_components)
        Sigma: The singular values, shape (n_components,)
        VT: The right singular vectors, shape (n_components, N)

    """

    od_flows = compute_od_flows(dfg, gks, total_duration, time_unit_exp, )
    
    return flow_pca(od_flows, n_components=n_components)
    

def compute_od_flows(dfg, gks, total_duration, time_unit_exp=-1, verbose=True):
    """
    Parameters
    ----------
    gks is a list of specific, filtered group keys (e.g. keys of groups of total packet count < 10), 
    dfg is the full groupy object.

    Returns
    ----------
    od_flows: A ndarray of shape (T, N) where T is #time intervals and N is #flows.

    Description
    ----------
    Given a groupby object and a list of group keys, compute the flow matrixs for flows whose group keys are in gks
    """
    # num_t = int(np.ceil(np.floor(total_duration / 10**time_unit_exp * 10) / 10))
    num_t = time_unit_exp2time_point_count(total_duration, time_unit_exp)

    num_flow = len(gks)

    print("Time unit={:.2e} with {} time intervals and {} flows".format(
        10**time_unit_exp, 
        num_t, 
        num_flow))

    # od_flows = np.zeros((num_t, num_flow))
    od_flows = lil_matrix((num_t, num_flow))
    print(od_flows.shape)

    # bins = np.arange(0, total_duration, 10**time_unit_exp)  

    print("Computing {} flows".format(len(gks)))
    for j, gk in tqdm(enumerate(gks)):
        g = dfg.get_group(gk)
        od_flow = df2flow(g, time_unit_exp, num_t)
        od_flows[:, j] = od_flow
    print()

    return od_flows.toarray()

def df2flow(df, time_unit_exp, num_t, verbose=False):
    return timestamps2flow(df["time"].values, time_unit_exp=time_unit_exp, num_t=num_t, verbose=verbose)


def timestamps2flow(timestamps, time_unit_exp, num_t, verbose=False):
    """
    Parameters:
    ----------
    timestamps: ndarray of timestamps
    """
    time_interval = 10 ** time_unit_exp

    # Convert timestamp to the nearest time interval and count packets
    intervals = np.floor(timestamps / time_interval).astype(int)
    # truncate intervals to num_t - 1
    # e.g. if df["time"].max() == total_duration, there will be an extra time interval
    # we want to merge the last two time intervals in such case
    intervals[intervals >= num_t] = num_t - 1


    try:
        packet_counts = np.bincount(intervals)
    except ValueError:
        print("ValueError: intervals.shape={}, num_t={}".format(
            intervals.shape, num_t
        ))
        print("intervals: {}".format(intervals))
        raise ValueError("intervals.shape and num_t do not match")
    
    if packet_counts.max() < 1:
        raise ValueError("packet_counts.max() = {} < 1\ntimestamp is\n{}".format(packet_counts.max(), timestamps ))


    # Initialize the time series with zeros
    od_flow = np.zeros(num_t)

    try:
        # Assign packet_counts to the corresponding intervals in od_flow
        od_flow[:len(packet_counts)] = packet_counts
    except ValueError:
        print("ValueError: packet_counts.shape={}, od_flow.shape={}".format(
            packet_counts.shape, od_flow.shape
        ))
        print("num_t = {}".format(num_t))
        # print all nonzero packet_counts and their indices
        print("Nonzero packet_counts: {}".format(packet_counts[packet_counts != 0]))
        print("Nonzero packet_counts indices: {}".format(np.nonzero(packet_counts)))
        raise ValueError("flow size and num_t do not match")

    if verbose:
        print("Time unit={:.2e} with {} time intervals".format(
            10**time_unit_exp, 
            num_t))
    

    # ignore the last entry of od_flow because it is the count of packets in the next time interval
    return od_flow

def compute_od_throughput(dfg, gks, field_name, total_duration, time_unit_exp=-1, verbose=True):
    # num_t = int(np.ceil(np.floor(total_duration / 10**time_unit_exp * 10) / 10))
    num_t = time_unit_exp2time_point_count(total_duration, time_unit_exp)

    num_flow = len(gks)

    print("Time unit={:.2e} with {} time intervals and {} flows".format(
        10**time_unit_exp, 
        num_t, 
        num_flow))

    # od_flows = np.zeros((num_t, num_flow))
    od_flows = lil_matrix((num_t, num_flow))
    print(od_flows.shape)

    # bins = np.arange(0, total_duration, 10**time_unit_exp)  

    for j, gk in tqdm(enumerate(gks)):
        g = dfg.get_group(gk)
        od_flow = df2throughput(g, field_name, time_unit_exp, num_t)
        od_flows[:, j] = od_flow
    print()

    return od_flows.toarray()

def df2throughput(df, field_name, time_unit_exp, num_t, verbose=False):
    return timestamps2throughput(df["time"].values, df[field_name].values, time_unit_exp=time_unit_exp, num_t=num_t, verbose=verbose)

def timestamps2throughput(timestamps, weights, time_unit_exp, num_t, verbose=False):
    """
    Parameters:
    ----------
    timestamps: ndarray of timestamps
    """
    time_interval = 10 ** time_unit_exp

    # Convert timestamp to the nearest time interval and count packets
    intervals = np.floor(timestamps / time_interval).astype(int)
    # truncate intervals to num_t - 1
    # e.g. if df["time"].max() == total_duration, there will be an extra time interval
    # we want to merge the last two time intervals in such case
    intervals[intervals >= num_t] = num_t - 1


    try:
        packet_counts = np.bincount(intervals, weights)
    except ValueError:
        print("ValueError: intervals.shape={}, num_t={}".format(
            intervals.shape, num_t
        ))
        print("intervals: {}".format(intervals))
        raise ValueError("intervals.shape and num_t do not match")
    
    if packet_counts.max() < 1:
        raise ValueError("packet_counts.max() = {} < 1\ntimestamp is\n{}".format(packet_counts.max(), timestamps ))


    # Initialize the time series with zeros
    od_flow = np.zeros(num_t)

    try:
        # Assign packet_counts to the corresponding intervals in od_flow
        od_flow[:len(packet_counts)] = packet_counts
    except ValueError:
        print("ValueError: packet_counts.shape={}, od_flow.shape={}".format(
            packet_counts.shape, od_flow.shape
        ))
        print("num_t = {}".format(num_t))
        # print all nonzero packet_counts and their indices
        print("Nonzero packet_counts: {}".format(packet_counts[packet_counts != 0]))
        print("Nonzero packet_counts indices: {}".format(np.nonzero(packet_counts)))
        raise ValueError("flow size and num_t do not match")

    if verbose:
        print("Time unit={:.2e} with {} time intervals".format(
            10**time_unit_exp, 
            num_t))
    

    # ignore the last entry of od_flow because it is the count of packets in the next time interval
    return od_flow

def periodicity_index(time_series, fs=1.0):
    """
    This index measures the level of periodicity of a time series. It is based on the following intuition:

        "no trend" + "has a dominent periodic component" => "visually periodic"
    
    To identify the trend of a time series, we #TODO: 

    To identify the strongest periodic component and does not guarantee that the time series is perfectly periodic. 
    It just provides a metric that can be interpreted as: the higher the value, the stronger the dominant periodic component is compared to the rest.
    """

    # a1 \in [0, 1] measures how much the time series is trending. For now, we only consider linear trend.
    # TODO: more trend?
    a1 = 1

    freqs, psd = scipy.signal.welch(time_series, fs=fs)
    max_peak = np.max(psd)
    # a2 \in [0, 1] measures how dominant is the strongest periodic component
    a2 = max_peak / (np.sum(psd))

    return a1 * a2 


def psd(time_series, fs=1.0):
    """
    This function is generated by GPT-4

    This function returns the power spectral density of a time series
    """
    freqs, psd = scipy.signal.welch(time_series, fs=fs)

    return psd


def jsd_dr(p, q, num_bins=100, dr_config={"dr_method": "pca", "seed": 42}):
    """
    !! The complexity of this computation grows exponentially with the number of dimensions !!
    Compute the Jensen-Shannon divergence between two multivariate distributions by computing the KL divergence between
        the histograms of the two distributions and then averaging the KL divergences.

    Note that you should always experiment with different bins_count to ensure that
        the comparison result is meaningful
 
    Parameters:
    p (numpy.ndarray): Samples from the first distribution, shape (#samples, #dim)
    q (numpy.ndarray): Samples from the second distribution, shape (#samples, #dim)
    num_bins (int): Number of bins to use in the histogram

    Returns:
    float: The Jensen-Shannon divergence
    """
    # same dim
    if p.shape[1] != q.shape[1]:
        raise ValueError("p and q must have the same dimension! But dim of p = {}, dim of q = {}".format(
            p.shape[1], q.shape[1]
        ))
    
    # Dimension Reduction
    dr_method = dr_config["dr_method"]
    print("Reducing dimension to 2d using {}...".format(dr_config))
    if dr_method == "pca":
        # use pca to reduce dimension
        reducer = PCA(n_components=2, random_state=dr_config["seed"])
        sample_size = int(p.shape[0] * 0.1)
        sample_indices = np.random.choice(p.shape[0], size=sample_size, replace=False)
        reducer.fit(p[sample_indices])
        p_2d = reducer.transform(p)
    elif dr_method == "tsne":
        raise ValueError("Distance between cluster is meaningless when using tsne to reduce dimension!")
        # use tsne to reduce dimension
        reducer = TSNE(
            n_components=2, 
            verbose=1, 
            perplexity=40, 
            n_iter=300, 
            random_state=dr_config["seed"])
        p_2d = reducer.fit_transform(p)
    elif dr_method == "umap":
        # use umap to reduce dimension
        reducer = UMAP(
            n_components=2, 
            verbose=True, 
            random_state=dr_config["seed"],
            **dr_config["params"])
        # reducer = UMAP(n_components=2, verbose=True, n_neighbors=30)
        # fit on 10% of the data to avoid weird umap results
        # randomly sample 10% of the data and fit
        sample_ratio = dr_config["sample_ratio"]
        sample_size = int(p.shape[0] * sample_ratio)
        sample_indices = np.random.choice(p.shape[0], size=sample_size, replace=False)
        reducer.fit(p[sample_indices])
        p_2d = reducer.transform(p)
    else:
        raise ValueError("Invalid dr_method: {}".format(dr_method))


    if dr_method == "tsne":
        q_2d = reducer.fit_transform(q)
    else:
        q_2d = reducer.transform(q)


    print("feature_vectors_2d.shape: {}".format(p_2d.shape))

    # Compute histograms
    print("Computing histogram of p (shape: {}) and q (shape: {})".format(p_2d.shape, q_2d.shape))
    # complexity: O(#samples * #dim)
    hist_p, edges = np.histogramdd(p_2d, bins=num_bins)
    hist_q, _ = np.histogramdd(q_2d, bins=edges)

    # Normalize histograms to obtain PDFs
    print("Normalizing histograms")
    pdf_p = hist_p / np.sum(hist_p)
    pdf_q = hist_q / np.sum(hist_q)

    # Compute the average distribution
    m = 0.5 * (pdf_p + pdf_q)

    # Avoid division by zero and logarithm of zero by adding a small constant
    epsilon = 1e-10
    pdf_p += epsilon
    pdf_q += epsilon
    m += epsilon

    # Compute the KL divergences, complexity: O(#bins^#dim)
    print("Computin KL divergence of the histograms")
    kl_p_m = entropy(pdf_p.flatten(), m.flatten())
    kl_q_m = entropy(pdf_q.flatten(), m.flatten())

    # Compute the Jensen-Shannon divergence
    jsd = 0.5 * (kl_p_m + kl_q_m)

    return jsd

# TODO: pdf and lofpdf of discrete data

def logpdf(samples, bins_count=100, bins_min=None, bins_max=None, truncate=True):
    """
    Compute the probability density function of samples in log scale (with truncation)
    For r.v. X, we compute the pdf of log(X) instead of X.

    Parameters:
    ----------

    """
    samples = np.clip(samples, a_min=eps, a_max=None)
    if bins_min is None:
        bins_min = np.min(samples)
    if bins_max is None:
        bins_max = np.max(samples)
    if bins_min == bins_max:
        warnings.warn("bins_min == bins_max, setting bins_max = bins_min + 2*eps")
        bins_max += 2*eps
    if bins_min == 0:
        warnings.warn("bins_min == 0, setting bins_min = eps")
        bins_min = eps

    if truncate:
        samples = np.clip(samples, a_min=bins_min, a_max=bins_max)
    bins_range = np.logspace(np.log10(bins_min), np.log10(bins_max), bins_count+1)
    # use -inf and inf to include values that are out of range
    bins_range = np.insert(bins_range, 0, -np.inf)
    bins_range = np.append(bins_range, np.inf)
    # replace 0 with eps to avoid log(0)
    # samples[samples == 0] = eps
    hist, bins = np.histogram(
        samples,
        bins=bins_range)
    hist = hist / np.sum(hist)
    return hist, bins


def dep_logpdf(samples, bins_count=100, bins_min=None, bins_max=None):
    """
    Compute the probability density function of samples in log scale (with truncation)
    For r.v. X, we compute the pdf of log(X) instead of X.

    Parameters:
    ----------

    """
    samples = np.clip(samples, a_min=eps, a_max=None)
    if bins_min is None:
        bins_min = np.min(samples)
    if bins_max is None:
        bins_max = np.max(samples)
    if bins_min == bins_max:
        warnings.warn("bins_min == bins_max, setting bins_max = bins_min + 2*eps")
        bins_max += 2*eps
    if bins_min == 0:
        warnings.warn("bins_min == 0, setting bins_min = eps")
        bins_min = eps
    bins_range = np.linspace(bins_min, bins_max, bins_count+1)
    # use -inf and inf to include values that are out of range
    bins_range = np.insert(bins_range, 0, -np.inf)
    bins_range = np.append(bins_range, np.inf)
    # replace 0 with eps to avoid log(0)
    # samples[samples == 0] = eps
    hist, bins = np.histogram(
        np.log(samples), 
        bins=bins_range,
        density=True)
    return hist, bins

def pdf(samples, bins_count, bins_min, bins_max, truncate=True):
    """
    Compute the probability density function of samples.

    Parameters:
    ----------

    """
    if bins_min is None:
        bins_min = np.min(samples)
    if bins_max is None:
        bins_max = np.max(samples)
    if bins_min == bins_max:
        warnings.warn("bins_min == bins_max, setting bins_max = bins_min + 2*eps")
        bins_max += 2*eps
    
    if truncate:
        samples = np.clip(samples, a_min=bins_min, a_max=bins_max)
    bins_range = np.linspace(bins_min, bins_max, bins_count+1)
    # use -inf and inf to include values that are out of range
    bins_range = np.insert(bins_range, 0, -np.inf)
    bins_range = np.append(bins_range, np.inf)
    hist, bins = np.histogram(
        samples, 
        bins=bins_range,
        density=True)
    return hist, bins

def pmf(samples, sample_range):
    """
    Compute the probability mass function of samples.

    Parameters:
    ----------
    samples: ndarray of samples
    sample_range: ndarray of sample range. This is necessary because 
        samples may not contain all possible values.
    """
    counter = Counter(samples)
    total_count = len(samples)
    probs = np.array(
        [counter.get(i, 0) / total_count for i in sample_range]
    )
    return probs


def jsd(sample_x, sample_y):
    """
    Compute the Jensen-Shannon divergence between two discrete distributions
    """
    sample_range = list(set(sample_x) | set(sample_y))

    pmf_x = pmf(sample_x, sample_range)
    pmf_y = pmf(sample_y, sample_range)

    return jensenshannon(pmf_x, pmf_y)


# def logjsd(sample_x, sample_y):
#     """
#     Given r.v. X and Y, compute the JSD between log(X) and log(Y)
#     """
#     # find minimum value in sample_x and sample_y. If the minimum <= 0, shift by -minimum + eps
#     min_x = np.min(sample_x)
#     min_y = np.min(sample_y)
#     min_xy = min(min_x, min_y)
#     if min_xy <= 0:
#         sample_x += -min_xy + eps
#         sample_y += -min_xy + eps
#     return jsd(np.log(sample_x), np.log(sample_y))


def emd(sample_x, sample_y):
    """
    Compute the Earth Mover's Distance between two continuous distributions
    """
    return wasserstein_distance(sample_x, sample_y)


def logemd(sample_x, sample_y):
    """
    Given r.v. X and Y, compute the EMD between log(X) and log(Y)
    """
    # find minimum value in sample_x and sample_y. If the minimum <= 0, shift by -minimum + eps
    min_x = np.min(sample_x)
    min_y = np.min(sample_y)
    min_xy = min(min_x, min_y)
    if min_xy <= 0:
        sample_x += -min_xy + eps
        sample_y += -min_xy + eps
    return emd(np.log(sample_x), np.log(sample_y))


def dfg2flow_level_interarrival(dfg, gks, asint=False):
    ts = []
    print("Computing flow-level interarrival time for each flow...")
    for gk in tqdm(gks):
        g = dfg.get_group(gk)
        ts.append(np.diff(g["time"]))
    # ts = np.concatenate(ts).astype(int)
    ts = np.concatenate(ts)
    if asint:
        ts = ts.astype(int)
    return ts


def dep_dfg2flow_level_interarrival(dfg, flow_filter=None):
    ts = []
    if flow_filter is None:
        flow_filter = {}
    for k in flow_filter:
        if k == "flowsize_range":
            continue
        print("Filter {} is not implemented yet".format(k))
    def filter_by(g, flow_filter):
        # "flowsize_range": None,
        # "flowDurationRatio_range": None,
        flag = True
        if "flowsize_range" in flow_filter and flow_filter["flowsize_range"] is not None:
            flowsize_range = flow_filter["flowsize_range"]
            flag = flowsize_range[0] <= len(g) < flowsize_range[1]
        for k in flow_filter:
            if k == "flowsize_range":
                continue
            print("Filter {} is not implemented yet".format(k))
        return flag
    print("Computing flow-level interarrival time for each flow...")
    for _, g in tqdm(dfg):
        if not filter_by(g, flow_filter):
            continue
        ts.append(np.diff(g["time"]))
        # ts.append(np.diff(g["time"], prepend=0))
    return np.concatenate(ts).astype(int)

def df2flow_level_interarrival(df, flow_tuple=five_tuple, flow_filter=None):
    df["time"] = df["time"] - df["time"].min()
    dfg = df.groupby(flow_tuple)
    gks = dfg.groups.keys() 
    return dfg2flow_level_interarrival(dfg, gks, flow_filter)

def dfg2flowstart(dfg, flow_filter=None):
    flowstarts = []
    if flow_filter is None:
        flow_filter = {}
    for k in flow_filter:
        if k == "flowsize_range":
            continue
        print("Filter {} is not implemented yet".format(k))
    def filter_by(g, flow_filter):
        # "flowsize_range": None,
        # "flowDurationRatio_range": None,
        flag = True
        if "flowsize_range" in flow_filter and flow_filter["flowsize_range"] is not None:
            flowsize_range = flow_filter["flowsize_range"]
            flag = flowsize_range[0] <= len(g) < flowsize_range[1]
        return flag
    print("Computing flow start time for each flow...")
    for _, g in tqdm(dfg):
        if not filter_by(g, flow_filter):
            continue
        flowstarts.append(g["time"].min())
    return np.array(flowstarts)
    # return np.array([g["time"].min() for _, g in dfg])

def df2flowstart(df, flow_tuple=five_tuple, flow_filter=None):
    df["time"] = df["time"] - df["time"].min()
    dfg = df.groupby(flow_tuple)
    return dfg2flowstart(dfg, flow_filter)

def dfg2timestamp_position_in_interarrival(dfg, trace_start_time, duration, time_point_count, flow_filter=None):
    """
    Given a trace, a time_point_count, we divide the trace duration into time_point_count intervals.
    For a timestamp, we can compute its position within interval by
    1. find the interval [t_a, t_b) that contains t
    2. compute the position of t within [t_a, t_b) by (t - t_a) / (t_b - t_a)
    """
    positions = []
    interval_length = 10 ** stats.time_point_count2time_unit_exp(duration, time_point_count)
    if flow_filter is None:
        flow_filter = {}
    for k in flow_filter:
        if k == "flowsize_range":
            continue
        print("Filter {} is not implemented yet".format(k))
    def filter_by(g, flow_filter):
        # "flowsize_range": None,
        # "flowDurationRatio_range": None,
        flag = True
        if "flowsize_range" in flow_filter and flow_filter["flowsize_range"] is not None:
            flowsize_range = flow_filter["flowsize_range"]
            flag = flowsize_range[0] <= len(g) < flowsize_range[1]
        for k in flow_filter:
            if k == "flowsize_range":
                continue
            print("Filter {} is not implemented yet".format(k))
        return flag
    print("Computing flow-level interarrival time for each flow...")
    for _, g in tqdm(dfg):
        if not filter_by(g, flow_filter):
            continue
        ts = g["time"].values - trace_start_time
        ps = []
        for t in ts:
            # find the interval [t_a, t_b) that contains t
            t_a = math.floor(t / duration * time_point_count)
            if t_a == time_point_count:
                t_a -= 1
            t_b = t_a + 1
            # assert(t_a >= 0 and t_b <= time_point_count)
            if t_a < 0 or t_b > time_point_count:
                print("Error: t_a={}, t_b={}, time_point_count={}".format(t_a, t_b, time_point_count))
                raise ValueError("Invalid value encountered when computing position in interarrival")
            t_a = t_a * interval_length
            t_b = t_b * interval_length
            # compute the position of t within [t_a, t_b) by (t - t_a) / (t_b - t_a)
            p = (t - t_a) / (t_b - t_a)
            ps.append(p)
        positions.extend(ps)
    positions = np.array(positions)
    return positions

def df2timestamp_position_in_interval(df, time_point_count, flow_tuple=five_tuple, flow_filter=None):
    df["time"] = df["time"] - df["time"].min()
    trace_start_time = df["time"].min()
    duration = df["time"].max() - trace_start_time
    dfg = df.groupby(flow_tuple)
    return dfg2timestamp_position_in_interarrival(dfg, trace_start_time, duration, time_point_count, flow_filter)

def perturb_df(df, df_dataset_type, perturb_coeff=0.1):
    # perturb each column of raw df by a gaussian noise
    #   perturb by a gaussian noise with mean 0 and std 0.1 * column std
    raw_df = df
    perturbed_df = df.copy(deep=True)

    # perturb all non-categorical columns except timestamp
    for col in raw_df.columns:
        if df_dataset_type == "pcap" and col not in ["pkt_len", ]:
            continue 
        elif df_dataset_type == "netflow" and col not in ["td", "pkt", "byt", ]:
            continue
        # get col values
        col_values = raw_df[col].values
        col_dtype = col_values.dtype
        # convert column to float64 type for perturbation
        col_values = col_values.astype(np.float64)
        # get std of col values
        col_std = np.std(col_values)
        # perturb by N(0, perturb_coeff * col_std)
        perturb = np.random.normal(0, perturb_coeff * col_std, col_values.shape)
        # add perturb to col values
        col_values += perturb
        # convert col back to its original type
        col_values = col_values.astype(col_dtype)
        # update col values
        perturbed_df[col] = col_values
        
    # perturb timestamp by perturbing the interarrival time of each packet
    time_col_values = raw_df["time"].values
    time_col_dtype = time_col_values.dtype
    time_col_values = time_col_values.astype(np.float64)
    # get interarrival time
    interarrival_time = np.diff(time_col_values)
    # get std of interarrival time
    interarrival_time_std = np.std(interarrival_time)
    # perturb by N(0, perturb_coeff * interarrival_time_std)
    perturb = np.random.normal(0, perturb_coeff * interarrival_time_std, interarrival_time.shape)
    # add perturb to interarrival time
    interarrival_time += perturb
    # update time_col_values
    time_col_values[1:] = np.cumsum(interarrival_time)
    time_col_values = time_col_values.astype(time_col_dtype)
    # update time column
    perturbed_df["time"] = time_col_values
    # sort by time again
    perturbed_df = perturbed_df.sort_values(by="time")

    return perturbed_df

def flows2hursts(flows):
    hurst_exponents = []
    print("Computing hurst exponent for each flow")
    # filter out all flows with less than 2 nonzero values
    # find the number of nonzero for each flow
    num_nonzero = np.count_nonzero(flows, axis=0)
    # find the indices of flows with less than 2 nonzero values
    indices = np.where(num_nonzero < 2)[0]
    # remove flows with less than 2 nonzero values
    flows = np.delete(flows, indices, axis=1)
    # compute hurst exponent for each flow
    for i in tqdm(range(flows.shape[1])):
        hurst_exponent = nolds.hurst_rs(flows[:, i])
        # hurst_exponent = nolds.dfa(flows[:, i])
        hurst_exponents.append(hurst_exponent)
    return np.array(hurst_exponents)
