import numpy as np
import os 
import nta.utils.data as data
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

from nta.utils.const import interarrival_eps as eps
import nta.utils.stats as stats

# a list of default colors

COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
COLORS = ['C{}'.format(i) for i in range(10)]

COLORS = [
    "tab:gray",
    "tab:orange",
    "tab:brown",
    "tab:blue",
    "tab:red",
    "tab:green",
    "tab:pink",
    "tab:purple",
    "tab:olive",
    "tab:cyan",
]

HATCHS = ["--", "//", "xx", "\\", "oo", "..", "+", "O", "*"]

# lines
LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]
MARKERS = ["o", "^", "s", "D"]

COLORS_DICT = {
    "CTGAN": "tab:green",
    "E-WGAN-GP": "tab:orange",
    "STAN": "tab:brown",
    "REaLTabFormer" : "tab:pink",
    "NetShare": "tab:blue",
    "NetDiffusion": "tab:purple",
    "CascadeNet": "tab:red",

    "CA": "tab:orange",
    "CAIDA": "tab:green",
    "DC": "tab:brown",
    "TON_IoT": "tab:blue",

    "EQ": "tab:purple",
    "SP": "tab:brown",
    "ML": "tab:red",

    "CN-w/o-ZI": "tab:olive",
    "CN-w/o-Cond": "tab:cyan",
    
    "CN-20": "tab:green",
    "CN-50": "tab:blue",
    "CN-100": "tab:orange",
    "CN-200": "tab:purple",
    "CN-500": "tab:pink",
    "CN-1000": "tab:olive",
    "CN-2000": "tab:brown",

    "CN-mlp-rnn": "tab:purple",
    "CN-mlp-mlp": "tab:blue",
    "CN-rnn-rnn": "tab:green",
    "CN-rnn-mlp": "tab:orange",

    "CN-w/o-NPF": "tab:blue",     
    "CN-w/o-DUR": "tab:orange",  
    "CN-w/o-NumA": "tab:green",   
    "CN-w/o-MaxPR": "tab:purple", 
    "CN-w/o-MeanPR": "tab:cyan",  
    "CN-w/o-StdPR": "tab:brown",   
    "CN-w/o-MeanI": "tab:olive", 

    "Generated": "tab:red",
}

def get_color(label):
    if label in COLORS_DICT:
        return COLORS_DICT[label]
    else:
        index = hash(label) % len(COLORS)
        return COLORS[index % len(COLORS)]
        
    
HATCHS_DICT = {
    "CTGAN": "x",
    "E-WGAN-GP": "*",
    "STAN": "o",
    "REaLTabFormer" : ".",
    "NetDiffusion": "+",
    "NetShare": "-",
    "CascadeNet": "/",
    "EQ": "+",
    "SP": "-",
    "ML": "/",

    "CN-w/o-ZI": "O",
    "CN-w/o-Cond": "o",

    "CN-20": "+",
    "CN-50": "*",
    "CN-100": "o",
    "CN-200": "/",
    "CN-500": "x",
    "CN-1000": "O",
    "CN-2000": "-",

    "CN-mlp-rnn": "+",
    "CN-mlp-mlp": "*",
    "CN-rnn-rnn": "-",
    "CN-rnn-mlp": "o",

    "CN-w/o-NPF": "/",
    "CN-w/o-DUR": "x",
    "CN-w/o-NumA": "+",
    "CN-w/o-MaxPR": "o",
    "CN-w/o-MeanPR": "*",
    "CN-w/o-StdPR": "-",
    "CN-w/o-MeanI": "\\",

    "Generated": "x",
}

def get_hatch(label):
    if label in HATCHS_DICT:
        return HATCHS_DICT[label]
    else:
        return HATCHS[hash(label) % len(HATCHS)]

def generate_line_style(i):
    """
    Generate a line style from index for matplotlib.pyplot.plot

    Parameters
    ----------
    i: int
        the index of the line, starting from 0

    Returns
    -------
    style: dict
        a dict of keyword arguments for matplotlib.pyplot.plot
    """
    
    style = {}
    style["color"] = COLORS[i % len(COLORS)]
    style["linestyle"] = LINESTYLES[i % len(LINESTYLES)]
    # style["linestyle"] = LINESTYLES[0]
    style["marker"] = MARKERS[i % len(MARKERS)] 
    style["markersize"] = 5
    return style


def plot_loghist(
        x, ax: plt.Axes,
        bins_count=10, bins_min=None, bins_max=None, bins_min_min=1e-6,
        density=True, cumsum=False, histtype="step", truncate_x=True, **kwargs):
    """
    Plot log histogram of x on ax.
    Zero values will be merged into the first bin (by clipping to bins_min_min)

    Parameters
    ----------
    x: array-like
        the data to plot
    ax: matplotlib.axes.Axes
        the axes to plot on
    bins_count: int
        number of bins
    bins_min: float, string or None
        the minimum value of the bins. If None, use np.min(x[x >= bins_min_min])
        If "cur", use the current x axis limit
    bins_max: float, string or None
        the maximum value of the bins. If None, use np.max(x)
        If "cur", use the current x axis limit
    bins_min_min: float
        the minimum value of bins_min. Values below bin_min_min will be clipped to bin_min_min
    density: bool
        whether to normalize the histogram
    cumsum: bool
        whether to plot the cumulative sum of the histogram
    truncate_x: bool
        whether to truncate x to [bins_min, bins_max]
    kwargs: dict
        keyword arguments passed to ax.step()
    """
    if bins_min is None:
        bins_min = np.min(x[x >= bins_min_min])
    elif bins_min == "cur":
        # get the bins_min from the current axis
        bins_min = ax.get_xlim()[0]

    if bins_max is None:
        bins_max = np.max(x)
    elif bins_max == "cur":
        # get the bins_max from the current axis
        bins_max = ax.get_xlim()[1]

    if bins_min == bins_max:
        warnings.warn("bins_min == bins_max, setting bins_max = bins_min + 2*eps")
        bins_max += 2*eps

    if np.min(x) < bins_min or np.max(x) > bins_max:
        if truncate_x:
            # truncate x to [bins_min, bins_max]
            warnings.warn("x is truncated to [{}, {}]".format(bins_min, bins_max))
            x = np.clip(x, bins_min, bins_max)
        else:
            warnings.warn("truncate_x is False, values outside of [{}, {}] will be ignored".format(
                bins_min, bins_max))

    logbins = np.logspace(np.log10(bins_min+eps), np.log10(bins_max+eps), bins_count)
    try:
        # Note: we can't use histogram's density parameter for loghist's density plot
        hist, bins = np.histogram(x, bins=logbins, density=False)
        if density:
            hist = hist / np.sum(hist)
    except ValueError:
        print("bins_min={}, bis_max={}, logbins={}".format(bins_min, bins_max, logbins))
        exit(0)
    if cumsum:
        hist = np.cumsum(hist)

    if histtype == "hist":
        ax.hist(bins[:-1], bins, weights=hist, **kwargs) 
    elif histtype == "step":
        ax.step(bins[:-1], hist, linestyle='-', linewidth=1, where='post', **kwargs)
    elif histtype == "plot":
        # plot as a line
        ax.plot(bins[:-1], hist, linestyle='-', **kwargs)
    else:
        raise ValueError("Histtype not supported: {}".format(histtype))
    # ax.step(bins[:-1], hist, width=np.diff(bins), align="edge", **kwargs)
    # ax.step(bins1[:-1],hist1,'k',linestyle='-',linewidth=2,where='post')
    # ax.bar(bins1[:-1],hist1,width=0.5,linewidth=0,facecolor='k',alpha=0.3)

    # set_xscale, but don't mess with xtick
    ax.set_xscale('log') 
    ax.set_xlim(bins_min, bins_max)

def decorate_loghist(ax: plt.Axes, tick_num=5, tick_size=10, label_size=10):
    """
    Add xtick and xticklabel of the form of 10^x to ax
    """
    xlim = ax.get_xlim()
    # clear minor and major tick and label
    ax.minorticks_off()
    ax.set_xticks([])
    ax.set_xticklabels([])
    # set 'tick_num' major ticks
    xticks = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), tick_num)
    ax.set_xticks(xticks, minor=False, fontsize=tick_size)
    xticklabels = ["$10^{{{:.0f}}}$".format(np.log10(x)) for x in xticks]
    xticklabels[0] = "<" + xticklabels[0]
    ax.set_xticklabels(xticklabels, fontsize=label_size)
    

def plot_hist(
        x, ax: plt.Axes,
        bins_count=10, bins_min=None, bins_max=None, bins_min_min=1e-6,
        truncate_x=True, density=True, cumsum=True, histtype="plot", 
        xlim=None, ylim=None, log_bins=False, **kwargs):
    """
    Plot histogram of x on ax.
    """
    if bins_min is None:
        bins_min = np.min(x[x >= bins_min_min])
    elif bins_min == "cur":
        # get the bins_min from the current axis
        bins_min = ax.get_xlim()[0]
        # Handle case where bins_min is zero or less
        if bins_min <= bins_min_min:
            bins_min = bins_min_min + eps
            
    if bins_max is None:
        bins_max = np.max(x)
    elif bins_max == "cur":
        # get the bins_max from the current axis
        bins_max = ax.get_xlim()[1]
    
    if bins_min == bins_max:
        warnings.warn("bins_min == bins_max, setting bins_max = bins_min + 2*eps")
        bins_max += 2*eps
    
    if np.min(x) < bins_min or np.max(x) > bins_max:
        if truncate_x:
            # truncate x to [bins_min, bins_max]
            warnings.warn("x is truncated to [{}, {}]".format(bins_min, bins_max))
            x = np.clip(x, bins_min, bins_max)
        else:
            warnings.warn("truncate_x is False, values outside of [{}, {}] will be ignored".format(
                bins_min, bins_max))
    
    if log_bins:
        bins = np.logspace(np.log10(bins_min), np.log10(bins_max), bins_count)
        ax.set_xscale('log') 
    else:
        bins = np.linspace(bins_min, bins_max, bins_count)
        
    hist, bins = np.histogram(x, bins=bins, density=False)
    if density:
        hist = hist / np.sum(hist)
    if cumsum:
        hist = np.cumsum(hist)

    if histtype == "hist":
        ax.hist(bins[:-1], bins, weights=hist, **kwargs)
    elif histtype == "step":
        ax.step(bins[:-1], hist, linestyle='-', where='post', **kwargs)
    elif histtype == "plot":
        # plot as a line
        ax.plot(bins[:-1], hist, **kwargs)
    else:
        raise ValueError("Histtype not supported: {}".format(histtype))
   
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(bins_min, bins_max)


def plot_all_flows_and_save(path, save_to=None, mode=("time_point_count", 200), flowsize_range=None, sample=300, need_divide_1e6=False, zero_threshold=0.5):
    """
    Parameters
    ----------
    path: str
        path to the .csv or .npz file
    save_to: str
        path to the folder to save the figures
    time_unit_exp: int
        the exponent of the time unit, e.g. -1 for 1/second, -6 for 1/microsecond
    flowsize_range: tuple
        (min, max) of the flow size
    sample: int
        number of flows to sample
    zero_threshold: float
        the threshold to add to the flows to avoid rounding to 0. Only used for npz file
    """
    if zero_threshold >= 1 or zero_threshold < 0:
        raise ValueError("zero_threshold must be in [0, 1)")
    
    # flows.shape = (n_timestamps, n_flows)
    if path.endswith(".csv"):
        print("Loading flows from this csv file:\n\t{}".format(path))
        flows = data.load_flow_from_csv(
            path, mode=mode, flowsize_range=flowsize_range, verbose=False, need_divide_1e6=need_divide_1e6,
            ).astype(int)
        print(flows.shape)
    elif path.endswith(".npz"):
        print("Loading flows from this npz file, ignoring time_unit_exp:\n\t{}".format(path))
        flows = np.load(path)["packetrate"].T + zero_threshold
        # TODO: temp use, delete later
        # flows = np.load(path)["output"].T * 55 + zero_threshold
        # add 0.5 before flooring to avoid rounding to 0
        flows = flows.astype(int)
        # keep only flows within flowsize_range (i.e. the sum of packetrate is within flowsize_range)
        print(flows.shape)
        if flowsize_range is not None:
            s = np.sum(flows, axis=0)
            flows = flows[:, (s >= flowsize_range[0]) & (s < flowsize_range[1])]
            del s     
        print("There are {} flows within flowsize_range".format(flows.shape[1]))
    else:
        raise ValueError("path must be a .csv or .npz file")

    # use unix time as folder name
    folder = "flows_vis_{}".format(int(time.time()))
    if save_to is None:
        save_to = folder
    else:
        save_to = os.path.join(save_to, folder)

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    print("Saving flow figures to\n\t{}".format(save_to))

    # write log file
    with open(os.path.join(save_to, "config.log"), "w") as log_file:
        log_file.write("path: {}\n".format(path))
        log_file.write("save_to: {}\n".format(save_to))
        log_file.write("mode: {}\n".format(mode))
        log_file.write("flowsize_range: {}\n".format(flowsize_range))
        log_file.write("sample: {}\n".format(sample))
        log_file.write("need_divide_1e6: {}\n".format(need_divide_1e6))
        log_file.write("zero_threshold: {}\n".format(zero_threshold))
        log_file.write("flows.shape: {}\n".format(flows.shape))
    
    # save all flow figures
    t = np.linspace(0, 1, flows.shape[0])
    
    # if sample is too large, sample all flows
    if sample > flows.shape[1]:
        sample_flow_indices = np.arange(flows.shape[1])
    else:
        sample_flow_indices = np.random.choice(flows.shape[1], sample, replace=False)

    for i in tqdm(sample_flow_indices):
        fig, ax = plt.subplots()
        plt.plot(flows[:, i])
        # save with low resolution
        fig.savefig(os.path.join(save_to, "flow_{}.png".format(i)), dpi=100)
        plt.close(fig)
    
    print("{} flow figures are saved to\n\t{}".format(sample, save_to))
    

def plot_trace_level_packet_rate(
        path, 
        df=None,
        ax=None, axes=None, 
        mode=("time_point_count", 200), 
        flow_tuple=stats.five_tuple,
        flowsize_range=None, flowDurationRatio_range=None,
        nonzeroIntervalCount_range=None, maxPacketrate_range = None,
        sample=300, 
        need_divide_1e6=False, zero_threshold=0.5,
        start_at_zero=True,
        set_xlim=False,
        read_csv_kwargs={
            "verbose": False,
            "need_divide_1e6": False,
        },
        **kwargs):
    
    if ax is None and axes is None:
        raise ValueError("ax and axes can't be both None")
    if ax is not None and axes is not None:
        raise ValueError("ax and axes can't be both not None")
    
    if ax is not None and axes is None:
        axes = [ax]
    
    time_range = list(axes[0].get_xlim())

    if zero_threshold >= 1 or zero_threshold < 0:
        raise ValueError("zero_threshold must be in [0, 1)")
    
    # flows.shape is (n_timestamps, n_flows)
    if path.endswith(".csv"):
        print("Loading flows from this csv file:\n\t{}".format(path))
        df = data.load_csv(
            path, 
            **read_csv_kwargs,
        )
        if start_at_zero:
            df["time"] = df["time"] - df["time"].min()
        if set_xlim:
            time_range = [df["time"].min(), df["time"].max()]
            for ax in axes:
                ax.set_xlim(time_range)
        
        # truncate df to the time range of ax
        df = df[(df["time"] >= axes[0].get_xlim()[0] * 1e6) & (df["time"] <= axes[0].get_xlim()[1] * 1e6)]
        if flowsize_range is None and flowDurationRatio_range is None and nonzeroIntervalCount_range is None and maxPacketrate_range is None:
            flows = stats.df2flow(
                df,
                time_unit_exp=stats.time_point_count2time_unit_exp(
                    total_duration=df["time"].max() - df["time"].min(),
                    time_point_count=mode[1]),
                num_t=mode[1],)
            flows = flows[:, np.newaxis].astype(int)
        else:
            flows = data.load_flow_from_df(
                df, 
                mode=mode, flow_tuple=flow_tuple,
                flowsize_range=flowsize_range, 
                flowDurationRatio_range=flowDurationRatio_range,
                nonzeroIntervalCount_range=nonzeroIntervalCount_range,
                maxPacketrate_range=maxPacketrate_range,
            ).astype(int)
        print(flows.shape)

    elif path.endswith(".npz"):
        print("Loading flows from this npz file, ignoring time_unit_exp:\n\t{}".format(path))
        flows = data.load_flow_from_npz(
            path,
            zero_threshold=zero_threshold,
            flowsize_range=flowsize_range,
            flowDurationRatio_range=flowDurationRatio_range,
            nonzeroIntervalCount_range=nonzeroIntervalCount_range,
            maxPacketrate_range=maxPacketrate_range,
        )
        # flows = np.load(path)["packetrate"].T + zero_threshold
        # # flows = np.load(path)["output"].T * 55 + zero_threshold
        # # add 0.5 before flooring to avoid rounding to 0
        # flows = flows.astype(int)
        # # keep only flows within flowsize_range (i.e. the sum of packetrate is within flowsize_range)
        # print(flows.shape)
        # if flowsize_range is not None:
        #     s = np.sum(flows, axis=0)
        #     flows = flows[:, (s >= flowsize_range[0]) & (s < flowsize_range[1])]
        #     del s     
        # print("There are {} flows within flowsize_range".format(flows.shape[1]))
    else:
        raise ValueError("path must be a .csv or .npz file")


    # save all flow figures
    t = np.linspace(0, 1, flows.shape[0])
    t = t * (time_range[1] - time_range[0]) + time_range[0]
    # convert to packets / s
    flows = flows / (t[1] - t[0])
    
    # sum all flows
    trace = np.sum(flows, axis=1)
    for ax in axes:
        ax.plot(t, trace, **kwargs)

def compute_subplots_shape(num_subplots, bp=4):
    """
    Compute subplots shape in a Bootstrap grid fashion

    Example
    -------
    >>> compute_subplots_shape(3)
    (1, 3)
    >>> compute_subplots_shape(6)
    (2, 4)
    >>> compute_subplots_shape(7)
    (2, 4)
    >>> compute_subplots_shape(8)
    (2, 4)
    >>> compute_subplots_shape(11)
    (3, 4)
    """

    if num_subplots <= bp:
        return (1, num_subplots)
    else:
        if bp == 1:
            return (num_subplots, 1)
        return (np.max([(num_subplots - 1) // bp + 1, 1]), bp)


def subplots(num_x, num_y, bp=4, figsize=(8, 6), **kwargs):
    """
    Matplotlib subplots with Bootstrap grid fashion, result is flattened

    num_x and num_y is set for compatibility with plt.subplots()
    """
    num_subplots = num_x * num_y
    shape = compute_subplots_shape(num_subplots, bp)
    fig, axes = plt.subplots(*shape, figsize=figsize, **kwargs)
    if num_subplots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    return fig, axes

def savefig(fig, save_config):

    VALID_FORMAT = ["png", "pdf", "svg"]

    if save_config is None:
        print("save_config is None, not saving figure")
        return 
    
    folder = save_config["folder"]
    filename = save_config["filename"]
    format = save_config["format"]

    # create folder if not exist
    os.makedirs(folder, exist_ok=True)

    if format not in VALID_FORMAT:
        raise ValueError("Invalid format: {}".format(format))

    fig.savefig(os.path.join(folder, "{}.{}".format(filename, format)), format=format) 
    print("Figure saved to\n\t{}".format(os.path.join(folder, "{}.{}".format(filename, format))))
