"""
Manually compute the characteristics (features) of flows in a dataset.

The input is assumed to be an array of packet rate series
    flows, shape = (#interval, #flow)

"""

import numpy as np
import nta.utils.feature_extraction.characteristics as char
import os 
import pickle

from torch import nn

class FeatureExtractor:
    def __init__(self, config, verbose=False):
        """
        Extract features from flows according to a config file
        """
        self.config = config 
        self.verbose = verbose 
        if self.verbose:
            print("FeatureExtractor config: {}".format(self.config))
        self._norm_params = {}

    # def _set_norm_params(self, feature_vectors):
    #     method = self.config["normalize_features"]
    #     if method is None:
    #         pass 
    #     elif method == "max":
    #         # assuming to be nonnegative
    #         if (feature_vectors < 0).any():
    #             raise ValueError("Cannot normalize using max if there are negative values")
    #         self._norm_params["max"] = feature_vectors.max(axis=0)
    #         # if max == 0, then we set max to 1
    #         self._norm_params["max"][self._norm_params["max"] == 0] = 1
    #     elif method == "minmax":
    #         self._norm_params["min"] = feature_vectors.min(axis=0)
    #         self._norm_params["max"] = feature_vectors.max(axis=0)
    #         # if min == max and max != 0, then we set min to 0
    #         # elif min == max and max == 0, then we set max to 1
    #         eq_max_min_indices = self._norm_params["min"] == self._norm_params["max"]
    #         zero_max_indices = self._norm_params["max"] == 0
    #         self._norm_params["min"][eq_max_min_indices & (~zero_max_indices)] = 0
    #         self._norm_params["max"][eq_max_min_indices & zero_max_indices] = 1

    #     elif method == "normal":
    #         self._norm_params["mean"] = feature_vectors.mean(axis=0)
    #         self._norm_params["std"] = feature_vectors.std(axis=0)
    #     else:
    #         raise ValueError("Invalid normalization method: {}".format(method))

    def compute_norm_params(self, feature_vectors, save_to_folder=None):
        """
        compute normalization parameters, save if folder is specified 
        """

        if save_to_folder is not None and not os.path.exists(save_to_folder):
            os.makedirs(save_to_folder)

        method = self.config["normalize_features"]
        if method is None:
            pass 
        elif method == "max":
            # assuming to be nonnegative
            if (feature_vectors < 0).any():
                raise ValueError("Cannot normalize using max if there are negative values")
            self._norm_params["max"] = feature_vectors.max(axis=0)
            # if max == 0, then we set max to 1
            self._norm_params["max"][self._norm_params["max"] == 0] = 1
        elif method == "minmax":
            self._norm_params["min"] = feature_vectors.min(axis=0)
            self._norm_params["max"] = feature_vectors.max(axis=0)
            # if min == max and max != 0, then we set min to 0
            # elif min == max and max == 0, then we set max to 1
            eq_max_min_indices = self._norm_params["min"] == self._norm_params["max"]
            zero_max_indices = self._norm_params["max"] == 0
            self._norm_params["min"][eq_max_min_indices & (~zero_max_indices)] = 0
            self._norm_params["max"][eq_max_min_indices & zero_max_indices] = 1

        elif method == "normal":
            self._norm_params["mean"] = feature_vectors.mean(axis=0)
            self._norm_params["std"] = feature_vectors.std(axis=0)
        else:
            raise ValueError("Invalid normalization method: {}".format(method))

        if save_to_folder is None:
            return 

        # save to folder if specified
        save_to_path = os.path.join(save_to_folder, "norm_params.pkl")
        print("Saving norm_params to:\n\t{}".format(save_to_path)) 
        with open(save_to_path, "wb") as f:
            pickle.dump(self._norm_params, f)

    def load_norm_params(self, norm_params_folder):
        if not os.path.exists(norm_params_folder):
            raise ValueError("norm_params_folder {} does not exist".format(norm_params_folder))
        norm_params_path = os.path.join(norm_params_folder, "norm_params.pkl")
        if not os.path.exists(norm_params_path):
            raise ValueError("norm_params_path {} does not exist".format(norm_params_path))
        
        print("Loading norm_params from:\n\t{}".format(norm_params_path))
        with open(norm_params_path, "rb") as f:
            self._norm_params = pickle.load(f)


    def normalize(self, feature_vectors):
        method = self.config["normalize_features"]
        if method is None:
            return feature_vectors
        elif method == "max":
            return feature_vectors / self._norm_params["max"]
        elif method == "minmax":
            return (feature_vectors - self._norm_params["min"]) / (self._norm_params["max"] - self._norm_params["min"])
        elif method == "normal":
            return (feature_vectors - self._norm_params["mean"]) / self._norm_params["std"]
        else:
            raise ValueError("Invalid normalization method: {}".format(method)) 

    def denormalize(self, feature_vectors):
        # print("Denormalizing this feature vector: {}, shape={}".format(feature_vectors, feature_vectors.shape))
        back_to_1dim = False
        if feature_vectors.ndim == 1:
            back_to_1dim = True
            # feature_vectors = feature_vectors[:, np.newaxis]
            feature_vectors = feature_vectors[np.newaxis, :]
            # print("One dim, pad, now denormalizing this feature vector: {}, shape={}".format(feature_vectors, feature_vectors.shape))
        method = self.config["normalize_features"]
        if method is None:
            result = feature_vectors
        elif method == "max":
            result = feature_vectors * self._norm_params["max"]
        elif method == "minmax":
            result = feature_vectors * (self._norm_params["max"] - self._norm_params["min"]) + self._norm_params["min"]
        elif method == "normal":
            result = feature_vectors * self._norm_params["std"] + self._norm_params["mean"]
        else:
            raise ValueError("Invalid normalization method: {}".format(method))
        
        # take expm1 if log1p is used
        # print("After denormalization, we get this feature vector: {}, shape={}".format(result, result.shape))
        methods = self.config["methods"]
        for i, method in enumerate(methods):
            if "use_log" in methods[method] and methods[method]["use_log"]:
                result[:, i] = np.expm1(result[:, i])
                # result[i, :] = np.expm1(result[i, :])
        if back_to_1dim:
            result = result[0, :]
        return result

    
    def extract(self, flows, dfg, gks, save_to_folder=None):

        # remove flows with all 0s
        flows = flows[:, (flows != 0).any(axis=0)]
        print("After removing flows with all 0s, flows.shape = {}".format(flows.shape))

        # call all test methods, assumed to be defined in nta.utils.feature_extraction.characteristics
        methods = self.config["methods"]
        test_results = dict.fromkeys(methods, None)
        for method in methods:
            kwargs = methods[method]
            test_results[method] = getattr(char, method)(flows, dfg, gks, **kwargs)
            if self.verbose:
                print(method)
                print(test_results[method].shape)
                print(test_results[method][:2])
                print() 

        feature_vectors = np.stack(list(test_results.values()), axis=1)
        feature_vectors_normalization = [[nn.Identity()], [feature_vectors.shape[-1]]]
        if self.config["normalize_features"]:
            # n = {
            #     "max": lambda v: v / v.max(axis=0),
            #     "normal": lambda v: (v - v.mean(axis=0)) / v.std(axis=0),
            # }

            # if self.config["normalize_features"] not in n:
            #     raise ValueError("Invalid normalization method: {}".format(self.config["normalize_features"]))
            # feature_vectors = n[self.config["normalize_features"]](feature_vectors)
            
            print("Normalizing features using {}".format(self.config["normalize_features"]))
            # self._set_norm_params(feature_vectors)
            self.compute_norm_params(feature_vectors, save_to_folder=save_to_folder)
            feature_vectors = self.normalize(feature_vectors)

            if self.config["normalize_features"] == "max" or "minmax":
                feature_vectors_normalization = [[nn.Sigmoid()], [feature_vectors.shape[-1]]]
        
        return feature_vectors, feature_vectors_normalization

# NOTE deming
def simply_zeros(flows, dfg, gks, use_log=False):
    return np.zeros(flows.shape[1])

def simply_zeros2(flows, dfg, gks, use_log=False):
    return np.zeros(flows.shape[1])

def number_of_packets(flows, dfg, gks, use_log=False):
    """
    Count the number of packets in each flow
    """
    if use_log:
        return np.log1p(flows.sum(axis=0))
    return flows.sum(axis=0)


def duration(flows, dfg, gks, interval_duration=1, use_log=False):
    """
    Compute the duration of each flow. Since the input is already converted to packet rate,
        we can only get an approximation of the duration.
    
    For each flow of shape (#interval, ), its duration is (index of last nonzero - index of first nonzero + 1) * interval_duration
    """
    flow_nonzero = flows != 0
    
    flow_duration = flows.shape[0] - flow_nonzero.argmax(axis=0) - flow_nonzero[::-1, :].argmax(axis=0)
    flow_duration = flow_duration * interval_duration

    if use_log:
        return np.log1p(flow_duration)
    return flow_duration


def number_of_nonzero_intervals(flows, dfg, gks, use_log=False):
    """
    Count the number of nonzero intervals in each flow
    """
    result = (flows != 0).sum(axis=0)

    if use_log:
        return np.log1p(result) 
    return result

def flowstart(flows, dfg, gks, interval_duration=1, use_log=False):
    """
    Compute the start time of each flow
    """

    #flow_nonzero = flows != 0
    #flow_start = flow_nonzero.argmax(axis=0) * interval_duration

    flow_start = np.zeros(flows.shape[1])
    for i, gk in enumerate(gks):
        flow_start[i] = np.min(dfg.get_group(gk)["time"])
    
    if use_log:
        return np.log1p(flow_start)
    return flow_start

def flowend(flows, dfg, gks, interval_duration=1, use_log=False):
    """
    Compute the end time of each flow
    """

    #flow_nonzero = flows != 0
    #flow_end = flow_nonzero[::-1, :].argmax(axis=0) * interval_duration

    flow_end = np.zeros(flows.shape[1])
    for i, gk in enumerate(gks):
        flow_end[i] = np.max(dfg.get_group(gk)["time"])
    
    if use_log:
        return np.log1p(flow_end)
    return flow_end

def median(flows, dfg, gks, interval_duration=1, use_log=False):
    """
    Compute the middle time of each flow: median = (flowend + flowstart) / 2
    """
    flow_nonzero = flows != 0
    flow_start = flow_nonzero.argmax(axis=0) * interval_duration
    flow_end = flow_nonzero[::-1, :].argmax(axis=0) * interval_duration
    flow_median = (flow_start + flow_end) / 2
    
    if use_log:
        return np.log1p(flow_median)
    return flow_median

def span(flows, dfg, gks, interval_duration=1, use_log=False):
    """
    Compute the span of each flow: span = (flowend - flowstart) / 2
    """
    flow_nonzero = flows != 0
    flow_start = flow_nonzero.argmax(axis=0) * interval_duration
    flow_end = flow_nonzero[::-1, :].argmax(axis=0) * interval_duration
    flow_span = (flow_end - flow_start) / 2
    
    if use_log:
        return np.log1p(flow_span)
    return flow_span

def max_packetrate(flows, dfg, gks, use_log=False):
    """
    Compute the maximum packet rate of each flow
    """
    if use_log:
        return np.log1p(flows.max(axis=0))
    return flows.max(axis=0)


def mean_packetrate(flows, dfg, gks, use_log=False):
    """
    Compute the mean packet rate of each flow
    """
    if use_log:
        return np.log1p(flows.mean(axis=0))
    return flows.mean(axis=0)


def std_packetrate(flows, dfg, gks, use_log=False):
    """
    Compute the standard deviation of packet rate of each flow
    """
    if use_log:
        return np.log1p(flows.std(axis=0))
    return flows.std(axis=0)

def mean_nonzero_packetrate(flows, dfg, gks, use_log=False):
    """
    Compute the mean packet rate of each flow
    """

    nonzero_flows = flows.copy().astype(float)
    nonzero_flows[nonzero_flows == 0] = np.nan
    if use_log:
        return np.log1p(np.nanmean(nonzero_flows, axis=0))
    # return np.nanmean(nonzero_flows, axis=0)
    result = np.nanmean(nonzero_flows, axis=0)
    # replace nan with 0
    result[np.isnan(result)] = 0
    return result

def return_zero(flows, dfg, gks, use_log=False):
    """
    Compute the mean packet rate of each flow
    """

    return np.zeros(flows.shape[1])



def std_nonzero_packetrate(flows, dfg, gks, use_log=False):
    """
    Compute the standard deviation of packet rate of each flow
    """
    nonzero_flows = flows.copy().astype(float)
    nonzero_flows[nonzero_flows == 0] = np.nan
    if use_log:
        return np.log1p(np.nanstd(nonzero_flows, axis=0))
    # return np.nanstd(nonzero_flows, axis=0)
    result = np.nanstd(nonzero_flows, axis=0)
    # replace nan with 0
    result[np.isnan(result)] = 0
    return result


def mean_gap_between_nonzero(flows, dfg, gks, use_log=False, interval_duration=1):
    """
    [0, 0, 1, 0, 2, 0, 0, 0, 0, 3, 5, 0, 0, 1, 0]
    =>
    [2, 1, 4, 2, 1]
    => 
    mean([2, 1, 4, 2, 1])
    """
    # get all nonzero indices
    result = []
    for i in range(flows.shape[1]):
        flow = flows[:, i]
        nonzero_indices = np.nonzero(flow)[0]
        if len(nonzero_indices) == 1:
            result.append(0)
            continue
        gaps = np.diff(nonzero_indices) - 1
        gaps = gaps[gaps != 0]
        if len(gaps) == 0:
            result.append(0)
            continue
        result.append(gaps.mean())
    result = np.array(result)
    if use_log:
        return np.log1p(result * interval_duration)
    return result * interval_duration


def std_gap_between_nonzero(flows, dfg, gks, use_log=False, interval_duration=1):
    """
    [0, 0, 1, 0, 2, 0, 0, 0, 0, 3, 5, 0, 0, 1, 0]
    =>
    [2, 1, 4, 2, 1]
    => 
    std([2, 1, 4, 2, 1])
    """
    # get all nonzero indices
    result = []
    for i in range(flows.shape[1]):
        flow = flows[:, i]
        nonzero_indices = np.nonzero(flow)[0]
        if len(nonzero_indices) == 1:
            result.append(0)
            continue
        gaps = np.diff(nonzero_indices) - 1
        gaps = gaps[gaps != 0]
        if len(gaps) == 0:
            result.append(0)
            continue
        result.append(gaps.std())
    result = np.array(result)
    if use_log:
        return np.log1p(result * interval_duration)
    return result * interval_duration


def nonzero_density(flows, dfg, gks, use_log=False):
    pass 


# def fft(flows):
#     pass 

# def steadiness(flows):
#     """
#     Evaluate the level of steadiness / burstiness of flows in a dataset
#     """
#     pass 


# def magnitude(flows):
#     """
#     Evaluate the level of magnitude of flows in a dataset
#     """
#     pass

# def duration(flows):
#     """
#     Evaluate the level of duration of flows in a dataset
#     """
#     pass

# def size(flows):
#     """
#     Evaluate the level of size of flows in a dataset
#     """
#     pass

# def periodicity(flows):
#     """
#     Evaluate the level of periodicity of flows in a dataset
#     """
#     pass

# def interval_regularity(flows):
#     """
#     Evaluate the level of interval regularity / irregularity of flows in a dataset
#     """
#     pass

# def packetrate_regularity(flows):
#     """
#     Evaluate the level of packet rate regularity / irregularity of flows in a dataset
#     """
#     pass


# def density(flows):
#     """
#     Evaluate the level of density of flows in a dataset.

#     This is purely a visual feature. Fix the time point count, if the flows looks dense,
#     then it's dense.

#     It features small intervals and packet rate of similar magnitude.

#     This partially overlaps with steadiness and regularity.
#     """
#     pass