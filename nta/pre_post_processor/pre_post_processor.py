"""
Preprocessing and Postprocessing
"""
import typing
import numpy as np
import pandas as pd 
import os 
import scipy.sparse as ss 
import pickle
import random
from tqdm import tqdm 
from gensim.models import Word2Vec
import warnings
from sklearn.utils.extmath import randomized_svd
import json

import nta.utils.data as data
import nta.utils.data
import nta.utils.stats as stats
import nta.utils.wavelet as wavelet
import nta.utils.const as const

from nta.pre_post_processor.field import *
from nta.pre_post_processor.embedding_helper import build_annoy_dictionary_word2vec, get_original_objs
from nta.pre_post_processor.word2vec_embedding import *
import nta.utils.feature_extraction.vae as fe_vae
import nta.utils.feature_extraction.characteristics as fe_char

from pprint import pprint
import math
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


output_type_options = [
    "global_normalization_w/o_FTA",
    "global_normalization_with_FTA_w/o_TAL"
]

class PrePostProcessor:
    """
    Class for both preprocessing and postprocessing
    """
    def __init__(self, config: typing.Mapping[str, typing.Any]) -> None:
        self.config = config

        self.input_folder = self.config["input_folder"]
        self.input_file = self.config["input_file"]
        self.input_path = os.path.join(self.input_folder, self.input_file)
        self.input_type = self.config["input_type"]
        if not os.path.exists(self.input_path):
            raise ValueError("Input file not found:\n\t{}".format(self.input_path))
        if self.input_type not in ["pcap", "netflow"]:
            raise ValueError("Unrecognized input type: {}".format(self.input_type))
        print("Using input file of trace type {}:\n\t{}".format(self.input_type, self.input_path))

        self.output_folder = self.config["output_folder"]
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        print("Using output folder:\n\t{}".format(self.output_folder))

        # check if time_unit_exp's setting is correct
        if "pkt_rate_sample_method" not in self.config:
            # for backward comaptibility purpose
            if "time_unit_exp" in self.config:
                # warn user
                warnings.warn("time_unit_exp will be deprecated in the future, please use pkt_rate_sample_method instead")
                self.time_unit_exp = self.config["time_unit_exp"]
                print("Using time_unit_exp={}".format(self.time_unit_exp))
            else:
                raise ValueError("pkt_rate_sample_method not found in config")
        else:
            pkt_rate_sample_method = self.config["pkt_rate_sample_method"]
            if pkt_rate_sample_method[0] == "time_unit_exp":
                print("Convert trace to packet rate time series with time_unit_exp={}".format(pkt_rate_sample_method[1]))
            elif pkt_rate_sample_method[0] == "time_point_count":
                if not isinstance(pkt_rate_sample_method[1], int):
                    raise ValueError("pkt_rate_sample_method[1] should be int if pkt_rate_sample_method[0] is 'time_point_count'")
                print("Convert trace to packet rate time series with time_point_count={}".format(pkt_rate_sample_method[1]))
            else:
                raise ValueError("Unrecognized pkt_rate_sample_method: {}".format(pkt_rate_sample_method))

        self.flow_tuple = [stats.five_tuple[i] for i in self.config["flow_tuple"]]
        

        # if "use_metaoutput" not in self.config:
        #     # default to use metaoutput
        #     self.config["use_metaoutput"] = True

        if "output_type" not in self.config or self.config["output_type"] not in output_type_options:
            raise ValueError("Unrecognized output_type: {}. Must be one of the following:\n\t{}".format(
                self.config["output_type"], output_type_options))

        if "use_time_interval" not in self.config:
            # default to use time interval
            self.config["use_time_interval"] = True
        
        if "encoding_packetrate" not in self.config:
            self.config["encoding_packetrate"] = {"method": None}
        
        if "feature_extraction" not in self.config:
            self.config["feature_extraction"] = {"methods": None}

        # self.fields is a dict of fields, each field is an instance of Field class. This is used for normalization and denormalization
        #    This is the method adopted by NetShare's implementation.
        self.fields = {}

        # self.normalizations is a dict that specifies the last layer of each dimension of output of each generator
        # Format: { generator_name: [act_fn_a, act_fn_b], [n_a, n_b]}
        #     - For output[:n_a], apply activation function act_fn_a
        #     - For output[n_a: n_a + n_b], apply activation function act_fn_b
        #     where generator_name is one of these: ["fivetuple", "condition", "packetrate", "packetfield"]
        self.normalizations = {}

        # create preprocess folder
        self.preprocess_folder = os.path.join(self.output_folder, "preprocess")
        if not os.path.exists(self.preprocess_folder):
            os.makedirs(self.preprocess_folder)

        # create preprocess folder
        self.postprocess_folder = os.path.join(self.output_folder, "postprocess")
        if not os.path.exists(self.postprocess_folder):
            os.makedirs(self.postprocess_folder)
        
        # compute dims
        self.WORD2VEC_SIZE = self.config["word2vec_size"]
        # embedding of ip (bit), port (word2vec), proto (word2vec)
        self.metadata_dim = 64 * 2 + 3 * self.WORD2VEC_SIZE
        # packetrate within time unit in trace 
        # max of in-flow max packetrate
        self.output_dim = 1          # packetrate within time unit in flow

        self.sample_unit = self.config.get("packetfield_sample_unit", "pkt")
        self.max_pkt_num_per_timestep = self.config.get("max_pkt_num_per_timestep", 2000)


    def pre_process(self):
        """
        Parameters
        ----------

        
        Returns
        ----------
        Return nothing, save result to self.preprocess_folder as npz file instead

        
        Description
        ----------
        
        There are 2 ways to convert a trace into an ndarray of flows;
        
        For a trace with 
        -   duration 1000s
        -   1328 flows
        -   at most 1800 #packet in a flow
        -   at most 780s duration in a flow (may not be the same flow as the one above)

        Let's choose time unit 10s, then there are 100 time intervals.

        We assume that, for a single flow, there are
        -   at most 4073 out of 100 time intervals that has non-zero value
        
        we can convert all flows into 3 kinds of unified format

        1.  "time series": Time series with evenly-spaved time interval
            
            flows.shape = (1328, 100, 1)  

            trace duration 1000s, time unit 0.1s => time steps 100
        
            flows[i, :, 0]: ith flow 
            flows[i, t, 0]: packet rate of ith flow at time t

        2. "sequence of timestamps": Sequence of timestamps of packets in a flow

            This is the method adopted by NetShare. This part of documentation is only for the sake of completeness, and we 
                do not implement this method in this project.

            flows.shape = (1328, 1800, 2)       # NetShare's implemenetation of `max_flow_len`

            a flow has maximum 1800 packets, each packet has two features: timestamp and byte count

            flows[i, :, :]: ith flow
            flows[i, j, 0]: timestamp/interarrival of the jth packet of ith flow
            flows[i, j, 1]: byte count of the jth packet of ith flow
        """

        # if input is pcap, convert to csv
        if self.input_file.endswith(".pcap"):
            print("Converting pcap file to csv file...")
            data.pcap2csv(self.input_path, self.input_path[:-5] + ".csv", is_ip2int=True)
            self.input_path = self.input_path[:-5] + ".csv"

        df = data.load_csv(self.input_path, is_ip2int=True, verbose=False, need_divide_1e6=False)
        self.trace_start_time = df["time"].min()
        self.total_duration = df["time"].max() - self.trace_start_time
        df["time"] = df["time"] - self.trace_start_time
        df = df[df["time"] <= self.total_duration * self.config["trace_truncate_ratio"]]
        self.total_duration = self.total_duration * self.config["trace_truncate_ratio"]

        print("Truncate data to {}% of original size with duration: {}s".format(
            self.config["trace_truncate_ratio"]*100, self.total_duration))

        # set time_unit_exp
        if "time_unit_exp" not in self.config:
            if self.config["pkt_rate_sample_method"][0] == "time_unit_exp":
                self.config["time_unit_exp"] = self.config["pkt_rate_sample_method"][1]
                self.time_unit_exp = self.config["time_unit_exp"]
            elif self.config["pkt_rate_sample_method"][0] == "time_point_count":
                # convert time_point_count to time_unit_exp
                # self.config["time_unit_exp"] = np.log10(self.total_duration / self.config["pkt_rate_sample_method"][1])
                self.config["time_unit_exp"] = stats.time_point_count2time_unit_exp(
                    total_duration = self.total_duration,
                    time_point_count = self.config["pkt_rate_sample_method"][1],
                )
                self.time_unit_exp = self.config["time_unit_exp"]

        # group flows by flow_tuple
        print("Grouping flows using flow_tuple: {}".format(self.flow_tuple))
        dfg = df.groupby(self.flow_tuple)
        gks = dfg.groups.keys()


        if "flowsize_min_threshold" in self.config or "flowsize_max_threshold" in self.config:
            raise ValueError("flowsize_min_threshold and flowsize_max_threshold is deprecated, please use `flow_filter: {'flowsize': {'min': flowsize_min, 'max': flowsize_max} }` instead")

        # filter flows by flow_filter
        # format: 
        # "flow_filter": {
        #   "filter_type_1":  {
        #       'arg1': val1, 
        #       'arg2': val2,
        #   },
        #  "filter_type_2":  {
        #       'arg1': val1,
        #   }
        # }
        if "flow_filter" in self.config:
            print("Before filtering, there are {} packets and {} flows".format(
                len(df), len(gks)))
            
            flow_filter = self.config["flow_filter"]
            # we gradually remove gk from filtered_gks based on elements in flow_filter
            filtered_gks = gks

            # filter the following using groupby object
            # - flowsize
            # - flow_duration_ratio

            if "flowsize" in flow_filter:
                flowsize_min = flow_filter["flowsize"]["min"]
                flowsize_max = flow_filter["flowsize"]["max"]
                if flowsize_min == "inf":
                    flowsize_min = np.inf 
                if flowsize_max == "inf":
                    flowsize_max = np.inf
                flowsizes = dfg.size()
                filtered_gks = [gk for gk in filtered_gks if flowsizes[gk] >= flowsize_min and flowsizes[gk] < flowsize_max] 
                print("Filtering flows by flowsize (#pkt), keeping {} flows with #pkt in [{}, {})...".format(
                    len(filtered_gks), flowsize_min, flowsize_max))

            if "flow_duration_ratio" in flow_filter:
                flow_duration_ratio_min = flow_filter["flow_duration_ratio"]["min"]
                flow_duration_ratio_max = flow_filter["flow_duration_ratio"]["max"]
                flow_durations = dfg["time"].max() - dfg["time"].min()
                flow_duration_ratio = flow_durations / self.total_duration
                filtered_gks = [gk for gk in filtered_gks if flow_duration_ratio[gk] >= flow_duration_ratio_min and flow_duration_ratio[gk] < flow_duration_ratio_max]
                print("Filtering flows by flow_duration_ratio, keeping {} flows with flow_duration_ratio in [{}, {})...".format(
                    len(filtered_gks), flow_duration_ratio_min, flow_duration_ratio_max))
 
            # filter the following using packet rate time series (df2flow)
            # - nonzero_interval_count
            # - max_packetrate

            if "nonzero_interval_count" in flow_filter:
                nonzero_interval_count_min = flow_filter["nonzero_interval_count"]["min"]
                nonzero_interval_count_max = flow_filter["nonzero_interval_count"]["max"]
                if nonzero_interval_count_min == "inf":
                    nonzero_interval_count_min = np.inf
                if nonzero_interval_count_max == "inf":
                    nonzero_interval_count_max = np.inf
                new_gks = []
                num_t = stats.time_unit_exp2time_point_count(self.total_duration, self.time_unit_exp)
                for gk in tqdm(filtered_gks):
                    single_flow = stats.df2flow(dfg.get_group(gk), time_unit_exp=self.time_unit_exp, num_t=num_t)
                    nonzero_interval_count = np.sum(single_flow > 0)
                    if nonzero_interval_count >= nonzero_interval_count_min and nonzero_interval_count < nonzero_interval_count_max:
                        new_gks.append(gk)
                filtered_gks = new_gks
                print("Filtering flows by nonzero_interval_count, keeping {} flows with nonzero_interval_count in [{}, {})...".format(
                    len(filtered_gks), nonzero_interval_count_min, nonzero_interval_count_max))


            if "max_packetrate" in flow_filter:
                max_packetrate_min = flow_filter["max_packetrate"]["min"]
                max_packetrate_max = flow_filter["max_packetrate"]["max"]
                if max_packetrate_min == "inf":
                    max_packetrate_min = np.inf
                if max_packetrate_max == "inf":
                    max_packetrate_max = np.inf
                new_gks = []
                num_t = stats.time_unit_exp2time_point_count(self.total_duration, self.time_unit_exp)
                for gk in tqdm(filtered_gks):
                    single_flow = stats.df2flow(dfg.get_group(gk), time_unit_exp=self.time_unit_exp, num_t=num_t)
                    single_flow_max = np.max(single_flow)
                    if single_flow_max >= max_packetrate_min and single_flow_max < max_packetrate_max:
                        new_gks.append(gk)
                filtered_gks = new_gks
                print("Filtering flows by max_packetrate, keeping {} flows with max_packetrate in [{}, {})...".format(
                    len(filtered_gks), max_packetrate_min, max_packetrate_max))

            gks = filtered_gks

            if len(gks) < 10:
                warnings.warn("Too few flows (only {}) kept after filtering by the following flow_filter config:\n{}".format(
                    len(gks), flow_filter))
            

            # if filtered, recompute df and dfg to match the filtered gks
            df = pd.concat([dfg.get_group(gk) for gk in gks])
            # sort by time
            df = df.sort_values(by="time")
            # reindex df
            df = df.reset_index(drop=True)
            # recompute dfg
            dfg = df.groupby(self.flow_tuple)
            print("After filtering, there are {} packets and {} flows left".format(
                len(df), len(gks)))


        # convert sequence of packets into packetrate time series
        flows = stats.compute_od_flows(
            dfg, gks,
            self.total_duration, 
            self.time_unit_exp,
        )
        
        self.num_t = flows.shape[0]
        self.num_flows = flows.shape[1]
        
        assert(self.num_t == stats.time_unit_exp2time_point_count(self.total_duration, self.time_unit_exp))
        # assert(self.num_t == int(np.ceil(np.floor(self.total_duration / 10**self.time_unit_exp * 10) / 10)))
        assert(self.num_flows == len(gks))

        # smooth the data
        if "smoothing" not in self.config:
            self.config["smoothing"] = None 
            print("No smoothing applied")
        elif self.config["smoothing"]["type"] == "simple_moving_average":
            window_size = self.config["smoothing"]["window_size"]
            for i in range(self.num_flows):
                flows[:, i] = data.weighted_moving_average(
                    flows[:, i], 
                    window_size=window_size, 
                    weight_type="simple")
            print("Smoothed using simple moving average with window_size={}".format(window_size))
        elif self.config["smoothing"]["type"] == "exponential_moving_average":
            window_size = self.config["smoothing"]["window_size"]
            for i in range(self.num_flows):
                flows[:, i] = data.weighted_moving_average(
                    flows[:, i], 
                    window_size=window_size, 
                    weight_type="exponential")
            print("Smoothed using exponential moving average with window_size={}".format(window_size))
        elif self.config["smoothing"]["type"] == "bump_moving_average":
            window_size = self.config["smoothing"]["window_size"]
            for i in range(self.num_flows):
                flows[:, i] = data.weighted_moving_average(
                    flows[:, i], 
                    window_size=window_size, 
                    weight_type="bump")
            print("Smoothed using bump moving average with window_size={}".format(window_size))
        else:
            raise ValueError("Unrecognized smoothing type: {}".format(self.config["smoothing"]["type"]))

        self._save(df, dfg, gks, flows)

        print("Configuration:")
        pprint(self.config)

    def _save(self, df, dfg, gks, flows):
        """
        Save flows and metadata as npz file, save fields as pkl file
        """
        self.max_len = self.num_t
        
        print(flows.shape)

        condition = self._flows2condition(flows, dfg, gks)
        metadata = self._gks2metadata(gks, df)
        output = self._flows2output(df, dfg, gks, flows)
        #packetinfo = self._packetrate2packetinfo(metadata, flows.T, use_time_interval=self.config["use_time_interval"])
        packetinfo, packetindex = self._packetrate2packetinfo(condition, metadata, output)
        packetfield = self._dfg2packetfield(df, dfg, gks, output)

        if condition is not None:
            print("condition.shape: {}".format(condition.shape))
        else:
            print("condition is None")
        print("metadata.shape: {}".format(metadata.shape))
        print("output.shape: {}".format(output.shape))
        print("packetinfo.shape: {}".format(packetinfo.shape))
        print("packetfield.shape: {}".format(packetfield.shape))

        # save metadata, output as npz, save fields as pkl
        np.savez(
            os.path.join(
                self.preprocess_folder, "raw_packetrate.npz"), 
            condition   =   condition,
            metadata    =   metadata,
            output      =   output,
            packetrate  =   flows,
            ) 
        np.savez(
            os.path.join(
                self.preprocess_folder, "raw_packetfield.npz"),
            packetinfo  =   packetinfo,
            packetfield =   packetfield,
            packetindex =   packetindex
            )

        with open(os.path.join(self.preprocess_folder, "fields.pkl"), "wb") as file:
            pickle.dump(self.fields, file) 
        with open(os.path.join(self.preprocess_folder, "feature_extractor.pkl"), "wb") as file:
            pickle.dump(self.feature_extractor, file) 
        with open(os.path.join(self.preprocess_folder, "normalizations.pkl"), "wb") as file:
            pickle.dump(self.normalizations, file) 
        with open(os.path.join(self.preprocess_folder, "other_attrs.pkl"), "wb") as file:
            pickle.dump({
                "time_unit_exp": self.time_unit_exp,
                "trace_start_time": self.trace_start_time,
                "total_duration": self.total_duration,
                "num_t": self.num_t,
                "num_flows": self.num_flows,
                "max_len": self.max_len,
                "input_path": self.input_path
            }, file)

    def _gks2metadata(self, gks, df):
        """
        Embed metadata (bit encoding for ip, ip2vec for (port, proto)), 
        Normalize, store inverse of normalize to self
        """
        metadata = np.zeros((self.num_flows, self.metadata_dim))

        self.normalizations["fivetuple"] = [[], []]

        # bit encoding of srcip, dstip
        self.fields["srcip"] = BitField(
            name="srcip",
            num_bits=32
        )

        self.normalizations["fivetuple"][0].extend([nn.Softmax(dim=-1)] * 32)
        self.normalizations["fivetuple"][1].extend([2] * 32)

        self.fields["dstip"] = BitField(
            name="dstip",
            num_bits=32
        )

        self.normalizations["fivetuple"][0].extend([nn.Softmax(dim=-1)] * 32)
        self.normalizations["fivetuple"][1].extend([2] * 32)

        # word2vec encoding of srcport, dstport, proto
        for i in range(self.WORD2VEC_SIZE):
            self.fields["srcport_{}".format(i)] = ContinuousField(
                name="srcport_{}".format(i),
                norm_option=Normalization.MINUSONE_ONE,
                dim_x=1
            )
            self.fields["dstport_{}".format(i)] = ContinuousField(
                name="dstport_{}".format(i),
                norm_option=Normalization.MINUSONE_ONE,
                dim_x=1
            )
            self.fields["proto_{}".format(i)] = ContinuousField(
                name="proto_{}".format(i),
                norm_option=Normalization.MINUSONE_ONE,
                dim_x=1
            )
        
        self.normalizations["fivetuple"][0].append(nn.Tanh())
        self.normalizations["fivetuple"][1].append(3 * self.WORD2VEC_SIZE)

        
        # word2vec encoding of srcport, dstport, proto
        # use pretrained word2vec model if exists
        embed_model_name = word2vec_train(
            df=df,
            out_dir=self.preprocess_folder,
            word_vec_size=self.WORD2VEC_SIZE,
            encode_IP='bit'
        )
        embed_model = Word2Vec.load(embed_model_name)

        # Building Annoy Index for each word2vec embedded fields
        # FIXME: For now, the word2vec cols are hard coded into the source code
        print("Building annoy dictionary word2vec...")
        dict_type_annDictPair = build_annoy_dictionary_word2vec(
            df=df,
            model_path=embed_model_name,
            word2vec_cols=["srcport", "dstport", "proto"],
            word2vec_size=self.WORD2VEC_SIZE,
            n_trees=100,
        )
        # save annoy dictionary
        output_folder = os.path.join(self.preprocess_folder, "annoy_dict")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for type, annDictPair in dict_type_annDictPair.items():
            annDictPair[0].save(os.path.join(
                output_folder, f"{type}_ann.ann"))
            with open(os.path.join(output_folder, f"{type}_dict.json"), 'w') as f:
                json.dump(annDictPair[1], f)


        # TODO: slow implementation, can be optimized with GPU
        print("Embedding metadata...")
        for i, gk in tqdm(enumerate(gks)):
            # srcip
            metadata[i, 0:64] = self.fields["srcip"].normalize(gk[0])

            # dstip
            metadata[i, 64:128] = self.fields["dstip"].normalize(gk[1])

            # srcport
            metadata[i, 128: 128 + self.WORD2VEC_SIZE] = list(self._convert_word_to_vector(embed_model,
                str(gk[2]), norm_option=True))

            # dstport
            metadata[i, 128 + self.WORD2VEC_SIZE: 128 + 2 * self.WORD2VEC_SIZE] = list(self._convert_word_to_vector(embed_model,
                str(gk[3]), norm_option=True))
                                                                                                   
            # proto
            metadata[i, 128 + 2 * self.WORD2VEC_SIZE: 128 + 3 * self.WORD2VEC_SIZE] = list(self._convert_word_to_vector(embed_model,
                str(gk[4]), norm_option=True))
        print("Metadata is embedded into an array of shape {}".format(metadata.shape))
        
        return metadata


    def _flows2condition(self, flows, dfg, gks):
        """
        Extract conditional input from raw data, and feed to both metadata generator and packetrate generator
        """
        if self.config["feature_extraction"]["methods"] is None:
            print("No feature extraction method specified, return None")
            return 
        # extract features from each flow
        print("Extracting features from packetrate...")
        # normalization of feature vectors is independent
        self.feature_extractor = fe_char.FeatureExtractor(self.config["feature_extraction"])
        feature_vectors, feature_vectors_normalization = self.feature_extractor.extract(flows, dfg, gks, save_to_folder = self.preprocess_folder)
        self.normalizations["condition"] = feature_vectors_normalization
        print("Extract an array of feature vectors of shape {}".format(feature_vectors.shape))
        return feature_vectors


    def _flows2output(self, df, dfg, gks, flows):
        num_t = flows.shape[0]
        num_flow = flows.shape[1]
        flows = flows.T

        self.normalizations["packetrate"] = [[], []]

        # if self.config["output_type"] is not None:
        print("Normalizing output with output type ({})...".format(self.config["output_type"]))

        extra_features = []
        extra_features_normalizations = []

        # extract extra time series features from each flow, and concate to output
        if self.config.get("timestamp_recovery", {}).get("method") == "median_and_span" or self.config.get("timestamp_recovery", {}).get("method") == "timestamp_in_unit":
            """
            For each flow, compute the median and span of each interval, and add them
              as extra features to the time series.
            median and span of the portion within the interval that actually has packets

            For example. let
            - flows[0, :] = [0, 0, 3, 0, 4, 0, 0, 2]
            - interval length = 3
            - flows w/ fivetuple gks[0] has timestamp
                dfg.get_group(gks[0]) = [6.1, 6.2, 6.5, 7.3, 7.4, 8.3, 8.4, 21.2, 21.3]
            
            Based on flows[0, :], we know that there are three groups of packets
            - flows[0, 2] ~ [6.1, 6.2, 6.5] in interval [6, 9]
                median[0, 2]: (6.2 - 6) / (9 - 6) = 0.0667
                span[0, 2]: (6.5 - 6.1) / (9 - 6) = 0.1333
            - flows[0, 4] ~ [7.3, 7.4, 8.3, 8.4] in interval [12, 15]
            - flows[0, 7] ~ [21.3, 21.4] in interval [21, 24]

            For flows[1, 2], we can compute its median and span as follows
            - flows[1, 2]: [6.1, 6.2, 6.5] in interval [6, 9]
                median[0, 2]: 
                    middle of the actual range [6.1, 6.5] is 6.3
                    It's ratio is 
                        ((6.5 + 6.1) / 2 - 6) / 3 = 0.1
                span[0, 2]: (6.5 - 6.1) / 2 = 0.2
                    span of the actual range is 0.2 (mid-span, mid+span) = (6.2, 6.5)
                    It's ratio is 
                        ((6.5 - 6.1) / 2) / 3 = 0.0667
            
            To recover flows[0, 2] from median[0, 2] and span[0, 2], we can do the following
            - find the interval, which is [6, 9]
            - find the median: 
                the ratio is 0.1, 
                the actual value is 6 + 0.1 * 3 = 6.3
            - find the span: 
                the ratio is 0.0667, 
                the actual value is 0.0667 * 3 = 0.2
            => The actual range is [6.3 - 0.2, 6.3 + 0.2] = [6.1, 6.5]
            - find the packet rate, which is 3
            - equally divide the interval [median - span, median + span] using three points
                [6.1, 6.3, 6.5]
              This corresponds to an interarrival of 2*span / (packetrate - 1), which is 
            
            """
            print("Computing median and span of each interval for each flow...")
            result = nta.utils.data.flow2median_and_span(
                dfg = dfg,
                gks = gks,
                flows = flows,
                num_flow = num_flow,
                num_t = num_t,
                interval_length = 10 ** self.time_unit_exp
            )
            
            self.fields["flow_medians"] = ContinuousField(
                name="flow_medians",
                norm_option=Normalization.ZERO_ONE,
                min_x=0,
                max_x=1,
                dim_x=2,
                log1p_norm=False
            )
            self.fields["flow_spans"] = ContinuousField(
                name="flow_spans",
                norm_option=Normalization.ZERO_ONE,
                min_x=0,
                max_x=0.5,
                dim_x=2,
                log1p_norm=False
            )
            
            medians = self.fields["flow_medians"].normalize(result["medians"]["feature"])
            spans = self.fields["flow_spans"].normalize(result["spans"]["feature"])
            
            extra_features.append(
                medians)
            extra_features_normalizations.append(
                result["medians"]["normalization"])
            extra_features.append(
                spans)
            extra_features_normalizations.append(
                result["spans"]["normalization"])
        
        # add throughput
        if self.config.get("throughput"):
            if self.input_type == "pcap":
                throughput = stats.compute_od_throughput(
                    dfg, gks, "pkt_len",
                    self.total_duration, 
                    self.time_unit_exp,
                )
                throughput = throughput.T[:, :, np.newaxis]

                self.fields["throughput"] = ContinuousField(
                    name="throughput",
                    norm_option=Normalization.ZERO_ONE,
                    min_x = np.min(throughput),
                    max_x = np.max(throughput),
                    dim_x=2,
                    log1p_norm=True
                )

                throughput_normalization = [nn.Sigmoid(), 1]

                extra_features.append(
                    self.fields["throughput"].normalize(throughput)
                )
                extra_features_normalizations.append(
                    throughput_normalization
                )
            elif self.input_type == "netflow":
                throughput1 = stats.compute_od_throughput(
                    dfg, gks, "pkt",
                    self.total_duration, 
                    self.time_unit_exp,
                )
                throughput2 = stats.compute_od_throughput(
                    dfg, gks, "byt",
                    self.total_duration, 
                    self.time_unit_exp,
                )
                throughput1 = throughput1.T[:, :, np.newaxis]
                throughput2 = throughput2.T[:, :, np.newaxis]

                self.fields["throughput1"] = ContinuousField(
                    name="throughput1",
                    norm_option=Normalization.ZERO_ONE,
                    min_x = np.min(throughput1),
                    max_x = np.max(throughput1),
                    dim_x=2,
                    log1p_norm=True
                )
                self.fields["throughput2"] = ContinuousField(
                    name="throughput2",
                    norm_option=Normalization.ZERO_ONE,
                    min_x = np.min(throughput2),
                    max_x = np.max(throughput2),
                    dim_x=2,
                    log1p_norm=True
                )

                throughput_normalization = [nn.Sigmoid(), 1]

                extra_features.append(
                    self.fields["throughput1"].normalize(throughput1)
                )
                extra_features_normalizations.append(
                    throughput_normalization
                )
                extra_features.append(
                    self.fields["throughput2"].normalize(throughput2)
                )
                extra_features_normalizations.append(
                    throughput_normalization
                )
            
        # concatenate extra features
        if len(extra_features) > 0:
            extra_features = np.concatenate(extra_features, axis=-1)
        else:
            extra_features = None
            extra_features_normalizations = None

        if self.config["output_type"] == "global_normalization_with_FTA_w/o_TAL":
            
            flow_start_time = np.argmax((flows > 0), axis=1, keepdims=True)
            flow_end_time = np.argmax(np.flip(flows > 0, axis=1), axis=1, keepdims=True)
            flow_duration = flows.shape[1] - flow_start_time - flow_end_time

            output = np.zeros((flows.shape[0], flows.shape[1], 1))
            output_activation = np.zeros((flows.shape[0], flows.shape[1], 2))
            for i in range(flows.shape[0]):
                output[i, :flow_duration[i,0], 0] = flows[i, flow_start_time[i,0]:(flow_start_time[i,0] + flow_duration[i,0])]
                if extra_features is not None:
                    extra_features[i, :flow_duration[i,0]] = extra_features[i, flow_start_time[i,0]:(flow_start_time[i,0] + flow_duration[i,0])]
                    extra_features[i, flow_duration[i,0]:] = 0
                output_activation[i, :flow_duration[i,0], 0] = 1
                output_activation[i, flow_duration[i,0]:, 1] = 1
            
            if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
                self.fields["output"] = ContinuousField(
                    name="output",
                    norm_option=Normalization.ZERO_ONE,
                    min_x = np.min(output[output > 0]),
                    max_x = np.max(output),
                    dim_x=2,
                    log1p_norm=True
                )
                
                # # normalization for output
                # self.normalizations["packetrate"][0].append(nn.Sigmoid())
                # self.normalizations["packetrate"][1].append(1)

                output_zero_flag = np.concatenate(
                    [output > 0, output <= 0], axis=-1
                ).astype(float)
                output_zero_flag = output_zero_flag * output_activation[:,:,[0]]

                # # normalization for output_zero_flag
                # self.normalizations["packetrate"][0].append(nn.Softmax(dim=-1))
                # self.normalizations["packetrate"][1].append(2)
            else:
                self.fields["output"] = ContinuousField(
                    name="output",
                    norm_option=Normalization.ZERO_ONE,
                    min_x = np.min(output),
                    max_x = np.max(output),
                    dim_x=2,
                    log1p_norm=True
                )

                # # normalization for output
                # self.normalizations["packetrate"][0].append(nn.Sigmoid())
                # self.normalizations["packetrate"][1].append(1)
            
            output = self.fields["output"].normalize(output)
            if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
                output[output < 0] = 0

            # concatenate extra features
            if extra_features is not None:
                output = np.concatenate([output, extra_features], axis=-1)

            # concatenate flags to the very end
            if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
                output = np.concatenate(
                    [output, output_zero_flag, output_activation], axis=-1
                )
            else:
                output = np.concatenate(
                    [output, output_activation], axis=-1
                )
            

            # assign normalization (activation function at last layer) to each dimension of packetrate)

            # normalization for output
            self.normalizations["packetrate"][0].append(nn.Sigmoid())
            self.normalizations["packetrate"][1].append(1)

            # normalization for extra features, as is specified by extra_features_normalizations
            if extra_features is not None:
                for i in range(extra_features.shape[-1]):
                    self.normalizations["packetrate"][0].append(extra_features_normalizations[i][0])
                    self.normalizations["packetrate"][1].append(extra_features_normalizations[i][1])

            # normalization for output_zero_flag
            if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
                self.normalizations["packetrate"][0].append(nn.Softmax(dim=-1))
                self.normalizations["packetrate"][1].append(2)
            
            # normalization for output_activation
            self.normalizations["packetrate"][0].append(nn.Softmax(dim=-1))
            self.normalizations["packetrate"][1].append(2)

        
        elif self.config["output_type"] == "global_normalization_w/o_FTA":
            flow_start_time = np.argmax((flows > 0), axis=1, keepdims=True)
            flow_end_time = np.argmax(np.flip(flows > 0, axis=1), axis=1, keepdims=True)
            flow_duration = flows.shape[1] - flow_start_time - flow_end_time
            
            output = np.zeros((flows.shape[0], flows.shape[1], 1))
            output_activation = np.zeros((flows.shape[0], flows.shape[1], 3))
            for i in range(flows.shape[0]):
                output[i, :, 0] = flows[i]
                output_activation[i, :flow_start_time[i,0], 0] = 1
                output_activation[i, flow_start_time[i,0]:(flow_start_time[i,0]+flow_duration[i,0]), 1] = 1
                output_activation[i, (flow_start_time[i,0]+flow_duration[i,0]):, 2] = 1
            
            if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
                self.fields["output"] = ContinuousField(
                    name="output",
                    norm_option=Normalization.ZERO_ONE,
                    min_x = np.min(output[output > 0]),
                    max_x = np.max(output),
                    dim_x=2,
                    log1p_norm=True
                )
                
                # # normalization for output
                # self.normalizations["packetrate"][0].append(nn.Sigmoid())
                # self.normalizations["packetrate"][1].append(1)

                output_zero_flag = np.concatenate(
                    [output > 0, output <= 0], axis=-1
                ).astype(float)
                output_zero_flag = output_zero_flag * output_activation[:,:,[1]]

                # # normalization for output_zero_flag
                # self.normalizations["packetrate"][0].append(nn.Softmax(dim=-1))
                # self.normalizations["packetrate"][1].append(2)
            else:
                self.fields["output"] = ContinuousField(
                    name="output",
                    norm_option=Normalization.ZERO_ONE,
                    min_x = np.min(output),
                    max_x = np.max(output),
                    dim_x=2,
                    log1p_norm=True
                )

                # # normalization for output
                # self.normalizations["packetrate"][0].append(nn.Sigmoid())
                # self.normalizations["packetrate"][1].append(1)
            
            output = self.fields["output"].normalize(output)
            if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
                output[output < 0] = 0

            # concatenate extra features
            if extra_features is not None:
                output = np.concatenate([output, extra_features], axis=-1)

            # concatenate flags to the very end
            if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
                output = np.concatenate(
                    [output, output_zero_flag, output_activation], axis=-1
                )
            else:
                output = np.concatenate(
                    [output, output_activation], axis=-1
                )
            

            # assign normalization (activation function at last layer) to each dimension of packetrate)

            # normalization for output
            self.normalizations["packetrate"][0].append(nn.Sigmoid())
            self.normalizations["packetrate"][1].append(1)

            # normalization for extra features, as is specified by extra_features_normalizations
            if extra_features is not None:
                for i in range(extra_features.shape[-1]):
                    self.normalizations["packetrate"][0].append(extra_features_normalizations[i][0])
                    self.normalizations["packetrate"][1].append(extra_features_normalizations[i][1])

            # normalization for output_zero_flag
            if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
                self.normalizations["packetrate"][0].append(nn.Softmax(dim=-1))
                self.normalizations["packetrate"][1].append(2)
            
            # normalization for output_activation
            self.normalizations["packetrate"][0].append(nn.Softmax(dim=-1))
            self.normalizations["packetrate"][1].append(3)
        
        return output
            
    
    def _packetrate2packetinfo(self, condition, fivetuple, packetrate):
        packetinfo, packetindex = self.to_packetinfo(
            torch.tensor(condition),
            torch.tensor(fivetuple),
            torch.tensor(packetrate),
            return_index=True
        )
        print("Trace is converted to an array of packetinfo of shape {}".format(packetinfo.shape))
        return packetinfo.cpu().numpy(), packetindex


    def _dfg2packetfield(self, df, dfg, gks, packetrate):
        """
        packetfield contains all fields of a packet except its timestamp
        At preprocessing stage, we store the packetfield of the raw trace to use as input to generator & discriminator 4
        
        Instead of extracting packet fields from df directly, we extract by the order of flows grouped by fivetuple, then shuffle the packet.
        This is because we need to follow the order of groupby object in _packetrate2packetinfo
        
        The order of flows to be extracted is determined by gks, which can be a filtered version of dfg.groups.keys()

        """
        # use df to determine the shape of packetfield
        if self.input_type == "pcap":
            flag_strings = pd.unique(df["flag"])
            field_dim = 5 + len(flag_strings)
            if self.config.get("timestamp_recovery", {}).get("method") == "timestamp_in_unit":
                field_dim += 1
            if self.sample_unit == "timestep":
                field_dim += 2
            with open(os.path.join(self.preprocess_folder, "flag_strings.pkl"), "wb") as flag_strings_file:
                pickle.dump(flag_strings, flag_strings_file)
        elif self.input_type == "netflow":
            type_strings = pd.unique(df["type"])
            field_dim = 3 + len(type_strings)
            if self.config.get("timestamp_recovery", {}).get("method") == "timestamp_in_unit":
                field_dim += 1
            if self.sample_unit == "timestep":
                field_dim += 2
            with open(os.path.join(self.preprocess_folder, "type_strings.pkl"), "wb") as type_strings_file:
                pickle.dump(type_strings, type_strings_file)
        else:
            raise ValueError("Unrecognized input type: {}".format(self.input_type))
        
        # save min and max timestamp
        self.fields["time"] = ContinuousField(
            name="timestamp",
            norm_option=Normalization.ZERO_ONE,
            min_x = np.min(df["time"]),
            max_x = np.max(df["time"]),
            dim_x=1
        )
        
        # sum of sizes to determine num_packets
        
        if self.sample_unit == "pkt":
            packetfield = np.zeros((len(df), field_dim))
        elif self.sample_unit == "timestep":
            denormed_packetrate = self.denormalize_packetrate(packetrate)
            packetfield = np.zeros(((denormed_packetrate > 0).sum(), self.max_pkt_num_per_timestep, field_dim))

        print("packetfield.shape: {}".format(packetfield.shape))

        self.normalizations["packetfield"] = [[], []]

        # store all fields for normalization and denormalization
        if self.input_type == "pcap":
            # fields: pkt_len,version,ihl,tos,id,flag,off,ttl
            # NetShare only uses: pkt_len, tos, id, flag, off, ttl

            self.fields["pkt_len"] = ContinuousField(
                name="pkt_len",
                norm_option=Normalization.ZERO_ONE,
                min_x = np.min(df["pkt_len"]),
                max_x = np.max(df["pkt_len"]),
                dim_x=1
            )

            self.fields["tos"] = ContinuousField(
                name="tos",
                norm_option=Normalization.ZERO_ONE,
                min_x = 0.0,
                max_x = 255.0,
                dim_x=1
            )

            self.fields["id"] = ContinuousField(
                name="id",
                norm_option=Normalization.ZERO_ONE,
                min_x = 0.0,
                max_x = 65535.0,
                dim_x=1
            )

            self.normalizations["packetfield"][0].append(nn.Sigmoid())
            self.normalizations["packetfield"][1].append(3)

            flag_strings = pd.unique(df["flag"])
            self.fields["flag"] = StringField(
                name="flag",
                strings=flag_strings,
            )

            self.normalizations["packetfield"][0].append(nn.Softmax(dim=-1))
            self.normalizations["packetfield"][1].append(len(flag_strings))

            self.fields["off"] = ContinuousField(
                name="off",
                norm_option=Normalization.ZERO_ONE,
                min_x = 0.0,
                max_x = 65535.0,
                dim_x=1
            )

            self.fields["ttl"] = ContinuousField(
                name="ttl",
                norm_option=Normalization.ZERO_ONE,
                min_x = 0.0,
                max_x = 255.0,
                dim_x=1
            )

            self.normalizations["packetfield"][0].append(nn.Sigmoid())
            self.normalizations["packetfield"][1].append(2)
            
            if self.config.get("timestamp_recovery", {}).get("method") == "timestamp_in_unit":
                self.normalizations["packetfield"][0].append(nn.Sigmoid())
                self.normalizations["packetfield"][1].append(1)
            
            if self.sample_unit == "timestep":
                self.normalizations["packetfield"][0].append(nn.Softmax(dim=-1))
                self.normalizations["packetfield"][1].append(2)
                
           
        elif self.input_type == "netflow":

            self.fields["td"] = ContinuousField(
                name="td",
                norm_option=Normalization.ZERO_ONE,
                min_x = np.min(df["td"]),
                max_x = np.max(df["td"]),
                dim_x=1
            )

            self.fields["pkt"] = ContinuousField(
                name="pkt",
                norm_option=Normalization.ZERO_ONE,
                min_x = np.min(df["pkt"]),
                max_x = np.max(df["pkt"]),
                dim_x=1
            )

            self.fields["byt"] = ContinuousField(
                name="byt",
                norm_option=Normalization.ZERO_ONE,
                min_x = np.min(df["byt"]),
                max_x = np.max(df["byt"]),
                dim_x=1
            )

            self.normalizations["packetfield"][0].append(nn.Sigmoid())
            self.normalizations["packetfield"][1].append(3)

            type_strings = pd.unique(df["type"])
            self.fields["type"] = StringField(
                name="type",
                strings=type_strings,
            )
            print("All types: {}".format(type_strings))

            self.normalizations["packetfield"][0].append(nn.Softmax(dim=-1))
            self.normalizations["packetfield"][1].append(len(type_strings))
            
            if self.config.get("timestamp_recovery", {}).get("method") == "timestamp_in_unit":
                self.normalizations["packetfield"][0].append(nn.Sigmoid())
                self.normalizations["packetfield"][1].append(1)
            
            if self.sample_unit == "timestep":
                self.normalizations["packetfield"][0].append(nn.Softmax(dim=-1))
                self.normalizations["packetfield"][1].append(2)

        # extract and normalize the fields
        start = 0
        end = 0
        print("Extracting packet fields...")
        if self.sample_unit == "pkt":
            for gk in tqdm(gks):
                start = end
                flow = dfg.get_group(gk)
                flow_len = len(flow)
                end = start + flow_len
                if self.input_type == "pcap":
                    packetfield[start: end, 0] = self.fields["pkt_len"].normalize(flow["pkt_len"])
                    packetfield[start: end, 1] = self.fields["tos"].normalize(flow["tos"])
                    packetfield[start: end, 2] = self.fields["id"].normalize(flow["id"])
                    packetfield[start: end, 3:3+len(flag_strings)] = self.fields["flag"].normalize(flow["flag"])
                    packetfield[start: end, 3+len(flag_strings)] = self.fields["off"].normalize(flow["off"])
                    packetfield[start: end, 3+len(flag_strings)+1] = self.fields["ttl"].normalize(flow["ttl"])
                    if self.config.get("timestamp_recovery", {}).get("method") == "timestamp_in_unit":
                        packetfield[start: end, 3+len(flag_strings)+2] = (flow["time"] - self.fields["time"].min_x) / (10 ** self.time_unit_exp) % 1
                        packetfield[start: end, 3+len(flag_strings)+2][flow["time"] >= self.fields["time"].max_x] = 1
                elif self.input_type == "netflow":
                    packetfield[start: end, 0] = self.fields["td"].normalize(flow["td"])
                    packetfield[start: end, 1] = self.fields["pkt"].normalize(flow["pkt"])
                    packetfield[start: end, 2] = self.fields["byt"].normalize(flow["byt"])
                    packetfield[start: end, 3:3+len(type_strings)] = self.fields["type"].normalize(flow["type"])
                    if self.config.get("timestamp_recovery", {}).get("method") == "timestamp_in_unit":
                        packetfield[start: end, 3+len(type_strings)] = (flow["time"] - self.fields["time"].min_x) / (10 ** self.time_unit_exp) % 1
                        packetfield[start: end, 3+len(type_strings)][flow["time"] >= self.fields["time"].max_x] = 1
        elif self.sample_unit == "timestep":
            seq_count = 0
            for i, gk in tqdm(enumerate(gks)):
                flow = dfg.get_group(gk)
                start = 0
                end = 0
                for j in range(denormed_packetrate.shape[1]):
                    if denormed_packetrate[i][j] > 0:
                        start = end
                        end = start + min(denormed_packetrate[i][j], self.max_pkt_num_per_timestep)

                        if self.input_type == "pcap":
                            packetfield[seq_count, :end-start, 0] = self.fields["pkt_len"].normalize(flow["pkt_len"][start:end])
                            packetfield[seq_count, :end-start, 1] = self.fields["tos"].normalize(flow["tos"][start:end])
                            packetfield[seq_count, :end-start, 2] = self.fields["id"].normalize(flow["id"][start:end])
                            packetfield[seq_count, :end-start, 3:3+len(flag_strings)] = self.fields["flag"].normalize(flow["flag"][start:end])
                            packetfield[seq_count, :end-start, 3+len(flag_strings)] = self.fields["off"].normalize(flow["off"][start:end])
                            packetfield[seq_count, :end-start, 3+len(flag_strings)+1] = self.fields["ttl"].normalize(flow["ttl"][start:end])
                            if self.config.get("timestamp_recovery", {}).get("method") == "timestamp_in_unit":
                                packetfield[seq_count, :end-start, 3+len(flag_strings)+2] = (flow["time"][start:end] - self.fields["time"].min_x) / (10 ** self.time_unit_exp) % 1
                                packetfield[seq_count, :end-start, 3+len(flag_strings)+2][flow["time"][start:end] >= self.fields["time"].max_x] = 1
                        elif self.input_type == "netflow":
                            packetfield[seq_count, :end-start, 0] = self.fields["td"].normalize(flow["td"][start:end])
                            packetfield[seq_count, :end-start, 1] = self.fields["pkt"].normalize(flow["pkt"][start:end])
                            packetfield[seq_count, :end-start, 2] = self.fields["byt"].normalize(flow["byt"][start:end])
                            packetfield[seq_count, :end-start, 3:3+len(type_strings)] = self.fields["type"].normalize(flow["type"][start:end])
                            if self.config.get("timestamp_recovery", {}).get("method") == "timestamp_in_unit":
                                packetfield[seq_count, :end-start, 3+len(type_strings)] = (flow["time"][start:end] - self.fields["time"].min_x) / (10 ** self.time_unit_exp) % 1
                                packetfield[seq_count, :end-start, 3+len(type_strings)][flow["time"][start:end] >= self.fields["time"].max_x] = 1
                        
                        packetfield[seq_count, :end-start, -2] = 1
                        packetfield[seq_count, end-start:, -1] = 1

                        if denormed_packetrate[i][j] > self.max_pkt_num_per_timestep:
                            end = start + denormed_packetrate[i][j]
                            print(f"max_pkt_num_per_timestep exceeded: {self.max_pkt_num_per_timestep} vs {denormed_packetrate[i][j]}") 
                        
                        seq_count += 1

        print("Trace is converted to an array of packetfield of shape {}".format(packetfield.shape))

        return packetfield
        

    def _convert_word_to_vector(self, model, word, norm_option=False):
        all_words_str = list(model.wv.vocab.keys())

        # Privacy-related
        # If word not in the vocabulary, replace with nearest neighbor
        # Suppose that protocol is covered
        #   while very few port numbers are out of range
        if word not in all_words_str:
            # print(f"{word} not in dict")
            all_words = []
            for ele in all_words_str:
                if ele.isdigit():
                    all_words.append(int(ele))
            all_words = np.array(all_words).reshape((-1, 1))
            nbrs = NearestNeighbors(
                n_neighbors=1, algorithm='ball_tree').fit(all_words)
            distances, indices = nbrs.kneighbors([[int(word)]])
            nearest_word = str(all_words[indices[0][0]][0])
            # print("nearest_word:", nearest_word)
            model.init_sims()
            return model.wv.word_vec(nearest_word, use_norm=norm_option)
        else:
            model.init_sims()
            return model.wv.word_vec(word, use_norm=norm_option)
    
    def _convert_vector_to_word(self, annDictType, vector):
        # load annoy dict for type (port or proto)
        type_ann = AnnoyIndex(self.WORD2VEC_SIZE, metric='angular')
        type_ann_path = os.path.join(
            self.preprocess_folder, 
            "annoy_dict",
            "{}_ann.ann".format(annDictType))
        if not os.path.exists(type_ann_path):
            raise ValueError("annoy_dict not found at {}".format(type_ann_path))
        type_ann.load(type_ann_path)

        # load dict for type (port or proto)
        type_dict_path = os.path.join(
            self.preprocess_folder, 
            "annoy_dict",
            "{}_dict.json".format(annDictType))
        if not os.path.exists(type_dict_path):
            raise ValueError("annoy_dict not found at {}".format(type_dict_path))
        with open(type_dict_path, "r") as f:
            type_dict = json.load(f)
        
        return np.asarray(get_original_objs(
            ann=type_ann,
            vectors=[vector[i] for i in range(vector.shape[0])],
            dic={int(k): v for k, v in type_dict.items()}
        ))
        

    def _metadata_post_process(self, metadata, ip_output_type="decimal"):
        """
        param metadata: numpy array of shape (num_flows, metadata_dim)
        """
        
        embed_model_name = os.path.join(self.preprocess_folder, "word2vec_vecSize_{}.model".format(self.WORD2VEC_SIZE))
        embed_model = Word2Vec.load(embed_model_name)

        # srcport
        srcport = self._convert_vector_to_word(
            annDictType="port", 
            vector=metadata[:, 128: 128 + self.WORD2VEC_SIZE])

        # dstport
        dstport = self._convert_vector_to_word(
            annDictType="port", 
            vector=metadata[:, 128 + self.WORD2VEC_SIZE: 128 + 2 * self.WORD2VEC_SIZE])

        # proto
        proto = self._convert_vector_to_word(
            annDictType="proto", 
            vector=metadata[:, 128 + 2 * self.WORD2VEC_SIZE: 128 + 3 * self.WORD2VEC_SIZE])

        result = []
        for i in tqdm(range(metadata.shape[0])):
            # srcip 
            srcip = self.fields["srcip"].denormalize(metadata[i, 0: 64], output_type=ip_output_type)

            # dstip
            dstip = self.fields["dstip"].denormalize(metadata[i, 64: 128], output_type=ip_output_type)

            result.append((srcip, dstip, srcport[i], dstport[i], proto[i]))

        return result
    
    def metadata_post_process(self, metadata_result, result_filename, ip_output_type="decimal"):

        print("Postprocessing metadata...")

        fivetuple, = metadata_result

        metadata = self._metadata_post_process(fivetuple, ip_output_type=ip_output_type)
        
        if not result_filename is None:
            np.savez(
                os.path.join(self.postprocess_folder, result_filename),
                fivetuple=fivetuple,
                metadata=metadata
            )
        return np.array(metadata)
    
    def flowlevel_post_process(self, flowlevel_result, result_filename="syn_flowlevel.npz"):
        
        condition, fivetuple = flowlevel_result
        
        if not result_filename is None:
            np.savez(
                os.path.join(self.postprocess_folder, result_filename),
                condition=condition,
                fivetuple=fivetuple
            )

    def packetrate_post_process(self, packetrate_result, result_filename="syn_packetrate.npz"):
        """
        Use fields to denormalize output
        Output as a numpy array (npz file) of shape (num_flows, max_len, output_dim)
        """
        output, = packetrate_result 

        self.num_flows = output.shape[0]

        print("Denormalizing output with output type ({})...".format(self.config["output_type"]))
        
        packetrate = self.denormalize_packetrate(output)

        # just in case
        # def mask2duration(mask):
        #     if np.min(mask) == 0 and np.max(mask) == 1:
        #         return np.argmin(mask)
        #     elif np.min(mask) == 1:
        #         return mask.shape[0]
        #     elif np.max(mask) == 0:
        #         return 0
        
        # if self.config["output_type"] == "global_normalization_with_FTA_w/o_TAL":
        #     packetrate = self.fields["output"].denormalize(output[:, :, 0])
        #     if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
        #         mask = ((output[:, :, -4] + output[:, :, -3]) > 0) * (output[:, :, -4] > output[:, :, -3])
        #         mask[:, 0] = 1 #force
        #         packetrate = packetrate * mask
        #     else:
        #         def continuous_mask(mask):
        #             return_mask = mask
        #             tmp_mask = return_mask[:, 0]
        #             for i in range(return_mask.shape[1]):
        #                 return_mask[:, i] = return_mask[:, i] * tmp_mask
        #                 tmp_mask = return_mask[:, i]
        #             return return_mask
        #         mask = continuous_mask(output[:, :, -2] > output[:, :, -1])
        #         mask[:, 0] = 1 #force
        #         packetrate = packetrate * mask
        #     packetrate = np.rint(packetrate).astype(int)
        #     # NOTE prevent invalid values
        #     packetrate[:, 0] = np.maximum(packetrate[:, 0], 1)
        
        # elif self.config["output_type"] == "global_normalization_w/o_FTA":
        #     #packetrate = self.fields["output"].denormalize(output[:, :, [0]])
        #     #if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
        #     #    packetrate = packetrate * (output[:, :, 1] + output[:, :, 2] > 0)[:, :, None]
        #     #    packetrate = packetrate * (output[:, :, 1] > output[:, :, 2])[:, :, None]
            
        #     packetrate = self.fields["output"].denormalize(output[:, :, 0])
        #     if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
        #         mask = ((output[:, :, -5] + output[:, :, -4]) > 0) * (output[:, :, -5] > output[:, :, -4])
        #         packetrate = packetrate * mask
        #     packetrate = np.rint(packetrate).astype(int)
        #     # NOTE prevent invalid values
        #     packetrate[:, 0] = np.maximum(packetrate[:, 0], 1)


        # invert smoothing if smoothing is used
        if "smoothing" not in self.config or not self.config["smoothing"]:
            pass 
        elif self.config["smoothing"]["type"] == "simple_moving_average":
            for i in range(self.num_flows):
                packetrate[i, :, 0] = data.rev_weighted_moving_average(
                    packetrate[i, :, 0],
                    self.config["smoothing"]["window_size"],
                    "simple" 
                )
        elif self.config["smoothing"]["type"] == "exponential_moving_average":
            for i in range(self.num_flows):
                packetrate[i, :, 0] = data.rev_weighted_moving_average(
                    packetrate[i, :, 0],
                    self.config["smoothing"]["window_size"],
                    "exponential"
                )
        elif self.config["smoothing"]["type"] == "bump_moving_average":
            for i in range(self.num_flows):
                packetrate[i, :, 0] = data.rev_weighted_moving_average(
                    packetrate[i, :, 0],
                    self.config["smoothing"]["window_size"],
                    "bump"
                )
        else:
            raise ValueError("Unknown smoothing type: {}".format(self.config["smoothing"]["type"]))

        if packetrate.ndim == 3:
            # remove 1 dim
            packetrate = packetrate[:, :, 0]
        
        packetrate = np.rint(packetrate).astype(int)

        if not result_filename is None:
            # save as npz file
            np.savez(
                os.path.join(self.postprocess_folder, result_filename),
                output=output,
                packetrate=packetrate,
            )
        return np.array(packetrate)

    def packetfield_post_process(self, packetfield_result, result_filename="syn_packetfield.npz"):
        
        packetinfo, packetfield = packetfield_result

        if not result_filename is None:
            np.savez(
                os.path.join(self.postprocess_folder, result_filename),
                packetinfo=packetinfo,
                packetfield=packetfield
            )
    
    def sample_unit_reform(self, packetrate_denormalized, packetinfo, packetfield):

        reformed_packetrate_denormalized = packetrate_denormalized.copy()
        reformed_packetinfo = []
        reformed_packetfield = []

        seq_count = 0

        for i in range(packetrate_denormalized.shape[0]):
            for j in range(packetrate_denormalized.shape[1]):
                if packetrate_denormalized[i][j] > 0:

                    def continuous_mask(mask):
                        return_mask = mask
                        tmp_mask = return_mask[0]
                        for i in range(return_mask.shape[0]):
                            return_mask[i] = return_mask[i] * tmp_mask
                            tmp_mask = return_mask[i]
                        return return_mask
                    mask = continuous_mask(packetfield[seq_count, :, -2] > packetfield[seq_count, :, -1])
                    mask[0] = 1 #force

                    # print(mask.sum())
                    # print(packetinfo[[seq_count], :].shape)
                    # print(packetinfo[[seq_count], :].repeat(mask.sum(), axis=0).shape)
                    # print(packetfield[seq_count, mask, :-2].shape)
                    
                    reformed_packetrate_denormalized[i][j] = mask.sum()
                    reformed_packetinfo.append(packetinfo[[seq_count], :].repeat(mask.sum(), axis=0))
                    reformed_packetfield.append(packetfield[seq_count, mask, :-2])
        
        reformed_packetinfo = np.concatenate(reformed_packetinfo, axis=0)
        reformed_packetfield = np.concatenate(reformed_packetfield, axis=0)

        print("Reformed packetinfo shape: ", reformed_packetinfo.shape)
        print("Reformed packetfield shape: ", reformed_packetfield.shape)
    
        return reformed_packetrate_denormalized, reformed_packetinfo, reformed_packetfield

    def trace_post_process(self, result, result_filename):
        """
        the overall post_processing

        merge metadata, timestamp, packet fields into a csv
        """
        # NOTE skip saving

        # packetrate: (num_flows, max_len, output_dim)
        condition, fivetuple, packetrate, packetinfo, packetfield = result
        # save processed result to npz
        dfg_metadata_denormalized = self.metadata_post_process((fivetuple, ), result_filename=result_filename+".metadata", ip_output_type="decimal")
        # self.flowlevel_post_process((condition, fivetuple), result_filename=result_filename+".flowlevel")
        packetrate_denormalized = self.packetrate_post_process((packetrate, ), result_filename=result_filename+".packetrate")
        # self.packetfield_post_process((packetinfo, packetfield), result_filename=result_filename+".packetfield")

        if self.sample_unit == "timestep":
            packetrate_denormalized, packetinfo, packetfield = self.sample_unit_reform(packetrate_denormalized, packetinfo, packetfield)
        
        # convert npz result to csv
        # metadata_result = np.load(os.path.join(self.postprocess_folder, result_filename+".metadata.npz"))
        # dfg_metadata_denormalized = metadata_result["metadata"]
        # use a dict of (fivetuple, dfg_metadata_normalized) to convert packetinfo to csv to save time
        dfg_metadata_denormalized_dict = {
            tuple(fivetuple[i]): tuple(dfg_metadata_denormalized[i])
            for i in range(dfg_metadata_denormalized.shape[0])
        }
        # packetrate_result = np.load(os.path.join(self.postprocess_folder, result_filename+".packetrate.npz"))
        # packetrate_denormalized = packetrate_result["packetrate"]

        # -------------------------
        # eliminate flows with same 5-tuple
        # -------------------------
        
        print("eliminating repeated data")

        unique_metadata = set()
        flow_index = np.ones(fivetuple.shape[0]).astype(bool)
        packet_index = np.ones(packetfield.shape[0]).astype(bool)
        j = 0
        for i in range(fivetuple.shape[0]):
            if tuple(dfg_metadata_denormalized[i]) in unique_metadata:
                flow_index[i] = False
                packet_index[j:j+packetrate_denormalized[i].sum()] = False
            j += packetrate_denormalized[i].sum()
            unique_metadata.add(tuple(dfg_metadata_denormalized[i]))
        
        fivetuple = fivetuple[flow_index]
        condition = condition[flow_index]
        packetrate = packetrate[flow_index]
        packetinfo = packetinfo[packet_index]
        packetfield = packetfield[packet_index]
        packetrate_denormalized = packetrate_denormalized[flow_index]

        print("Converting all data to csv...")


        # packetinfo.shape = (num_packet, condition_dim + fivetuple_dim + packetrate_dim * window_size)
        df_metadata = packetinfo[:, condition.shape[1]: condition.shape[1]+fivetuple.shape[1]]
        df_metadata_denormalized = []
        df_timestamp_denormalized = []
        df_fields_denormalized = []

        # metadata
        for i in tqdm(range(packetinfo.shape[0])):
            single_metadata = tuple(df_metadata[i, :])
            if single_metadata not in dfg_metadata_denormalized_dict:
                raise ValueError("Packetinfo {} not in dfg_metadata_normalized_dict".format(single_metadata))
            df_metadata_denormalized.append(dfg_metadata_denormalized_dict[single_metadata])
        
        #def compute_flowstarts(condition, packetrate):
        #    flowstarts = []
        #    flow_condition = self.feature_extractor.denormalize(condition)
        #    flowstarts_cond = np.rint(flow_condition[:, -2]).astype(int)
        #    flowends_cond = np.rint(flow_condition[:, -1]).astype(int)
        #    for i in range(packetrate.shape[0]):
        #        
        #        flow_start_time = flowstarts_cond[i]
        #        flow_end_time = flowends_cond[i]
        #        flow_duration = packetrate.shape[1] - np.flip(packetrate[i] > 0).argmax()
        #        bias = (packetrate.shape[1] - flow_start_time - flow_end_time) - flow_duration
        #        flow_start_time += bias // 2
        #        flow_end_time += (bias + 1) // 2
        #        flowstarts.append(flow_start_time)
        # 
        #    return flowstarts
        
        #flowstarts = compute_flowstarts(condition, packetrate_denormalized)
        
        if self.config["output_type"] == "global_normalization_with_FTA_w/o_TAL":
            flow_condition = self.feature_extractor.denormalize(condition)
            flowmid = (flow_condition[:, -2] + flow_condition[:, -1]) / 2 / (10 ** self.time_unit_exp)

        # timestamp
        if "timestamp_recovery" not in self.config:
            self.config["timestamp_recovery"] = { "method": "equidistant" }
        rec_config = self.config["timestamp_recovery"]

        if rec_config["method"] == "equidistant":
            timestamps = data.packetrate2timestamp(
                packetrate_denormalized.T, 
                interval_duration = 1, 
                mode = rec_config["method"],
            )
            
        elif rec_config["method"] == "median_and_span":
            medians = self.fields["flow_medians"].denormalize(packetrate[:, :, 1])
            spans = self.fields["flow_spans"].denormalize(packetrate[:, :, 2])
            timestamps = data.packetrate2timestamp(
                packetrate_denormalized.T, 
                interval_duration = 1, 
                mode=rec_config["method"],
                params={
                    "medians": medians,
                    "spans": spans
                }
            )
            
        elif rec_config["method"] == "timestamp_in_unit":
            timestamps = data.packetrate2timestamp(
                packetrate_denormalized.T, 
                interval_duration = 1, 
                mode=rec_config["method"],
                params={
                    "timestamp_in_unit": packetfield[:, -1]
                }
            )
        
        #timestamps = [ flowstarts[i] + timestamps[i] for i in range(len(timestamps)) ] 
        if self.config["output_type"] == "global_normalization_with_FTA_w/o_TAL":
            timestamps = [ flowmid[i] + timestamps[i] - (np.max(timestamps[i]) + np.min(timestamps[i])) / 2 for i in range(len(timestamps)) ] 
        # concatneate timestamp
        timestamps = np.concatenate(timestamps)
        assert(timestamps.shape[0] == packetinfo.shape[0])
        df_timestamp_denormalized = self.fields["time"].denormalize(timestamps[:, None] / self.num_t) 
        df_timestamp_denormalized = np.rint(df_timestamp_denormalized).astype(int)
        df_timestamp_denormalized += int(self.trace_start_time)

        # packetfield
        if self.input_type == "pcap":
            with open(os.path.join(self.preprocess_folder, "flag_strings.pkl"), "rb") as flag_strings_file:
                flag_strings = pickle.load(flag_strings_file)
        elif self.input_type == "netflow":
            with open(os.path.join(self.preprocess_folder, "type_strings.pkl"), "rb") as type_strings_file:
                type_strings = pickle.load(type_strings_file)
        for i in tqdm(range(packetfield.shape[0])):
            single_packetfield = packetfield[i, :]
            if self.input_type == "pcap":
                df_fields_denormalized.append([
                    self.fields["pkt_len"].denormalize(single_packetfield[0]),
                    self.fields["tos"].denormalize(single_packetfield[1]),
                    self.fields["id"].denormalize(single_packetfield[2]),
                    self.fields["flag"].denormalize(single_packetfield[3:3+len(flag_strings)]),
                    self.fields["off"].denormalize(single_packetfield[3+len(flag_strings)]),
                    self.fields["ttl"].denormalize(single_packetfield[3+len(flag_strings)+1]),
                ])
            elif self.input_type == "netflow":
                df_fields_denormalized.append([
                    self.fields["td"].denormalize(single_packetfield[0]),
                    self.fields["pkt"].denormalize(single_packetfield[1]),
                    self.fields["byt"].denormalize(single_packetfield[2]),
                    self.fields["type"].denormalize(single_packetfield[3:3+len(type_strings)]),
                ])  
        
        # merge and convert to csv        
        if self.input_type == "pcap":
            # srcip,dstip,srcport,dstport,proto,time,pkt_len,version,ihl,tos,id,flag,off,ttl
            col_names = ["srcip", "dstip", "srcport", "dstport", "proto", "time", "pkt_len", "tos", "id", "flag", "off", "ttl"] 
            # convert to csv
            df = pd.DataFrame(
                data=((
                    *df_metadata_denormalized[i], *df_timestamp_denormalized[i], *df_fields_denormalized[i]) 
                    for i in range(packetinfo.shape[0])),
                columns=col_names
                )
            df = df.sort_values(by=["time"])
            # only keep packets with timestamp inside of raw time range
            df = df[(df["time"] >= self.trace_start_time) & (df["time"] <= self.trace_start_time + self.total_duration)]
            print("Saving result...")
            df.to_csv(os.path.join(self.postprocess_folder, result_filename+".csv"), index=False)
        elif self.input_type == "netflow":
            col_names = ["srcip", "dstip", "srcport", "dstport", "proto", "ts", "td", "pkt", "byt", "type"]
            # convert to csv
            df = pd.DataFrame(
                data=((
                    *df_metadata_denormalized[i], *df_timestamp_denormalized[i], *df_fields_denormalized[i]) 
                    for i in range(packetinfo.shape[0])),
                columns=col_names
                )
            df = df.sort_values(by=["ts"])
            df = df[(df["ts"] >= self.trace_start_time) & (df["ts"] <= self.trace_start_time + self.total_duration)]
            print("Saving result...")
            df.to_csv(os.path.join(self.postprocess_folder, result_filename+".csv"), index=False)

    def prob_post_process(self, prob):
        
        prob_fivetuple, prob_packetrate_addi, prob_packetrate, prob_fivetuple_hat, prob_packetrate_addi_hat, prob_packetrate_hat = prob

        np.savez(
            os.path.join(self.postprocess_folder, "debug.npz"),
            prob_fivetuple=prob_fivetuple,
            prob_packetrate_addi=prob_packetrate_addi,
            prob_packetrate=prob_packetrate,
            prob_fivetuple_hat=prob_fivetuple_hat,
            prob_packetrate_addi_hat=prob_packetrate_addi_hat,
            prob_packetrate_hat=prob_packetrate_hat
        )
    
    def denormalize_packetrate(self, output):

        denormed_packetrate = self.fields["output"].denormalize(output[:, :, 0])
        
        if self.config["output_type"] == "global_normalization_with_FTA_w/o_TAL":
            if "output_zero_flag" in self.config and self.config["output_zero_flag"]:
                mask = ((output[:, :, -4] + output[:, :, -3]) > 0) * (output[:, :, -4] > output[:, :, -3])
                mask[:, 0] = 1 #force
            else:
                def continuous_mask(mask):
                    return_mask = mask
                    tmp_mask = return_mask[:, 0]
                    for i in range(return_mask.shape[1]):
                        return_mask[:, i] = return_mask[:, i] * tmp_mask
                        tmp_mask = return_mask[:, i]
                    return return_mask
                mask = continuous_mask(output[:, :, -2] > output[:, :, -1])
                mask[:, 0] = 1 #force
        elif self.config["output_type"] == "global_normalization_w/o_FTA":
            mask = ((output[:, :, -5] + output[:, :, -4]) > 0) * (output[:, :, -5] > output[:, :, -4])
        denormed_packetrate = denormed_packetrate * mask
        denormed_packetrate = np.rint(denormed_packetrate).astype(int)
        # NOTE deming
        denormed_packetrate[:, 0] = np.maximum(denormed_packetrate[:, 0], 1)

        return denormed_packetrate

    
    # convert condition, fivetuple, packetrate into packetinfo
    # single_sample controls whether just pick one packet in each flow
    def to_packetinfo(self, condition, fivetuple, packetrate, single_sample=False, return_index=False):
        """
        return: packetinfo of shape (num_packet, condition_dim + fivetuple_dim + packetrate_dim * window_size)
        """
        window_size = self.config["window_size"]

        #expand packetrate
        if window_size // 2 > 0:
            expand_packetrate = torch.cat([
                torch.zeros(packetrate.shape[0], window_size//2, packetrate.shape[2]).to(packetrate.device),
                packetrate,
                torch.zeros(packetrate.shape[0], window_size//2, packetrate.shape[2]).to(packetrate.device),
            ], dim=1)
            expand_packetrate[:, :(window_size//2), -1] = 1
            expand_packetrate[:, -(window_size//2):, -1] = 1
        else:
            expand_packetrate = packetrate
        
        output = packetrate.detach().cpu().numpy().astype(np.float64)
        denormed_packetrate = self.denormalize_packetrate(output)
        
        if single_sample: # a fast mode that randomly pick one packet in each flow
            packetinfo_list = []
            for i in range(condition.shape[0]):
                if self.sample_unit == "pkt":
                    random_index = random.randint(0, max(denormed_packetrate[i].sum(), 1))
                    j = (np.cumsum(denormed_packetrate[i]) >= random_index).argmax()
                elif self.sample_unit == "timestep":
                    random_index = random.randint(0, max((denormed_packetrate[i] > 0).sum(), 1))
                    j = (np.cumsum(denormed_packetrate[i] > 0) >= random_index).argmax()
                packetinfo_list.append(
                    torch.cat([
                        condition[i][None, :],
                        fivetuple[i][None, :],
                        expand_packetrate[i, j:(j+window_size), :].flatten()[None, :]
                    ], dim=-1)
                )
            return torch.cat(packetinfo_list, dim=0)
        else: # a slower mode that generate all packets
            if return_index:
                packetindex = np.zeros(condition.shape[0]+1).astype(int)
            packetinfo_list = []
            for i in range(condition.shape[0]):
                #denormed_packetrate[i, 0] = max(denormed_packetrate[i, 0], 1) #force only on ca?
                if return_index:
                    if self.sample_unit == "pkt":
                        packetindex[i+1] = packetindex[i] + denormed_packetrate[i].sum()
                    elif self.sample_unit == "timestep":
                        packetindex[i+1] = packetindex[i] + (denormed_packetrate[i] > 0).sum()
                for j in range(denormed_packetrate.shape[1]):
                    if denormed_packetrate[i][j] > 0:
                        if self.sample_unit == "pkt":
                            packetinfo_list.append(
                                torch.cat([
                                    condition[i].expand(denormed_packetrate[i][j], -1),
                                    fivetuple[i].expand(denormed_packetrate[i][j], -1),
                                    expand_packetrate[i, j:(j+window_size), :].flatten().expand(denormed_packetrate[i][j], -1)
                                ], dim=-1)
                            )
                        elif self.sample_unit == "timestep":
                            packetinfo_list.append(
                                torch.cat([
                                    condition[[i]],
                                    fivetuple[[i]],
                                    expand_packetrate[i, j:(j+window_size), :].flatten()[np.newaxis, :]
                                ], dim=-1)
                            )
        
            if return_index:
                return torch.cat(packetinfo_list, dim=0), packetindex
            else:
                return torch.cat(packetinfo_list, dim=0)
    
    def load(self):

        with open(os.path.join(self.preprocess_folder, "fields.pkl"), "rb") as file:
            self.fields = pickle.load(file)
        with open(os.path.join(self.preprocess_folder, "feature_extractor.pkl"), "rb") as file:
            self.feature_extractor = pickle.load(file)
        with open(os.path.join(self.preprocess_folder, "normalizations.pkl"), "rb") as file:
            self.normalizations = pickle.load(file)
        with open(os.path.join(self.preprocess_folder, "other_attrs.pkl"), "rb") as file:
            other_attrs = pickle.load(file)
            self.time_unit_exp = other_attrs["time_unit_exp"]
            self.trace_start_time = other_attrs["trace_start_time"]
            self.total_duration = other_attrs["total_duration"]
            self.num_t = other_attrs["num_t"]
            self.num_flows = other_attrs["num_flows"]
            self.max_len = other_attrs["max_len"]
            self.input_path = other_attrs["input_path"]
        
