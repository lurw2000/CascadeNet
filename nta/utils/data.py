import pandas as pd 
import os
from tqdm import tqdm
from scapy.all import IP, TCP, UDP, ICMP
from scapy.all import wrpcap, PcapReader
from scipy.stats import rankdata
import ipaddress
import socket
import struct
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
# to avoid cropping axis labels
rcParams.update({'figure.autolayout': True})
import pickle
from getpass import getpass
from torch import nn
import copy

import nta.utils.stats as stats
from nta.utils.const import interarrival_eps as eps
from nta.utils.const import PROTO2INT, INT2PROTO

password = ""
def input_password():
    global password
    password = getpass("Enter Password: ")


def load_csv(
        path, 
        is_ip2int=False, 
        is_proto2int=False,
        verbose=True, 
        need_divide_1e6=True, 
        return_interval=False,
        unify_timestamp_fieldname=True,
        ):
    """
    Load csv data of network traffic from path

    :param path to the csv file.
    :param is_ip2int: whether to convert ip address to int
    :return: Pandas dataframe of the network traffic
    """

    # Load the data
    if path == "missing_file_placeholder":
        df = pd.DataFrame()  # Use an empty DataFrame for the placeholder
    elif not path.endswith(".csv"):
        raise Exception(f"Input file must be a csv file:\n\t{path}")
    
    print(f"Loading data from:\n\t{path}")

    df = pd.read_csv(path)

    # rename time column 
    if 'time' not in df.columns:
        df = df.rename(columns={'ts': 'time'})

    if "time" not in df.columns:
        print(f"Warning: 'time' column not found in the CSV file: {path}")
        print("Adding fake time column with negative values.")
        df["time"] = np.arange(len(df)) * -1  # Use negative values as impossible time
    
    # process time field (divide by 1e6, minus the first time stamp)
    if need_divide_1e6 == "auto":
        warnings.warn("need_divide_1e6 is set to auto, which will be deprecated in the future. Please set it to True or False explicitly.")
        need_divide_1e6 = False
        # if is is integer value, divide by 1e6
        if isinstance(df["time"][0], (int, np.integer)):
            print("timestamp is integer, thus divide it by 1e6")
            need_divide_1e6 = True
        # if it's a floating number, don't divide by 1e6
        elif isinstance(df["time"][0], (float, np.floating)):
            # we assume all datasets are captured after year 1980 (around 3e9). Thus, if the floating number 
            #  is larger than 1e9, we assume it's in microseconds, and divide it by 1e6
            if df["time"][0] > 3e9:
                print("timestamp is floating number, but larger than 3e9 (year 1980), thus divide it by 1e6")
                need_divide_1e6 = True
        else:
            print("Can't decide. Don't divide timestamp by 1e6")

    if isinstance(need_divide_1e6, bool) and need_divide_1e6:
        df["time"] = df["time"] / 1e6
    df = df.sort_values(by="time")
    total_interval = np.array([df["time"].min(), df["time"].max()])
    #df["time"] = df["time"] - df["time"].min()

    print(f"Number of packets: {len(df)}")
    print(f"Trace duration: {df['time'].max() - df['time'].min()} seconds")

    if is_ip2int:
        df["srcip"] = df["srcip"].apply(ip2int)
        df["dstip"] = df["dstip"].apply(ip2int)

    if is_proto2int:
        df["proto"] = df["proto"].apply(proto2int)
    
    if unify_timestamp_fieldname:
        if "ts" in df.columns:
            df = df.rename(columns={"ts": "time"})

    if verbose:
        print(df)
    
    if return_interval:
        return df, total_interval
    return df

def ip2int(ip):
    # ip should always be ipv4
    if isinstance(ip, str) and "." in ip and len(ip.split(".")) == 4:
        return sum([256**j*int(i) for j,i in enumerate(ip.split(".")[::-1])])
    # # mac 
    # elif isinstance(ip, str) and (":" in ip or "-" in ip) and len(ip.split(":")) == 6:
    #     return int(ip.replace(':', '').replace('-', ''), 16)
    # # ipv6
    # elif isinstance(ip, str) and ":" in ip and len(ip.split(":")) == 8:
    #     return sum([65536**j*int(i, 16) for j, i in enumerate(ip.split(":")[::-1])])
    # assumed to have been converted by ip2int, but stored as a str
    elif isinstance(ip, str):
        return int(ip)
    elif isinstance(ip, int):
        return ip
    else:
        try:
            return int(ip)
        except ValueError:
            raise ValueError("ip should be either string or int")

def int2ip(num):
    return ".".join([str((num//(256**i))%256) for i in range(3,-1,-1)])

def proto2int(proto):
    # if already int, return itself
    if proto in INT2PROTO:
        return proto
    elif isinstance(proto, (int, np.integer)):
        raise ValueError("proto number {} not found in INT2PROTO dict (valid numbers are {})".format(
            proto, list(INT2PROTO.keys())
        ))
    # if convertable to int, return int
    try:
        p = int(proto)
        proto = p
        if proto in INT2PROTO:
            return proto
    except ValueError:
        pass 
    proto = proto.upper()
    # for unknown protocol, return -1
    result = PROTO2INT.get(proto, -1)
    return result

def int2proto(num):
    # if already str, return itself
    if num in PROTO2INT:
        return PROTO2INT[num]
    num = int(num)
    # for unknown protocol, return "unknown"
    result = INT2PROTO.get(num, "unknown")
    return result


def IP_str2int(IP_str):
    return int(ipaddress.ip_address(IP_str))


def pcap2csv(input_path, output_path=None, is_ip2int=True, need_divide_1e6=False, keep_frag=False, verbose=False):
    """
    Convert pcap file to csv file

    :param input_path: Path to the input pcap file. 
        The csv file should the following format:
            srcip,dstip,srcport,dstport,proto,time,pkt_len
    :param output_path: Path to the output csv file
    """

    print("Converting pcap file to csv file...")
    print("Input path (pcap file):\n\t{}".format(input_path))
    if output_path is None:
        output_path = input_path.replace(".pcap", ".csv")
    print("Output path (csv file):\n\t{}".format(output_path))
    print("Convert ip address to int: {}".format(is_ip2int))
    count = 0   
    discard_count = 0
    frag_count = 0
    if keep_frag:
        print("Keeping fragmented packets (assign to a fragment packet the port number of the first packet)")
        id2ports = {}
    with open(output_path, "w") as file:
        file.write("srcip,dstip,srcport,dstport,proto,time,pkt_len,version,ihl,tos,id,flag,off,ttl\n")
        for packet in tqdm(PcapReader(input_path)): 
            # if count % 20000 == 0:   
            #     print(f"\r#pkt scanned: {count}", end="")
            count += 1
            try:
                if keep_frag:
                    if packet.haslayer(IP) and packet[IP].frag == 0:
                        id2ports[packet[IP].id] = (packet.sport, packet.dport)
                    else:
                        frag_count += 1
                        if packet[IP].id not in id2ports:
                            # skip frag if the first packet is not captured
                            continue 
                        packet.sport, packet.dport = id2ports[packet[IP].id]
                        if verbose:
                            print("Fragmented packet with id = {}".format(packet[IP].id))
                            print("flags={}, frag={}".format(packet.flags, packet.frag))
                            print()

                if not is_ip2int:
                    srcip = packet.getlayer(IP).src
                    dstip = packet.getlayer(IP).dst
                else:
                    srcip = ip2int(packet.getlayer(IP).src)
                    dstip = ip2int(packet.getlayer(IP).dst)
                
                srcport = packet.sport
                dstport = packet.dport
                proto = packet.getlayer(IP).proto

                time = packet.time
                if not need_divide_1e6:
                    time = int(time * 1e6)

                file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    srcip, dstip,
                    srcport, dstport,
                    proto,
                    time,
                    packet.wirelen,
                    packet.version, packet.ihl, packet.tos, packet.id, packet.flags, packet.frag, packet.ttl
                ))
 
                # if not is_ip2int: 
                #     file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                #         packet.getlayer(IP).src, packet.getlayer(IP).dst,
                #         packet.sport, packet.dport,
                #         packet.getlayer(IP).proto,
                #         time,
                #         packet.wirelen,
                #         packet.version, packet.ihl, packet.tos, packet.id, packet.flags, packet.frag, packet.ttl
                #     ))
                # else:
                #     file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                #         ip2int(packet.getlayer(IP).src), ip2int(packet.getlayer(IP).dst),
                #         packet.sport, packet.dport,
                #         packet.getlayer(IP).proto,
                #         time,
                #         packet.wirelen,
                #         packet.version, packet.ihl, packet.tos, packet.id, packet.flags, packet.frag, packet.ttl
                #     ))  
                    # print("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    #     ip2int(packet.getlayer(IP).src), ip2int(packet.getlayer(IP).dst),
                    #     packet.sport, packet.dport,
                    #     packet.getlayer(IP).proto,
                    #     time,
                    #     packet.wirelen,
                    #     packet.version, packet.ihl, packet.tos, packet.id, packet.flags, packet.frag, packet.ttl
                    # ))  
                                

            except Exception as e:
                # skip all non tcp/udp packets
                # print("Error in packet {}\nException: {}".format(count, e))
                discard_count += 1
                # if fragmented, skip this packet
                if packet.haslayer(IP) and packet[IP].frag != 0:
                    # print("Fragmented with offset = {}".format(packet[IP].frag))
                    frag_count += 1
                # print(packet.getlayer(IP))
                # print("Error in packet {} w/ proto: {}\nException: {}".format(count, packet.proto, e))
                continue
        
            # if count > 1000:
            #     break

    print(f"\r#pkt scanned and converted: {count}")
    print("Number of discarded packets: {}".format(discard_count))
    print("Number of fragmented packets: {}".format(frag_count))
    return output_path
    

def csv2pcap(input_path, output_path):
    """
    Convert a csv file to a pcap file

    :param input: Pandas dataframe of the csv file
    :param output: Path to the output pcap file
    """
    print("Converting csv file to pcap file...")
    print("Input path (csv file):\n\t{}".format(input_path))
    print("Output path (pcap file):\n\t{}".format(output_path))

    df = pd.read_csv(input_path)
    df = df.sort_values(["time"])

    packets = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        time = float(row["time"] / 10**6)
        if isinstance(row["srcip"], str):
            srcip = IP_str2int(row["srcip"])
            dstip = IP_str2int(row["dstip"])
            src = socket.inet_ntoa(struct.pack('!L', srcip))
            dst = socket.inet_ntoa(struct.pack('!L', dstip))
        else:
            src = socket.inet_ntoa(struct.pack('!L', row["srcip"]))
            dst = socket.inet_ntoa(struct.pack('!L', row["dstip"]))

        srcport = row["srcport"]
        dstport = row["dstport"]
        proto = row["proto"]
        pkt_len = int(row["pkt_len"])

        try:
            proto = int(proto)
        except BaseException:
            raise NotImplementedError("This conversion is deprecated, and will be fixed in the future")
            if proto == "TCP":
                proto = 6
            elif proto == "UDP":
                proto = 17
            elif proto == "ICMP":
                proto = 1
            else:
                proto = 0

        ip = IP(src=src, dst=dst, len=pkt_len, proto=proto)
        if proto == 1:
            p = ip / ICMP()
        elif proto == 6:
            tcp = TCP(sport=srcport, dport=dstport)
            p = ip / tcp
        elif proto == 17:
            udp = UDP(sport=srcport, dport=dstport)
            p = ip / udp
        else:
            p = ip

        p.time = time
        p.len = pkt_len
        p.wirelen = pkt_len + 4

        packets.append(p)

    wrpcap(output_path, packets)

def discard_payload(input_path, output_path):
    """
    Discard the payload of a pcap file
    """
    print("Discarding payload...")
    print("Input path (pcap file):\n\t{}".format(input_path))
    print("Output path (pcap file):\n\t{}".format(output_path))

    packets = []
    count = 0
    for packet in tqdm(PcapReader(input_path)):
        count += 1
        if packet.haslayer(IP):
            if packet.haslayer(TCP):
                packet[IP][TCP].remove_payload()
            elif packet.haslayer(UDP):
                packet[IP][UDP].remove_payload()
            # elif packet.haslayer(ICMP):
            #     packet[IP][ICMP].remove_payload()
            else:
                # skip this packet
                print("Skipping #{} packet of protocol {}".format(count, packet.proto))
                continue 
        packets.append(packet)
        # if count > 1000:
        #     break
    wrpcap(output_path, packets)


def weighted_moving_average(x, window_size, weight_type):
    """
    Compute the weighted moving average of a list of values. This is equivalent to 1d convolution

    Parameter:
    ----------
    x: list of values
    window_size: size of the window
    weight_type: type of the weight function
    """
    kernel = np.ones(window_size)

    # determine the weight / kernel by weight_type 
    if weight_type == "simple":
        pass 
    elif weight_type == "exponential":
        kernel = np.exp(np.linspace(-1., 0., window_size))
    elif weight_type == "bump":
        for i in range(window_size):
            x_i = 1 - 2 * (window_size - i) / (window_size + 1)
            kernel[i] = np.exp(-1/(1-x_i*x_i))
    else:
        raise ValueError("Invalid weight type: {}".format(weight_type))
    kernel = kernel / np.sum(kernel)

    # pad the input with zeros so that it can be inverted
    x = np.pad(x, (0, window_size-1), 'constant', constant_values=(0, 0))

    # compute the weighted moving average
    y = np.convolve(x, kernel[::-1], mode='valid')

    return y 


def rev_weighted_moving_average(y, window_size, weight_type):
    """
    Invert the weighted moving average of a list of values by back-substitution
    """
    kernel = np.ones(window_size)
    if weight_type == "simple":
        pass 
    elif weight_type == "exponential":
        kernel = np.exp(np.linspace(-1., 0., window_size))
    elif weight_type == "bump":
        for i in range(window_size):
            x_i = 1 - 2 * (window_size - i) / (window_size + 1)
            kernel[i] = np.exp(-1/(1-x_i*x_i))
    else:
        raise ValueError("Invalid weight type: {}".format(weight_type))
    kernel = kernel / np.sum(kernel)

    # pad x
    x = np.zeros(len(y) + window_size - 1)

    for i in range(len(y)-1, -1, -1):
        # x[i] = 3*y[i] - x[i+1] - x[i+2]
        x[i] = 1/kernel[0] * (y[i] - x[i+1: i+window_size].dot(kernel[1:]))
        # print("i={}, x[i]={:.2}, y[i]={:.2}, x[i+1]={:.2}, x[i+2]={:.2}".format(i, x[i], y[i], x[i+1], x[i+2]))
        

    return x[:len(y)]

def prepare_packetinfo(fivetuple, packetrate, use_time_interval):
    # fivetuple:  num_flows x fivetuple_dim
    # packetrate: num_flows x length

    # convert packetrate to int
    packetrate = packetrate.astype(np.int32)

    packet_fivetuple = []
    for i in range(packetrate.shape[0]):
        for j in range(packetrate.shape[1]):
            if packetrate[i][j] > 0:
                # print("packetrate[{}][{}] = {}".format(i, j, packetrate[i][j]))
                packet_fivetuple.append(np.tile(fivetuple[i], (packetrate[i][j], 1)))
    packet_fivetuple = np.concatenate(packet_fivetuple, axis=0)
    # packet_fivetuple: num_packets x fivetuple_dim
    if not use_time_interval:
        return packet_fivetuple
    
    time_last_interval = []
    time_next_interval = []
    for i in range(packetrate.shape[0]):
        singleflow_interval = []
        current_interval = 0
        for j in range(packetrate.shape[1]):
            if packetrate[i][j] > 0:
                # FIXME:
                singleflow_interval.append(np.ones((packetrate[i][j])) / packetrate[i][j])
                singleflow_interval[-1][0] += current_interval
                current_interval = singleflow_interval[-1][-1]
            else:
                current_interval += 1
        if len(singleflow_interval) > 0:
            singleflow_interval = np.concatenate(singleflow_interval)[1:]
            time_last_interval.append(np.concatenate([np.zeros(1), singleflow_interval]))
            time_next_interval.append(np.concatenate([singleflow_interval, np.zeros(1)]))
    time_last_interval = np.concatenate(time_last_interval)
    time_next_interval = np.concatenate(time_next_interval)
    packetinfo = np.concatenate([
        packet_fivetuple,
        time_last_interval[:, np.newaxis],
        time_next_interval[:, np.newaxis],
    ], axis=1)
    # packetinfo: num_packets x (fivetuple_dim + 2)
    return packetinfo


def load_flow_from_df(
        df, 
        mode=("time_point_count", 200), flow_tuple=stats.five_tuple,
        flowsize_range=None, flowDurationRatio_range=None, 
        nonzeroIntervalCount_range=None, maxPacketrate_range=None,
        return_all=False,
        ):
    df["time"] = df["time"] - df["time"].min()
    df = df.sort_values(by="time")
    dfg = df.groupby(flow_tuple)
    gks = list(dfg.groups.keys())
    num_flow_before_filtering = len(gks)
    print("Number of flows before filtering: {}".format(num_flow_before_filtering))
    
    if flowsize_range is not None:
        if isinstance(flowsize_range, (float, np.floating)):
            # keep only top percentages
            print("Filtering flows with top {}% flow size".format(flowsize_range * 100))
            flowsizes = {
                gk: len(dfg.get_group(gk))
                for gk in gks
            }
            # sort by flow sizes, keeping only the top percent of flows
            sorted_flowsizes = sorted(flowsizes.items(), key=lambda x: x[1], reverse=True)
            gks = [gk for gk, _ in sorted_flowsizes[:int(len(sorted_flowsizes) * flowsize_range)]]
        else:
            if len(flowsize_range) != 2:
                raise ValueError("flowsize_range must be a tuple/list/numpy array of (min_flowsize, max_flowsize)")
            print("Filtering flows with number of packets in [{}, {})".format(flowsize_range[0], flowsize_range[1]))
            flowsizes = {
                gk: len(dfg.get_group(gk))
                for gk in gks
            }
            old_gks = copy.deepcopy(gks)
            gks = [gk for gk in old_gks if flowsize_range[0] <= flowsizes[gk] < flowsize_range[1]]

    if flowDurationRatio_range is not None:
        if isinstance(flowDurationRatio_range, (float, np.floating)):
            # keep only top percentages
            print("Filtering flows with top {}% flow duration".format(flowDurationRatio_range * 100))
            flow_durations = {
                gk: dfg.get_group(gk)["time"].max() - dfg.get_group(gk)["time"].min()
                for gk in gks
            }
            # sort by flow sizes, keeping only the top percent of flows
            sorted_flow_durations = sorted(flow_durations.items(), key=lambda x: x[1], reverse=True)
            gks = [gk for gk, _ in sorted_flow_durations[:int(len(sorted_flow_durations) * flowDurationRatio_range)]]
        else:
            if len(flowDurationRatio_range) != 2:
                raise ValueError("flowduration_range must be a tuple/list/numpy array of (min_flowduration, max_flowduration)")
            if flowDurationRatio_range[0] < 0 or flowDurationRatio_range[1] > 1:
                raise ValueError("flowduration_range must be in [0, 1]")
            flowDurationRatio_range = list(flowDurationRatio_range)
            print("Filtering flows with flow duration ratio in [{}, {})".format(flowDurationRatio_range[0], flowDurationRatio_range[1])) 
            if flowDurationRatio_range[1] == 1:
                flowDurationRatio_range[1] = np.inf
            flow_durations = {
                gk: dfg.get_group(gk)["time"].max() - dfg.get_group(gk)["time"].min()
                for gk in gks
            }
            total_duration = df["time"].max() - df["time"].min()
            flowDuration_range = (flowDurationRatio_range[0] * total_duration, flowDurationRatio_range[1] * total_duration)
            old_gks = copy.deepcopy(gks)
            gks = [gk for gk in old_gks if flowDuration_range[0] <= flow_durations[gk] < flowDuration_range[1]]

    if mode[0] == "time_unit_exp":
        time_unit_exp = mode[1]
        print("Using time_unit_exp = {}".format(time_unit_exp))
    elif mode[0] == "time_point_count":
        time_point_count = mode[1]
        total_duration = df["time"].max() - df["time"].min()
        time_unit_exp = stats.time_point_count2time_unit_exp(total_duration, time_point_count)
        # time_unit_exp = math.log10(total_duration / time_point_count) 
        print("Using time point count = {}, corespoinding time_unit_exp = {}".format(time_point_count, time_unit_exp))
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    flows = stats.compute_od_flows(
        dfg, gks,
        total_duration = df["time"].max() - df["time"].min(),
        time_unit_exp=time_unit_exp)
    
    if nonzeroIntervalCount_range is not None:
        """
        Count the number of nonzero intervals in each flow
        """
        if len(nonzeroIntervalCount_range) != 2:
            raise ValueError("nonzeroInterval_range must be a tuple/list/numpy array of (min_nonzeroInterval, max_nonzeroInterval)")
        nonzeroIntervalCount_range = list(nonzeroIntervalCount_range)
        print("Filtering flows with #nonzero interval in [{}, {})".format(nonzeroIntervalCount_range[0], nonzeroIntervalCount_range[1])) 
        nonzeroCount = np.count_nonzero(flows, axis=0)
        indices = np.where(np.logical_and(nonzeroIntervalCount_range[0] <= nonzeroCount, nonzeroCount < nonzeroIntervalCount_range[1]))[0]
        gks = [gks[i] for i in indices]
        flows = flows[:, indices]
    
    if maxPacketrate_range is not None:
        if len(maxPacketrate_range) != 2:
            raise ValueError("maxPacketrate_range must be a tuple/list/numpy array of (min_maxPacketrate, max_maxPacketrate)")
        maxPacketrate_range = list(maxPacketrate_range)
        print("Filtering flows with max packet rate in [{}, {})".format(maxPacketrate_range[0], maxPacketrate_range[1]))
        maxPacketrate = np.max(flows, axis=0)
        indices = np.where(np.logical_and(maxPacketrate_range[0] <= maxPacketrate, maxPacketrate < maxPacketrate_range[1]))[0]
        gks = [gks[i] for i in indices]
        flows = flows[:, indices]
    
    num_flow_after_filtering = len(gks)
    print("Number of flows after filtering: {}, {}% of the original flows".format(
        num_flow_after_filtering, num_flow_after_filtering / num_flow_before_filtering * 100))

    if len(gks) == 0:
        warnings.warn("No flow left after filtering")

    if return_all: 
        return dfg, gks, flows
    return flows   

def load_flow_from_csv(
        path, mode=("time_point_count", 200), flow_tuple=stats.five_tuple,
        flowsize_range=None, flowDurationRatio_range=None, 
        nonzeroIntervalCount_range=None, maxPacketrate_range = None,
        read_csv_kwargs={},
        ):
    """
    Convert a csv file to a numpy array of flows
    csv => df => df.groupby => numpy array of packet rate
    
    Parameters:
    -----------
    csv_path: path to the csv file (converted from pcap file by nta.utils.data.pcap2csv)
    **load_csv_kwargs: arguments for nta.utils.data.load_csv 
    # filter flows with number of packets in [min_flowsize, max_flowsize)
    "flowsize_range": (min_flowsize, max_flowsize),
    # filter flows with flow duration ratio in [min_flowduration, max_flowduration)
    "flowDurationRatio_range": (min_flowDurationRatio, max_flowDurationRatio),
    # filter flows with #nonzero interval in [min_nonzeroIntervalCount, max_nonzeroIntervalCount)
    "nonzeroIntervalCount_range": (min_nonzeroIntervalCount, max_nonzeroIntervalCount),
    # filter flows with max packet rate in [min_maxPacketrate, max_maxPacketrate)
    "maxPacketrate_range": (min_maxPacketrate, max_maxPacketrate),

    Returns:
    --------
    flows: a numpy array of size (n_interval, n_flow)

    """
    df = load_csv(
        path, 
        **read_csv_kwargs, 
        )
    df["time"] = df["time"] - df["time"].min()
    df = df.sort_values(by="time")
    dfg = df.groupby(flow_tuple)
    gks = dfg.groups.keys()
    num_flow_before_filtering = len(gks)
    print("Number of flows before filtering: {}".format(num_flow_before_filtering))

    if flowsize_range is not None:
        if len(flowsize_range) != 2:
            raise ValueError("flowsize_range must be a tuple/list/numpy array of (min_flowsize, max_flowsize)")
        print("Filtering flows with number of packets in [{}, {})".format(flowsize_range[0], flowsize_range[1]))
        flowsizes = dfg.size()
        gks = [gk for gk in dfg.groups.keys() if flowsize_range[0] <= flowsizes[gk] < flowsize_range[1]]

    if flowDurationRatio_range is not None:
        if len(flowDurationRatio_range) != 2:
            raise ValueError("flowduration_range must be a tuple/list/numpy array of (min_flowduration, max_flowduration)")
        if flowDurationRatio_range[0] < 0 or flowDurationRatio_range[1] > 1:
            raise ValueError("flowduration_range must be in [0, 1]")
        flowDurationRatio_range = list(flowDurationRatio_range)
        if flowDurationRatio_range[1] == 1:
            flowDurationRatio_range[1] = np.inf
        print("Filtering flows with flow duration ratio in [{}, {})".format(flowDurationRatio_range[0], flowDurationRatio_range[1])) 
        flowdurations = dfg["time"].max() - dfg["time"].min()
        total_duration = df["time"].max() - df["time"].min()
        flowDuration_range = (flowDurationRatio_range[0] * total_duration, flowDurationRatio_range[1] * total_duration)
        gks = [gk for gk in dfg.groups.keys() if flowDuration_range[0] <= flowdurations[gk] < flowDuration_range[1]]

    if mode[0] == "time_unit_exp":
        time_unit_exp = mode[1]
        print("Using time_unit_exp = {}".format(time_unit_exp))
    elif mode[0] == "time_point_count":
        time_point_count = mode[1]
        total_duration = df["time"].max() - df["time"].min()
        time_unit_exp = stats.time_point_count2time_unit_exp(total_duration, time_point_count)
        # time_unit_exp = math.log10(total_duration / time_point_count) 
        print("Using time point count = {}, corespoinding time_unit_exp = {}".format(time_point_count, time_unit_exp))
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    flows = stats.compute_od_flows(
        dfg, gks, 
        total_duration = df["time"].max() - df["time"].min(),
        time_unit_exp=time_unit_exp)
    
    if nonzeroIntervalCount_range is not None:
        """
        Count the number of nonzero intervals in each flow
        """
        if len(nonzeroIntervalCount_range) != 2:
            raise ValueError("nonzeroInterval_range must be a tuple/list/numpy array of (min_nonzeroInterval, max_nonzeroInterval)")
        nonzeroIntervalCount_range = list(nonzeroIntervalCount_range)
        print("Filtering flows with #nonzero interval in [{}, {})".format(nonzeroIntervalCount_range[0], nonzeroIntervalCount_range[1])) 
        nonzeroCount = np.count_nonzero(flows, axis=0)
        indices = np.where(np.logical_and(nonzeroIntervalCount_range[0] <= nonzeroCount, nonzeroCount < nonzeroIntervalCount_range[1]))[0]
        flows = flows[:, indices]
    
    if maxPacketrate_range is not None:
        if len(maxPacketrate_range) != 2:
            raise ValueError("maxPacketrate_range must be a tuple/list/numpy array of (min_maxPacketrate, max_maxPacketrate)")
        maxPacketrate_range = list(maxPacketrate_range)
        print("Filtering flows with max packet rate in [{}, {})".format(maxPacketrate_range[0], maxPacketrate_range[1]))
        maxPacketrate = np.max(flows, axis=0)
        indices = np.where(np.logical_and(maxPacketrate_range[0] <= maxPacketrate, maxPacketrate < maxPacketrate_range[1]))[0]
        flows = flows[:, indices]
    
    num_flow_after_filtering = flows.shape[1]
    print("Number of flows after filtering: {}, {}% of the original flows".format(
        num_flow_after_filtering, num_flow_after_filtering / num_flow_before_filtering * 100))

    return flows         
        

def load_flow_from_npz(
        path, 
        zero_threshold=0,
        time_point_count=200,
        flowsize_range=None, flowDurationRatio_range=None, 
        nonzeroIntervalCount_range=None, maxPacketrate_range = None,
    ):
    f = open(path)
    flows = np.load(f)["packetrate"].T + zero_threshold
    f.close()

    flows = flows.astype(int)
    # check if we should transpose the flows
    if flows.shape[0] != time_point_count and flows.shape[1] == time_point_count:
        flows = flows.T

    num_flow_before_filtering = flows.shape[1]
    print("Number of flows before filtering: {}".format(num_flow_before_filtering))

    # keep only flows within flowsize_range (i.e. the sum of packetrate is within flowsize_range)
    print(flows.shape)
    if flowsize_range is not None:
        if len(flowsize_range) != 2:
            raise ValueError("flowsize_range must be a tuple/list/numpy array of (min_flowsize, max_flowsize)")
        print("Filtering flows with number of packets in [{}, {})".format(flowsize_range[0], flowsize_range[1]))
        s = np.sum(flows, axis=0)
        flows = flows[:, (s >= flowsize_range[0]) & (s < flowsize_range[1])]
        
    if flowDurationRatio_range is not None:
        # given a vector of packet rate, we find the first and last nonzero index, and compute the duration as the difference of the two indices
        if len(flowDurationRatio_range) != 2:
            raise ValueError("flowduration_range must be a tuple/list/numpy array of (min_flowduration, max_flowduration)")
        if flowDurationRatio_range[0] < 0 or flowDurationRatio_range[1] > 1:
            raise ValueError("flowduration_range must be in [0, 1]")
        flowDurationRatio_range = list(flowDurationRatio_range)
        print("Filtering flows with flow duration ratio in [{}, {})".format(flowDurationRatio_range[0], flowDurationRatio_range[1]))
        if flowDurationRatio_range[1] == 1:
            flowDurationRatio_range[1] = np.inf
        flowdurations = []
        for i in range(flows.shape[1]):
            nonzero_idx = np.nonzero(flows[:, i])[0]
            if len(nonzero_idx) == 0:
                flowdurations.append(0)
            else:
                # +1 in case the there is only one nonzero index
                flowdurations.append(nonzero_idx[-1] - nonzero_idx[0] + 1)
        flowdurations = np.array(flowdurations)
        total_duration = flows.shape[0]
        flowDuration_range = (flowDurationRatio_range[0] * total_duration, flowDurationRatio_range[1] * total_duration)
        flows = flows[:, (flowDuration_range[0] <= flowdurations) & (flowdurations < flowDuration_range[1])]

    if nonzeroIntervalCount_range is not None:
        """
        Count the number of nonzero intervals in each flow
        """
        if len(nonzeroIntervalCount_range) != 2:
            raise ValueError("nonzeroInterval_range must be a tuple/list/numpy array of (min_nonzeroInterval, max_nonzeroInterval)")
        nonzeroIntervalCount_range = list(nonzeroIntervalCount_range)
        print("Filtering flows with #nonzero interval in [{}, {})".format(nonzeroIntervalCount_range[0], nonzeroIntervalCount_range[1])) 
        nonzeroCount = np.count_nonzero(flows, axis=0)
        indices = np.where(np.logical_and(nonzeroIntervalCount_range[0] <= nonzeroCount, nonzeroCount < nonzeroIntervalCount_range[1]))[0]
        flows = flows[:, indices]

    if maxPacketrate_range is not None:
        if len(maxPacketrate_range) != 2:
            raise ValueError("maxPacketrate_range must be a tuple/list/numpy array of (min_maxPacketrate, max_maxPacketrate)")
        maxPacketrate_range = list(maxPacketrate_range)
        print("Filtering flows with max packet rate in [{}, {})".format(maxPacketrate_range[0], maxPacketrate_range[1]))
        maxPacketrate = np.max(flows, axis=0)
        indices = np.where(np.logical_and(maxPacketrate_range[0] <= maxPacketrate, maxPacketrate < maxPacketrate_range[1]))[0]
        flows = flows[:, indices]

    num_flow_after_filtering = flows.shape[1]
    print("Number of flows after filtering: {}, {}% of the original flows".format(
        num_flow_after_filtering, num_flow_after_filtering / num_flow_before_filtering * 100))
    return flows
 

# to study the distribution of each condition



def flow_downsample(flows, downsample_to=10):
    """
    Downsample a time series to a specific size `downsample_to`

    e.g. Given `flow` of shape (235, 1), we want to downsample it to `flow_ds` of shape (50, 1)
    agg_num = 235 / 50 = 4.7, ceil(4.7) = 5 ?????? TODO:
    flow_ds[0] = np.sum(flow[0*agg_num: 1*agg_num])
    flow_ds[i] = np.sum(flow[i*agg_num: (i+1)*agg_num])
    ...
    flow_ds[49] = np.sum(flow[49 * agg_num:])

    Parameters:
    -----------
    flows: a numpy array of size (n_interval, n_flow)
    downsample_to: the target size of the time series

    Returns:
    --------
    flows: a numpy array of size (downsample_to, n_flow)
    """
    raise NotImplementedError("This method is not implemented yet")
    if downsample_to >= flows.shape[0]:
        warnings.warn("downsample_to >= flows.shape[0], no downsampling is performed")
        return flows
    pass 


def flow2median_and_span(dfg, gks, flows, num_flow, num_t, interval_length):
    medians = np.zeros((num_flow, num_t))
    spans = np.zeros((num_flow, num_t))
    medians_normalizations = (nn.Sigmoid(), 1)
    spans_normalizations = (nn.Sigmoid(), 1)

    for i, gk in tqdm(enumerate(gks)):
        single_flow = dfg.get_group(gk)
        single_flow = single_flow.sort_values(by="time")
        single_flow = single_flow.reset_index(drop=True)
        single_flow = single_flow["time"].values

        single_flow_index = 0
        for j in np.nonzero(flows[i, :])[0]:
            # get packet rate value
            num_packets = int(flows[i, j])
            # get interval
            intervals = single_flow[single_flow_index: single_flow_index+num_packets]

            sample_interval = (j * interval_length, (j+1) * interval_length)
            medians[i, j] = ((intervals.max() + intervals.min()) / 2 - sample_interval[0]) / interval_length
            spans[i, j] = (intervals.max() - intervals.min()) / 2 / interval_length
            single_flow_index += num_packets

    medians = medians[:, :, np.newaxis]
    spans = spans[:, :, np.newaxis]
    return {
        "medians": {
            "feature": medians,
            "normalization": medians_normalizations
        },
        "spans": {
            "feature": spans,
            "normalization": spans_normalizations
        }
    }


def packetrate2timestamp(packetrate, interval_duration=1, mode="equal", params=None):
    """
    Convert an array of packet rate into a sequence of interarrival time. The conversion method is determined by `mode`
    - mode == "equidistant": interarrival time is equally divided by the number of packets in the interval
    - mode == "resample_flow_gmm": fit a Gaussian mixture model to the interarrival time and sample from the model
    - mode == "resample_flow_hist"
    - mode == "resample_trace": resample the flow-level interarrival time from the original trace if the reconstructed
        interarrival is less than interval_duration
    - mode == "resample_trace_refined"
    - mode == "resample_trace_hist"
    - mode == "resample_actual_interval_duration_ratio":
        estimate the portion of interval that actually has packets sending.
        e.g. an interval of 2s, only 0.3~0.4s has 5 packets, then the actual duration is 0.1s
            This is stored in params["actual_interval_duration_ratio"] as 0.1 / 2 = 0.05 \in [0, 1]
            To reconstruct packet interarrival, we randomly select a start time in interval (e.g. 0.6s), then equally 
            divide actual_duration, which gives timestamps [0.6, 0.62, 0.64, 0.66, 0.68]
    Note that timestamp is assumed to be represented in unix timestamp, which is an integer
    Parameters:
    -----------
    packetrate: a numpy array of size (n_interval, n_flow)
    interval_duration: the duration of each interval
    mode: the conversion method
    params:
    - sample_num: the number of samples to draw from the GMM
    - sample_method: the method to convert samples to reconstructed interarrival. 
        - "first"
            take the first sample below interval duration as the reconstructed interarrival
        - "min"
            take the minimum sample below interval duration as the reconstructed interarrival
        - "mean"
            take the mean of the samples below interval duration as the reconstructed interarrival

    Returns:
    --------
    interarrival: a list of numpy arrays where each array is a sequence of interarrival started with flow_start
    e.g. [
        [0.333, 0.333, 0.333],              # start with time 0.333
        [99, 0.2, 0.2, 0.2, 0.2, 0.2],      # start with time 99
        ...
    ]
    """
    # packetrate.shape = (n_interval, n_flow)
    packetrate = packetrate.astype(int)
    print("packetrate2timestamp mode: {}".format(mode))
    if mode == "equidistant":
        timestamp = []
        for flow_index in range(packetrate.shape[1]):
            singleflow_packetrate = packetrate[:, flow_index]
            singleflow_timestamp = []

            for interval_index in np.nonzero(singleflow_packetrate)[0]:
                num_packets = singleflow_packetrate[interval_index]
                # compute the actual interval normalized to 1 duration
                interval = np.array([interval_index, interval_index + 1])
                # equally divide the interval
                t = np.linspace(interval[0], interval[1], num_packets+2)
                t = t[1:-1]
                singleflow_timestamp.extend(t)
            timestamp.append(singleflow_timestamp)
        return timestamp
        # interarrival = []
        # for flow_index in range(packetrate.shape[1]):
        #     singleflow_packetrate = packetrate[:, flow_index]
        #     nonzero_packetrate_interval = np.nonzero(singleflow_packetrate)[0]
        #     nonzero_packetrate = singleflow_packetrate[nonzero_packetrate_interval]
        #     # given a nonzero_packetrate n and a nonzero_packetrate_interval k, 
        #     # the result is np.ones(n) / n + k * interval_duration
        #     singleflow_interarrival = []
        #     for n, k in zip(nonzero_packetrate, nonzero_packetrate_interval):
        #         t = np.ones((n)) / n * interval_duration
        #         t[0] += k * interval_duration
        #         t = np.cumsum(t)
        #         singleflow_interarrival.append(t)
        #     if len(singleflow_interarrival) > 0:
        #         singleflow_interarrival = np.concatenate(singleflow_interarrival)
        #         singleflow_interarrival = np.sort(singleflow_interarrival)
        #         singleflow_interarrival[1:] = np.diff(np.sort(singleflow_interarrival))

        #     else:
        #         warnings.warn("#{} flow has no packet!".format(flow_index))
        #     breakpoint()
        #     interarrival.append(singleflow_interarrival)      
        # return interarrival
    elif mode == "median_and_span":
        # The recovered interval is 
        #   [median - span, median + span] * interval_duration + interval_start, 
        #   truncated to [interval_start, intervala_end]
        medians = params["medians"]
        spans = params["spans"]
        timestamp = []
        for flow_index in tqdm(range(packetrate.shape[1])):
            singleflow_packetrate = packetrate[:, flow_index]
            singleflow_timestamp = []

            for interval_index in np.nonzero(singleflow_packetrate)[0]:
                num_packets = singleflow_packetrate[interval_index]
                median = medians[flow_index, interval_index]
                span = spans[flow_index, interval_index]
                # compute the actual interval normalized to 1 duration
                interval = np.array([median - span, median + span])
                interval = np.clip(interval, 0, 1)
                # compute the actual interval with real duration
                interval = (interval + interval_index) * interval_duration
                # equally divide the interval
                t = np.linspace(interval[0], interval[1], num_packets)
                singleflow_timestamp.extend(t)
            timestamp.append(singleflow_timestamp)
        return timestamp

    elif mode == "timestamp_in_unit":
        timestamp_in_unit = params["timestamp_in_unit"]
        timestamp = []
        packet_index = 0
        for flow_index in tqdm(range(packetrate.shape[1])):
            singleflow_packetrate = packetrate[:, flow_index]
            singleflow_timestamp = []

            for interval_index in np.nonzero(singleflow_packetrate)[0]:
                num_packets = singleflow_packetrate[interval_index]
                t = timestamp_in_unit[packet_index:packet_index+num_packets] + interval_index
                singleflow_timestamp.extend(t)
                packet_index += num_packets
            timestamp.append(singleflow_timestamp)
        return timestamp

    elif mode == "resample_flow_gmm":
        if "gmm_params" not in params:
            raise ValueError("gmm_params not found in params for mode: '{}'".format(mode))
        interarrival = []
        for flow_index in range(packetrate.shape[1]):
            # skip flows with only one packet
            singleflow_packetrate = packetrate[:, flow_index]
            if np.sum(singleflow_packetrate) <= 1:
                continue
            nonzero_packetrate_interval = np.nonzero(singleflow_packetrate)[0]
            nonzero_packetrate = singleflow_packetrate[nonzero_packetrate_interval]
            # given a nonzero_packetrate n and a nonzero_packetrate_interval k, 
            # the result is np.ones(n) / n + k * interval_duration
            singleflow_interarrival = []
            for n, k in zip(nonzero_packetrate, nonzero_packetrate_interval):
                t = np.ones((n)) / n * interval_duration
                t[0] += k * interval_duration
                t = np.cumsum(t)
                singleflow_interarrival.append(t)
            if len(nonzero_packetrate_interval) > 3:
                print("", end="")
            if len(singleflow_interarrival) > 0:
                singleflow_interarrival = np.concatenate(singleflow_interarrival)
                singleflow_interarrival = np.sort(singleflow_interarrival)
                singleflow_interarrival[1:] = np.diff(np.sort(singleflow_interarrival))
            else:
                warnings.warn("#{} flow has no packet!".format(flow_index))
            interarrival.append(singleflow_interarrival)      
        


        gmm_params = params["gmm_params"]
        raw_interarrival = params["raw_interarrival"]
        sample_num = params["sample_num"]
        sample_method = params["sample_method"]

        print("gmm_params:")
        # print(gmm_params)
        print("\tgmm_weight.shape: {}, gmm_means.shape: {}, gmm_covariances.shape: {}".format(
            gmm_params[0][0].shape, gmm_params[0][1].shape, gmm_params[0][2].shape))
        print("sample_num: {}, sample_method: {}".format(sample_num, sample_method))

        print("Sampling interarrival from gmm...")
        for interval_index, t in tqdm(enumerate(interarrival)):
            t = t[1:]
            # skip flows with only one packet
            if len(t) == 0:
                continue
            # we use 4 n_component to fit gmm by default
            # find all indices whose value is <= interval_duration
            to_be_resample = np.nonzero(t <= interval_duration)[0]
            # if no interarrival is <= interval_duration, skip this flow
            if len(to_be_resample) == 0:
                continue 
            gmm_weights, gmm_means, gmm_covariances = gmm_params[interval_index]
            if gmm_weights is None:
                # skip this one
                continue 
            gmm_precision = np.linalg.inv(gmm_covariances)
            # initialize gmm w/ the given parameters, and sample from it
            gmm = GaussianMixture(
                n_components=len(gmm_weights), covariance_type='full',
                weights_init=gmm_weights,
                means_init=gmm_means,
                precisions_init=gmm_precision,)
            # bypass fitting the gmm
            gmm._set_parameters((
                gmm_weights, 
                gmm_means, 
                gmm_covariances,
                gmm_precision,))
            gmm.converged_ = True
            for k in to_be_resample:
                # cnt = 0
                # while True:
                #     # keep sampling until a sample <= t[k] is found.
                #     # This is because the raw interarrival can't be greater than t[k]
                #     r = gmm.sample()[0][0]
                #     r = np.exp(r)
                #     if r <= t[k] or cnt > 10:
                #         if cnt > 10:
                #             print("cnt > 10, t[k] = t[{}] = {}".format(
                #                 k, t[k]))
                #         break 
                #     cnt += 1

                r = gmm.sample(sample_num)[0]
                r = np.exp(r) - eps
                # r = np.expm1(r)
                # r = np.exp(np.exp(r) - 1e-12) - 1e-12
                # r = r[r <= t[k]]
                # in case the raw interarrival is slightly above t[k], we set 1.1*t[k] to be the threshold
                # r = r[np.logical_and(1e-6 <= r, r <= t[k])]
                r = r[np.logical_and(1e-8 <= r, r <= t[k])]
                if len(r) == 0:
                    continue 
                    # no sample <= t[k] is found, keep the 'equal' interarrival
                    # print("No sample <= t[k] is found, t[k] = t[{}] = {}".format(
                    #     k, t[k]))
                    # print("raw interarrival = {}".format(raw_interarrival[i]))
                    # print("rec interarrival = {}".format(t))
                    # print()
                    # continue
                    # # plot the pdf of the gmm
                    # fig, ax = plt.subplots(1, 1)
                    # # plot raw interarrival distribution
                    # vis.plot_loghist(
                    #     # add eps in case raw_interarrival is all 0
                    #     x = raw_interarrival[i] + eps,
                    #     ax = ax,
                    #     bins_count = 100,
                    #     bins_min = 1e-8,
                    #     bins_max = 1e0,
                    #     truncate_x = True,
                    #     density = True,
                    #     alpha=0.5,
                    #     label="raw"
                    # )
                    # # plot the pdf of the fitted GMM model
                    # gmm_sample = gmm.sample(len(t))[0]
                    # gmm_sample = np.exp(gmm_sample)
                    # vis.plot_loghist(
                    #     x = gmm_sample,
                    #     ax = ax,
                    #     bins_count = 100,
                    #     bins_min = 1e-8,
                    #     bins_max = 1e0,
                    #     truncate_x = True,
                    #     density = True,
                    #     alpha=0.5,
                    #     label="resample_flow_gmm"
                    # )
                    # ax.legend()
                    # plt.show()
                    # # break
                    # continue
                if sample_method == "first":
                    # take the first sample
                    r = r[0]
                elif sample_method == "min":
                    # take the eman
                    r = np.min(r)
                elif sample_method == "mean":
                    # take the mean of exponential
                    r = np.log(np.mean(np.exp(r)) + 1e-12)
                else:
                    raise ValueError("Invalid sample_method: {}".format(sample_method))
                # r = np.log1p(np.mean(np.expm1(r)))
                # r = np.max([r[0]/10, 1e-6])
                t[k] = r
                
                # if relative error > 50%, print sth
                rel_err = np.abs(raw_interarrival[interval_index][k] - t[k]) / (raw_interarrival[interval_index][k] + eps)
                rel_err_threshold = 10
                if rel_err > rel_err_threshold:
                    continue
                    # print("relative error = {:.2f}% > {}%\n\tt[k] = t[{}] = {}\n\traw[i][k] = raw[{}][{}] = {}".format(
                    #     rel_err * 100,
                    #     rel_err_threshold * 100,
                    #     k, t[k],
                    #     i, k, raw_interarrival[i][k]))
                    # print("#interarrival = {}".format(
                    #     len(raw_interarrival[i])
                    # ))
                    # continue
                    # # print("raw interarrival = {}".format(raw_interarrival[i]))
                    # # print("rec interarrival = {}".format(t))
                    #                     # plot the pdf of the gmm
                    # fig, ax = plt.subplots(1, 1)
                    # # plot raw interarrival distribution
                    # vis.plot_loghist(
                    #     # add eps in case raw_interarrival is all 0
                    #     x = raw_interarrival[i] + eps,
                    #     ax = ax,
                    #     bins_count = 9,
                    #     bins_min = 1e-8,
                    #     bins_max = 1e0,
                    #     truncate_x = True,
                    #     density = True,
                    #     alpha=0.5,
                    #     label="raw"
                    # )
                    # # plot the pdf of the fitted GMM model
                    # gmm_sample = gmm.sample(len(t))[0]
                    # gmm_sample = np.exp(gmm_sample)
                    # vis.plot_loghist(
                    #     x = gmm_sample,
                    #     ax = ax,
                    #     bins_count = 9,
                    #     bins_min = 1e-8,
                    #     bins_max = 1e0,
                    #     truncate_x = True,
                    #     density = True,
                    #     alpha=0.5,
                    #     label="resample_flow_gmm"
                    # )
                    # ax.legend()
                    # ax.set_yscale("log")
                    # plt.show()
                    # # user_input = input("Continue? (y/n)")
                    # # if user_input == "n":
                    # #     exit()
                    # break
                    
            # t[to_be_resample] = np.random.choice(raw_interarrival_concat, size=np.sum(to_be_resample))
            interarrival[interval_index][1:] = t      
        return interarrival  
        
    elif mode == "resample_flow_hist":
        """
        For each flow, record its log-log histogram of interarrival time
        e.g. assign a list of 10 number to each flow representing the log density of 
            bin of range (0, 1e-7), (1e-7, 1e-6), ..., (1e-2, 1e-1), (1e-1, 1e0)
        
        Note that, for better reconstruction accuracy
        - we should use only compute histogram of interarrival time <= interval_duration
        - we shouldn't use np.linspace or np.logspace to compute the bins. Instead, the bins_edges
            should be based on the density of raw flow-level interarrival time
        """
        if "hist_params" not in params:
            raise ValueError("hist_params not found in params for mode: '{}'".format(mode))
        interarrival = []
        for flow_index in range(packetrate.shape[1]):
            # skip flows with only one packet
            singleflow_packetrate = packetrate[:, flow_index]
            if np.sum(singleflow_packetrate) <= 1:
                continue
            nonzero_packetrate_interval = np.nonzero(singleflow_packetrate)[0]
            nonzero_packetrate = singleflow_packetrate[nonzero_packetrate_interval]
            # given a nonzero_packetrate n and a nonzero_packetrate_interval k, 
            # the result is np.ones(n) / n + k * interval_duration
            singleflow_interarrival = []
            for n, k in zip(nonzero_packetrate, nonzero_packetrate_interval):
                t = np.ones((n)) / n * interval_duration
                t[0] += k * interval_duration
                t = np.cumsum(t)
                singleflow_interarrival.append(t)
            if len(nonzero_packetrate_interval) > 3:
                print("", end="")
            if len(singleflow_interarrival) > 0:
                singleflow_interarrival = np.concatenate(singleflow_interarrival)
                singleflow_interarrival = np.sort(singleflow_interarrival)
                singleflow_interarrival[1:] = np.diff(np.sort(singleflow_interarrival))
            else:
                warnings.warn("#{} flow has no packet!".format(flow_index))
            interarrival.append(singleflow_interarrival)      
        

        hist_params = params["hist_params"]
        # bins_min_exp = float(hist_params["bins_min_exp"])
        # bins_max_exp = float(hist_params["bins_max_exp"])
        bins_edges = hist_params["bins_edges"]
        bins_count = int(hist_params["bins_count"])
        hists = hist_params["hists"]
        if "true_interarrival" in params:
            true_interarrival = params["true_interarrival"]
        sample_num = params["sample_num"]
        sample_method = params["sample_method"]
        print("hist_params:")
        print("\tbins_min_exp: {}, bins_max_exp: {}, bins_count: {}".format(
            get_sci_exp(np.min(bins_edges[bins_edges > 0]).astype(float)),
            get_sci_exp(np.max(bins_edges[bins_edges > 0]).astype(float)),
            bins_count,))
        print("sample_num: {}, sample_method: {}".format(
            sample_num, sample_method))

        print("Sampling interarrival from histogram...")
        # large_bin_indices = np.nonzero(bins_edges[1:] > interval_duration)[0] - 1

        # for i, t in tqdm(enumerate(interarrival)):
        #     t = t[1:]
        #     # skip flows with only one packet
        #     if len(t) == 0:
        #         continue
        #     # find all indices whose value is <= interval_duration
        #     to_be_resample = np.nonzero(t <= interval_duration)[0]
        #     # if no interarrival is <= interval_duration, skip this flow
        #     if len(to_be_resample) == 0:
        #         continue 

        #     # read hist params
        #     hist = hists[i]
        #     # we set the value of bins whose range is larger than interval_duration to 0,
        #     # and then normalize the hist
        #     hist[large_bin_indices] = 0
        #     hist = hist / np.sum(hist)
            

        #     for k in tqdm(to_be_resample, leave=False):
        #         r = np.random.choice(
        #             a = bins_edges[1:],
        #             size = sample_num,
        #             p = hist,)

        #         r = r[r <= t[k]]

        #         # if no sample <= t[k] is found, keep the 'equal' interarrival
        #         if len(r) == 0:
        #             continue 
                
        #         if sample_method == "first":
        #             # take the first sample
        #             r = r[0]
        #             # add a small perturbation at r's exponent
        #             r = 10 ** (np.log10(r) + np.random.uniform(-0.1, 0.1))
        #         elif sample_method == "min":
        #             # take the mean of exponential
        #             r = np.min(r)
        #             r = 10 ** (np.log10(r) + np.random.uniform(-0.1, 0.1))
        #         elif sample_method == "mean":
        #             # take the mean of exponent
        #             r = 10 ** (np.mean(np.log10(r)) + np.random.uniform(-0.1, 0.1))
        #         else:
        #             raise ValueError("Invalid sample_method: {}".format(sample_method))
        #         t[k] = r
                
        #         # # if relative error > 50%, print sth
        #         # rel_err = np.abs(true_interarrival[i][k] - t[k]) / (true_interarrival[i][k] + eps)
        #         # rel_err_threshold = 10
        #         # if rel_err > rel_err_threshold:
        #         #     continue
                   
                    
        #     # t[to_be_resample] = np.random.choice(raw_interarrival_concat, size=np.sum(to_be_resample))
        #     interarrival[i][1:] = t      

        # Assuming `bins_edges`, `sample_num`, `large_bin_indices`, and `interval_duration` 
        # are already defined in your code.

        a_choice = bins_edges[1:]

        for interval_index, t in tqdm(enumerate(interarrival)):
            t = t[1:]
            
            if t.size == 0:  # skip flows with only one packet
                continue

            to_be_resample = np.nonzero(t <= interval_duration)[0]

            # Skip if none are to be resampled
            if len(to_be_resample) == 0:
                continue

            hist = hists[interval_index]
            # hist[large_bin_indices] = 0
            # hist /= np.sum(hist)

            # Sample for all `to_be_resample` indices at once
            r_all = np.random.choice(a=a_choice, size=(len(to_be_resample), sample_num), p=hist)
            
            for idx, k in enumerate(to_be_resample):
                r = r_all[idx]
                r = r[r <= t[k]]
                
                if r.size == 0:
                    continue

                if sample_method == "first":
                    r = r[0]
                elif sample_method == "min":
                    r = np.min(r)
                elif sample_method == "mean":
                    r = 10 ** np.mean(np.log10(r))
                else:
                    raise ValueError(f"Invalid sample_method: {sample_method}")

                t[k] = 10 ** (np.log10(r) + np.random.uniform(-0.1, 0.1))

            interarrival[interval_index] = np.concatenate(([interarrival[interval_index][0]], t))


        return interarrival      

    elif mode == "resample_trace":
        """
        sample directly from the raw interarrival if reconstructed interarrival <= interval_duration
        gives good pdf of interarrival, but totally mismatched with the original trace's interarrival
        """
        interarrival = []
        for flow_index in range(packetrate.shape[1]):
            singleflow_packetrate = packetrate[:, flow_index]
            nonzero_packetrate_interval = np.nonzero(singleflow_packetrate)[0]
            nonzero_packetrate = singleflow_packetrate[nonzero_packetrate_interval]
            # given a nonzero_packetrate n and a nonzero_packetrate_interval k, 
            # the result is np.ones(n) / n + k * interval_duration
            singleflow_interarrival = []
            for n, k in zip(nonzero_packetrate, nonzero_packetrate_interval):
                t = np.ones((n)) / n * interval_duration
                t[0] += k * interval_duration
                t = np.cumsum(t)
                singleflow_interarrival.append(t)
            if len(singleflow_interarrival) > 0:
                singleflow_interarrival = np.concatenate(singleflow_interarrival)
                singleflow_interarrival = np.sort(singleflow_interarrival)
                singleflow_interarrival[1:] = np.diff(np.sort(singleflow_interarrival))

            else:
                warnings.warn("#{} flow has no packet!".format(flow_index))
            interarrival.append(singleflow_interarrival)

        raw_interarrival = params["raw_interarrival"]
        raw_interarrival_concat = np.concatenate(raw_interarrival)
        # only resample from raw interarrival that is <= interval_duration
        raw_interarrival_concat = raw_interarrival_concat[raw_interarrival_concat <= interval_duration]

        for interval_index in range(len(interarrival)):
            t = interarrival[interval_index][1:]
            # skip flows with only one packet
            if len(t) == 0:
                continue
            to_be_resample = t <= interval_duration
            # if no interarrival is <= interval_duration, skip this flow
            if len(to_be_resample) == 0:
                continue 
            t[to_be_resample] = np.random.choice(raw_interarrival_concat, size=np.sum(to_be_resample))
            interarrival[interval_index][1:] = t
            
        return interarrival
    elif mode == "resample_trace_refined":
        """
        ! much computational extensive than resample_raw
        sample directly from the raw interarrival if reconstructed interarrival <= interval_duration
        """
        interarrival = []
        for flow_index in range(packetrate.shape[1]):
            singleflow_packetrate = packetrate[:, flow_index]
            nonzero_packetrate_interval = np.nonzero(singleflow_packetrate)[0]
            nonzero_packetrate = singleflow_packetrate[nonzero_packetrate_interval]
            # given a nonzero_packetrate n and a nonzero_packetrate_interval k, 
            # the result is np.ones(n) / n + k * interval_duration
            singleflow_interarrival = []
            for n, k in zip(nonzero_packetrate, nonzero_packetrate_interval):
                t = np.ones((n)) / n * interval_duration
                t[0] += k * interval_duration
                t = np.cumsum(t)
                singleflow_interarrival.append(t)
            if len(singleflow_interarrival) > 0:
                singleflow_interarrival = np.concatenate(singleflow_interarrival)
                singleflow_interarrival = np.sort(singleflow_interarrival)
                singleflow_interarrival[1:] = np.diff(np.sort(singleflow_interarrival))

            else:
                warnings.warn("#{} flow has no packet!".format(flow_index))
            interarrival.append(singleflow_interarrival)

        raw_interarrival = params["raw_interarrival"]
        raw_interarrival_concat = np.concatenate(raw_interarrival)
        # only resample from raw interarrival that is <= interval_duration
        raw_interarrival_concat = raw_interarrival_concat[raw_interarrival_concat <= interval_duration]

        # for i in tqdm(range(len(interarrival))):
        #     t = interarrival[i][1:]
            
        #     # skip flows with only one packet
        #     if len(t) == 0:
        #         continue

        #     # find all indices whose value is <= interval_duration
        #     to_be_resample = np.nonzero(t <= interval_duration)[0]
            
        #     # if no interarrival is <= interval_duration, skip this flow
        #     if len(to_be_resample) == 0:
        #         continue 

        #     # The goal is to vectorize the operations inside the inner loop
        #     # Step 1: Use broadcasting to get a mask of valid values in raw_interarrival_concat for all t[k]
        #     valid_choices_mask = raw_interarrival_concat[:, None] <= t[to_be_resample]

        #     # Step 2: Filter values using this mask
        #     valid_choices = [raw_interarrival_concat[mask] for mask in valid_choices_mask.T]
            
        #     # Step 3: Use numpy random choice for each of the valid choice arrays
        #     r_values = [np.random.choice(choices, size=1) if choices.size > 0 else np.array([0]) for choices in valid_choices]
            
        #     # Step 4: Convert list of arrays to single array
        #     r_values = np.array(r_values).squeeze()

        #     # Step 5: Update t values at once using advanced indexing
        #     t[to_be_resample] = r_values

        #     interarrival[i][1:] = t



        for interval_index in tqdm(range(len(interarrival))):
            t = interarrival[interval_index][1:]
            # skip flows with only one packet
            if len(t) == 0:
                continue
            # find all indices whose value is <= interval_duration
            to_be_resample = np.nonzero(t <= interval_duration)[0]
            # if no interarrival is <= interval_duration, skip this flow
            if len(to_be_resample) == 0:
                continue 
            # for k in tqdm(to_be_resample):
            #     raw_interarrival_concat_refined = raw_interarrival_concat[raw_interarrival_concat <= t[k]]
            #     if len(raw_interarrival_concat_refined) == 0:
            #         continue
            #     r = np.random.choice(raw_interarrival_concat_refined, size=1)
            #     t[k] = r
            
            t[to_be_resample] = np.random.choice(raw_interarrival_concat, size=to_be_resample.size)
            interarrival[interval_index][1:] = t
            
        return interarrival
    elif mode == "resample_trace_hist":
        """
        for interarrival <= interval_duration,
            - bin the raw interarrival into a historgram
            - sample a bin based on the probability of the bin, and uniformly sample from the bin
            - add a bin for very small interarrival (e.g. 1e-7, 0)
        """
        raise NotImplementedError("resample_trace_hist is not implemented yet")
        interarrival = []
        for flow_index in range(packetrate.shape[1]):
            singleflow_packetrate = packetrate[:, flow_index]
            nonzero_packetrate_interval = np.nonzero(singleflow_packetrate)[0]
            nonzero_packetrate = singleflow_packetrate[nonzero_packetrate_interval]
            # given a nonzero_packetrate n and a nonzero_packetrate_interval k, 
            # the result is np.ones(n) / n + k * interval_duration
            singleflow_interarrival = []
            for n, k in zip(nonzero_packetrate, nonzero_packetrate_interval):
                t = np.ones((n)) / n * interval_duration
                t[0] += k * interval_duration
                t = np.cumsum(t)
                singleflow_interarrival.append(t)
            if len(nonzero_packetrate_interval) > 3:
                print("", end="")
            if len(singleflow_interarrival) > 0:
                singleflow_interarrival = np.concatenate(singleflow_interarrival)
                singleflow_interarrival = np.sort(singleflow_interarrival)
                singleflow_interarrival[1:] = np.diff(np.sort(singleflow_interarrival))

            else:
                warnings.warn("#{} flow has no packet!".format(flow_index))
            interarrival.append(singleflow_interarrival)

        raw_interarrival = params["raw_interarrival"]
        raw_interarrival_concat = np.concatenate(raw_interarrival)
        # only resample from raw interarrival that is <= interval_duration
        raw_interarrival_concat = raw_interarrival_concat[raw_interarrival_concat <= interval_duration]

        # bin the raw interarrival
        if bins_count not in params:
            bins_count = 100
            bins = np.logspace()
            
        return interarrival
    elif mode == "resample_actual_interval_duration_ratio":
        actual_interval_duration_ratio = params["actual_interval_duration_ratio"]
        assert(actual_interval_duration_ratio.shape == packetrate.shape)
        interarrival = []
        for flow_index in range(packetrate.shape[1]):
            singleflow_packetrate = packetrate[:, flow_index]
            singleflow_actual_interval_duration_ratio = actual_interval_duration_ratio[:, flow_index]
            nonzero_packetrate_interval = np.nonzero(singleflow_packetrate)[0]
            nonzero_packetrate = singleflow_packetrate[nonzero_packetrate_interval]
            # given a nonzero_packetrate n and a nonzero_packetrate_interval k, 
            # the result is np.ones(n) / n + k * interval_duration
            singleflow_interarrival = []
            for n, k in zip(nonzero_packetrate, nonzero_packetrate_interval):
                # get ratio of actual interval duration
                # as an example, let 
                # - k = 2
                # - packetrate = 5
                # - interval_duration = 1, 
                # - actual_interval_duration_ratio = 0.05
                # the naive "equal" mode gives [2, 2.2, 2.4, 2.6, 2.8]
                interval_duration_ratio = singleflow_actual_interval_duration_ratio[k]
                assert(0 <= interval_duration_ratio <= 1)
                # uniformly sample start time in the interval
                # interval_start = [0, 1] * (1 - 0.05) * 1 = [0, 0.95], let it be 0.6
                interval_start = np.random.uniform() * (1 - interval_duration_ratio) * interval_duration
                # t = [1, 1, 1, 1, 1] / 5 * 1 * 0.05 = [0.01, 0.01, 0.01, 0.01, 0.01]
                t = np.ones((n)) / (n + 1) * interval_duration * interval_duration_ratio
                # t[0] += 2 * 1 + 0.6, t = [2.61, 0.01, 0.01, 0.01, 0.01]
                t[0] += k * interval_duration + interval_start
                # t = [2.61, 2.62, 2.63, 2.64, 2.65]
                t = np.cumsum(t)
                singleflow_interarrival.append(t)
            if len(singleflow_interarrival) > 0:
                singleflow_interarrival = np.concatenate(singleflow_interarrival)
                singleflow_interarrival = np.sort(singleflow_interarrival)
                singleflow_interarrival[1:] = np.diff(np.sort(singleflow_interarrival))

            else:
                warnings.warn("#{} flow has no packet!".format(flow_index))
            if flow_index == 0:
                print(np.where(singleflow_packetrate > 0))
                print(singleflow_packetrate[singleflow_packetrate > 0])
                print(singleflow_actual_interval_duration_ratio[singleflow_packetrate > 0])
                print(singleflow_interarrival[1:])
            interarrival.append(singleflow_interarrival)    
        return interarrival
    else:
        raise ValueError("Unknown mode: '{}'".format(mode))





def get_sci_exp(x):
    """
    Return the exponent of a number in scientific notation
    
    Example:
    --------
    >>> a = 1.234e-5
    >>> get_sci_exp(a)
    -5
    >>> type(get_sci_exp(a))
    <class 'int'>
    """
    if isinstance(x, np.ndarray):
        return np.floor(np.log10(x.astype(float))).astype(int)
    elif isinstance(x, float):
        return int(np.floor(np.log10(x)))
    return int(np.floor(np.log10(float(x))))


def interarrival2gmm(interarrival, output_path=None):
    """
    Given a list of flow-level interarrival, fit a Gaussian mixture model to the interarrival time of each flow.

    Parameters:
    -----------
    interarrival: a list of numpy arrays where each array is a sequence of interarrival started with flow_start.
        Note that interarrival[i] and interarrival[j] may not have the same length b/c different flows have different #pkt

    Returns:
    --------
    gmm_params: a list of tuples of ndarray (gmm_weights, gmm_means, gmm_covariances) where each tuple is the parameters of a GMM
    """
    if output_path is None:
        warnings.warn("output_path is None, gmm_params will not be saved to disk")
    elif os.path.exists(output_path):
        print("Loading gmm parameters from {}".format(output_path))
        with open(output_path, "rb") as f:
            gmm_params = pickle.load(f)
        return gmm_params 
    
    #TODO: Find out what types of flow-level pkt rate contribute to the misalignment at 1e-5, 1e-6 level
    # fit gmm for each flow, and save its parameters
    gmm_params = []
    gmm = GaussianMixture(n_components=2, covariance_type="full")
    print("Fitting gmm to raw flow-level interarrival...")
    for t in tqdm(interarrival):
        # for flows with only a few packets, the gmm degenerates to memorizing the data using the mean value
        if len(t) < 2:
            gmm_params.append((None, None, None))
            continue
        t_train = t.reshape(-1, 1)
        t_train_log = np.log(t_train + eps)
        # t_train_log = np.log1p(t_train)
        # t_train_log = np.log(np.log(t_train + eps) + eps)
        gmm.fit(t_train_log)
        gmm_params.append((gmm.weights_, gmm.means_, gmm.covariances_))
        
    # return without saving
    if output_path is None:
        return {"gmm_params": gmm_params}

    # save gmm_params and return
    with open(output_path, "wb") as f:
        print("Saving gmm parameters to {}".format(output_path))
        pickle.dump(gmm_params, f)
    return {"gmm_params": gmm_params}

def interarrival2hist(interarrival, bins_count, edge_mode="linspace", output_path=None):
    """
    Given a list of flow-level interarrival, compute the histogram of interarrival time of each flow.

    Parameters:
    -----------
    interarrival: same as nta.utils.data.interarrival2gmm
    bins_count: the number of bins in the histogram
    edge_mode: the method to compute the edges of the histogram
        Let the m be interarrival.min(), and M be interarrival.max()
        - "linspace": use np.linspace(m, M, bins_count) to compute the edge
        - "logspace": use np.logspace(np.log10(m), np.log10(M), bins_count) to compute the edge
        - "density": ??? 
    
    Returns:
    --------
    A dictionary consisting of the following keys:
    - "bins_count": int, the number of bins in the histogram
    - "bins_edges": a numpy array of size (bins_count + 1), the edges of the histogram
    - "hists": a numpy array of size (n_flow, bins_count), the histogram of each flow
    """

    if edge_mode not in ["linspace", "logspace", "density"]:
        raise ValueError("Invalid edge_mode: {}".format(edge_mode))

    if output_path is None:
        warnings.warn("output_path is None, hist_params will not be saved to disk")
    elif os.path.exists(output_path):
        print("Loading hist parameters from {}".format(output_path))
        with open(output_path, "rb") as f:
            hist_params = pickle.load(f)
        return hist_params

    hist_params = {
        "bins_count": bins_count,
        "bins_edges": None,
        "hists": [],
    }

    # If the smallest nonzero interarrival is of the order of 1e-7, 
    #   and the largest interarrival is of the order of 1e1, and bins_count = 10,
    #   then there should be 10 bins of the following range
    #       [0, 1e-7), [1e-7, 1e-6), ..., [1e-3, 1e-2), 
    #       [1e-2, 1e-1), [1e-1, 1e0), [1e0, 1e1), [1e1, 1e2]
    #   where [0, 1e-7) is meant to capture both 1e-7 and 0 interarrival
    #   and [1e1, 1e2) is meant to capture interarrval > 1e1
    # The corresponding bins_edges for np.histogram is
    #       [0, 1e-7, ..., 1e0, 1e1, 1e2]
    interarrival_concat = np.concatenate(interarrival)
    interarrival_concat_nonzero = interarrival_concat[interarrival_concat > 0]
    if len(interarrival_concat_nonzero) == 0:
        print("All interarrival are 0, use eps as the minimum and 10 * eps as the maximum")
        interarrival_min = eps 
        interarrival_max = eps * 10
    else:
        # find global nonzero minimum and global nonzero maximum of interarrival
        interarrival_min = interarrival_concat_nonzero.min()
        interarrival_max = interarrival_concat_nonzero.max()
    bins_min_exp = np.log10(interarrival_min)
    bins_max_exp = np.log10(interarrival_max)
    print("interarriva min: {}, nonzero_min: {}, max: {}".format(
        interarrival_min, 
        interarrival_concat_nonzero.min(),
        interarrival_max))
    print("bins_min_exp: {}, bins_max_exp: {}".format(bins_min_exp, bins_max_exp))

    # define the right edge (included) of each bin
    if edge_mode == "linspace":
        bins_edges = np.linspace(interarrival_min, interarrival_max, num=bins_count)
        # insert 0 to the beginning of bins_edges
        bins_edges = np.insert(bins_edges, 0, 0)
    elif edge_mode == "logspace":
        bins_edges = np.logspace(bins_min_exp, bins_max_exp, num=bins_count)
         # insert 0 to the beginning of bins_edges
        bins_edges = np.insert(bins_edges, 0, 0)
    elif edge_mode == "density":
        # TODO:
        raise NotImplementedError("edge_mode = 'density' is not implemented yet")

    hist_params["bins_edges"] = bins_edges
    # compute histogram for each flow        
    print("Computing histogram of raw flow-level interarrival...")
    for t in tqdm(interarrival):
        # for flows with only a few packets, the histogram degenerates to memorizing the data using the mean value
        hist, bins = np.histogram(t, bins=bins_edges, density=False)
        hist_n = hist / np.sum(hist)
        hist_params["hists"].append(hist_n)

        continue
        if edge_mode == "logspace":
            print(t)
            print(bins_edges)
            print(hist)
            print(hist_n)
            fig, ax = plt.subplots(1, 1)
            
            # plot hist
            ax.hist(t, bins=bins_edges, density=False, alpha=0.5, label="hist", color="C0")

            ax.legend()
            ax.set_xscale("log")
            # ax.set_yscale("log")

            xtick_range = [-8, -7, -6, -5, -4, -3, -2, -1, 0]
            tick_fontsize = 12
            ax.xaxis.set_major_locator(ticker.FixedLocator(ax.get_xticks())) 
            ax.yaxis.set_major_locator(ticker.FixedLocator(ax.get_yticks()))
            ax.set_xticks([10**x for x in xtick_range])
            ax.set_xticklabels(["$10^{" + str(x) + "}$" for x in xtick_range], fontsize=tick_fontsize)  
            ax.set_yticklabels(ax.get_yticks(), fontsize=tick_fontsize)
            ax.set_yticklabels(["{:1.1e}".format(x) for x in ax.get_yticks()], fontsize=tick_fontsize)            

            
            plt.show(block=False)
            user_input = input("Press enter to continue, or q to quit: ")
            if user_input == "q":
                break
            elif user_input == "qq":
                exit(0)
            # close the figure
            plt.close(fig)
    
    # return without saving
    if output_path is None:
        return hist_params 
    
    # save histogram and return
    with open(output_path, "wb") as f:
        print("Saving hist parameters to {}".format(output_path))
        pickle.dump(hist_params, f)
    return hist_params


def timestamp2actual_interval_duration_ratio(timestamp, time_unit_exp, num_t):
    # of the same shape as interarrival
    actual_interval_duration_ratio = np.ones((num_t, len(timestamp)))
    interval_duration = 10**time_unit_exp

    print("Converting interarrival and packetrate to interval_duration_ratio")
    for i in tqdm(range(len(timestamp))):
        flow_timestamp = timestamp[i]
        # if len(flow_interarrival) == 1:
        #     actual_interval_duration_ratio.append(1)
        #     continue 
        flow_packetrate = stats.timestamps2flow(flow_timestamp, time_unit_exp, num_t, verbose=False)
        assert(flow_timestamp.size == flow_packetrate.sum())
        nonzero_flow_packetrate = np.where(flow_packetrate > 0)[0]
        start = 0
        for j in nonzero_flow_packetrate:
            k = int(flow_packetrate[j])
            # if k == 1:
            #     # ignore interval with only 1 interarrival, i.e. 2 packets
            #     actual_interval_duration_ratio[j, i] = 1
            # else:
            #     actual_d = flow_timestamp[start + k - 1] - flow_timestamp[start]
            #     actual_interval_duration_ratio[j, i] = actual_d / interval_duration
            actual_d = flow_timestamp[start + k - 1] - flow_timestamp[start]
            actual_interval_duration_ratio[j, i] = actual_d / interval_duration
            start += k
        r = actual_interval_duration_ratio[:, i]
        if i == 0:
            print(np.diff(flow_timestamp))
            print(np.where(flow_packetrate > 0))
            print(flow_packetrate[flow_packetrate > 0])
            print(r[flow_packetrate > 0])
        
        assert(np.logical_and(r <= 1, r >= 0).sum() == num_t)
    return actual_interval_duration_ratio

def categories2number(x, category_map=None):
    """
    Convert a list of categories to a list of numbers by assiging a number to each category.
    """

    if category_map is None:
        category_map = {}
        for i, c in enumerate(np.unique(x)):
            category_map[c] = i
    return np.array([category_map[c] for c in x]), category_map