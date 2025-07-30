import pandas as pd
from scapy.all import rdpcap, wrpcap
from collections import defaultdict
import os
from tqdm import tqdm
import socket
import struct
from scapy.layers.inet import IP, TCP, UDP, ICMP

def IP_str2int(ip_str):
    """Convert an IP string to an integer."""
    return struct.unpack("!L", socket.inet_aton(ip_str))[0]

def csv2pcap(input_path, output_path):
    """
    Convert a CSV file to a PCAP file.

    :param input_path: Path to the input CSV file.
    :param output_path: Path to save the output PCAP file.
    """
    print("Converting CSV file to PCAP file...")
    print(f"Input CSV: {input_path}")
    print(f"Output PCAP: {output_path}")

    df = pd.read_csv(input_path)
    df = df.sort_values(["time"])

    packets = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Converting CSV to PCAP"):
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
        except ValueError:
            raise NotImplementedError("Non-integer protocol types are not supported.")

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
            p = ip  # For other protocols

        p.time = time
        p.len = pkt_len
        p.wirelen = pkt_len + 4  # Adjust as per your requirement

        packets.append(p)

    wrpcap(output_path, packets)
    print(f"Successfully converted {input_path} to {output_path}")

def split_pcap_by_flows(input_pcap, output_dir):
    """
    Split a PCAP file into multiple PCAP files based on network flows.

    :param input_pcap: Path to the input PCAP file.
    :param output_dir: Directory to save flow-specific PCAP files.
    :return: List of flow-specific PCAP file paths.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read all packets from the input PCAP file
    packets = rdpcap(input_pcap)
    flow_map = defaultdict(list)

    # Process each packet and categorize them into flows
    for packet in tqdm(packets, desc="Processing Packets", unit="packet"):
        if 'IP' in packet:
            proto = packet['IP'].proto
            if 'TCP' in packet:
                src_port = packet['TCP'].sport
                dst_port = packet['TCP'].dport
            elif 'UDP' in packet:
                src_port = packet['UDP'].sport
                dst_port = packet['UDP'].dport
            else:
                src_port = None
                dst_port = None
            # Define flow_id based on source IP, destination IP, protocol, source port, and destination port
            flow_id = (packet['IP'].src, packet['IP'].dst, proto, src_port, dst_port)
            flow_map[flow_id].append(packet)
    
    flow_files = []
    # Write each flow's packets to a separate PCAP file
    for i, (flow_id, flow_packets) in enumerate(tqdm(flow_map.items(), desc="Writing Flow Files", unit="flow")):
        flow_file = os.path.join(output_dir, f'flow_{i}.pcap')
        wrpcap(flow_file, flow_packets)
        flow_files.append(flow_file)

    return flow_files


def count_flows_in_pcap(input_pcap):
    """
    Count the total number of unique flows in a PCAP file.

    :param input_pcap: Path to the input PCAP file.
    :return: Total number of unique flows.
    """
    print(f"Counting flows in PCAP file: {input_pcap}")
    
    # Read all packets from the input PCAP file
    packets = rdpcap(input_pcap)
    flow_map = defaultdict(list)

    # Process each packet and categorize them into flows
    for packet in tqdm(packets, desc="Processing Packets", unit="packet"):
        if 'IP' in packet:
            proto = packet['IP'].proto
            if 'TCP' in packet:
                src_port = packet['TCP'].sport
                dst_port = packet['TCP'].dport
            elif 'UDP' in packet:
                src_port = packet['UDP'].sport
                dst_port = packet['UDP'].dport
            else:
                src_port = None
                dst_port = None
            # Define flow_id based on source IP, destination IP, protocol, source port, and destination port
            flow_id = (packet['IP'].src, packet['IP'].dst, proto, src_port, dst_port)
            flow_map[flow_id].append(packet)
    
    total_flows = len(flow_map)
    print(f"Total number of unique flows: {total_flows}")
    
    return total_flows

if __name__ == "__main__":
    input_csv = "../../../data/ton_iot/normal_1.csv"
    input_pcap = "../../../result/netdiffusion/normal_1.pcap"
    output_dir = "../../../result/netdiffusion/fine_tune_pcaps"
    csv2pcap(input_csv, input_pcap)
    split_pcap_by_flows(input_pcap, output_dir)
