#!/usr/bin/env python3
import os
import sys
import struct
import socket
import pandas as pd
import numpy as np
from scapy.all import rdpcap, wrpcap, PcapReader, IP, TCP, UDP, ICMP
from tqdm import tqdm

# =======================
# Configuration Variables
# =======================

# Directory containing all the pcap files to be merged
INPUT_DIR = "../../../result/netdiffusion/replayable_generated_pcaps"

# Output paths
MERGED_PCAP = "../../../result/netdiffusion/reconstructed_ton.pcap"
CSV_OUTPUT = "../../../result/netdiffusion/reconstructed_ton.csv"

# CSV Header
CSV_HEADER = "srcip,dstip,srcport,dstport,proto,time,pkt_len,version,ihl,tos,id,flag,off,ttl\n"

# =======================
# Helper Functions
# =======================

def ip_str_to_int(ip):
    """
    Convert an IPv4 address from string format to integer format.
    
    :param ip: IPv4 address as a string.
    :return: IPv4 address as an integer.
    """
    return struct.unpack("!I", socket.inet_aton(ip))[0]

def merge_pcap_files(input_dir, output_pcap):
    """
    Merge all pcap files in the specified directory into a single pcap file.

    :param input_dir: Directory containing input pcap files.
    :param output_pcap: Path to the output merged pcap file.
    """
    print("=== Step 1: Merging PCAP Files ===")
    pcap_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pcap')]
    
    if not pcap_files:
        print(f"No pcap files found in the directory: {input_dir}")
        sys.exit(1)
    
    all_packets = []
    print(f"Found {len(pcap_files)} pcap files. Starting to merge...")
    
    for pcap_file in tqdm(pcap_files, desc="Processing PCAP files"):
        try:
            packets = rdpcap(pcap_file)
            all_packets.extend(packets)
        except Exception as e:
            print(f"Error reading {pcap_file}: {e}")
    
    print(f"Total packets before sorting: {len(all_packets)}")
    # Sort all packets by their timestamp to maintain chronological order
    all_packets.sort(key=lambda pkt: pkt.time)
    
    # Write the merged packets to the output pcap file
    wrpcap(output_pcap, all_packets)
    print(f"Merged pcap saved to: {output_pcap}\n")

def pcap_to_csv(input_pcap, output_csv, header):
    """
    Convert a pcap file to a CSV file with the specified header.

    :param input_pcap: Path to the input pcap file.
    :param output_csv: Path to the output CSV file.
    :param header: CSV header string.
    """
    print("=== Step 2: Converting PCAP to CSV ===")
    print(f"Input PCAP: {input_pcap}")
    print(f"Output CSV: {output_csv}\n")
    
    with open(output_csv, "w") as csv_file:
        # Write the header
        csv_file.write(header)
        
        # Initialize packet counter
        packet_count = 0
        discarded_packets = 0
        
        # Read packets using PcapReader for memory efficiency
        with PcapReader(input_pcap) as pcap_reader:
            for packet in tqdm(pcap_reader, desc="Converting packets"):
                try:
                    if not packet.haslayer(IP):
                        discarded_packets += 1
                        continue  # Skip non-IP packets

                    ip_layer = packet[IP]
                    src_ip = ip_layer.src
                    dst_ip = ip_layer.dst
                    src_port = None
                    dst_port = None
                    proto = ip_layer.proto

                    # Determine protocol and extract ports
                    if proto == 6 and packet.haslayer(TCP):
                        tcp_layer = packet[TCP]
                        src_port = tcp_layer.sport
                        dst_port = tcp_layer.dport
                    elif proto == 17 and packet.haslayer(UDP):
                        udp_layer = packet[UDP]
                        src_port = udp_layer.sport
                        dst_port = udp_layer.dport
                    elif proto == 1 and packet.haslayer(ICMP):
                        # ICMP does not have ports
                        src_port = 0
                        dst_port = 0
                    else:
                        # Unsupported protocol
                        discarded_packets += 1
                        continue

                    # Packet metadata
                    timestamp = packet.time
                    pkt_len = len(packet)
                    version = ip_layer.version
                    ihl = ip_layer.ihl
                    tos = ip_layer.tos
                    pkt_id = ip_layer.id
                    flags = ip_layer.flags
                    frag = ip_layer.frag
                    ttl = ip_layer.ttl

                    # Convert IPs to integers
                    src_ip_int = ip_str_to_int(src_ip)
                    dst_ip_int = ip_str_to_int(dst_ip)

                    # Prepare CSV row
                    csv_row = f"{src_ip_int},{dst_ip_int},{src_port},{dst_port},{proto},{timestamp},{pkt_len},{version},{ihl},{tos},{pkt_id},{flags},{frag},{ttl}\n"
                    csv_file.write(csv_row)
                    packet_count += 1

                except Exception as e:
                    # Handle any unexpected errors during packet processing
                    discarded_packets += 1
                    continue

    print(f"Total packets converted: {packet_count}")
    print(f"Total packets discarded: {discarded_packets}")
    print(f"CSV file created at: {output_csv}\n")

# =======================
# Main Execution
# =======================

def main():
    # Step 1: Merge all pcap files into one large pcap file
    merge_pcap_files(INPUT_DIR, MERGED_PCAP)
    
    # Step 2: Convert the merged pcap file to a CSV file with the specified header
    pcap_to_csv(MERGED_PCAP, CSV_OUTPUT, CSV_HEADER)
    
    print("=== Processing Completed Successfully ===")

if __name__ == "__main__":
    main()
