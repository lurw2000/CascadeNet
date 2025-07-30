from netml.pparser.parser import PCAP
from netml.utils.tool import dump_data, load_data
from sklearn.model_selection import train_test_split
from netml.ndm.model import MODEL
from netml.ndm.ocsvm import OCSVM
from netml.ndm.iforest import IF
from netml.ndm.ae import AE 
from netml.ndm.gmm import GMM
from netml.ndm.pca import PCA
from netml.ndm.kde import KDE
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from scapy.all import *
import socket
import csv

from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def parse_args():
    parser = ArgumentParser(description='anomly detection by netml')
    parser.add_argument('--input', type=str, action='append', help='path of input pcap or csv file')
    parser.add_argument('--raw', type=str, help='path of input pcap or csv file')
    parser.add_argument('--out_dir', type=str, help='base output directory')
    parser.add_argument('--generator', type=str, action='append', help='generator name (e.g., CascadeNet)')
    parser.add_argument('--dataset', type=str, help='dataset name (e.g., caida)')
    parser.add_argument('--ndm', type=str, help='specific NDM to use (e.g., OCSVM, IForest, GMM, AE, PCA, KDE). If not specified, all NDMs will be used.')
    parser.add_argument('--iters', type=int, help='number of iterations', default=1)
    
    return parser.parse_args()

def generate_raw_pcap(raw_csv, dataset_name, out_dir):
    """Generate raw PCAP file if it doesn't exist."""
    base_dir = Path(out_dir) / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)
    raw_pcap = base_dir / f"{dataset_name}_raw.pcap"
    
    if not raw_pcap.exists() and raw_csv.endswith('.csv'):
        print(f"Generating raw PCAP file: {raw_pcap}")
        csv2pcap(raw_csv, str(raw_pcap))
    
    return str(raw_pcap)

def csv2pcap(csv_file, pcap_file):
    df = pd.read_csv(csv_file)
    df.sort_values("time")
    df.reset_index()

    packets = []
    skip = 0
    total = 0
    for index, row in df.iterrows():
        if row["proto"] == "TCP":
            row["proto"] = 6
        elif row["proto"] == "UDP":
            row["proto"] = 17

        #print(socket.inet_ntoa(int(row["srcip"]).to_bytes(4, byteorder="big")))
        #print(socket.inet_ntoa(int(row["dstip"]).to_bytes(4, byteorder="big")))
        #print(int(float(row["pkt_len"])))
        #print(int(row["proto"]))
        #print(int(row["srcport"]))
        #print(int(row["dstport"]))
        #print(float(row["time"]) / 1e6)
        #print()

        total += 1
        if int(row["srcip"]) < 0 or int(row["srcip"]) >= 1<<32:
            #print("srcip error: "+str(row["srcip"]))
            skip += 1
            continue
        
        if int(row["dstip"]) < 0 or int(row["dstip"]) >= 1<<32:
            #print("dstip error: "+str(row["dstip"]))
            skip += 1
            continue
        
        if int(row["srcport"]) < 0 or int(row["srcport"]) >= 1<<16:
            #print("srcport error: "+str(row["srcport"]))
            skip += 1
            continue
        
        if int(row["dstport"]) < 0 or int(row["dstport"]) >= 1<<16:
            #print("dstport error: "+str(row["dstport"]))
            skip += 1
            continue
        
        if int(row["proto"]) != 6 and int(row["proto"]) != 17:
            #print("proto error: "+str(row["proto"]))
            skip += 1
            continue

        ip = IP(
            src=socket.inet_ntoa(int(row["srcip"]).to_bytes(4, byteorder="big")),
            dst=socket.inet_ntoa(int(row["dstip"]).to_bytes(4, byteorder="big")),
            #version=4,
            #ihl=5,
            #tos=int(float(row["tos"])),
            len=int(float(row["pkt_len"])),
            #id=int(float(row["id"])),
            #flags=row["flag"],
            #frag=int(float(row["off"])),
            #ttl=int(float(row["ttl"])),
            proto=int(row["proto"]),
        )
        if int(row["proto"]) == 6:
            tcp = TCP(
                sport=int(row["srcport"]),
                dport=int(row["dstport"]),
            )
            pkt = ip/tcp
        elif int(row["proto"]) == 17:
            udp = UDP(
                sport=int(row["srcport"]),
                dport=int(row["dstport"]),
            )
            pkt = ip/udp

        pkt.time = float(row["time"]) / 1e6
        
        packets.append(pkt)

        
    
    print("skip count: " + str(skip))
    print("total count: " + str(total))
    
    wrpcap(pcap_file, packets)

def load_pcap_data(pcap_file, raw_file, feature_type, random_state):

    pcap_raw = PCAP(
        raw_file,
        flow_ptks_thres=2,
        random_state=random_state,
        verbose=0,
    )
    pcap_raw.pcap2flows(interval=100000, tcp_timeout=100000, udp_timeout=100000)
    pcap_raw.q_interval = 0.9
    if feature_type in ["SAMP_NUM", "SAMP_SIZE"]:
        time = [ pkt.time for fid, pkts in pcap_raw.flows for pkt in pkts ]
        pcap_raw.flow2features(feature_type, dim=200, sampling_rate=(np.max(time) - np.min(time)) / 200, fft=False, header=False)
    else:
        pcap_raw.flow2features(feature_type, fft=False, header=False)

    # print(pcap_raw.features.shape)

    pcap = PCAP(
        pcap_file,
        flow_ptks_thres=2,
        random_state=random_state,
        verbose=0,
    )

    pcap.pcap2flows(interval=100000, tcp_timeout=100000, udp_timeout=100000)
    if feature_type in ["SAMP_NUM", "SAMP_SIZE"]:
        pcap.flow2features(feature_type, dim=200, sampling_rate=(np.max(time) - np.min(time)) / 200, fft=False, header=False)
    else:
        pcap.flow2features(feature_type, dim=pcap_raw.dim, fft=False, header=False)
    
    # print(pcap.features.shape)
    

    #(
    #    features_train,
    #    features_test
    #) = train_test_split(pcap.features, test_size=0.1, random_state=random_state)

    features_test = pcap_raw.features
    features_train = pcap.features

    # print(features_test.shape)
    # print(features_train.shape)
    #if features_train.shape[1] > features_test.shape[1]:
    #    features_train = features_train[:, :features_test.shape[1]]
    #elif features_train.shape[1] < features_test.shape[1]:
    #    features_train = np.concatenate([features_train, np.zeros((features_train.shape[0], features_test.shape[1] - features_train.shape[1]))], axis=1)
    return features_test, features_train


# OCSVM
def predict_ocsvm(features_train, features_test, random_state):
    ocsvm = OCSVM(kernel='rbf', nu=0.5, random_state=random_state)
    ocsvm.name = 'OCSVM'
    ndm = MODEL(ocsvm, score_metric='auc', verbose=0, random_state=random_state)

    ocsvm_raw = OCSVM(kernel='rbf', nu=0.5, random_state=random_state)
    ocsvm_raw.name = 'OCSVM_raw'
    ndm_raw = MODEL(ocsvm_raw, score_metric='auc', verbose=0, random_state=random_state)

    ndm.train(features_train)
    ndm_raw.train(features_test)

    result = ocsvm.predict(features_test)
    result_raw = ocsvm_raw.predict(features_test)

    return calculate_metrics_0(result, result_raw)


# IForest
def predict_iforest(features_train, features_test, random_state):
    iforest = IF(n_estimators=100, contamination=0.1, random_state=random_state)
    iforest.name = 'IForest'
    ndm = MODEL(iforest, score_metric='auc', verbose=0, random_state=random_state)

    iforest_raw = IF(n_estimators=100, contamination=0.1, random_state=random_state)
    iforest_raw.name = 'IForest_raw'
    ndm_raw = MODEL(iforest_raw, score_metric='auc', verbose=0, random_state=random_state)

    ndm.train(features_train)
    ndm_raw.train(features_test)

    result = iforest.decision_function(features_test)
    result_raw = iforest_raw.decision_function(features_test)

    return calculate_metrics_0(result, result_raw)



# GMM
def predict_gmm(features_train, features_test, random_state):
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)
    gmm = GMM(n_components=1, reg_covar=1e6, contamination=0.1, random_state=random_state) 
    gmm.name = 'GMM'
    ndm = MODEL(gmm, score_metric='auc', verbose=0, random_state=random_state)

    gmm_raw = GMM(n_components=1, reg_covar=1e6, contamination=0.1, random_state=random_state) 
    gmm_raw.name = 'GMM_raw'
    ndm_raw = MODEL(gmm_raw, score_metric='auc', verbose=0, random_state=random_state)

    ndm.train(features_train)
    ndm_raw.train(features_test)

    result = gmm.decision_function(features_test)
    result_raw = gmm_raw.decision_function(features_test)

    return calculate_metrics(result, result_raw)


# AE
def predict_ae(features_train, features_test, random_state):
    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(features_train)
    scaled_features_test = scaler.transform(features_test)

    ae = AE(epochs=100, batch_size=32, random_state=random_state)
    ae.name = 'Autoencoder'
    ndm = MODEL(ae, score_metric='auc', verbose=0, random_state=random_state)

    ae_raw = AE(epochs=100, batch_size=32, random_state=random_state)
    ae_raw.name = 'Autoencoder_raw'
    ndm_raw = MODEL(ae_raw, score_metric='auc', verbose=0, random_state=random_state)

    ndm.train(scaled_features_train)
    ndm_raw.train(scaled_features_test)

    result = ae.decision_function(scaled_features_test)
    result_raw = ae_raw.decision_function(scaled_features_test)
    
    return calculate_metrics(result, result_raw)


# PCA
def predict_pca(features_train, features_test, random_state):
    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(features_train)
    scaled_features_test = scaler.transform(features_test)
    pca = PCA(n_components=1, contamination=0.1, random_state=random_state) 
    pca.name = 'PCA'
    ndm = MODEL(pca, score_metric='auc', verbose=0, random_state=random_state)

    pca_raw = PCA(n_components=1, contamination=0.1, random_state=random_state) 
    pca_raw.name = 'PCA_raw'
    ndm_raw = MODEL(pca_raw, score_metric='auc', verbose=0, random_state=random_state)

    ndm.train(scaled_features_train)
    ndm_raw.train(scaled_features_test)

    result = pca.decision_function(scaled_features_test)
    result_raw = pca_raw.decision_function(scaled_features_test)

    return calculate_metrics(result, result_raw)


# KDE
def predict_kde(features_train, features_test, random_state):
    scaler = StandardScaler()
    scaled_features_train = scaler.fit_transform(features_train)
    scaled_features_test = scaler.transform(features_test)
    kde = KDE(kernel='gaussian', bandwidth=1.0, contamination=0.1, random_state=random_state)
    kde.name = 'KDE'
    ndm = MODEL(kde, score_metric='auc', verbose=0, random_state=random_state)

    kde_raw = KDE(kernel='gaussian', bandwidth=1.0, contamination=0.1, random_state=random_state)
    kde_raw.name = 'KDE_raw'
    ndm_raw = MODEL(kde_raw, score_metric='auc', verbose=0, random_state=random_state)

    ndm.train(scaled_features_train)
    ndm_raw.train(scaled_features_test)

    result = kde.decision_function(scaled_features_test)
    result_raw = kde_raw.decision_function(scaled_features_test)

    return calculate_metrics(result, result_raw)


def calculate_metrics_0(result, result_raw):
    tn = ((result < 0) & (result_raw < 0)).sum()
    tp = ((result >= 0) & (result_raw >= 0)).sum()
    fn = ((result < 0) & (result_raw >= 0)).sum()
    fp = ((result >= 0) & (result_raw < 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("precision, recall, f1: "+ str((precision, recall, f1)))

    return precision, recall, f1


def calculate_metrics(result, result_raw):
    threshold = np.percentile(result_raw, 100 * (1 - 0.1)) # 1-contamination

    # Classify based on the threshold
    predictions = (result > threshold).astype(int)
    true_labels = (result_raw > threshold).astype(int)

    # Calculate TN, TP, FN, FP
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    tp = ((predictions == 1) & (true_labels == 1)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()


    # Calculate precision, recall, f1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("precision, recall, f1: "+ str((precision, recall, f1)))

    return precision, recall, f1


def predict_pcap(pcap_file, raw_file, feature_type, random_state, ndm):
    features_test, features_train = load_pcap_data(pcap_file, raw_file, feature_type, random_state)

    if ndm == "OCSVM":
        return predict_ocsvm(features_train, features_test, random_state)
    elif ndm == "IForest":
        return predict_iforest(features_train, features_test, random_state)
    elif ndm == "GMM":
        return predict_gmm(features_train, features_test, random_state)
    elif ndm == "AE":
        return predict_ae(features_train, features_test, random_state)
    elif ndm == "PCA":
        return predict_pca(features_train, features_test, random_state)
    elif ndm == "KDE":
        return predict_kde(features_train, features_test, random_state)
    else:
        raise ValueError(f"Unknown model type: {ndm}")

if __name__ == "__main__":
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    args = parse_args()
    print(args)

    if not args.input or not args.generator or len(args.input) != len(args.generator):
        raise ValueError("Number of inputs must match number of generators")

    # Generate base directory
    base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate raw PCAP first
    raw_file = base_dir / f"{args.dataset}_raw.pcap"
    if not raw_file.exists() and args.raw.endswith('.csv'):
        print(f"Generating raw PCAP file: {raw_file}")
        csv2pcap(args.raw, str(raw_file))

    # Process each generator
    for input_csv, generator_name in zip(args.input, args.generator):
        print(f"\nProcessing generator: {generator_name}")
        
        # Create generator directory
        generator_dir = base_dir / generator_name
        generator_dir.mkdir(parents=True, exist_ok=True)

        # Define pcap file paths
        if input_csv.endswith(".pcap"):
            pcap_file = input_csv
        else:
            pcap_file = str(generator_dir / f"{args.dataset}_{generator_name}.pcap")
            if not os.path.exists(pcap_file):
                print(f"Converting {generator_name} CSV to PCAP")
                csv2pcap(input_csv, pcap_file)

        # Determine which NDMs to use
        if args.ndm:
            if args.ndm not in ["OCSVM", "IForest", "GMM", "AE", "PCA", "KDE"]:
                raise ValueError(f"Invalid NDM specified: {args.ndm}. Must be one of: OCSVM, IForest, GMM, AE, PCA, KDE")
            ndms = [args.ndm]
        else:
            ndms = ["OCSVM", "IForest", "GMM", "AE", "PCA", "KDE"]

        # Process each NDM
        for ndm in ndms:
            print(f"Processing NDM: {ndm}")
            dfs = []
            
            for model in ["SAMP_NUM", "SAMP_SIZE", "IAT", "SIZE", "IAT_SIZE", "STATS"]:
                print(f"Model: {model}")
                precisions = []
                recalls = []
                f1s = []
                for i in tqdm(range(args.iters)):
                    precision, recall, f1 = predict_pcap(pcap_file, str(raw_file), model, i, ndm)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)
                dfs.append(
                    pd.DataFrame({
                        "model": [model] * args.iters,
                        "precision": precisions,
                        "recall": recalls,
                        "f1_score": f1s
                    })
                )
            
            df = pd.concat(dfs)
            # Save results for each NDM separately
            output_file = generator_dir / f"{args.dataset}_{generator_name}_{ndm}.csv"
            df.to_csv(output_file, index=False)

    print("Finished")