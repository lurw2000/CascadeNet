#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from gensim.models import Word2Vec
from gensim.similarities.annoy import AnnoyIndexer

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, TensorDataset

# ====================================================
# Main Directories and Argument Parsing
# ====================================================
# Base directories (modify these if needed)
DATA_DIR = "../../data"
RESULTS_DIR = "../../result/e-wgan-gp"

# Parse dataset argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['caida', 'ca', 'dc', 'ton_iot'],
                    help="Dataset to use. Options: caida, ca, dc, ton_iot")
args = parser.parse_args()
dataset = args.dataset

# Define dataset-specific paths
dataset_path = os.path.join(DATA_DIR, dataset)
result_dir = os.path.join(RESULTS_DIR, dataset)
log_dir = os.path.join(result_dir, "log")

# Create directories if they don't exist
os.makedirs(result_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print(f"Using dataset: {dataset}")
print(f"Dataset path: {dataset_path}")
print(f"Result directory: {result_dir}")

# ====================================================
# Read Data
# ====================================================
if dataset == "ton_iot":
    df = pd.read_csv(os.path.join(dataset_path, "normal_1.csv")).iloc[:]
else:
    df = pd.read_csv(os.path.join(dataset_path, "raw.csv")).iloc[:]
display(df)

'''
The dataset contains network traffic data with fields such as:
srcip, dstip, srcport, dstport, proto, time, pkt_len, etc.
'''

# Preprocessing strategy:
# - Use Word2Vec for: srcip, dstip, srcport, dstport, ttl, pkt_len
# - One-hot encoding for: proto
# - Normalize time (MinMaxScaler)
# ====================================================
# Preprocessing
# ====================================================
postprocessed_file = os.path.join(result_dir, "postprocessed.pkl")
if os.path.exists(postprocessed_file):
    print("Found postprocessed.pkl. Using cached preprocessed data.")
    with open(postprocessed_file, 'rb') as f:
        postprocessed = pickle.load(f)
        data = postprocessed['data']
        w2v_model = postprocessed['w2v_model']
        one_hot_encoder = postprocessed['one_hot_encoder']
        scaler = postprocessed['scaler']
else:
    print("Preprocessing data...")
    # Prepare Word2Vec sentences for selected fields
    w2v_sentences = df[['srcip', 'dstip', 'srcport', 'dstport', 'ttl', 'pkt_len']].copy()
    for column, kind in [('srcip', 'ip'), ('dstip', 'ip'),
                         ('srcport', 'port'), ('dstport', 'port'),
                         ('ttl', 'ttl'), ('pkt_len', 'pkt_len')]:
        w2v_sentences[column] = kind + ' ' + w2v_sentences[column].astype(str)
    print("Example sentences:", w2v_sentences.values.tolist()[:5])
    
    print("Training Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=w2v_sentences.values.tolist(),
        vector_size=10, window=5, min_count=1, sg=1
    )
    print("Embedding sentences...")
    w2v_embedded = w2v_sentences.applymap(lambda kind_word: w2v_model.wv[kind_word])
    w2v_embedded = np.array(w2v_embedded.values.tolist()).reshape([-1, 6 * 10])
    print("Embedded shape:", w2v_embedded.shape)

    print("One-hot encoding for protocol...")
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = one_hot_encoder.fit_transform(df[['proto']]).astype('float32')

    print("Normalizing time...")
    scaler = MinMaxScaler()
    time_normalized = scaler.fit_transform(df[['time']]).astype('float32')

    print("Concatenating features...")
    data = torch.tensor(np.hstack([w2v_embedded, one_hot_encoded, time_normalized]), dtype=torch.float32)
    print("Data tensor shape:", data.shape)

    print("Saving preprocessed data to postprocessed.pkl")
    with open(postprocessed_file, 'wb') as f:
        pickle.dump({
            'data': data,
            'w2v_model': w2v_model,
            'one_hot_encoder': one_hot_encoder,
            'scaler': scaler
        }, f)
    
    del w2v_sentences, w2v_embedded, one_hot_encoded, time_normalized

# ====================================================
# Prepare Decoding Tools
# ====================================================
print("Splitting out vectors for different data kinds...")
w2v_models = {'ip': {}, 'port': {}, 'ttl': {}, 'pkt_len': {}}
for kind_word in w2v_model.wv.index_to_key:
    vec = w2v_model.wv[kind_word]
    kind, word = kind_word.split(' ')
    w2v_models[kind][word] = vec

print("Vocab sizes:", {kind: len(model) for kind, model in w2v_models.items()})
for kind in w2v_models:
    dic = w2v_models[kind]
    new_model = Word2Vec(vector_size=10, window=5, min_count=1, sg=1)
    new_model.build_vocab([list(dic.keys())])
    new_model.wv.vectors = np.array([dic[word] for word in new_model.wv.index_to_key])
    w2v_models[kind] = new_model

print("Building annoy indices...")
annoy_indices = {kind: AnnoyIndexer(w2v_model, 10) for kind, w2v_model in w2v_models.items()}

def int_to_ip(ip_int):
    octets = [(ip_int >> 24) & 0xFF, (ip_int >> 16) & 0xFF, (ip_int >> 8) & 0xFF, ip_int & 0xFF]
    return '.'.join(map(str, octets))

def postprocess(tensor):
    tensor = tensor.numpy()
    w2v_dim = w2v_model.vector_size
    start, end = 0, 0
    w2v_decoded = []
    kinds = ['ip', 'ip', 'port', 'port', 'ttl', 'pkt_len']
    for i in range(6):
        start = end
        end += w2v_dim
        feature_vecs = tensor[:, start:end]
        closest_words = [w2v_models[kind].wv.most_similar([vec], topn=1, indexer=annoy_indices[kind])[0][0]
                         for vec in feature_vecs]
        w2v_decoded.append(closest_words)
    
    srcip_decoded, dstip_decoded, srcport_decoded, dstport_decoded, ttl_decoded, pkt_len_decoded = w2v_decoded
    
    start = end
    end = start + len(one_hot_encoder.categories_[0])
    one_hot_encoded = tensor[:, start:end]
    proto_decoded = one_hot_encoder.inverse_transform(one_hot_encoded)
    
    start = end
    end = start + 1
    time_normalized = tensor[:, start:end]
    time_decoded = time_normalized
    
    df_decoded = pd.DataFrame({
        'srcip': srcip_decoded,
        'dstip': dstip_decoded,
        'srcport': srcport_decoded,
        'dstport': dstport_decoded,
        'ttl': ttl_decoded,
        'pkt_len': pkt_len_decoded,
        'proto': proto_decoded.flatten(),
        'time': time_decoded.flatten(),
    })
    return df_decoded

proto_dim = len(one_hot_encoder.categories_[0])
embed_dim = 60 + proto_dim + 1  # 60 for w2v features, proto_dim for protocol, 1 for time
test_tensor = torch.randn(1, embed_dim)
display(postprocess(test_tensor))

print("Sampled real data for comparison:")
display(postprocess(data[torch.randint(0, data.size(0), (20,))]))

import socket
def get_service(src_port, dst_port):
    service = []
    for port in [src_port, dst_port]:
        try:
            service.append(socket.getservbyport(port))
        except:
            pass
    return service

# ====================================================
# Training Settings and Model Definition
# ====================================================
print("Starting Training:")
proto_dim = len(one_hot_encoder.categories_[0])
embed_dim = 60 + proto_dim + 1  # 60 for w2v features, proto_dim for protocol, 1 for time
z_dim = 10
gp_factor = 10
batch_size = 256
lr = 1e-4
n_critic = 2
epochs = 25

dataset_tensor = TensorDataset(data)
dataloader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)
torch.cuda.empty_cache()

writer = SummaryWriter(log_dir)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )
    def forward(self, z):
        output = self.model(z)
        # Split the output:
        # - First 60 values for the Word2Vec features
        # - Next proto_dim values for protocol; use softmax for a proper probability distribution
        # - Last 1 value for time; use sigmoid to normalize it
        output = torch.cat([
            output[:, :60],
            F.softmax(output[:, 60:60+proto_dim], dim=1),
            torch.sigmoid(output[:, 60+proto_dim:60+proto_dim+1])
        ], dim=1)
        return output

D = Discriminator().cuda()
G = Generator().cuda()

G_path = os.path.join(result_dir, "G.pth")
D_path = os.path.join(result_dir, "D.pth")
if os.path.exists(G_path) and os.path.exists(D_path):
    print("Loading saved model weights...")
    G.load_state_dict(torch.load(G_path))
    D.load_state_dict(torch.load(D_path))
else:
    print("Training new weights from scratch.")

optimizer_D = optim.Adam(D.parameters(), lr=lr)
optimizer_G = optim.Adam(G.parameters(), lr=lr)

def gradient_penalty(D, real_data, fake_data):
    batch_size_local = real_data.size(0)
    alpha = torch.rand(batch_size_local, 1, 1, 1).cuda()
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    gradients = grad(outputs=d_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(d_interpolates.size()).cuda(),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

start_time = time.time()
for epoch in range(epochs):
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)
    for i, (pkgs,) in enumerate(dataloader):
        real_pkgs = pkgs.cuda()
        # Train Discriminator
        for _ in range(n_critic):
            optimizer_D.zero_grad()
            z = torch.randn(real_pkgs.size(0), z_dim).cuda()
            fake_pkgs = G(z).detach()
            real_validity = D(real_pkgs)
            fake_validity = D(fake_pkgs)
            gp = gradient_penalty(D, real_pkgs, fake_pkgs)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gp_factor * gp
            d_loss.backward()
            optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, z_dim).cuda()
        fake_pkgs = G(z)
        fake_validity = D(fake_pkgs)
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        optimizer_G.step()

        real_score = real_validity.mean().item()
        fake_score = fake_validity.mean().item()
        writer.add_scalars('score', {'real': real_score, 'fake': fake_score},
                           epoch * len(dataloader) + i)
        writer.add_scalar('gp', gp.item(), epoch * len(dataloader) + i)
        if i % 25 == 0:
            with torch.no_grad():
                z = torch.randn(1, z_dim).cuda()
                pkgs = G(z).cpu()
            pkgs = postprocess(pkgs)
            if i % 1000 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[real_score: {real_score:.4f}] [fake_score: {fake_score:.4f}] [gp: {gp.item():.4f}]")
                display(pkgs)
            pkgs = pkgs.iloc[0]
            service = get_service(int(pkgs['srcport']), int(pkgs['dstport']))
            writer.add_text('example', 
                f"{pkgs['srcip']} {pkgs['dstip']} {pkgs['srcport']} {pkgs['dstport']} "
                f"{service} ttl={pkgs['ttl']} pkt_len={pkgs['pkt_len']} proto={pkgs['proto']} time={pkgs['time']}",
                epoch * len(dataloader) + i
            )

training_time = time.time() - start_time
torch.save(G.state_dict(), G_path)
torch.save(D.state_dict(), D_path)

print("Generating new data:")
batch_size_gen = 8192
batch_num = 500
new_data = []
G.eval()
start_time = time.time()
for i in range(batch_num):
    print(f"Generating batch {i+1}/{batch_num}")
    with torch.no_grad():
        z = torch.randn(batch_size_gen, z_dim).cuda()
        pkgs = G(z).cpu()
    pkgs = postprocess(pkgs)
    new_data.append(pkgs)

new_data = pd.concat(new_data, ignore_index=True)
new_data['time'] = scaler.inverse_transform(new_data[['time']].astype('float64')).astype('int64')
generated_csv = os.path.join(result_dir, "syn.csv")
new_data.to_csv(generated_csv, index=False)

generating_time = time.time() - start_time
print("Dataset:", dataset, "Training time:", training_time, "Generating time:", generating_time)
