# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

import joblib
import torch

import random

# set seed
np.random.seed(0)
random.seed(0)

# load loss functions
import auraloss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stft_loss = auraloss.freq.STFTLoss(
    w_log_mag=1.0, 
    w_lin_mag=1.0, 
    w_sc=0.0,
    device=device
)
sample_rate = 44100

class KmeansDistance(object):
    def __init__(self, km_path, k=5, dist_type="cosine"):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.k = k
        self.dist_type = dist_type

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if self.dist_type == "cosine":
            # Normalize x
            x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)

            # Normalize centroids
            C_norm = self.C_np / np.linalg.norm(self.C_np, axis=0, keepdims=True)

            # Calculate cosine similarity (dot product)
            similarity = np.dot(x_norm, C_norm)

            # Convert similarity to distances (subtract from 1 for cosine distance)
            dist = 1 - similarity

            # Find indices of top k nearest neighbors
            k_nearest_indices = np.argsort(dist, axis=1)[:, :self.k]

            # Extract distances of top k nearest neighbors
            k_nearest_distances = np.take_along_axis(dist, k_nearest_indices, axis=1)

            # Calculate the average distance of the top k nearest neighbors
            avg_distances = np.mean(k_nearest_distances, axis=1)
            
            return avg_distances


        elif self.dist_type == "euclidean":
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )

            # Find indices of top k nearest neighbors
            k_nearest_indices = np.argsort(dist, axis=1)[:, :self.k]

            # Extract distances of top k nearest neighbors
            k_nearest_distances = np.take_along_axis(dist, k_nearest_indices, axis=1)

            # Calculate the average distance of the top k nearest neighbors
            avg_distances = np.mean(k_nearest_distances, axis=1)
            
            return avg_distances


        

class KNearestStem(object):
    def __init__(self, stems, km_path, k=5, dist_type="cosine"):
        self.stem_predictors = {}
        for stem in stems:
            self.stem_predictors[stem] = KmeansDistance(f'{km_path}/{stem}', k, dist_type) # TODO: maybe don't use dicts

    def __call__(self, feat_list):
        stem_dist = {}
        for stem, predictor in self.stem_predictors.items():
            stem_dist[stem] = predictor(feat_list)
            # print(f'Average distance for {stem}: {np.mean(stem_dist[stem])}')
        return stem_dist
    


stems = ['vocals', 'drums', 'bass', 'other'] # TODO: make sure this in the right order
km_path = '/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/km_models'

model = KNearestStem(stems, km_path, k=16, dist_type='cosine')
        
feat_path = '/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/feats/test/mixture_0_1.npy'
leng_path = '/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/feats/test/mixture_0_1.len'
manifest = '/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/manifest/test/mixture.tsv'
scenes_root = '/mnt/parscratch/users/cadenza/data/cad_icassp_2024/audio/scene_music/valid'

def build_lookup_table(tsv_path):
    lookup_table = {}
    root_dir = None
    
    with open(tsv_path, 'r') as file:
        for index, line in enumerate(file):
            if index == 0:  # First line is the root directory
                root_dir = line.strip()
                continue
            
            relative_path, _ = line.strip().split('\t')
            full_path = f"{root_dir}{relative_path}"
            lookup_table[full_path] = index - 1  # Subtract 1 to exclude the root line index
    
    return lookup_table

lookup_table = build_lookup_table(manifest)

with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

all_feats = np.load(feat_path, mmap_mode="r")

def path_to_feat(path):
    # turn path into feature

    i = lookup_table[str(path)]

    leng = lengs[i]
    i_start = offsets[i]
    i_end = i_start + leng

    feat = all_feats[i_start:i_end]

    return feat

def produce_hypothesis(source, features, gains):
    # gains['mixture'] = 0 # TODO: firgure out if we can use mixture distance for anything

    stem_predictions = model(features)

    hypothesis = source.copy()

    n_frames = len(stem_predictions['vocals'])

    weights_list = np.zeros((n_frames, 4))
    overall_gains_list = np.zeros(n_frames)

    for frame_n in range(n_frames):
        stem_weights = {}
        for stem in stems:
            stem_weights[stem] = stem_predictions[stem][frame_n]

        # Convert to numpy array
        weights_array = np.array(list(stem_weights.values()))

        # Calculate softmax using the softmax function
        weights = 1- weights_array # weight is bigger when distance is smaller
        # print(np.std(weights))
        weights = weights # * np.std(weights) # high std means more confidence

        weights_list[frame_n] = weights
        
        # Convert dB volumes to linear scale gains
        stem_gains = [10 ** (volume / 20) for volume in gains.values()]

        # Calculate weighted sum of gains
        weighted_sum_gains = np.dot(weights, stem_gains)

        # Normalize the overall gain
        overall_gain = weighted_sum_gains / sum(stem_weights.values())

        overall_gains_list[frame_n] = overall_gain

        # print(f'Overall gain: {overall_gain}')

        hypothesis[frame_n*44:((frame_n*44) + 44)] = hypothesis[frame_n*44:((frame_n*44) + 44)] * overall_gain #*  window # assume 220 samples per frame, shifting by 44

    return hypothesis

def do_kmeans(source, path, gains):
    features = path_to_feat(path)

    hypothesis = produce_hypothesis(source, features, gains)

    return hypothesis