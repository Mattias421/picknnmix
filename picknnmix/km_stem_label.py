# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import json

import numpy as np

import joblib
import torch
import tqdm

import matplotlib.pyplot as plt
import random

import librosa
import soundfile as sf

from recipes.cad_icassp_2024.baseline.evaluate import (
    apply_gains,
    remix_stems,
    load_reference_stems,
    level_normalisation
)
from haaqi_score import do_haaqi


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")

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
melstft_loss = auraloss.freq.MelSTFTLoss(sample_rate)
mrstft_loss = auraloss.freq.MultiResolutionSTFTLoss(
    scale="mel", 
    n_bins=64,
    sample_rate=sample_rate,
    device=device)
sadstft_loss = auraloss.freq.SumAndDifferenceSTFTLoss(
    fft_sizes=[1024, 2048, 8192],
    hop_sizes=[256, 512, 2048],
    win_lengths=[1024, 2048, 8192],
    perceptual_weighting=True,
    sample_rate=44100,
    scale="mel",
    n_bins=128,
)


def print_loss(target, source, hypothesis):
    base_mse = np.mean((source - target) ** 2)
    mse = np.mean((hypothesis - target) ** 2)
    # print(f'Base MSE: {base_mse}')
    # print(f'Enhanced MSE: {mse}')
    # print(f'MSE difference: {mse - base_mse}')

    target = torch.from_numpy(target).T.unsqueeze(0).to(dtype=torch.float32)
    source = torch.from_numpy(source).T.unsqueeze(0).to(dtype=torch.float32)
    hypothesis = torch.from_numpy(hypothesis).T.unsqueeze(0).to(dtype=torch.float32)
    # to gpu
    target = target.to(device)
    source = source.to(device)
    hypothesis = hypothesis.to(device)

    loss_stft = stft_loss(target, source)
    loss_melstft = melstft_loss(target, source)
    loss_mrstft = mrstft_loss(target, source)
    loss_sadstft = sadstft_loss(target, source)
    # print(f'\nBase')
    # print(f'STFT loss: {loss_stft}')
    # print(f'MelSTFT loss: {loss_melstft}')
    # print(f'MultiResolutionSTFT loss: {loss_mrstft}')
    # print(f'SumAndDifferenceSTFT loss: {loss_sadstft}')

    loss_stft_hyp = stft_loss(target, hypothesis)
    loss_melstft_hyp = melstft_loss(target, hypothesis)
    loss_mrstft_hyp = mrstft_loss(target, hypothesis)
    loss_sadstft_hyp = sadstft_loss(target, hypothesis)
    # print(f'\nEnhanced')
    # print(f'STFT loss: {loss_stft_hyp}')
    # print(f'MelSTFT loss: {loss_melstft_hyp}')
    # print(f'MultiResolutionSTFT loss: {loss_mrstft_hyp}')
    # print(f'SumAndDifferenceSTFT loss: {loss_sadstft_hyp}')

    # print(f'\nLoss difference')
    # print(f'STFT loss: {loss_stft_hyp - loss_stft}')
    # print(f'MelSTFT loss: {loss_melstft_hyp - loss_melstft}')
    # print(f'MultiResolutionSTFT loss: {loss_mrstft_hyp - loss_mrstft}')
    # print(f'SumAndDifferenceSTFT loss: {loss_sadstft_hyp - loss_sadstft}')

    return [base_mse, loss_stft.cpu(), loss_melstft.cpu(), loss_mrstft.cpu(), loss_sadstft.cpu()], [mse, loss_stft_hyp.cpu(), loss_melstft_hyp.cpu(), loss_mrstft_hyp.cpu(), loss_sadstft_hyp.cpu()]
                                        



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
        if isinstance(x, torch.Tensor):
            logger.info("warning: x is a torch tensor")
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            
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
    

def iterate_feat():
    feat = np.load(feat_path, mmap_mode="r")
    assert feat.shape[0] == (offsets[-1] + lengs[-1])
    for offset, leng in zip(offsets, lengs):
        yield feat[offset: offset + leng]

# TODO: need to do normalisation and stuff, maybe
def softmax(x):
    # Apply the log-sum-exp trick for stability
    max_val = np.max(x)
    exp_vals = np.exp(x - max_val)  # Subtract the maximum value for numerical stability
    softmax_vals = exp_vals / np.sum(exp_vals)
    return softmax_vals

def plot_weights(weights_list, i, labels):
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    plt.rcParams['lines.linewidth'] = 0.5

    colours = ['r', 'b', 'g', 'y']

    for j in range(4):
        axs[j].plot(weights_list[:, j], label=labels[j], c=colours[j])
        axs[j].legend()
        axs[j].set_ylabel('Weight')
        axs[j].set_xlabel('Frame')
        axs[j].set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(f'output/knn_weights_{i}.png')


stems = ['vocals', 'drums', 'bass', 'other'] # TODO: make sure this in the right order
km_path = '/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/km_models'

model = KNearestStem(stems, km_path, k=16, dist_type='cosine')
        
feat_path = '/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/feats/valid/mixture_0_1.npy'
leng_path = '/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/feats/valid/mixture_0_1.len'
manifest = '/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/manifest/valid/mixture.tsv'
scenes_root = '/mnt/parscratch/users/cadenza/data/cad_icassp_2024/audio/scene_music/valid'

with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

with open(manifest, "r") as f:
    root = f.readline().rstrip()
    print(root)
    lines = [line.rstrip() for line in f]

    gains_path = '/mnt/parscratch/users/cadenza/data/cad_icassp_2024/metadata/gains.json'

    # Load the JSON file
    with open(gains_path, 'r') as file:
        gains_data = json.load(file)

    gains_list = [value for value in gains_data.values()]

    def iterate_wav():
        for line in lines:
            path, leng = line.split('\t')
            stems_path = f'{root}{os.path.dirname(path)}'

            # choose random gain
            gains = random.choice(gains_list)

            # Load reference signals
            reference_stems, original_mixture = load_reference_stems(
                stems_path
            )
            reference_stems = apply_gains(
                reference_stems, 44100, gains
            )
            reference_mixture = remix_stems(
                reference_stems, original_mixture, 44100
            )

            yield original_mixture, reference_mixture, gains
                   
loss_array = []
loss_base_array = []
window = np.vstack((np.hamming(220), np.hamming(220))).T

for i, (features, (source, target, gains)) in enumerate(zip(iterate_feat(), iterate_wav())):

    # gains['mixture'] = 0 # TODO: firgure out if we can use mixture distance for anything

    logger.info(f'gains for {i}: {gains}')

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

    plot_weights(weights_list, i, stems)
    # plt.clf()
    # plt.plot(overall_gains_list)
    # plt.savefig(f'knn_gains_{i}.png')
    hypothesis = level_normalisation(hypothesis, source, 44100)

    base_losses, losses = print_loss(target, source, hypothesis)
    loss_array.append(losses)
    loss_base_array.append(base_losses)

    sf.write(f'output/knn_{i}.wav', hypothesis, 44100) # TODO: maybe apply level_normalisation from recipes.cad_icassp_2024.baseline.evaluate
    sf.write(f'output/knn_{i}_target.wav', target, 44100)
    sf.write(f'output/knn_{i}_source.wav', source, 44100)
    if i > 100:
        break

print(f'Average loss: {np.mean(loss_array, axis=0)}')
print(f'Average base loss: {np.mean(loss_base_array, axis=0)}')
print(f'Average loss difference: {np.mean(loss_array, axis=0) - np.mean(loss_base_array, axis=0)}')



    # print_loss(target, source, hypothesis)

    # left, right = do_haaqi(target, hypothesis)
    # loss = (left + right) / 2

    # left, right = do_haaqi(target, source)
    # base = (left + right) / 2
    # print(f'Base loss: {base}')
    # print(f'Enhanced loss: {loss}')
    # print(f'Loss difference: {loss - base}')



        

    # plt.clf()
    # for stem in stems:
    #     plt.plot(stem_predictions[stem], label=stem)
    # plt.legend()
    # plt.ylabel('Distance')
    # plt.xlabel('Frame')
    # plt.savefig(f'knn_{i}.png')
    

# # Given stem volumes in dB and respective weights
# stem_volumes = {'vocals': -3, 'bass': -5, 'drums': -4, 'other': -6}
# stem_weights = {'vocals': 0.5, 'bass': 0.3, 'drums': 0.4, 'other': 0.2}

# # Convert dB volumes to linear scale gains
# stem_gains = {stem: 10 ** (volume / 20) for stem, volume in stem_volumes.items()}

# # Calculate weighted sum of gains
# weighted_sum_gains = sum(stem_gains[stem] * weight for stem, weight in stem_weights.items())

# # Normalize the overall gain
# overall_gain = weighted_sum_gains / sum(stem_weights.values())
