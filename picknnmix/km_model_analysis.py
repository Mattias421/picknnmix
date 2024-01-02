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

from sklearn.manifold import TSNE


class KmeansDistance(object):
    def __init__(self, km_path, k=5, dist_type="cosine"):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.stem = os.path.basename(km_path)

        self.k = k
        self.dist_type = dist_type

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def plot(self):
        cluster_centers = self.C_np.T

        # Apply t-SNE to reduce dimensionality to 2D
        tsne = TSNE(n_components=2, random_state=42)
        cluster_centers_2d = tsne.fit_transform(cluster_centers)

        # Plotting clusters in 2D
        if self.stem == 'all':
            print(cluster_centers.shape)
            print(cluster_centers_2d.shape)
            plt.clf()
            plt.scatter(cluster_centers_2d[:128, 0], cluster_centers_2d[:128, 1], c='r', label='vocals')
            plt.scatter(cluster_centers_2d[128:256, 0], cluster_centers_2d[128:256, 1], c='b', label='drums')
            plt.scatter(cluster_centers_2d[256:384, 0], cluster_centers_2d[256:384, 1], c='g', label='bass')
            plt.scatter(cluster_centers_2d[384:512, 0], cluster_centers_2d[384:512, 1], c='y', label='other')
            plt.scatter(cluster_centers_2d[512:, 0], cluster_centers_2d[512:, 1], c='k', label='mixture')
            plt.title('t-SNE Visualization of all stems')
            plt.legend()

        else:
            plt.clf()
            plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], cmap='viridis')
            plt.title('t-SNE Visualization of Clusters')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.savefig(f'tsne_{self.stem}.png')


stems = ['vocals', 'drums', 'bass', 'other'] # TODO: make sure this in the right order
km_path = '/mnt/parscratch/users/cadenza/cadenza_friendz/clarity/recipes/cad_icassp_2024/baseline/k_means/km_models'

vocals = KmeansDistance(os.path.join(km_path, 'vocals'))
drums = KmeansDistance(os.path.join(km_path, 'drums'))
bass = KmeansDistance(os.path.join(km_path, 'bass'))
other = KmeansDistance(os.path.join(km_path, 'other'))
mixture = KmeansDistance(os.path.join(km_path, 'mixture'))
all = KmeansDistance(os.path.join(km_path, 'vocals'))

# concatinate centroids
all.stem = 'all'
print(vocals.C_np.shape)
all.C_np = np.concatenate((vocals.C_np, drums.C_np, bass.C_np, other.C_np, mixture.C_np), axis=1)
print(all.C_np.shape)

vocals.plot()
drums.plot()
bass.plot()
other.plot()
mixture.plot()
all.plot()

# print(f'mean of centroids:')
# print(f'vocals: {vocals.C_np.mean(axis=1)}')
# print(f'drums: {drums.C_np.mean(axis=1)}')
# print(f'bass: {bass.C_np.mean(axis=1)}')
# print(f'other: {other.C_np.mean(axis=1)}')

# print(f'std of centroids:')
# print(f'vocals: {vocals.C_np.std(axis=1)}')
# print(f'drums: {drums.C_np.std(axis=1)}')
# print(f'bass: {bass.C_np.std(axis=1)}')
# print(f'other: {other.C_np.std(axis=1)}')

# print(f'max of centroids:')
# print(f'vocals: {vocals.C_np.max(axis=1)}')
# print(f'drums: {drums.C_np.max(axis=1)}')
# print(f'bass: {bass.C_np.max(axis=1)}')
# print(f'other: {other.C_np.max(axis=1)}')

# print(f'min of centroids:')
# print(f'vocals: {vocals.C_np.min(axis=1)}')
# print(f'drums: {drums.C_np.min(axis=1)}')
# print(f'bass: {bass.C_np.min(axis=1)}')
# print(f'other: {other.C_np.min(axis=1)}')

# print(f'range of centroids:')
# print(f'vocals: {vocals.C_np.max(axis=1) - vocals.C_np.min(axis=1)}')
# print(f'drums: {drums.C_np.max(axis=1) - drums.C_np.min(axis=1)}')
# print(f'bass: {bass.C_np.max(axis=1) - bass.C_np.min(axis=1)}')
# print(f'other: {other.C_np.max(axis=1) - other.C_np.min(axis=1)}')



