import hashlib
import json
import logging
import warnings
from pathlib import Path

import hydra
import numpy as np
import pyloudnorm as pyln
from numpy import ndarray
from omegaconf import DictConfig

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.audiogram import Audiogram, Listener
from clarity.utils.file_io import read_signal
from clarity.utils.flac_encoder import read_flac_signal
from clarity.utils.results_support import ResultsFile
from clarity.utils.signal_processing import compute_rms, resample

from recipes.cad_icassp_2024.baseline.evaluate import apply_ha

enhancer = NALR(nfir=220, sample_rate=44100)

# assume default listener

audiogram_freqs = [
        125,
        250,
        500,
        750,
        1000,
        1500,
        2000,
        3000,
        4000,
        6000,
        8000,
        10000,
        12000,
        14000,
        16000,
    ]

levels = np.array([5, 10, 15, 18, 19, 22, 25, 28, 31, 35, 38, 40, 40, 45, 50])
audiogram = Audiogram(levels=levels, frequencies=audiogram_freqs)

def do_haaqi(reference_mixture, enhanced_signal):

# Apply hearing aid to reference signals
        left_reference = apply_ha(
            enhancer=enhancer,
            compressor=None,
            signal=reference_mixture[:, 0],
            audiogram=audiogram,
            apply_compressor=False,
        )
        right_reference = apply_ha(
            enhancer=enhancer,
            compressor=None,
            signal=reference_mixture[:, 1],
            audiogram=audiogram,
            apply_compressor=False,
        )

        left_enhanced = apply_ha(
            enhancer=enhancer,
            compressor=None,
            signal=enhanced_signal[:, 0],
            audiogram=audiogram,
            apply_compressor=False,
        )
        right_enhanced = apply_ha(
            enhancer=enhancer,
            compressor=None,
            signal=enhanced_signal[:, 1],
            audiogram=audiogram,
            apply_compressor=False,
        )

        # Compute the scores
        left_score = compute_haaqi(
            processed_signal=resample(
                left_enhanced,
                44100,
                24000,
            ),
            reference_signal=resample(
                left_reference, 44100, 24000
            ),
            processed_sample_rate=24000,
            reference_sample_rate=24000,
            audiogram=audiogram,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 0])),
        )

        right_score = compute_haaqi(
            processed_signal=resample(
                right_enhanced,
                44100,
                24000,
            ),
            reference_signal=resample(
                right_reference, 44100, 24000
            ),
            processed_sample_rate=24000,
            reference_sample_rate=24000,
            audiogram=audiogram,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 1])),
        )

        return left_score, right_score