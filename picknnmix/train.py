from dora import hydra_main
from .dump_mfcc import main as dump_mfcc
import os
import logging


def get_features(split, feat_dir, sample_rate, frame_length=25, frame_shift=10):
    if os.path.exists(feat_dir):
        logging.info(f"skip {feat_dir}")
        return
    else:
        logging.info(f"WARNING: feat dir doesn't exist: {feat_dir}")

@hydra_main(config_path="conf", config_name="config")
def main(cfg):
    # TODO: preprocess data

    # TODO: check if feats exists
    
    # TODO: check if km exists (within xp dir maybe?)

    # TODO: evaluate knn

    pass
