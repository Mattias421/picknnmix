from dora import hydra_main

@hydra_main(config_path="conf", config_name="config")
def main(cfg):
    # TODO: check if feats exists
    
    # TODO: check if km exists (within xp dir maybe?)

    # TODO: evaluate knn

    pass
