# script to train the pepo model, using a hydra config file, log the training process to wb and save the model to huggingface

import hydra
from omegaconf import DictConfig

# import set_seed from root/src/utils/general.py
from pepo.utils import set_seed


@hydra.main(
    config_path="../configs/train", config_name="pepo_base.yaml", version_base="1.1"
)
def main(cfg: DictConfig):
    print(cfg)
    set_seed(cfg.seed)


if __name__ == "__main__":
    main()
