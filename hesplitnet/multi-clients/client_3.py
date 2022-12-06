from pathlib import Path
import logging
import socket

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

project_path = Path(__file__).parents[2].absolute()
log = logging.getLogger(__name__)


@hydra.main(config_path=project_path/"conf", config_name="config_multiclient")
def main(cfg : DictConfig) -> None:
    log.info('I am client 3')


if __name__ == "__main__":
    main()