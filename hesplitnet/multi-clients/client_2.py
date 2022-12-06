from pathlib import Path
import logging
import socket

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import tenseal as ts

project_path = Path(__file__).parents[2].absolute()
log = logging.getLogger(__name__)

class Client2:
    """
    The class that represents the second client in the protocol.    
    """    
    def __init__(self) -> None:
        # paths to files and directories
        self.socket = None
        self.train_loader = None
        self.test_loader = None
        self.context = None
        self.ecg_model = None
        self.device = None

    def init_socket(self, host, port) -> None:
        """Connect to the server's socket 

        Args:
            host ([str]): [description]
            port ([int]): [description]
        """
        self.socket = socket.socket()
        self.socket.connect((host, port))  # connect to a remote [server] address,


@hydra.main(config_path=project_path/"conf", config_name="config_multiclient")
def main(cfg : DictConfig) -> None:
    # logging info
    log.info(f'project path: {project_path}')
    log.info(f'tenseal version: {ts.__version__}')
    log.info(f'torch version: {torch.__version__}')
    output_dir = Path(HydraConfig.get().run.dir)
    log.info(f'output directory: {output_dir}')
    log.info(f'hyperparameters: \n{OmegaConf.to_yaml(cfg)}')

    # establish the connection with the server
    client = Client2()
    client.init_socket(host='localhost', port=int(cfg['port']))
    log.info('connected to the server')


if __name__ == "__main__":
    main()