from pathlib import Path
import logging
import socket
import time
from _thread import *

import torch
import tenseal as ts

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


project_path = Path(__file__).parents[2].absolute()
log = logging.getLogger(__name__)


class Client1:
    """
    The class that represents the first client in the protocol.    
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
    client = Client1()
    client.init_socket(host='localhost', port=int(cfg['port']))
    log.info('connected to the server')
    time.sleep(3)

    # send something to the server
    while True:
        Input = input("Say something")
        client.socket.send(str.encode(Input))
        response = client.socket.recv(1024)
        print(response.decode("utf-8"))
    

if __name__ == "__main__":
    main()