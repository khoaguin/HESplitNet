from pathlib import Path
import logging
import socket
import time
from _thread import start_new_thread

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import tenseal as ts


project_path = Path(__file__).parents[2].absolute()
log = logging.getLogger(__name__)


class Server:
    def __init__(self) -> None:
        self.socket = None
        self.device = None
        self.model = None
        self.context = None
        self.connection = None

    def init_socket(self, host, port):
        """Initialize the connection with the client

        Args:
            host ([str]): [description]
            port ([int]): [description]
        """
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))  # associates the socket with its local address
        self.socket.listen()
        self.connection, addr = self.socket.accept()  # wait for the client to connect
    
    def client_thread(self):
        self.connection.send(str.encode('welcome to the server'))
        while True:
            data = self.connection.recv(1024)
            reply = "Hello I am server" + data.decode("utf-8")
            if not data:
                break
            self.connection.sendall(str.encode(reply))
        self.connection.close()


@hydra.main(config_path=project_path/"conf", config_name="config_multiclient")
def main(cfg : DictConfig) -> None:
    # log info to log files
    log.info(f'project path: {project_path}')
    log.info(f'tenseal version: {ts.__version__}')
    log.info(f'torch version: {torch.__version__}')
    output_dir = Path(HydraConfig.get().run.dir)
    log.info(f'output directory: {output_dir}')
    log.info(f'hyperparameters: \n{OmegaConf.to_yaml(cfg)}')
    
    # establish the connection with the clients, send the hyperparameters
    server = Server()
    server.init_socket(host='localhost', port=int(cfg['port']))
    log.info('connected to the client')
    # client thread
    start_new_thread(server.client_thread, (client, ))

if __name__ == "__main__":
    main()