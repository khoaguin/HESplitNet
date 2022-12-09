from pathlib import Path
import logging
import time
import socket
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
from torch.utils.data import DataLoader
import tenseal as ts

from hesplitnet.utils import MultiMITBIH
from hesplitnet.utils import send_msg, recv_msg


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
        try:
            self.socket.connect((host, port))  # connect to a remote [server] address,
        except socket.error as e:
            log.error(str(e))

    def load_dataset(self, dataset, batch_size):
        """
        Create the train and test dataloader
        """
        if dataset == 'MIT-BIH':
            train_path = project_path / 'data' / 'multiclient_mitbih_train.hdf5'
            test_path = project_path / 'data' / 'multiclient_mitbih_test.hdf5' 
            train_dataset = MultiMITBIH(train_path, test_path, client=2, train=True)
            test_dataset = MultiMITBIH(train_path, test_path, client=2, train=False)
        if dataset == 'PTB-XL':
            pass
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)



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
    client2 = Client2()
    client2.init_socket(host='localhost', port=int(cfg['port']))
    log.info('connected to the server')

    # load the dataset
    client2.load_dataset(dataset=cfg['dataset'], batch_size=2)

    welcome = client2.socket.recv(1024)
    print(welcome.decode('utf-8'))
    # send something to the server
    # while True:
    #     # Input = input("Let's send something to the server: ")
    #     # client.socket.send(str.encode(Input))
    #     client2.socket.send(str.encode('I am client 2'))
    #     time.sleep(2)
    #     response = client2.socket.recv(1024)
    #     print(response.decode("utf-8"))
    for i, batch in enumerate(client2.train_loader):
        x, y = batch
        send_msg(sock=client2.socket, msg=pickle.dumps(x))
        # response = client1.socket.recv(1024)
        # response, _ = recv_msg(client2.socket)
        # print(pickle.loads(response))
        # break
        print(i)


if __name__ == "__main__":
    main()