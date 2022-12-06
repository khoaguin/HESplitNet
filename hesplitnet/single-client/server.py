import math
import pickle
import socket
from pathlib import Path
from typing import Union
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import tenseal as ts
import torch
from torch import Tensor
from tenseal.enc_context import Context
from tenseal.tensors.ckkstensor import CKKSTensor

from hesplitnet.utils import send_msg, recv_msg, set_random_seed
from hesplitnet.models import Server1DCNN

log = logging.getLogger(__name__)  # A logger for this file
project_path = Path(__file__).parents[2].absolute()


class Server:
    """The class that represents the server in the protocol
    """
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

    def recv_ctx(self):
        """Receive the context (in bytes) from the client 
        and load it into the tenseal context
        """
        client_ctx_bytes, _ = recv_msg(sock=self.connection)
        self.context: Context = Context.load(client_ctx_bytes)

    def build_model(self, 
                    init_weight_path: Union[str, Path]) -> None:
        """Build the neural network model for the server
        """
        self.model = Server1DCNN(init_weight_path)

    def train(self, hyperparams: dict):
        if hyperparams['dataset'] == 'MIT-BIH':
            total_batch = math.ceil(13245 / hyperparams["batch_size"])
        else:  # PTB-XL dataset
            total_batch = math.ceil(19267 / hyperparams["batch_size"])
        # set random seed for reproduceible results
        set_random_seed(hyperparams["seed"])

        log.info('---- Training ---- ')
        for e in range(hyperparams["epoch"]):
            log.info(f"---- Epoch {e+1} ----")
            self.training_loop(total_batch, hyperparams["verbose"], hyperparams["lr"],
                hyperparams["batch_encrypted"], hyperparams['dry_run'])
            train_status, _ = recv_msg(self.connection)
            log.info(pickle.loads(train_status))

    def training_loop(self, 
                    total_batch: int, 
                    verbose: bool, 
                    lr: float, 
                    batch_encrypted: bool,
                    dry_run: bool) -> None:
        """The client's training function for each epoch

        Args:
            total_batch (int): the total number of data batches for one epoch 
            verbose (bool): if True, then print out more info
            lr (float): the learning rate
            batch_encrypted (bool): if True, then homomorphically encrypt the
                data using batching 
            dry_run (bool): if True, only train on one batch of data
        """
        for i in range(total_batch):
            if verbose: print(f"Batch {i+1}")
            self.model.clear_grad_and_cache()
            enc_a, recv_size1 = recv_msg(sock=self.connection)
            enc_a = CKKSTensor.load(context=self.context,
                                    data=enc_a)
            if verbose: print(f"📨 Received enc_a of shape {enc_a.shape} "
                              f"and size {recv_size1} Mb from the client")
            enc_a_t, recv_size2 = recv_msg(sock=self.connection)
            enc_a_t = CKKSTensor.load(context=self.context,
                                      data=enc_a_t)
            if verbose: print(f"📨 Received enc_a_t of shape {enc_a_t.shape} "
                              f"and size {recv_size2} Mb from the client")

            if verbose: print("Forward pass ---")
            enc_a2: CKKSTensor = self.model.forward(enc_a)
            send_size1 = send_msg(sock=self.connection, msg=enc_a2.serialize())
            if verbose: print(f"📨 Sending enc_a2 of shape {enc_a2.shape} "
                              f"and size {send_size1} Mb to the client")

            if verbose: print("Backward pass --- ")
            dJda2, recv_size3 = recv_msg(sock=self.connection)
            dJda2: Tensor = pickle.loads(dJda2)
            if verbose: print(f"📨 Received dJda2 of shape {dJda2.shape} "
                              f"and size {recv_size3} Mb from the client")

            dJda: Tensor = self.model.backward(dJda2, enc_a_t)
            send_size2 = send_msg(sock=self.connection, msg=pickle.dumps(dJda))
            if verbose: print(f"📨 Sending dJda of shape {dJda.shape} "
                              f"and size {send_size2} Mb to the client")

            self.model.encrypt_weights(self.context, batch_encrypted)
            enc_Wt = self.model.update_params(lr=lr) # updating the parameters

            # send the encrypted weights to the client and 
            # get back the decrypted weights to reset noise
            send_size3 = send_msg(sock=self.connection, msg=enc_Wt.serialize())
            if verbose: print(f"📨 Sending encrypted W of shape {enc_Wt.shape} "
                              f"and size {send_size3} Mb to the client")
            Wt, recv_size4 = recv_msg(sock=self.connection)
            Wt = pickle.loads(Wt)
            if verbose: print(f"📨 Received decrypted W of shape {Wt.shape} "
                              f"and size {recv_size4} Mb from the client")
            self.model.set_weights(Wt.T)

            if dry_run: break


@hydra.main(config_path=project_path/"conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # log info to log files
    log.info(f'project path: {project_path}')
    log.info(f'tenseal version: {ts.__version__}')
    log.info(f'torch version: {torch.__version__}')
    output_dir = Path(HydraConfig.get().run.dir)
    log.info(f'output directory: {output_dir}')
    log.info(f'hyperparameters: \n{OmegaConf.to_yaml(cfg)}')
    
    # establish the connection with the client, send the hyperparameters
    server = Server()
    server.init_socket(host='localhost', port=int(cfg['port']))
    log.info('connected to the client')
    
    # the TenSeal HE context
    server.recv_ctx()
    log.info("📨 received the TenSeal context from the Client")
    
    # build and train the model
    if cfg['dataset'] == 'MIT-BIH':
        init_weight_path = project_path/'weights/init_weight_mitbih.pth'
    elif cfg['dataset'] == 'PTB-XL':
        init_weight_path = project_path/'weights/init_weight_ptbxl.pth'
    else:
        raise ValueError("dataset must be 'MIT-BIH' or 'PTB-XL'")
    server.build_model(init_weight_path)
    server.train(cfg)

    # save the model to .pth file
    if cfg["save_model"]:
        saved_params = {
            'W': server.model.params['W'],
            'b': server.model.params['b'],
        }
        torch.save(saved_params,
                   output_dir / 'trained_server.pth')


if __name__ == "__main__":
    main()