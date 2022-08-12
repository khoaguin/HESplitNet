import pickle
import socket
import time
from pathlib import Path
import math
import logging
from typing import Union, Tuple, Dict

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam

import tenseal as ts
from tenseal.enc_context import Context
from tenseal.tensors.ckkstensor import CKKSTensor

from utils import send_msg, recv_msg, ECGDataset, set_random_seed
from models import ClientCNN256

log = logging.getLogger(__name__)
project_path = Path(__file__).parents[1].absolute()


class Client:
    """
    The class that represents the client in the protocol.    
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
    
    def load_ecg_dataset(self, 
                         train_path: str, 
                         test_path: str,
                         batch_size: int) -> None:
        """[summary]

        Args:
            train_path (str): [description]
            test_path (str): [description]
            batch_size (int): [description]
        """
        train_dataset = ECGDataset(train_path, test_path, train=True)
        test_dataset = ECGDataset(train_path, test_path, train=False)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)

    def make_tenseal_context(self, 
                             he_context: Dict) -> Context:
        """Generate the TenSeal context to encrypt the activation maps

        Args:
            he_context (Dict): a dictionary that contains the
                HE params, namely poly_modulus_degree (int), 
                coeff_mod_bit_sizes (List[int]), and the scaling factor (int)

        Returns:
            Context: the TenSeal HE context object 
        """
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS, 
            poly_modulus_degree=he_context["P"],
            coeff_mod_bit_sizes=he_context["C"]
        )
        self.context.global_scale = he_context["Delta"]
        # self.context.generate_galois_keys()
    
    def send_context(self, send_secret_key) -> None:
        """
        Function used to send the context to the server
        Args:
            send_secret_key (bool): if True, send the secret key to the server (to debug)
        """
        _ = send_msg(sock=self.socket,
                     msg=self.context.serialize(save_secret_key=send_secret_key))

    def build_model(self, init_weight_path: Union[str, Path]) -> None:
        """Build the neural network model for the client

        Raises:
            TypeError: if the tenseal context needed to encrypt the activation 
                        map is None, then raise an error
        """
        if self.context == None:
            raise TypeError("Tenseal Context is None")
        if torch.cuda.is_available():
            self.device = torch.device('cuda') 
            log.info(f'Client device: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu') 
            log.info(f'Client device: {self.device}')
        self.ecg_model = ClientCNN256(context=self.context, 
                                      init_weight_path=init_weight_path)
        self.ecg_model.to(self.device)

    def train(self, hyperparams: dict) -> None:
        if hyperparams['dataset'] == 'MIT-BIH':
            total_batch = math.ceil(13245 / hyperparams["batch_size"])
        # set random seed
        set_random_seed(hyperparams['seed'])

        train_losses, train_accs = list(), list()
        train_comms, train_times = list(), list()
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(self.ecg_model.parameters(), lr=hyperparams['lr'])
        
        log.info('---- Training ----')
        for e in range(hyperparams["epoch"]):
            log.info(f"---- Epoch {e+1} ----")
            start = time.time()
            e_train_loss, e_correct, e_samples, e_comm = \
                self.training_loop(hyperparams["verbose"], loss_func, optimizer, 
                    hyperparams['batch_encrypted'], hyperparams['dry_run'])
            end = time.time()
            train_losses.append(e_train_loss / total_batch)
            train_accs.append(e_correct / e_samples)
            train_times.append(end-start)
            train_comms.append(e_comm)
            train_status = f"training loss: {train_losses[-1]:.4f}, "\
                           f"training acc: {train_accs[-1]*100:.2f}, "\
                           f"training time: {end-start:.2f}s, "\
                           f"training communication: {e_comm:.2f} Mb"
            log.info(train_status)
            send_msg(sock=self.socket, msg=pickle.dumps(train_status))

        return train_losses, train_accs, train_times, train_comms

    def training_loop(self, 
                    verbose: bool, 
                    loss_func, 
                    optimizer, 
                    batch_encrypted: bool,
                    dry_run: bool):
        """The server's training function for each epoch

        Args:
            verbose (bool): _description_
            loss_func (): a pytorch loss fucntion 
            optimizer (_type_): a pytorch optimizer
            batch_encrypted (_type_): _description_
            dry_run (bool): if true, only train on one batch of data for each epoch
        Returns:
            _type_: _description_
        """
        epoch_train_loss = 0.0
        epoch_correct = 0
        epoch_total_samples = 0
        epoch_communication = 0
        for i, batch in enumerate(self.train_loader):
            if verbose: print(f"Batch {i+1}\nForward Pass ---")
            start = time.time()
            optimizer.zero_grad()
            x, y = batch  # get the input data and ground-truth output in the batch
            x, y = x.to(self.device), y.to(self.device)  # put to cuda or cpu
            a = self.ecg_model.forward(x)
            enc_a, enc_a_t = self.ecg_model.encrypt(a, batch_encrypted)
            if verbose: print(f"📨 Sending enc_a of shape {enc_a.shape} to the server")
            send_size1 = send_msg(sock=self.socket, msg=enc_a.serialize())
            if verbose: print(f"📨 Sending enc_a_t of shape {enc_a_t.shape} to the server")
            send_size2 = send_msg(sock=self.socket, msg=enc_a_t.serialize())

            enc_a2, recv_size1 = recv_msg(sock=self.socket)
            enc_a2 = CKKSTensor.load(context=self.context,
                                     data=enc_a2)
            if verbose: print(f"📨 Received he_a2 of shape {enc_a2.shape} from the server")
            a2 = enc_a2.decrypt().tolist() # the client decrypts he_a2
            a2 = torch.tensor(a2, requires_grad=True)
            a2 = a2.squeeze(dim=1).to(self.device)
            a2.retain_grad()
            y_hat: Tensor = F.softmax(a2, dim=1)  # apply softmax
            # y_hat = torch.squeeze(y_hat, 1).type(torch.LongTensor)
            # the client calculates the training loss (J) and accuracy
            batch_loss: Tensor = loss_func(y_hat, y)
            epoch_train_loss += batch_loss.item()
            epoch_correct += torch.sum(y_hat.argmax(dim=1) == y).item()
            epoch_total_samples += len(y)
            
            if verbose: print("Backward Pass ---")
            batch_loss.backward()
            dJda2: Tensor = a2.grad.clone().detach().to('cpu')
            send_size3 = send_msg(sock=self.socket, msg=pickle.dumps(dJda2))
            if verbose: print(f"📨 Sending dJda2 of shape {dJda2.shape} to the server")
            
            dJda, recv_size2 = recv_msg(sock=self.socket)
            dJda = pickle.loads(dJda)
            if verbose: print(f"📨 Received dJda of shape {dJda.shape} from the server")
            dJda = dJda.to(self.device)
            assert dJda.shape == a.shape, "dJ/da and a have different shape"

            a.backward(dJda)  # calculating the gradients w.r.t the conv layers
            optimizer.step()  # updating the parameters

            server_Wt, recv_size3 = recv_msg(sock=self.socket)
            server_Wt = CKKSTensor.load(context=self.context, data=server_Wt)
            if verbose: print(f"📨 Received encrypted W of shape {server_Wt.shape} from the server")
            server_Wt = torch.tensor(server_Wt.decrypt().tolist()).squeeze()
            send_size4 = send_msg(sock=self.socket, msg=pickle.dumps(server_Wt))
            if verbose: print(f"📨 Send decrypted W of shape {server_Wt.shape} to the server")
            
            end = time.time()
            if verbose: print(f'Batch {i+1} loss: {batch_loss:.4f}')
            if verbose: print(f"Training time for batch {i+1}: {end-start:.2f}s\n")

            # calculate communication overhead
            communication = recv_size1 + recv_size2 + recv_size3 +\
                 send_size1 + send_size2 + send_size3 + send_size4
            if verbose: print(f"Communication for batch {i+1}: {communication:.2f} (Mb)\n")
            epoch_communication += communication

            if dry_run: break

        return epoch_train_loss, epoch_correct, epoch_total_samples, epoch_communication


# def main():
#     # establish the connection with the server
#     client = Client()
#     client.init_socket(host='localhost', port=1025)
    
#     # receive the hyperparameters from the server
#     hyperparams, _ = recv_msg(sock=client.socket)
#     hyperparams = pickle.loads(hyperparams)
#     # print("📨 Received the hyperparameters from the Server")
#     print(f'hyperparams: {hyperparams}')

#     # construct the tenseal context to encrypt data homomorphically
#     # he_context: Dict = {
#     #     "P": 8192,  # polynomial_modulus_degree
#     #     "C": [40, 21, 21, 21, 40],  # coeff_modulo_bit_sizes
#     #     "Delta": pow(2, 21)  # the global scaling factor
#     # }
#     he_context: Dict = {
#         "P": 16384,  # polynomial_modulus_degree
#         "C": [40, 21, 21, 21, 40],  # coeff_modulo_bit_sizes
#         "Delta": pow(2, 21)  # the global scaling factor
#     }
#     output_dir = project_path / 'outputs' / hyperparams["output_dir"]
#     output_dir.mkdir(parents=True, exist_ok=True)
#     write_params(output_dir/'params.txt', he_context, hyperparams)

#     client.make_tenseal_context(he_context)

#     send_sk = True if hyperparams["debugging"] else False 
#     client.send_context(send_secret_key=send_sk)  # only send the public context (private key dropped)
#     print(f"HE Context: {he_context}")
#     # print(f"📨 Sending the context to the server. Sending the secret key: {send_sk}")
    
#     # load the dataset
#     client.load_ecg_dataset(train_name=project_path/"data/train_ecg.hdf5",
#                             test_name=project_path/"data/test_ecg.hdf5",
#                             batch_size=hyperparams["batch_size"])
#     # build the model and start training
#     client.build_model(project_path/'weights/init_weight.pth')
#     train_losses, train_accs, train_times, train_comms = client.train(hyperparams)

#     # after the training is done, save the results and the trained models
#     if hyperparams["save_model"]:    
#         df = pd.DataFrame({  # save model training process into csv file
#             'train_losses': train_losses,
#             'train_accs': train_accs,
#             'train_times (s)': train_times,
#             'train_comms (Mb)': train_comms
#         })
#         df.to_csv(output_dir / 'train_results.csv')
#         torch.save(client.ecg_model.state_dict(), 
#                    output_dir / 'trained_client.pth')


@hydra.main(version_base=None, config_path=project_path/"conf", config_name="config")
def main(cfg : DictConfig) -> None:
    log.info(f'project path: {project_path}')
    log.info(f'tenseal version: {ts.__version__}')
    log.info(f'torch version: {torch.__version__}')
    output_dir = Path(HydraConfig.get().run.dir)
    log.info(f'output directory: {output_dir}')
    log.info(f'hyperparameters: \n{OmegaConf.to_yaml(cfg)}')
    # establish the connection with the server
    client = Client()
    client.init_socket(host='localhost', port=int(cfg['port']))
    log.info('connected to the server')
    # the TenSeal HE context
    client.make_tenseal_context(he_context=cfg['he'])
    send_sk = True if cfg['debugging'] else False 
    client.send_context(send_secret_key=send_sk)  # only send the public context (private key dropped)
    log.info(f"📨 sending the context to the server. Sending the secret key: {send_sk}")
    # load the dataset
    if cfg['dataset'] == 'MIT-BIH':
        train_path = project_path/'data/mitbih_train.hdf5'
        test_path = project_path/'data/mitbih_test.hdf5'
    client.load_ecg_dataset(train_path=train_path,
                            test_path=test_path,
                            batch_size=cfg['batch_size'])
    # build the model and start training
    client.build_model(project_path/'weights/init_weight.pth')
    train_losses, train_accs, train_times, train_comms = client.train(cfg)
    # after the training is done, save the results and the trained models
    if cfg["save_model"]:    
        df = pd.DataFrame({  # save model training process into csv file
            'train_losses': train_losses,
            'train_accs': train_accs,
            'train_times (s)': train_times,
            'train_comms (Mb)': train_comms
        })
        df.to_csv(output_dir / 'train_results.csv')
        torch.save(client.ecg_model.state_dict(), 
                   output_dir / 'trained_client.pth')


if __name__ == "__main__":
    main()
    # print('client')