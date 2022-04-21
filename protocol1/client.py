import pickle
import socket
import time
from pathlib import Path
from typing import Union, Tuple, Dict

from sockets import send_msg, recv_msg

import h5py
import numpy as np
import tenseal as ts
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from icecream import ic

ic.configureOutput(includeContext=True)
from tenseal.enc_context import Context
from tenseal.tensors.ckkstensor import CKKSTensor
from tenseal.tensors.ckksvector import CKKSVector
from tenseal.tensors.plaintensor import PlainTensor
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset

project_path = Path(__file__).parents[1].absolute()
print(f'project dir: {project_path}')


class ECG(Dataset):
    """The class used by the client to load the dataset

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, train_name, test_name, train=True):
        if train:
            with h5py.File(train_name, 'r') as hdf:
                self.x = hdf['x_train'][:]
                self.y = hdf['y_train'][:]
        else:
            with h5py.File(test_name, 'r') as hdf:
                self.x = hdf['x_test'][:]
                self.y = hdf['y_test'][:]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), \
               torch.tensor(self.y[idx])


class EcgClient256(nn.Module):
    """The client's 1D CNN model

    Args:
        nn ([torch.Module]): [description]
    """
    def __init__(self, 
                 context: Context, 
                 init_weight_path: Union[str, Path]):
        super(EcgClient256, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, 
                               out_channels=16, 
                               kernel_size=7, 
                               padding=3,
                               stride=1)  # 128 x 16
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  # 64 x 16
        self.conv2 = nn.Conv1d(in_channels=16, 
                               out_channels=8, 
                               kernel_size=5, 
                               padding=2)  # 64 x 8
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2)  # 32 x 8 = 256
        
        self.load_init_weights(init_weight_path)
        self.context = context

    def load_init_weights(self, init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.conv1.weight.data = checkpoint["conv1.weight"]
        self.conv1.bias.data = checkpoint["conv1.bias"]
        self.conv2.weight.data = checkpoint["conv2.weight"]
        self.conv2.bias.data = checkpoint["conv2.bias"]

    def forward(self, x: Tensor, batch_encrypted: bool) -> Tuple[Tensor, CKKSTensor]:
        x = self.conv1(x)  
        x = self.relu1(x)
        x = self.pool1(x)  
        x = self.conv2(x) 
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 256)  # [batch_size, 256]
        if batch_encrypted:
            # if batch=True, then enc_x.shape = [256]
            enc_x: CKKSTensor = ts.ckks_tensor(self.context, x.tolist(), batch=True)
        else:
            enc_x: CKKSTensor = ts.ckks_tensor(self.context, x.tolist(), batch=False)

        return x, enc_x


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
        print(self.socket)
    
    def load_ecg_dataset(self, 
                         train_name: str, 
                         test_name: str,
                         batch_size: int) -> None:
        """[summary]

        Args:
            train_name (str): [description]
            test_name (str): [description]
            batch_size (int): [description]
        """
        train_dataset = ECG(train_name, test_name, train=True)
        test_dataset = ECG(train_name, test_name, train=False)
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
            print(f'Client device: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu') 
            print(f'Client device: {self.device}')
        self.ecg_model = EcgClient256(context=self.context, 
                                      init_weight_path=init_weight_path)
        self.ecg_model.to(self.device)

    def train(self, hyperparams: dict) -> None:
        seed = hyperparams["seed"]
        verbose = hyperparams["verbose"]
        lr = hyperparams["lr"]
        total_batch = hyperparams["total_batch"]
        epoch = hyperparams["epoch"]
        batch_encrypted = hyperparams['batch_encrypted']
        # set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        train_losses = list()
        train_accs = list()
        # test_losses = list()
        # test_accs = list()
        # best_test_acc = 0  # best test accuracy
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(self.ecg_model.parameters(), lr=lr)
        for e in range(epoch):
            print(f"---- Epoch {e+1} ----")
            start = time.time()
            e_train_loss, e_correct, e_samples = \
                self.training_loop(verbose, loss_func, optimizer, batch_encrypted)
            end = time.time()
            train_losses.append(e_train_loss / total_batch)
            train_accs.append(e_correct / e_samples)
            train_status = f"training loss: {train_losses[-1]:.4f}, "\
                           f"training acc: {train_accs[-1]*100:.2f}, "\
                           f"training time: {end-start:.2f}s"
            print(train_status)
            _ = send_msg(sock=self.socket, msg=pickle.dumps(train_status))
                
        return train_losses, train_accs

    def training_loop(self, verbose, loss_func, optimizer, batch_encrypted):
        epoch_train_loss = 0.0
        epoch_correct = 0
        epoch_total_samples = 0
        for i, batch in enumerate(self.train_loader):
            if verbose: print("Forward Pass ---")
            start = time.time()
            optimizer.zero_grad()
            x, y = batch  # get the input data and ground-truth output in the batch
            x, y = x.to(self.device), y.to(self.device)  # put to cuda or cpu
            a, he_a = self.ecg_model.forward(x, batch_encrypted)
            if verbose: print("\U0001F601 Sending he_a to the server")
            send_msg(sock=self.socket, msg=he_a.serialize())
            he_a2, _ = recv_msg(sock=self.socket)
            he_a2 = CKKSTensor.load(context=self.context,
                                            data=he_a2)
            if verbose: print("\U0001F601 Received he_a2 from the server")
            a2 = he_a2.decrypt().tolist() # the client decrypts he_a2
            a2 = torch.tensor(a2, requires_grad=True)
            a2 = a2.squeeze(dim=1).to(self.device)
            a2.retain_grad()
            y_hat: Tensor = F.softmax(a2, dim=1)  # apply softmax
            # the client calculates the training loss (J) and accuracy
            batch_loss: Tensor = loss_func(y_hat, y)
            epoch_train_loss += batch_loss.item()
            epoch_correct += torch.sum(y_hat.argmax(dim=1) == y).item()
            epoch_total_samples += len(y)
            if verbose: print(f'batch {i+1} loss: {batch_loss:.4f}')
            
            if verbose: print("Backward Pass ---")
            batch_loss.backward()
            dJda2: Tensor = a2.grad.clone().detach().to('cpu')
            if verbose: print("\U0001F601 Sending dJda2 to the server")
            send_msg(sock=self.socket, msg=pickle.dumps(dJda2))
            dJda, _ = recv_msg(sock=self.socket)
            if verbose: print("\U0001F601 Received dJda from the server")
            dJda = CKKSTensor.load(context=self.context, data=dJda)
            dJda = dJda.decrypt().tolist()
            # dJda, _ = recv_msg(sock=self.socket)
            # dJda = pickle.loads(dJda)
            dJda = torch.Tensor(dJda).to(self.device)
            print(f'dJda shape: {dJda.shape}')
            if dJda.shape != a.shape:
                dJda = dJda.sum(dim=0)
            a.backward(dJda)  # calculating the gradients w.r.t the conv layers
            optimizer.step()  # updating the parameters
            end = time.time()
            if verbose: print(f"training time for 1 batch: {end-start:.2f}s")

        return epoch_train_loss, epoch_correct, epoch_total_samples


def main():
    # establish the connection with the server
    client = Client()
    client.init_socket(host='localhost', port=10080)
    
    # receive the hyperparameters from the server
    hyperparams, _ = recv_msg(sock=client.socket)
    hyperparams = pickle.loads(hyperparams)
    if hyperparams["verbose"]:
        print("\U0001F601 Received the hyperparameters from the Server")
        print(hyperparams)

    # load the dataset
    client.load_ecg_dataset(train_name=project_path/"data/train_ecg.hdf5",
                            test_name=project_path/"data/test_ecg.hdf5",
                            batch_size=hyperparams["batch_size"])
    
    # construct the tenseal context to encrypt data homomorphically
    he_context: Dict = {
        "P": 8192,  # polynomial_modulus_degree
        "C": [40, 21, 21, 21, 21, 21, 21, 40],  # coeff_modulo_bit_sizes
        "Delta": pow(2, 21)  # the global scaling factor
    }
    client.make_tenseal_context(he_context)
    if hyperparams["debugging"]:
        send_sk = True
    client.send_context(send_secret_key=send_sk)  # only send the public context (private key dropped)
    if hyperparams["verbose"]:
        print(f"HE Context: {he_context}")
        print(f"\U0001F601 Sending the context to the server. Sending the secret key: {send_sk}")
    # build the model and start training
    client.build_model(project_path/'protocol1/weights/init_weight.pth')
    train_losses, train_accs = client.train(hyperparams)

    # # after the training is done, save the results and the trained models
    # df = pd.DataFrame({  # save model training process into csv file
    #         'train_losses': train_losses,
    #         'train_accs': train_accs,
    #     })
    # if hyperparams["save_model"]:
    #     df.to_csv(project_path/'protocol1/outputs/loss_and_acc.csv')
    #     torch.save(client.ecg_model.state_dict(), 
    #                project_path/'protocol1/weights/trained_client.pth')


if __name__ == "__main__":
    main()