import math
import pickle
import socket
from pathlib import Path
from typing import Union

from sockets import send_msg, recv_msg

import numpy as np
import tenseal as ts
import torch
from torch import Tensor
from icecream import ic
ic.configureOutput(includeContext=True)
from tenseal.enc_context import Context
from tenseal.tensors.ckkstensor import CKKSTensor
print(f'tenseal version: {ts.__version__}')
print(f'torch version: {torch.__version__}')

project_path = Path(__file__).parents[1].absolute()
print(f'project dir: {project_path}')


class ECGServer256:

    def __init__(self, init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.params = dict(
            W = checkpoint["linear.weight"],  # [5, 256] ([output dimension, hidden dimension])
            b = checkpoint["linear.bias"]  # [5]
        )
        self.grads = dict()
        self.cache = dict()

    def enc_linear(self, 
                   enc_x: CKKSTensor, 
                   W: Union[Tensor, CKKSTensor], 
                   b: Tensor,
                   batch_encrypted: bool,
                   batch_size: int):
        """
        The linear layer on homomorphic encrypted data
        Based on https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        """
        if batch_encrypted:
            enc_x.reshape_([1, 256])
        if type(W) is Tensor:
            y: CKKSTensor = enc_x.mm(W.T) + b
            print('forward with plaintext W')
        else: # W is a CKKS Tensor
            y: CKKSTensor = enc_x.mm(W.transpose()) + b
            print('forward with encrypted W')
        dydW = enc_x
        dydx = W
        return y, dydW, dydx

    def forward(self, 
                he_a: CKKSTensor,
                batch_encrypted: bool,
                batch_size: int) -> CKKSTensor:
        # a2 = a*W' + b
        he_a2, _, W = self.enc_linear(he_a, 
                                      self.params["W"],
                                      self.params["b"],
                                      batch_encrypted,
                                      batch_size)
        self.cache["da2da"] = W
        return he_a2

    def backward(self, 
                 dJda2: Tensor,
                 he_a: CKKSTensor, 
                 context: Context) -> CKKSTensor:
        """Calculate the gradients of the loss function w.r.t the bias
           and the weights of the server's linear layer
           Also calculate the gradients of the loss function w.r.t the 
           client's activation map (dJda)     

        Args:
            dJda2 (Tensor): the derivative of the loss function w.r.t the output
                            of the linear layer. shape: [batch_size, 5]
            he_a (CKKSTensor): the encrypted activation map received from the client.
                               
            context (Context): the tenseal context, used to encrypt the output

        Returns:
            dJda (CKKSTensor): the derivative of the loss function w.r.t the
                          activation map received from the client. 
                          This will be sent to the client so he can calculate
                          the gradients w.r.t the conv layers weights.
        """
        # calculate dJdb (b: the server's bias)
        self.grads["dJdb"] = dJda2.sum(0)  # sum accross all samples in a batch
        assert self.grads["dJdb"].shape == self.params["b"].shape, \
            "dJdb and b must have the same shape"

        # calculate dJdW (W: the server's weights)
        _dJda2 = dJda2.mean(dim=0).reshape(1,5)
        _he_a = he_a.transpose()
        self.grads["dJdW"] = (_he_a.mm(_dJda2)).transpose()
        assert self.grads["dJdW"].shape == list(self.params["W"].shape), \
            f"dJdW and W must have the same shape"
        
        # calculate dJda (a: the client's activation map)
        if type(self.cache["da2da"]) is Tensor:
            dJda: Tensor = torch.matmul(dJda2, self.cache["da2da"])
            # dJda = pickle.dumps(dJda.detach().to('cpu'))
            dJda: CKKSTensor = ts.ckks_tensor(context, dJda.tolist(), batch=True)
        else:  # it is CKKSTensor
            temp = self.cache["da2da"].transpose()
            print(f'type of W: {type(self.cache["da2da"])}')
            print(f'shape of W: {self.cache["da2da"].shape}')
            dJda: CKKSTensor = temp.mm(dJda2.T)
            dJda = dJda.transpose()

        print(f'dJda type: {type(dJda)}, dJda shape: {dJda.shape}')
        
        return dJda

    def clear_grad_and_cache(self):
        """Clear the cache dictionary and make all grads zeros for the 
           next forward pass on a new batch
        """
        self.grads = dict()
        self.cache = dict()

    def update_params(self, lr: float):
        """
        Update the parameters based on the gradients calculated in backward()
        """
        self.params["W"] = self.params["W"] - lr*self.grads["dJdW"]
        self.params["b"] = self.params["b"] - lr*self.grads["dJdb"]


class Server:
    def __init__(self) -> None:
        self.socket = None
        self.device = None
        self.ecg_model = None
        self.client_ctx = None
        self.connection = None

    def init_socket(self, host, port):
        """[summary]

        Args:
            host ([str]): [description]
            port ([int]): [description]
        """
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))  # associates the socket with its local address
        self.socket.listen()
        print('Listening on', (host, port))
        self.connection, addr = self.socket.accept()  # wait for the client to connect
        print(f'Connection: {self.connection} \nAddress: {addr}')

    def recv_ctx(self):
        client_ctx_bytes, _ = recv_msg(sock=self.connection)
        self.client_ctx: Context = Context.load(client_ctx_bytes)

    def build_model(self, 
                    init_weight_path: Union[str, Path]) -> None:
        """Build the neural network model for the server
        """
        print("building the model")
        self.ecg_model = ECGServer256(init_weight_path)

    def train(self, hyperparams: dict):
        seed = hyperparams["seed"]
        verbose = hyperparams["verbose"]
        lr = hyperparams["lr"]
        total_batch = hyperparams["total_batch"]
        epoch = hyperparams["epoch"]
        batch_encrypted = hyperparams["batch_encrypted"]
        batch_size = hyperparams["batch_size"]
        # set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        for e in range(epoch):
            print(f"---- Epoch {e+1} ----")
            self.training_loop(total_batch, verbose, lr, batch_encrypted, epoch, batch_size)
            train_status, _ = recv_msg(self.connection)
            print(pickle.loads(train_status))

    def training_loop(self, total_batch, verbose, lr, batch_encrypted, epoch, batch_size):
        epoch_communication = 0
        for i in range(total_batch):
            self.ecg_model.clear_grad_and_cache()
            he_a, recv_size1 = recv_msg(sock=self.connection)
            he_a = CKKSTensor.load(context=self.client_ctx,
                                   data=he_a)
            if verbose: print("\U0001F601 Received he_a from the client")
            if verbose: print("Forward pass ---")
            he_a2: CKKSTensor = self.ecg_model.forward(he_a, batch_encrypted, batch_size)
            if verbose: print("\U0001F601 Sending he_a2 to the client")
            send_size1 = send_msg(sock=self.connection, msg=he_a2.serialize())
            
            if verbose: print("Backward pass --- ")
            dJda2, recv_size2 = recv_msg(sock=self.connection)
            dJda2 = pickle.loads(dJda2)
            # self.ecg_model.check_update_grads(dJda2)
            dJda = self.ecg_model.backward(dJda2, he_a, 
                                           self.client_ctx)
            if verbose: print("\U0001F601 Sending dJda to the client")
            send_size2 = send_msg(sock=self.connection, msg=dJda.serialize())
            self.ecg_model.update_params(lr=lr) # updating the parameters
            # calculate communication overhead
            communication = recv_size1 + recv_size2 + send_size1 + send_size2
            epoch_communication += communication
            if verbose: print(f"Communication for batch {i}: {communication} (Mb)")

        print(f"Communication for epoch {epoch}: "
              f"{epoch_communication*1e-6 :.4f} (Tb)")


def main(hyperparams):
    # establish the connection with the client, send the hyperparameters
    server = Server()
    server.init_socket(host='localhost', port=10080)
    if hyperparams["verbose"]:
        print(f"Hyperparams: {hyperparams}")
        print("\U0001F601 Sending the hyperparameters to the Client")
    send_msg(sock=server.connection, msg=pickle.dumps(hyperparams))

    # receive the tenseal context from the client
    server.recv_ctx()
    if hyperparams["verbose"]:
        print("\U0001F601 Received the TenSeal context from the Client")

    # build and train the model
    server.build_model(project_path/'protocol1/weights/init_weight.pth')
    server.train(hyperparams)

    # save the model to .pth file
    # if hyperparams["save_model"]:
    #     torch.save(server.ecg_model.params, 
    #                './weights/trained_server_8192.pth')


if __name__ == "__main__":
    hyperparams = {
        'verbose': True,
        'batch_size': 2,
        'total_batch': math.ceil(13245/2),
        'epoch': 10,
        'lr': 0.001,
        'seed': 0,
        'batch_encrypted': True,
        'save_model': True
    }
    main(hyperparams)