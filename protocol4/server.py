import math
import pickle
import socket
from pathlib import Path
from typing import Tuple, Union, List
import sys

from matplotlib.style import context

from utils import send_msg, recv_msg

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

project_path = Path(__file__).parents[0].absolute()
print(f'project path: {project_path}')


class ECGServer256:

    def __init__(self, init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.params = dict(
            W = checkpoint["linear.weight"],  # [5, 256] ([output dimension, hidden dimension])
            b = checkpoint["linear.bias"]  # [5]
        )
        self.grads = dict()
        self.cache = dict()
    
    def set_weights(self, W):
        assert self.params['W'].shape == W.shape, "shapes do not match"
        self.params['W'] = W

    def forward(self, 
                enc_a: CKKSTensor) -> CKKSTensor:
        """The server's forward pass on encrypted data 
        Currently only have one linear layer

        Args:
            enc_a (CKKSTensor): the encrypted activation maps from the client

        Returns:
            CKKSTensor: the encrypted outputs
        """
        W, b = self.params['W'], self.params['b']
        enc_a2: CKKSTensor = enc_a.mm(W.T) + b
        self.cache["da2da"] = W  # save this for backward pass

        return enc_a2

    def backward(self, 
                 dJda2: Tensor,
                 enc_a_t: CKKSTensor) -> CKKSTensor:
        """Calculate the gradients of the loss function w.r.t the bias
           and the weights of the server's linear layer
           Also calculate the gradients of the loss function w.r.t the 
           client's activation map (dJda)     

        Args:
            dJda2 (Tensor): the derivative of the loss function w.r.t the output
                            of the linear layer. shape: [batch_size, 5]
            enc_a (CKKSTensor): the encrypted transpose activation map received 
                            from the client. shape: [1, batch_size]
                               
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

        # calculate the encrypted dJ/dWt (Wt: the server's weights transposed)
        enc_dJdWt = enc_a_t.mm(dJda2)
        self.grads['enc_dJdWt'] = enc_dJdWt

        # calculate dJda to send to the client
        # we have: dJ/da = dJ/da2 * da2/da
        #                = dJ/da2 * W
        dJda: Tensor = dJda2.matmul(self.cache['da2da'])
 
        return dJda

    def clear_grad_and_cache(self):
        """Clear the cache dictionary and make all grads zeros for the 
           next forward pass on a new batch
        """
        self.grads = dict()
        self.cache = dict()

    def encrypt_weights(self, context, batch_enc):
        """Encrypt the weights

        Args:
            context (_type_): _description_
            batch_encrypted (_type_): _description_
        """
        W = self.params['W']
        enc_Wt = ts.CKKSTensor(context, W.T, batch=batch_enc)
        enc_Wt.reshape_([1, enc_Wt.shape[0]])
        self.params['enc_Wt'] = enc_Wt

    def update_params(self, lr: float):
        """
        Update the parameters based on the gradients calculated in backward()
        Return the encrypted weights
        """
        self.params['enc_Wt'] = self.params['enc_Wt'] - lr * self.grads["enc_dJdWt"]
        self.params["b"] = self.params["b"] - lr * self.grads["dJdb"]
        
        return self.params['enc_Wt']

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
        print('Listening on', (host, port))
        self.connection, addr = self.socket.accept()  # wait for the client to connect
        print(f'Connection: {self.connection} \nAddress: {addr}')

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
        self.model = ECGServer256(init_weight_path)

    def train(self, hyperparams: dict):
        seed = hyperparams["seed"]
        verbose = hyperparams["verbose"]
        lr = hyperparams["lr"]
        total_batch = math.ceil(13245 / hyperparams["batch_size"])
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
        
        # encrypt the weights before training
        # self.model.encrypt_weights(self.context, batch_encrypted)  
        # self.model.encrypt_bias(self.context, batch_encrypted)
        for e in range(epoch):
            print(f"---- Epoch {e+1} ----")
            self.training_loop(total_batch, verbose, lr, batch_encrypted, epoch, batch_size)
            train_status, _ = recv_msg(self.connection)
            print(pickle.loads(train_status))

    def training_loop(self, 
                    total_batch: int, 
                    verbose: bool, 
                    lr: float, 
                    batch_encrypted: bool, 
                    epoch: int, 
                    batch_size: int) -> None:
        """The client's training function for each epoch

        Args:
            total_batch (int): _description_
            verbose (bool): _description_
            lr (float): _description_
            batch_encrypted (bool): _description_
            epoch (int): _description_
            batch_size (int): _description_
        """
        epoch_communication = 0
        for i in range(total_batch):
            if verbose: print(f"Batch {i+1}")
            self.model.clear_grad_and_cache()
            enc_a, recv_size1 = recv_msg(sock=self.connection)
            enc_a = CKKSTensor.load(context=self.context,
                                    data=enc_a)
            if verbose: print(f"\U0001F601 Received enc_a of shape {enc_a.shape} "
                              f"and size {recv_size1} Mb from the client")
            enc_a_t, recv_size2 = recv_msg(sock=self.connection)
            enc_a_t = CKKSTensor.load(context=self.context,
                                      data=enc_a_t)
            if verbose: print(f"\U0001F601 Received enc_a_t of shape {enc_a_t.shape} "
                              f"and size {recv_size2} Mb from the client")

            if verbose: print("Forward pass ---")
            enc_a2: CKKSTensor = self.model.forward(enc_a)
            send_size1 = send_msg(sock=self.connection, msg=enc_a2.serialize())
            if verbose: print(f"\U0001F601 Sending enc_a2 of shape {enc_a2.shape} "
                              f"and size {send_size1} Mb to the client")

            if verbose: print("Backward pass --- ")
            dJda2, recv_size3 = recv_msg(sock=self.connection)
            dJda2: Tensor = pickle.loads(dJda2)
            if verbose: print(f"\U0001F601 Received dJda2 of shape {dJda2.shape} "
                              f"and size {recv_size2} Mb from the client")

            dJda: Tensor = self.model.backward(dJda2, enc_a_t)
            send_size2 = send_msg(sock=self.connection, msg=pickle.dumps(dJda))
            if verbose: print(f"\U0001F601 Sending dJda of shape {dJda.shape} "
                              f"and size {send_size2} Mb to the client")

            self.model.encrypt_weights(self.context, batch_encrypted)
            enc_Wt = self.model.update_params(lr=lr) # updating the parameters

            # send the encrypted weights to the client and 
            # get back the decrypted weights to reset noise
            send_size3 = send_msg(sock=self.connection, msg=enc_Wt.serialize())
            if verbose: print(f"\U0001F601 Sending encrypted W of shape {enc_Wt.shape} "
                              f"and size {send_size3} Mb to the client")
            Wt, recv_size4 = recv_msg(sock=self.connection)
            Wt = pickle.loads(Wt)
            if verbose: print(f"\U0001F601 Received decrypted W of shape {Wt.shape} "
                              f"and size {recv_size4} Mb from the client")
            self.model.set_weights(Wt.T)

            # calculate communication overhead
            communication = recv_size1 + recv_size2 + recv_size3 + recv_size4 +\
                 send_size1 + send_size2 + send_size3
            if verbose: print(f"Communication for batch {i+1}: {communication} (Mb)\n")
            epoch_communication += communication

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

    # # build and train the model
    server.build_model(project_path/'weights/init_weight.pth')
    server.train(hyperparams)

    # save the model to .pth file
    output_dir = project_path / 'outputs' / hyperparams["output_dir"]
    if hyperparams["save_model"]:
        torch.save(server.model.params, 
                   output_dir / 'trained_server.pth')


if __name__ == "__main__":
    hyperparams = {
        'verbose': False,
        'batch_size': 16,
        # 'total_batch': math.ceil(13245/2),
        'epoch': 10,
        'lr': 0.001,
        'seed': 0,
        'batch_encrypted': True,
        'save_model': True,
        'debugging': False,
        'output_dir': 'Jul_15_8192_batch16'
    }
    main(hyperparams)