import math
import pickle
import socket
from pathlib import Path
from typing import Tuple, Union, List
import sys

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
    
    def weights(self):
        return self.params["W"]
    
    def set_weights(self, W):
        self.params["W"] = W

    def enc_linear(self, 
                   enc_a_t: CKKSTensor,
                   enc_W_t: CKKSTensor, 
                   b: Tensor,
                   batch_encrypted: bool) \
            -> Tuple[CKKSTensor, CKKSTensor, CKKSTensor]:
        """
        The linear layer on homomorphic encrypted data
        """
        enc_a2: CKKSTensor = (enc_a_t.transpose()).mm(enc_W_t) + b
        # y: CKKSTensor = enc_a_t.mm(W.transpose()) + b
        # dydW = enc_x
        # dydx = W
        return enc_a2

    def forward(self, 
                enc_a_t: CKKSTensor,
                context: Context,
                batch_encrypted: bool) -> CKKSTensor:
        enc_W_t = self.transpose_encrypt_weights(context, batch_encrypted)
        he_a2, _, W = self.enc_linear(enc_a_t, 
                                      enc_W_t,
                                      self.params["b"],
                                      batch_encrypted)
        self.cache["da2da"] = W
        return he_a2

    def backward(self, 
                 dJda2: Tensor,
                 he_a: CKKSTensor) -> CKKSTensor:
        """Calculate the gradients of the loss function w.r.t the bias
           and the weights of the server's linear layer
           Also calculate the gradients of the loss function w.r.t the 
           client's activation map (dJda)     

        Args:
            dJda2 (Tensor): the derivative of the loss function w.r.t the output
                            of the linear layer. shape: [batch_size, 5]
            he_a (CKKSTensor): the encrypted activation map received from the client.
                               
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
        
        # calculate dJda to send to the client
        # we have: dJ/da = dJ/da2 * da2/da
        # because we can't multiply a torch tensor with a CKKSTensor
        # but we can do the reverse way, we use the rule (dJ/da)' = (dJ/da2)' * (da2/da)'
        da2da_transpose = self.cache["da2da"].transpose()  # da2da is W
        dJda: CKKSTensor = da2da_transpose.mm(dJda2.T)
        dJda = dJda.transpose()
 
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
        self.params["W"] = self.params["W"] - lr * self.grads["dJdW"]
        self.params["b"] = self.params["b"] - lr * self.grads["dJdb"]

    def encrypt_weights(self, 
                    context: Context,
                    batch_enc: bool) -> None:
        self.params["W"] = ts.ckks_tensor(context=context, 
                                tensor=self.params["W"].tolist(), 
                                batch=batch_enc)
        if batch_enc:
            self.params["W"].reshape_([1, 256])

    def transpose_encrypt_weights(self, 
                    context: Context,
                    batch_enc: bool) -> None:
        W_t = self.params["W"].T
        enc_W_t = ts.ckks_tensor(context=context, 
                                tensor=W_t.tolist(), 
                                batch=batch_enc)
        enc_W_t.reshape_([1, enc_W_t.shape[0]])

        return enc_W_t

    def encrypt_bias(self, 
                context: Context,
                batch_enc: bool) -> None:
        self.params["b"] = ts.ckks_tensor(context=context, 
                                tensor=self.params["b"].tolist(), 
                                batch=batch_enc)
        if batch_enc:
            self.params["b"].reshape_([1])


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
            enc_a_t, recv_size1 = recv_msg(sock=self.connection)
            enc_a_t = CKKSTensor.load(context=self.context,
                                   data=enc_a_t)
            if verbose: print(f"\U0001F601 Received he_a_t of shape {enc_a_t.shape} from the client")
            
            if verbose: print("Forward pass ---")
            enc_a2: CKKSTensor = self.model.forward(enc_a_t, self.context, batch_encrypted)
            if verbose: print("\U0001F601 Sending he_a2 to the client")
            send_size1 = send_msg(sock=self.connection, msg=enc_a2.serialize())

            if verbose: print("Backward pass --- ")
            dJda2, recv_size2 = recv_msg(sock=self.connection)
            dJda2: Tensor = pickle.loads(dJda2)
            if verbose: print("\U0001F601 Received dJda2 from the client")

            dJda: CKKSTensor = self.model.backward(dJda2, enc_a_t)
            send_size2 = send_msg(sock=self.connection, msg=dJda.serialize())
            if verbose: print("\U0001F601 Sending dJda to the client")
            
            W: CKKSTensor = self.model.weights()
            send_size3 = send_msg(sock=self.connection, msg=W.serialize())
            if verbose: print("\U0001F601 Sending encrypted W to the client")
            W, recv_size3 = recv_msg(sock=self.connection)
            if verbose: print("\U0001F601 Received decrypted W from the client")
            W = pickle.loads(W)
            self.model.set_weights(W)
            self.model.encrypt_weights()

            self.model.update_params(lr=lr) # updating the parameters
            # calculate communication overhead
            communication = recv_size1 + recv_size2 + recv_size3 +\
                 send_size1 + send_size2 + send_size3
            epoch_communication += communication
            if verbose: print(f"Communication for batch {i+1}: {communication} (Mb)\n")

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
    server.build_model(project_path/'protocol2/weights/init_weight.pth')
    server.train(hyperparams)

    # save the model to .pth file
    output_dir = project_path / 'protocol2/outputs' / hyperparams["output_dir"]
    if hyperparams["save_model"]:
        torch.save(server.model.params, 
                   output_dir / 'trained_server.pth')


if __name__ == "__main__":
    hyperparams = {
        'verbose': True,
        'batch_size': 4,
        # 'total_batch': math.ceil(13245/2),
        'epoch': 10,
        'lr': 0.001,
        'seed': 0,
        'batch_encrypted': True,
        'save_model': True,
        'debugging': True,
        'output_dir': 'Jul_13_8192_batch1'
    }
    main(hyperparams)