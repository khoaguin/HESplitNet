from typing import Union, Tuple
from pathlib import Path

import torch
import torch.nn as nn

import tenseal as ts
from tenseal.enc_context import Context
from tenseal.tensors.ckkstensor import CKKSTensor


class Client1DCNN(nn.Module):
    def __init__(self, 
                 context: Context, 
                 init_weight_path: Union[str, Path],
                 in_channels: int,
                 hidden_dim: int):
        """Initializing the layers, load the initial weights 
        and the context of the model

        Args:
            context (Context): the TenSeal context
            init_weight_path (Union[str, Path]): the initial weights of the model
            in_channels (int): the number of input channels
            hidden_dim (int): the hidden dimension (output dimension of the client model)
        """
        super(Client1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                               out_channels=16, 
                               kernel_size=7, 
                               padding=3,
                               stride=1)  
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)  
        self.conv2 = nn.Conv1d(in_channels=16, 
                               out_channels=8, 
                               kernel_size=5, 
                               padding=2)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2)  
        
        self.load_init_weights(init_weight_path)
        self.context = context
        self.hidden_dim = hidden_dim

    def load_init_weights(self, init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.conv1.weight.data = checkpoint["conv1.weight"]
        self.conv1.bias.data = checkpoint["conv1.bias"]
        self.conv2.weight.data = checkpoint["conv2.weight"]
        self.conv2.bias.data = checkpoint["conv2.bias"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  
        x = self.relu1(x)
        x = self.pool1(x)  
        x = self.conv2(x) 
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, self.hidden_dim)  # [batch_size, hidden_dim]
        
        return x

    def encrypt(self, a: torch.Tensor, batch_enc: bool) \
                -> Tuple[CKKSTensor, CKKSTensor]:
        """Encrypt the input tensor

        Args:
            a (Tensor): The plaintext activation maps
                from the forward function
            batch_enc (bool): if true, encrypt using batching 

        Returns:
            enc_a (CKKSTensor): the encrypted activation maps
            enc_a_t (CKKSTensor): the encrypted transpose activation maps
        """
        enc_a: CKKSTensor = ts.CKKSTensor(self.context, 
                        a.tolist(),
                        batch=batch_enc)
        enc_a.reshape_([1, enc_a.shape[0]])

        enc_a_t: CKKSTensor = ts.CKKSTensor(self.context,
                        a.T.tolist(),
                        batch=batch_enc)
        enc_a_t.reshape_([1, enc_a_t.shape[0]])

        return enc_a, enc_a_t


class Server1DCNN:
    """The 1D CNN model on the server side that has input activation map
    of 256 time steps
    """

    def __init__(self, init_weight_path: Union[str, Path]):
        checkpoint = torch.load(init_weight_path)
        self.params = dict(
            W = checkpoint["linear.weight"],  # [5, hidden dimension]
            b = checkpoint["linear.bias"]  # [5]
        )
        self.grads = dict()
        self.cache = dict()
    
    def set_weights(self, W: torch.Tensor):
        """Set the weights of the model

        Args:
            W (torch.Tensor): the updated weights
        """
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
                 dJda2: torch.Tensor,
                 enc_a_t: CKKSTensor) -> torch.Tensor:
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
            dJda (torch.Tensor): the deriv\ative of the loss function w.r.t the
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
        dJda: torch.Tensor = dJda2.matmul(self.cache['da2da'])
 
        return dJda

    def clear_grad_and_cache(self):
        """Clear the cache dictionary and make all grads zeros for the 
           next forward pass on a new batch
        """
        self.grads = dict()
        self.cache = dict()

    def encrypt_weights(self, context: Context, batch_enc: bool, noise):
        """Add some noise and then encrypt the weights

        Args:
            context (Context): The TenSeal context
            batch_encrypted (bool): If true, encrypt using batching
            noise (Tensor): The noise tensor
        """
        W = self.params['W']
        noisy_Wt = W.T + noise
        # enc_Wt = ts.CKKSTensor(context, W.T, batch=batch_enc)
        enc_Wt = ts.CKKSTensor(context, noisy_Wt, batch=batch_enc)
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