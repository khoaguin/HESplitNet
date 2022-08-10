"""
Socket functions to transfer data
"""

import pickle
import socket
import struct
from pathlib import Path
import datetime
from typing import Dict
import json
import sys


def send_msg(sock, msg):
    '''
    Send the message in bytes, return the message's size in Mb
    '''
    # prefix each message with a 4-byte length in network byte order
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)
    
    return sys.getsizeof(msg) / 10**6

def recv_msg(sock):
    '''
    Receive the message and return it in bytes 
    as well as the size in Mb
    '''
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg_bytes = recvall(sock, msglen)
    recv_size = sys.getsizeof(msg_bytes) / 10**6
    
    return msg_bytes, recv_size

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data 

# def get_output_dir(prefix: Path):
#     """Get the output dir based on date time

#     Args:
#         prefix (Path): _description_

#     Returns:
#         _type_: _description_
#     """
#     today = datetime.datetime.now()
#     year = today.strftime("%Y")
#     month = today.strftime("%m")
#     day = today.strftime("%d")
#     hour = today.strftime("%H")
#     output = "outputs/" + year +"_" + month + "_" + day + "_" + hour
#     output_dir = prefix / output
    
#     return output_dir

def write_params(file_path: Path, 
            he_params: Dict, 
            hyperparams: Dict) -> None:
    """Write the parameters into a text file

    Args:
        output_dir (Path): _description_
        he_params (Dict): _description_
        hyperparams (Dict): _description_
    """
    with open(file_path, 'w') as f:
        f.write('HE parameters: ')
        f.write(json.dumps(he_params))
        f.write('\n')
        f.write('Neural net hyperparameters: ')
        f.write(json.dumps(hyperparams))

