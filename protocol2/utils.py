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


def send_msg(sock, msg):
    '''
    Send the message in bytes, return the its size in Mb
    '''
    # prefix each message with a 4-byte length in network byte order
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)
    communication = len(msg) / 10**6
    return communication

def recv_msg(sock):
    '''
    Receive the message and return it in bytes 
    as well as the size in Mb
    '''
    # read message length and unpack it into an integer
    raw_msglen, _ = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg_bytes, communication = recvall(sock, msglen)
    return msg_bytes, communication

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    count = 0
    while len(data) < n:
        # print(n-len(data))
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
        count += len(data)
    communication = count / 10**6
    # print(f'amount of data received: {count / 10**6} Mb')
    return data, communication

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

