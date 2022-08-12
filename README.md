# HESplit
2-party Privacy-preserving Neural Network Training using Split Learning and Homomorphic Encryption (CKKS Scheme).

## Requirements
`python==3.9.7`  
`tenseal==0.3.10`  
`pytorch==1.8.1+cu102`  
`icecream==2.1.2`  
`h5py==3.7.0`  
`hydra-core==1.1.1`  

## Protocol Explanation
![protocol](./images/protocol.png)

## Code
### Structure
```
├── conf              # hold the config file
│   ├── config.yaml 
├── data  
├── images 
├── notebooks 
├── outputs
├── src  
|   ├── client.py
|   ├── server.py
└── weights
 ```

### Running
Specify the hyperparamters in `conf/config.yaml`.  
Run `python src/server.py`. The server will be started and waiting for the client.  
Then run `python src/client.py`. The training process will start.
After the training is done, the logs and output files will be saved in the directory `outputs/<year_month_day>/<output_dir>` where `output_dir` is defined in `conf/config.yaml`.