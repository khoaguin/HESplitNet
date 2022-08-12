# HESplit
2-party Privacy-preserving Neural Network Training using Split Learning and Homomorphic Encryption (CKKS Scheme).

## Requirements
`tenseal==0.3.10`  
`pytorch==1.10.0+cu102`  
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
Go into the directory, then run `python src/server.py` and `python src/client.py`.  
The outputs will be in the directory `outputs/<year_month_day>/<output_dir>` where `output_dir` is defined in `conf/config.yaml`.