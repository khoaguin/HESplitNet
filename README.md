# HESplit
2-party Privacy-preserving Neural Network Training using Split Learning and Homomorphic Encryption (CKKS Scheme).

## Requirements
`python==3.9.7`  
`tenseal==0.3.10`  
`torch==1.8.1+cu102`  
`icecream==2.1.2`  
`h5py==3.7.0`  
`hydra-core==1.1.1`  
`pandas==1.5.2`  

<!-- ## Protocol
![protocol](./images/protocol.png) -->

## Repository Structure
```
├── conf              
│   ├── config.yaml  # hold the configurations (dataset to use, hyperparameters)
├── data  
│   ├── ptbxl_processing.ipynb  # code to process the PTB-XL dataset
├── images 
├── notebooks 
├── outputs
├── src  
|   ├── client.py  # code for the client
|   ├── server.py  # code for the server
└── weights
 ```
## Data
You can run the file `data/ptbxl_processing.ipynb` to produce the `.hdf5` files for the PTB-XL dataset, but it is recommended that you just download them [here](https://zenodo.org/record/7006692) and put the `.hdf5` files in the `data` folder.

## Running
- Specify the hyperparamters in `conf/config.yaml`.  
- Run `python src/server.py`. The server will be started and waiting for the client.  
- Then run `python src/client.py` in a new tab. The training process will start.  
- After the training is done, the logs and output files will be saved in the directory `outputs/<year_month_day>/<output_dir>` where `output_dir` is defined in `conf/config.yaml`.
- After training, go to `notebooks/test_mitbih.ipynb` or `notebooks/test_ptbxl.ipynb` to run and inspect the testing procedure for the MIT-BIH or PTB-XL dataset respectively.