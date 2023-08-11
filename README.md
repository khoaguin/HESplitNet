# HESplitNet
Two-party Privacy-preserving Neural Network Training using Split Learning and Homomorphic Encryption in CKKS Scheme.

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
├── images  # images to be used in notebooks and README.md
├── notebooks  # contains jupyter notebooks 
├── outputs  # will be automatically created after running the protocols
├── hesplitnet  # contains the code
|   ├── multi-clients  # code for multi clients protocol
|   ├── single-client  # code for the single client protocol
└── weights  # contains the initial and trained weights
 ```
## Data
The processed [MIT-BIH dataset](https://physionet.org/content/mitdb/1.0.0/) for the one-client protocol (`mitbih_train.hdf5` and `mitbih_test.hdf5`) and multi-client protocol (`multiclient_mitbih_train.hdf5` and `multiclient_mitbih_test.hdf5`) are in the `data/` folder.

For the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.0/), you can run the file `data/ptbxl_processing.ipynb` to produce the `.hdf5` files, but it is recommended that you just download them [here](https://zenodo.org/record/7006692) and put the `.hdf5` files in the `data` folder.

## Running
1. Create a new conda environment, for example `conda create -n hesplitnet python=3.9.7` and activate it using `conda activate hesplitnet`
2. Install the required packages in the `requirements.txt`, e.g. `pip install -r requirements.txt`
3. Install `hesplitnet` as a package using `pip install -e .`
4. Specify the hyperparamters for your protocol in `conf/config.yaml`  
5. Running the protocol:
    - In the terminal, run `python hesplitnet/single-client/server.py`. The server will be started and waiting for the client.  
    - Open a new tab in the terminal and run `python hesplitnet/single-client/client.py`. The training process will start.
    - After the training is done, the logs and output files will be saved in the directory `outputs/<year_month_day>/<output_dir>` where `output_dir` is defined in `conf/config.yaml`.
    - After training, go to `notebooks/test_mitbih.ipynb` or `notebooks/test_ptbxl.ipynb` to run and inspect the testing procedure for the MIT-BIH or PTB-XL dataset respectively.
