# A Multi-View Graph Convolutional Network Framework Based on Adaptive Adjacency Matrix and Multi-Strategy Fusion Mechanism for Identifying Spatial Domains

![Model](https://github.com/Fuyh0628/STMGAMF/blob/master/Model.tif)

## Requirements

You'll need to install the following packages in order to run the codes.

* Python==3.8.10
* numpy==1.22.0
* pandas==1.4.4
* scipy==1.10.1
* stlearn==0.4.8
* pytorch==2.4.0
* torch_geometric==2.5.3
* torch_sparse==0.6.18
* torch_scatter==2.1.2
* matplotlib==3.7.5

## Usage

### Raw Data Preparation

Place the raw spatial transcriptomics data (e.g., DLPFC) in the folder ***data***.

### Data Preprocessing

Run ***STMGAMF/DLPFC_generate_data.py*** to preprocess the raw DLPFC data:

`python DLPFC_generate_data.py`

### Model Training and Testing

Run ***STMGAMF/DLPFC__test.py*** to train and test the STMGAMF model:

`python DLPFC__test.py`

### Results
 
The results, including predictions and evaluation metrics, are saved in the folder ***result***.


