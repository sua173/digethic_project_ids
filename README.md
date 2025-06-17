# Digethic Project IDS

This repository contains the implementation of an Intrusion Detection System (IDS) using both supervised (RandomForest) and unsupervised (OC-SVM) machine learning models. The CIC-IDS2017 dataset and network traffic captured via Docker containers are used for training and evaluation.

## Installation

### Python Version

Use Python 3.12.11:

```bash
pyenv install 3.12.11
cd ~/project
pyenv local 3.12.11
```

### Virtual Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python Dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Install `nfstream`

Installation via pip (version 6.5.4) may fail. Please build and install from source:

For platform-specific prerequisites, please refer to the [nfstream Installation Guide](https://www.nfstream.org/docs/#installation-guide)

```bash
mkdir tmp
cd tmp
git clone --recurse-submodules https://github.com/nfstream/nfstream.git
cd nfstream
python3 -m pip install -r dev_requirements.txt
python3 -m pip install .
```

## Data Preparation

Instructions are provided in: [`data/README.md`](data/README.md)

## IDS Models

### Random Forest

To train and evaluate the RandomForest model, either run all cells in jupyter notebook [`notebooks/RandomForest.ipynb`](notebooks/RandomForest.ipynb) or run the python script:

```bash
python src/RandomForest.py
```

### One-Class SVM (OC-SVM)

**Important:** Run the RandomForest model first. It generates feature importances that are saved to `models/rf/importance_df.pkl`.

Then, either run all cells in jupyter notebook [`notebooks/OC-SVM.ipynb`](notebooks/OC-SVM.ipynb)  or run the python script:

```bash
python src/OC-SVM.py
```

### Output

The output files (in HTML and PDF) from `RandomForest.ipynb` and `OC-SVM.ipynb` are saved to the `output/` directory.

## Docker Environment for Generating Evaluation traffic data

Docker containers are used to generate synthetic network traffic for IDS evaluation, including both normal and attack scenarios.
The necessary configuration files for building and running the containers are located in the [`docker/ids`](docker/ids/) directory.

## Dataset License

This project uses the CIC-IDS2017 dataset, which is publicly available from the Canadian Institute for Cybersecurity (CIC).  
For dataset access and license terms, please refer to:  
[CIC-IDS2017 Dataset Website](https://www.unb.ca/cic/datasets/ids-2017.html)

The dataset is used solely for academic and research purposes in accordance with the terms provided by the dataset authors.