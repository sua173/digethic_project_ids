# Data Preparation

There are two ways to prepare a dataset for model training and evaluation: you can either use pickle files of DataFrames or generate labeled flow files from PCAP files.

## Using Pickle Files

To run the model scripts (RandomForest/OC-SVM), you will need two DataFrame pickle files.  
The file `tcpdump_dataframe.pkl.zip` is available directly in this folder—please unzip it.  
The second file is too large to be stored here. Please download it from the separately provided URL and unzip it into the `data/` folder.

Afterward, the `data/` folder should contain the following:

```
data/
├── cic_dataframe.pkl (1.8GB)
└── tcpdump_dataframe.pkl (46MB)
```

## Extracting from PCAP

Alternatively, you can generate all flow CSV files from the PCAP files and then convert them into DataFrames.

The tcpdump PCAP files are zipped in `data/tcpdump/pcap/tcpdump-pcap.zip`. Please unzip it.  
The CIC-IDS2017 PCAP files must be downloaded from:  
http://cicresearch.ca/CICDataset/CIC-IDS-2017/

You will need the following five PCAP files from:  
`/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/PCAPs`

- Monday-WorkingHours.pcap
- Tuesday-WorkingHours.pcap
- Wednesday-workingHours.pcap
- Thursday-WorkingHours.pcap
- Friday-WorkingHours.pcap

After downloading and extracting, the `data/` directory structure should look like this:

```
data/
├── CIC/
│   ├── pcap/
│   │   ├── Monday-WorkingHours.pcap (10GB)
│   │   ├── Tuesday-WorkingHours.pcap (10GB)
│   │   ├── Wednesday-workingHours.pcap (12GB)
│   │   ├── Thursday-WorkingHours.pcap (7.7GB)
│   │   └── Friday-WorkingHours.pcap (8.2GB)
│   └── nfstream/
└── tcpdump/
    ├── pcap/
    │   ├── normal_01.pcap (28MB)
    │   ├── normal_02.pcap (57MB)
    │   ├── normal_and_attack_01.pcap (42MB)
    │   ├── normal_and_attack_02.pcap (47MB)
    │   ├── normal_and_attack_03.pcap (38MB)
    │   ├── normal_and_attack_04.pcap (19MB)
    │   └── normal_and_attack_05.pcap (34MB)
    └── nfstream/
```

Then, either run all cells in the Jupyter notebook [`notebooks/Prepare_Dataset.ipynb`](../notebooks/Prepare_Dataset.ipynb) or run the Python script directly:

```bash
python src/Prepare_Dataset.py
```
to generate the flow CSV files and the labeled files in the `nfstream/` subfolders. Once completed, you can proceed with running the model scripts.


## Final Folder Structure

```
data/
├── README.md
├── cic_dataframe.pkl (1.8GB)
├── tcpdump_dataframe.pkl (46MB)
├── CIC/
│   ├── pcap/
│   │   ├── Monday-WorkingHours.pcap (10GB)
│   │   ├── Tuesday-WorkingHours.pcap (10GB)
│   │   ├── Wednesday-workingHours.pcap (12GB)
│   │   ├── Thursday-WorkingHours.pcap (7.7GB)
│   │   └── Friday-WorkingHours.pcap (8.2GB)
│   └── nfstream/
│       ├── Monday-WorkingHours.pcap_nfstream.csv (271MB)
│       ├── Tuesday-WorkingHours.pcap_nfstream.csv (235MB)
│       ├── Wednesday-WorkingHours.pcap_nfstream.csv (353MB)
│       ├── Thursday-WorkingHours.pcap_nfstream.csv (246MB)
│       ├── Friday-WorkingHours.pcap_nfstream.csv (322MB)
│       ├── Monday-WorkingHours.pcap_nfstream_labeled.csv (275MB)
│       ├── Tuesday-WorkingHours.pcap_nfstream_labeled.csv (239MB)
│       ├── Wednesday-WorkingHours.pcap_nfstream_labeled.csv (359MB)
│       ├── Thursday-WorkingHours.pcap_nfstream_labeled.csv (89MB)
│       └── Friday-WorkingHours.pcap_nfstream_labeled.csv (326MB)
└── tcpdump/
    ├── pcap/
    │   ├── normal_01.pcap (28MB)
    │   ├── normal_02.pcap (57MB)
    │   ├── normal_and_attack_01.pcap (42MB)
    │   ├── normal_and_attack_02.pcap (47MB)
    │   ├── normal_and_attack_03.pcap (38MB)
    │   ├── normal_and_attack_04.pcap (19MB)
    │   └── normal_and_attack_05.pcap (34MB)
    └── nfstream/
        ├── normal_01.csv (5.2MB)
        ├── normal_02.csv (10MB)
        ├── normal_and_attack_01.csv (3.2MB)
        ├── normal_and_attack_02.csv (2.6MB)
        ├── normal_and_attack_03.csv (2.8MB)
        ├── normal_and_attack_04.csv (2.3MB)
        ├── normal_and_attack_05.csv (4.3MB)
        ├── normal_01_labeled.csv (5.3MB)
        ├── normal_02_labeled.csv (11MB)
        ├── normal_and_attack_01_labeled.csv (3.2MB)
        ├── normal_and_attack_02_labeled.csv (2.6MB)
        ├── normal_and_attack_03_labeled.csv (2.9MB)
        ├── normal_and_attack_04_labeled.csv (2.4MB)
        └── normal_and_attack_05_labeled.csv (4.4MB)
```
