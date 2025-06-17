# %% [markdown]
# # Prepare labeled flow dataset from pcap

# %%
import warnings
import sys
import os
import pandas as pd
from nfstream import NFStreamer

warnings.filterwarnings("ignore")

try:
    from config_manager import get_config
    from data_processing import DataProcessor
except ImportError:
    if "__file__" in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_dir) == "src":
            project_root = os.path.dirname(script_dir)
        else:
            project_root = script_dir
    else:
        current_dir = os.getcwd()
        project_root = (
            os.path.dirname(current_dir)
            if current_dir.endswith("notebooks")
            else current_dir
        )

    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    from config_manager import get_config
    from data_processing import DataProcessor

if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "src":
        project_root = os.path.dirname(script_dir)
    else:
        project_root = script_dir
else:
    current_dir = os.getcwd()
    project_root = (
        os.path.dirname(current_dir)
        if current_dir.endswith("notebooks")
        else current_dir
    )

data_dir = os.path.join(project_root, "data")

print(f"Project root directory: {project_root}\nData directory: {data_dir}")

idle_timeout, active_timeout = get_config()
print(f"idle_timeout: {idle_timeout}, active_timeout: {active_timeout}")

# %% [markdown]
# ## Extract flows from CIC pcap files (without label)


# %%
def pcap_to_nfstream(pcap_file, idle_timeout=10, active_timeout=120, save_csv=True):
    """
    Convert a pcap file to a NFStreamer DataFrame.
    Args:
        pcap_file (str): Path to the pcap file.
    Returns:
        pd.DataFrame: DataFrame containing the NFStreamer data.
    """
    print("Converting pcap file to NFStreamer DataFrame...")
    # check if pcap file exists
    if not os.path.exists(pcap_file):
        raise FileNotFoundError(f"The pcap file {pcap_file} does not exist.")
    streamer = NFStreamer(
        source=pcap_file,
        statistical_analysis=True,
        n_dissections=20,
        idle_timeout=idle_timeout,
        active_timeout=active_timeout,
    )
    print("New streamer instance is created")
    dataframe = streamer.to_pandas()

    if save_csv:
        print("DataFrame created from NFStreamer")
        # makedir if not exist
        subdirectory = os.path.join(data_dir, "CIC", "nfstream")
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)
        filename = os.path.basename(pcap_file)
        weekday = filename.split("-")[0]
        print(f"writing to {weekday}-WorkingHours.pcap_nfstream.csv")
        dataframe.to_csv(
            f"{subdirectory}/{weekday}-WorkingHours.pcap_nfstream.csv", index=False
        )
        print(f"{pcap_file} converted to NFStreamer DataFrame and saved as CSV.")

    return dataframe


# %%
pcap_files = [
    os.path.join(data_dir, "CIC", "pcap", f"{day}-WorkingHours.pcap")
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
]

for pcap_file in pcap_files:
    try:
        filename = os.path.basename(pcap_file)
        weekday = filename.split("-")[0]
        print(f"Found pcap file of {weekday}")
        pcap_to_nfstream(
            pcap_file,
            idle_timeout=idle_timeout,
            active_timeout=active_timeout,
            save_csv=True,
        )
        print()
    except FileNotFoundError:
        print(f"File {pcap_file} not found.")
        exit(1)

# %% [markdown]
# ## Label CIC flows dataset

# %% [markdown]
# Time Zone (Canada, Saint John): ADT (UTC-3) same as America/Halifax

# %% [markdown]
# ### Monday 2017-07-03
# All flows labeled as  "benign"

# %%
print("Labeling flows in Monday-WorkingHours.pcap_nfstream.csv")

# read Monday-WorkingHours.pcap_nfstream.csv to dataframe
df_src_mon = pd.read_csv(
    os.path.join(data_dir, "CIC", "nfstream", "Monday-WorkingHours.pcap_nfstream.csv"),
    low_memory=False,
    encoding="utf-8",
)

# add a column 'label' with value 'benign' for all rows
df_src_mon["label"] = "benign"

# save to file Monday-WorkingHours.pcap_nfstream_labeled.csv
df_src_mon.to_csv(
    os.path.join(
        data_dir, "CIC", "nfstream", "Monday-WorkingHours.pcap_nfstream_labeled.csv"
    ),
    index=False,
    encoding="utf-8",
)

print("Labeling of Monday-WorkingHours.pcap_nfstream.csv is completed")

# %% [markdown]
# ### Tuesday 2017-07-04
#
# labeled as 'ftp_patator'
#     src_addr: 172.16.0.1
#     dst_addr: 192.168.10.50
#     Start: 1499170620000
#     End: 1499175000000
#
# labeled as 'ssh_patator'
#     src_addr: 172.16.0.1
#     dst_addr: 192.168.10.50
#     Start: 1499188140000
#     End: 1499191860000
#

# %%
print("Labeling flows in Tuesday-WorkingHours.pcap_nfstream.csv")

# read Tuesday-WorkingHours.pcap_nfstream.csv to dataframe
df_src_tue = pd.read_csv(
    os.path.join(data_dir, "CIC", "nfstream", "Tuesday-WorkingHours.pcap_nfstream.csv"),
    low_memory=False,
    encoding="utf-8",
)

# ftp_parator
ftp_period = (df_src_tue["bidirectional_first_seen_ms"] >= 1499170620000) & (
    df_src_tue["bidirectional_last_seen_ms"] <= 1499175000000
)
subset_ftp = (df_src_tue["src_ip"] == "172.16.0.1") & (
    df_src_tue["dst_ip"] == "192.168.10.50"
)
df_src_tue.loc[ftp_period & subset_ftp, "label"] = "ftp_patator"

# ssh_parator
ssh_period = (df_src_tue["bidirectional_first_seen_ms"] >= 1499188140000) & (
    df_src_tue["bidirectional_last_seen_ms"] <= 1499191860000
)
subset_ssh = (df_src_tue["src_ip"] == "172.16.0.1") & (
    df_src_tue["dst_ip"] == "192.168.10.50"
)
df_src_tue.loc[ssh_period & subset_ssh, "label"] = "ssh_patator"

# benign
df_src_tue.loc[df_src_tue["label"].isnull(), "label"] = "benign"

# save to file Tuesday-WorkingHours.pcap_nfstream_labeled.csv
df_src_tue.to_csv(
    os.path.join(
        data_dir, "CIC", "nfstream", "Tuesday-WorkingHours.pcap_nfstream_labeled.csv"
    ),
    index=False,
    encoding="utf-8",
)

print("Labeling of Tuesday-WorkingHours.pcap_nfstream.csv is completed")

# %% [markdown]
# ### Wednesday 2017-07-05
#
# labeled as 'dos_slowloris'
#     src_addr: 172.16.0.1
#     dst_addr: 192.168.10.50
#     Start: 1499256060000
#     End:   1499260260000
#     Start: 1499275440000
#     End:   1499275500000
#
# labeled as 'dos_slowhttptest'
#     src_addr: 172.16.0.1
#     dst_addr: 192.168.10.50
#     Start: 1499260500000
#     End: 1499261820000
#
# labeled as 'dos_hulk'
#     src_addr: 172.16.0.1
#     dst_addr: 192.168.10.50
#     dst_port: 80
#     Start: 1499262180000
#     End: 1499263620000
#
# labeled as 'dos_goldeneye'
#     src_addr: 172.16.0.1
#     dst_addr: 192.168.10.50
#     dst_port: 80
#     Start: 1499263800000
#     End: 1499264340000
#
# labeled as 'heartbleed'
#     src_addr: 172.16.0.1
#     src_port: 45022
#     dst_addr: 192.168.10.51
#     dst_port:444
#

# %%
print("Labeling flows in Wednesday-WorkingHours.pcap_nfstream.csv")

# read Wednesday-WorkingHours.pcap_nfstream.csv to dataframe
df_src_wed = pd.read_csv(
    os.path.join(
        data_dir, "CIC", "nfstream", "Wednesday-WorkingHours.pcap_nfstream.csv"
    ),
    low_memory=False,
    encoding="utf-8",
)

# dos_slowloris
slowloris_period = (
    (df_src_wed["bidirectional_first_seen_ms"] >= 1499256060000)
    & (df_src_wed["bidirectional_last_seen_ms"] <= 1499260260000)
) | (
    (df_src_wed["bidirectional_first_seen_ms"] <= 1499275440000)
    & (df_src_wed["bidirectional_last_seen_ms"] <= 1499275500000)
)
subset_slowloris = (
    (df_src_wed["src_ip"] == "172.16.0.1")
    & (df_src_wed["dst_ip"] == "192.168.10.50")
    & (df_src_wed["dst_port"] == 80)
)
df_src_wed.loc[slowloris_period & subset_slowloris, "label"] = "dos_slowloris"

# dos_slowhttptest
slowhttptest_period = (df_src_wed["bidirectional_first_seen_ms"] >= 1499260500000) & (
    df_src_wed["bidirectional_last_seen_ms"] <= 1499261820000
)
subset_slowhttptest = (
    (df_src_wed["src_ip"] == "172.16.0.1")
    & (df_src_wed["dst_ip"] == "192.168.10.50")
    & (df_src_wed["dst_port"] == 80)
)
df_src_wed.loc[slowhttptest_period & subset_slowhttptest, "label"] = "dos_slowhttptest"

# dos_hulk
hulk_period = (df_src_wed["bidirectional_first_seen_ms"] >= 1499262180000) & (
    df_src_wed["bidirectional_last_seen_ms"] <= 1499263620000
)
subset_hulk = (
    (df_src_wed["src_ip"] == "172.16.0.1")
    & (df_src_wed["dst_ip"] == "192.168.10.50")
    & (df_src_wed["dst_port"] == 80)
)
df_src_wed.loc[hulk_period & subset_hulk, "label"] = "dos_hulk"

# dos_goldeneye
goldeneye_period = (df_src_wed["bidirectional_first_seen_ms"] >= 1499263800000) & (
    df_src_wed["bidirectional_last_seen_ms"] <= 1499264340000
)
subset_goldeneye = (
    (df_src_wed["src_ip"] == "172.16.0.1")
    & (df_src_wed["dst_ip"] == "192.168.10.50")
    & (df_src_wed["dst_port"] == 80)
)
df_src_wed.loc[goldeneye_period & subset_goldeneye, "label"] = "dos_goldeneye"

# heartbleed
subset_heartbleed = (
    (df_src_wed["src_ip"] == "172.16.0.1")
    & (df_src_wed["dst_ip"] == "192.168.10.51")
    & (df_src_wed["src_port"] == 45022)
    & (df_src_wed["dst_port"] == 444)
)
# df_src_wed.loc[heartbleed_period & subset_heartbleed, 'label'] = 'heartbleed'
df_src_wed.loc[subset_heartbleed, "label"] = "heartbleed"

# otherwise: add a column 'label' with value 'benign' for all rows
df_src_wed.loc[df_src_wed["label"].isnull(), "label"] = "benign"

# save to file Wednesday-WorkingHours.pcap_nfstream_labeled.csv
df_src_wed.to_csv(
    os.path.join(
        data_dir, "CIC", "nfstream", "Wednesday-WorkingHours.pcap_nfstream_labeled.csv"
    ),
    index=False,
    encoding="utf-8",
)

print("Labeling of Wednesday-WorkingHours.pcap_nfstream.csv is completed")

# %% [markdown]
# ### Thursday 2017-07-06
#
# labeled as 'webattack_bruteforce'
#     src_addr: 172.16.0.1
#     dst_addr: 192.168.10.50
#     ip_prot: 6
#     Start: 1499343300000
#     End:   1499346000000
#
# labeled as 'webattack_xss'
#     src_addr: 172.16.0.1
#     dst_addr: 192.168.10.50
#     ip_prot: 6
#     Start: 1499346900000
#     End: 1499348100000
#
# labeled as 'webattack_sql_injection'
#     src_addr: 172.16.0.1
#     dst_addr: 192.168.10.50
#     ip_prot: 6
#     Start: 1499348400000
#     End: 1499348520000
#
# drop all flow of Thursday afternoon
#     'timestamp' >= 1499353200000000

# %%
print("Labeling flows in Thursday-WorkingHours.pcap_nfstream.csv")

# read Thursday-WorkingHours.pcap_nfstream.csv to dataframe
df_src_thr = pd.read_csv(
    os.path.join(
        data_dir, "CIC", "nfstream", "Thursday-WorkingHours.pcap_nfstream.csv"
    ),
    low_memory=False,
    encoding="utf-8",
)

# webattack_bruteforce
bf_period = (df_src_thr["bidirectional_first_seen_ms"] >= 1499343300000) & (
    df_src_thr["bidirectional_last_seen_ms"] <= 1499346000000
)
subset_bf = (
    (df_src_thr["src_ip"] == "172.16.0.1")
    & (df_src_thr["dst_ip"] == "192.168.10.50")
    & (df_src_thr["protocol"] == 6)
)
df_src_thr.loc[bf_period & subset_bf, "label"] = "webattack_bruteforce"

# webattack_xss
xss_period = (df_src_thr["bidirectional_first_seen_ms"] >= 1499346900000) & (
    df_src_thr["bidirectional_last_seen_ms"] <= 1499348100000
)
subset_xss = (
    (df_src_thr["src_ip"] == "172.16.0.1")
    & (df_src_thr["dst_ip"] == "192.168.10.50")
    & (df_src_thr["protocol"] == 6)
)
df_src_thr.loc[xss_period & subset_xss, "label"] = "webattack_xss"

# webattack_sql_injection
xss_period = (df_src_thr["bidirectional_first_seen_ms"] >= 1499348400000) & (
    df_src_thr["bidirectional_last_seen_ms"] <= 1499348520000
)
subset_xss = (
    (df_src_thr["src_ip"] == "172.16.0.1")
    & (df_src_thr["dst_ip"] == "192.168.10.50")
    & (df_src_thr["protocol"] == 6)
)
df_src_thr.loc[xss_period & subset_xss, "label"] = "webattack_sql_injection"

# drop all flow of Thursday afternoon
drop_period = df_src_thr["bidirectional_first_seen_ms"] >= 1499353200000
df_src_thr.drop(df_src_thr[drop_period].index, axis=0, inplace=True)

# otherwise: add a column 'label' with value 'benign' for all rows
df_src_thr.loc[df_src_thr["label"].isnull(), "label"] = "benign"

# save to file Thursday-WorkingHours.pcap_nfstream_labeled.csv
df_src_thr.to_csv(
    os.path.join(
        data_dir, "CIC", "nfstream", "Thursday-WorkingHours.pcap_nfstream_labeled.csv"
    ),
    index=False,
    encoding="utf-8",
)

print("Labeling of Thursday-WorkingHours.pcap_nfstream.csv is completed")

# %% [markdown]
# ### Friday 2017-07-07
#
# drop dst_addr = "205.174.165.73" after 1499436193000
#
# drop flows with ip addr 52.6.13.28 and 52.7.235.158
#
# labeled as 'bot'
# dst_addr: 205.174.165.73
# Start: 1499430840000
# End: 1499443140000
#
# labeled as 'ddos'
# src_addr: 172.16.0.1
# dst_addr: 192.168.10.50
# ip_prot: 6
# Start: 1499453760000
# End: 1499454960000
#
# labeled as 'portscan'
# src_addr: 172.16.0.1
# dst_addr: 192.168.10.50
# ip_prot:  6
# Start: 1499443500000
# End: 1499451780000

# %%
print("Labeling flows in Friday-WorkingHours.pcap_nfstream.csv")

# read Friday-WorkingHours.pcap_nfstream.csv to dataframe
df_src_fri = pd.read_csv(
    os.path.join(data_dir, "CIC", "nfstream", "Friday-WorkingHours.pcap_nfstream.csv"),
    low_memory=False,
    encoding="utf-8",
)

# drop dst_addr = "205.174.165.73" after 1499436193000
drop_wrong_labels = (df_src_fri["dst_ip"] == "205.174.165.73") & (
    df_src_fri["bidirectional_first_seen_ms"] > 1499436193000
)
df_src_fri.drop(df_src_fri[drop_wrong_labels].index, axis=0, inplace=True)
# print out number of dropped rows
print(f"Dropped {drop_wrong_labels.sum()} rows")

# drop flows with ip addr
drop_subset = (df_src_fri["dst_ip"] == "52.6.13.28") | (
    df_src_fri["dst_ip"] == "52.7.235.158"
)
df_src_fri.drop(df_src_fri[drop_subset].index, axis=0, inplace=True)
# print out number of dropped rows
print(f"Dropped {drop_subset.sum()} rows")

# processing bot attacks
bot_period = (df_src_fri["bidirectional_first_seen_ms"] >= 1499430840000) & (
    df_src_fri["bidirectional_last_seen_ms"] <= 1499443140000
)
subset_bot = df_src_fri["dst_ip"] == "205.174.165.73"
df_src_fri.loc[bot_period & subset_bot, "label"] = "bot"

# ddos
ddos_period = (df_src_fri["bidirectional_first_seen_ms"] >= 1499453760000) & (
    df_src_fri["bidirectional_last_seen_ms"] <= 1499454960000
)
subset_ddos = (
    (df_src_fri["src_ip"] == "172.16.0.1")
    & (df_src_fri["dst_ip"] == "192.168.10.50")
    & (df_src_fri["protocol"] == 6)
)
df_src_fri.loc[ddos_period & subset_ddos, "label"] = "ddos"

# portscan
portscan_period = (df_src_fri["bidirectional_first_seen_ms"] >= 1499443500000) & (
    df_src_fri["bidirectional_last_seen_ms"] <= 1499451780000
)
subset_portscan = (
    (df_src_fri["src_ip"] == "172.16.0.1")
    & (df_src_fri["dst_ip"] == "192.168.10.50")
    & (df_src_fri["protocol"] == 6)
)
df_src_fri.loc[portscan_period & subset_portscan, "label"] = "portscan"

# otherwise: add a column 'label' with value 'benign' for all rows
df_src_fri.loc[df_src_fri["label"].isnull(), "label"] = "benign"

# save to file Friday-WorkingHours.pcap_nfstream_labeled.csv
df_src_fri.to_csv(
    os.path.join(
        data_dir, "CIC", "nfstream", "Friday-WorkingHours.pcap_nfstream_labeled.csv"
    ),
    index=False,
    encoding="utf-8",
)

print("Labeling of Friday-WorkingHours.pcap_nfstream.csv is completed")

# %% [markdown]
# ## Extract and label flows from tcpdump pcap

# %%
pcap_dir = os.path.join(data_dir, "tcpdump", "pcap")
# list of pcap files in pcap_dir, only filenames withaout extension
filenames = [f[:-5] for f in os.listdir(pcap_dir) if f.endswith(".pcap")]
filenames.sort()

for filename in filenames:
    pcap_file = os.path.join(pcap_dir, f"{filename}.pcap")
    csv_file = os.path.join(data_dir, "tcpdump", "nfstream", f"{filename}.csv")
    csv_labeled_file = os.path.join(
        data_dir, "tcpdump", "nfstream", f"{filename}_labeled.csv"
    )

    if os.path.exists(csv_labeled_file):
        print(f"File {filename}_labeled.csv already exists")
        df = pd.read_csv(csv_labeled_file, low_memory=False)
        print(df["label"].value_counts())
        print()
        continue

    if not os.path.exists(pcap_file):
        print(f"File {filename}.pcap does not exist.")
        print()
        continue

    if not os.path.exists(csv_file):
        print(f"File {filename}.csv does not exist.")
        print(f"{os.path.basename(pcap_file)} -> {os.path.basename(csv_file)}")
        df = DataProcessor.pcap2nfstream(
            pcap_file,
            csv_file,
            idle_timeout=idle_timeout,
            active_timeout=active_timeout,
            save_csv=True,
        )
    else:
        print(f"File {filename}.csv already exists, loading from CSV.")
        df = pd.read_csv(csv_file, low_memory=False)

    print(df["src_ip"].value_counts())

    ip_anomalous = "172.28.1.3"
    ip_benign = "172.28.1.4"

    df["label"] = "unknown"
    df.loc[df["src_ip"] == ip_anomalous, "label"] = "anomalous"
    df.loc[df["src_ip"] == ip_benign, "label"] = "benign"
    print(df["label"].value_counts())

    unknown_count = df[df["label"] == "unknown"].shape[0]
    print(f"Before delete: {df.shape[0]}")
    print(f"Number of 'unknown' rows: {unknown_count}")

    # delete 'unknown' rows
    df = df[df["label"] != "unknown"]
    print(f"After delete: {df.shape[0]}")
    print(df["label"].value_counts())

    df.to_csv(csv_labeled_file, index=False)
    print(f"Saved labeled file: {csv_labeled_file}")

    print(f"#rows: {df.shape[0]}")
    print(f"#columns: {df.shape[1]}")

    print()
