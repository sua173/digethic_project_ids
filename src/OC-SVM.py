# %%
import warnings
import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

try:
    from data_processing import DataProcessor
    from model_evaluation import evaluate_model
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

    from data_processing import DataProcessor
    from model_evaluation import evaluate_model

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

cic_pkl_file_name = os.path.join(data_dir, "cic_dataframe.pkl")
cic_file_paths = [
    os.path.join(data_dir, f"CIC/nfstream/{day}-WorkingHours.pcap_nfstream_labeled.csv")
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
]

tcpdump_pkl_file_name = os.path.join(data_dir, "tcpdump_dataframe.pkl")
tcpdump_file_paths = [
    os.path.join(data_dir, f"tcpdump/nfstream/{filename}_labeled.csv")
    for filename in [
        "normal_01",
        "normal_02",
        "normal_and_attack_01",
        "normal_and_attack_02",
        "normal_and_attack_03",
        "normal_and_attack_04",
        "normal_and_attack_05",
    ]
]

# Constants
NORMAL_LABEL = 1
ANOMALY_LABEL = -1

# Configuration
test_size = 0.2
random_state = 42
cache_size = 2000
scaled = True
encode_categorical = True
shap_enabled = True
dev_mode = False
corr_threshold = 0.95

# %%
# 1. Load and prepare the data
print("\nStep 1: Load and prepare the data")
if os.path.exists(cic_pkl_file_name):
    print(f"Loading dataframe from {cic_pkl_file_name}")
    dataframe = pd.read_pickle(cic_pkl_file_name)
else:
    print(f"Creating dataframe from pcap files and saving to {cic_pkl_file_name}")
    dataframe = DataProcessor.get_dataframe(file_paths=cic_file_paths)
    dataframe.to_pickle(cic_pkl_file_name)

# Add new featueres:
print("Adding new features to the dataframe")
dataframe = DataProcessor.add_new_features(dataframe)

# Drop object columns and handle categorical data
print("Dropping object columns except for some categorical columns")
df_without_object, available_categorical = DataProcessor.drop_object_columns(
    dataframe, encode_categorical=encode_categorical
)

# Split into features and labels
print("Splitting data into features (X) and labels (y)")
X, y = DataProcessor.split_to_X_y(df_without_object)

# Clean the data
print("Cleaning data")
DataProcessor.clean_data(X)

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")

# Split the data into training and test sets
print(f"Splitting data into train and test sets with test_size={test_size}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Handle categorical encoding
print("Handling categorical encoding")
X_train, X_test, categorical_encoder = DataProcessor.one_hot_encode_categorical(
    X_train, X_test, available_categorical, None
)

print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")

# Scaling the data
if scaled:
    print("Scaleing the data")
    scaler = MinMaxScaler()
    print("New MinMaxScaler instance is created")
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
else:
    scaler = None

# Label conversion
print("Converting labels: benign to 1 and anomalous to -1")
y_train = y_train.map(lambda x: 1 if x == "benign" else -1)
y_test = y_test.map(lambda x: 1 if x == "benign" else -1)

# Feature selection
features_to_drop = DataProcessor.get_features_to_drop()
print(f"Always drop id, src, timestamp...: {features_to_drop}")
X_train = X_train.drop(columns=features_to_drop)
X_test = X_test.drop(columns=features_to_drop)
print(f"Droped {len(features_to_drop)} features")

# Remove highly correlated features
print(f"Dropping highly correlated features with threshold={corr_threshold}")
X_train, dropped_corr = DataProcessor.remove_highly_correlated_features(
    X_train, threshold=corr_threshold
)
X_test = X_test.drop(columns=dropped_corr)
print(f"Droped {len(dropped_corr)} features: {dropped_corr}")

print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"y_test.shape: {y_test.shape}")

# reset index to ensure consistent indexing
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# %%
# 2. Load Feature Importance from Random Forest
print("\nStep 2: Load Feature Importance from Random Forest")
feature_importance_file = os.path.join(
    project_root, "models", "rf", "importance_df.pkl"
)
if os.path.exists(feature_importance_file):
    print(f"Loading feature importance from {feature_importance_file}")
    feature_importance_df = pd.read_pickle(feature_importance_file)
else:
    print(f"Feature importance file not found: {feature_importance_file}")
    exit(1)
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
)
print("Feature importance loaded and sorted by importance")

# %%
# 3. Performance validation with different feature counts
print("\nStep 3: Performance evaluation with different feature counts")
if dev_mode:
    target_features_list = [5, 10]
else:
    target_features_list = [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
    ]
results_comparison = []

for n_features in target_features_list:
    print(f"\nValidating with {n_features} features")

    # select top n_features based on F-scores
    top_features = feature_importance_df.head(n_features)["feature"].tolist()

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train,
        y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train,
    )

    X_train_split_selected = X_train_split[top_features]
    X_val_split_selected = X_val_split[top_features]

    X_train_split_selected_normal = X_train_split_selected[
        y_train_split == NORMAL_LABEL
    ]

    print(f"Training samples (normal): {len(X_train_split_selected_normal)}")
    print(f"Validate samples: {len(X_val_split_selected)}")

    # Reduce number of training samples
    max_samples = 10000
    X_train_sample = (
        X_train_split_selected_normal.sample(n=max_samples, random_state=random_state)
        if len(X_train_split_selected_normal) > max_samples
        else X_train_split_selected_normal
    )

    print(f"Training samples used: {len(X_train_sample)}")

    # Train OneClass SVM
    try:
        oc_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")

        oc_svm.fit(X_train_sample)

        y_pred_val = oc_svm.predict(X_val_split_selected)

        y_true_binary = (y_val_split != NORMAL_LABEL).astype(
            int
        )  # normal: 0, anomaly: 1
        y_pred_binary = (y_pred_val != NORMAL_LABEL).astype(
            int
        )  # normal: 0, anomaly: 1

        # Metrics calculation
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # Normal and anomaly detection rates
        normal_total = np.sum(y_true_binary == 0)
        anomaly_total = np.sum(y_true_binary == 1)

        if normal_total > 0:
            normal_detection_rate = (
                np.sum((y_true_binary == 0) & (y_pred_binary == 0)) / normal_total
            )
        else:
            normal_detection_rate = 0.0

        if anomaly_total > 0:
            anomaly_detection_rate = (
                np.sum((y_true_binary == 1) & (y_pred_binary == 1)) / anomaly_total
            )
        else:
            anomaly_detection_rate = 0.0

        # outlier fraction
        outlier_fraction = np.sum(y_pred_val == -1) / len(y_pred_val)

        results_comparison.append(
            {
                "n_features": n_features,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "normal_detection_rate": normal_detection_rate,
                "anomaly_detection_rate": anomaly_detection_rate,
                "outlier_fraction": outlier_fraction,
                "training_samples": len(X_train_sample),
            }
        )

        print(
            f"  F1: {f1:.4f}, Normal Det.: {normal_detection_rate:.4f}, "
            f"Anomaly Det.: {anomaly_detection_rate:.4f}, Outlier: {outlier_fraction:.4f}"
        )

    except Exception as e:
        print(f"  Error: {e}")
        results_comparison.append(
            {
                "n_features": n_features,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "normal_detection_rate": 0.0,
                "anomaly_detection_rate": 0.0,
                "outlier_fraction": 0.0,
                "training_samples": 0,
            }
        )

print(f"\nCompleted evaluation for {len(results_comparison)} feature configurations.")

# isualization and analysis
print("Results visualization")

results_df = pd.DataFrame(results_comparison)
print("\nResults Summary:")
print(results_df.round(4))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# F1-Score
ax1.plot(
    results_df["n_features"], results_df["f1_score"], "bo-", linewidth=2, markersize=8
)
ax1.set_xlabel("Number of Features")
ax1.set_ylabel("F1 Score")
ax1.set_title("OneClass SVM: F1 Score vs Number of Features")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Detection Rates
ax2.plot(
    results_df["n_features"],
    results_df["normal_detection_rate"],
    "go-",
    label="Normal Detection Rate",
    linewidth=2,
    markersize=8,
)
ax2.plot(
    results_df["n_features"],
    results_df["anomaly_detection_rate"],
    "ro-",
    label="Anomaly Detection Rate",
    linewidth=2,
    markersize=8,
)
ax2.set_xlabel("Number of Features")
ax2.set_ylabel("Detection Rate")
ax2.set_title("OneClass SVM: Detection Rates vs Number of Features")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

# Precision vs Recall
ax3.plot(
    results_df["n_features"],
    results_df["precision"],
    "mo-",
    label="Precision",
    linewidth=2,
    markersize=8,
)
ax3.plot(
    results_df["n_features"],
    results_df["recall"],
    "co-",
    label="Recall",
    linewidth=2,
    markersize=8,
)
ax3.set_xlabel("Number of Features")
ax3.set_ylabel("Score")
ax3.set_title("OneClass SVM: Precision/Recall vs Number of Features")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1)

# Outlier Fraction
ax4.plot(
    results_df["n_features"],
    results_df["outlier_fraction"],
    "ko-",
    linewidth=2,
    markersize=8,
)
ax4.set_xlabel("Number of Features")
ax4.set_ylabel("Outlier Fraction")
ax4.set_title("OneClass SVM: Outlier Fraction vs Number of Features")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=False)

print("Optimal feature selection")

# Calculate balanced score
results_df["balanced_score"] = (
    results_df["f1_score"]
    + results_df["normal_detection_rate"]
    + results_df["anomaly_detection_rate"]
) / 3

# NaN values handling
valid_results = results_df[results_df["balanced_score"] > 0]

if len(valid_results) > 0:
    best_idx = valid_results["balanced_score"].idxmax()
    best_result = valid_results.loc[best_idx]

    optimal_n_features = int(best_result["n_features"])
    optimal_features_list = feature_importance_df.head(optimal_n_features)[
        "feature"
    ].tolist()

    print(f"Optimal number of features: {optimal_n_features}")
    print(f"F1 Score: {best_result['f1_score']:.4f}")
    print(f"Normal Detection Rate: {best_result['normal_detection_rate']:.4f}")
    print(f"Anomaly Detection Rate: {best_result['anomaly_detection_rate']:.4f}")
    print(f"Balanced Score: {best_result['balanced_score']:.4f}")
    print(f"Outlier Fraction: {best_result['outlier_fraction']:.4f}")

    print("\nSelected features for OneClass SVM:")
    for i, (_, row) in enumerate(
        feature_importance_df.head(optimal_n_features).iterrows(), 1
    ):
        print(f"{i:2d}. {row['feature']:<35} {row['importance']:.4f}")

X_train_optimal = X_train[optimal_features_list]
X_test_optimal = X_test[optimal_features_list]

print(f"\nX_train_optimal.shape: {X_train_optimal.shape}")
print(f"X_test_optimal.shape: {X_test_optimal.shape}")

# %%
# 4. Hyperparameter tuning for OneClassSVM
print("\nStep 4: Hyperparameter tuning for OneClass SVM")
# Hyperparameter grid for OneClassSVM
if dev_mode:
    param_grid = {"kernel": ["rbf"], "nu": [0.05], "gamma": ["scale"]}
else:
    param_grid = {
        "kernel": ["linear", "rbf"],
        "nu": [0.01, 0.05, 0.1, 0.15, 0.2],
        "gamma": ["scale", 0.001, 0.01, 0.1, 1],
    }
best_score = -np.inf
best_params = None
best_model = None

total_combinations = sum(1 for _ in ParameterGrid(param_grid))

with tqdm(
    total=total_combinations, desc="Hyperparameter Tuning", file=sys.stdout
) as pbar:
    for i, params in enumerate(ParameterGrid(param_grid), 1):
        model = OneClassSVM(**params)
        # split X_train_optimal
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_optimal,
            y_train,
            test_size=test_size,
            random_state=random_state,
            stratify=y_train,
        )

        sample_size = 10000
        X_train_split_benign = X_train_split[y_train_split == NORMAL_LABEL]
        X_train_split_benign = (
            X_train_split_benign.sample(n=sample_size, random_state=random_state)
            if len(X_train_split_benign) > sample_size
            else X_train_split_benign
        )

        model.fit(X_train_split_benign)
        y_pred = model.predict(X_val_split)
        score = f1_score(y_val_split, y_pred, pos_label=-1)

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

        # update the progress bar description
        pbar.set_description(f"F1: {score:.4f} | Best: {best_score:.4f}")
        # write the current params and score to the progress bar
        pbar.write(
            f"[{i}/{total_combinations}] Params: {params}, F1 Score: {score:.4f}"
        )
        pbar.update(1)

print("Best params:", best_params)
print("Best F1 score:", best_score)

# %%
# 5. Find the best optimal sample size
print("\nStep 5: Find the best optimal sample size")

nu = float(best_params["nu"])
kernel = best_params["kernel"]
gamma = best_params["gamma"]

if dev_mode:
    sample_sizes = [100, 500, 1000, 5000]
else:
    sample_sizes = [100, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]

results = []

with tqdm(total=len(sample_sizes), desc="Sample Size Tuning", file=sys.stdout) as pbar:
    for i, n_samples in enumerate(sample_sizes, 1):
        # split X_train_optimal
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_optimal,
            y_train,
            test_size=test_size,
            random_state=random_state,
            stratify=y_train,
        )

        X_train_split_normal = X_train_split[y_train_split == NORMAL_LABEL]
        # randomly take samples of n_samples
        X_train_split_normal_sampled = X_train_split_normal.sample(
            n=min(n_samples, len(X_train_split_normal)), random_state=random_state
        )

        # OneClassSVM instance
        ocsvm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)

        # train
        t0 = time.time()
        ocsvm.fit(X_train_split_normal_sampled)
        train_time = time.time() - t0

        # test
        t0 = time.time()
        y_pred = ocsvm.predict(X_val_split)
        test_time = time.time() - t0

        # OneClassSVM predict: normal=1, anomaly=-1
        # metric for anomaly
        y_val_split_bin = (y_val_split != NORMAL_LABEL).astype(
            int
        )  # convert to binary: normal=0, anomaly=1
        y_pred_label = (y_pred != NORMAL_LABEL).astype(
            int
        )  # convert to binary: normal=0, anomaly=1

        # calculate metrics
        f1 = f1_score(y_val_split_bin, y_pred_label)
        y_score = ocsvm.decision_function(X_val_split)
        auc = roc_auc_score(y_val_split_bin, -y_score)
        precision = precision_score(y_val_split_bin, y_pred_label, zero_division=0)
        recall = recall_score(y_val_split_bin, y_pred_label, zero_division=0)
        # Calculate false positive rate (FPR) and false negative rate (FNR)
        # FPR: proportion of normal samples incorrectly classified as anomaly
        # FNR: proportion of anomaly samples incorrectly classified as normal
        fp = np.sum((y_val_split_bin == 0) & (y_pred_label == 1))
        tn = np.sum((y_val_split_bin == 0) & (y_pred_label == 0))
        fn = np.sum((y_val_split_bin == 1) & (y_pred_label == 0))
        tp = np.sum((y_val_split_bin == 1) & (y_pred_label == 1))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        results.append([n_samples, train_time, test_time, f1, auc, precision, recall])

        # update the progress bar description
        pbar.set_description(f"n_samles: {n_samples}")
        # write the current params and score to the progress bar
        pbar.write(f"[{i}/{len(sample_sizes)}] n_samples: {n_samples}")
        pbar.write(f"  Training (sec): {train_time:.1f}")
        pbar.write(f"  Predict (sec): {test_time:.1f}")
        pbar.write(f"  Precision: {precision:.3f}")
        pbar.write(f"  Recall: {recall:.3f}")
        pbar.write(f"  False Positive Rate: {fpr:.4f}")
        pbar.write(f"  False Negative Rate: {fnr:.4f}")
        pbar.write(f"  F1: {f1:.2f}")
        pbar.write(f"  AUC: {auc:.3f}")
        pbar.update(1)

# DataFrame
df_results = pd.DataFrame(
    results,
    columns=[
        "Samples",
        "Training (sec)",
        "Prediction (sec)",
        "F1",
        "AUC",
        "Precision",
        "Recall",
    ],
)
print(df_results)

# plot results
plt.figure(figsize=(8, 6))
plt.plot(df_results["Samples"], df_results["F1"], marker="o", label="F1 score")
plt.plot(df_results["Samples"], df_results["AUC"], marker="o", label="AUC")
plt.plot(df_results["Samples"], df_results["Precision"], marker="o", label="Precision")
plt.plot(df_results["Samples"], df_results["Recall"], marker="o", label="Recall")
plt.xlabel("Number of training samples")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("OC-SVM Scores vs. Training Samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(8, 6))
plt.plot(
    df_results["Samples"],
    df_results["Training (sec)"],
    marker="o",
    label="Training Time (sec)",
)
plt.plot(
    df_results["Samples"],
    df_results["Prediction (sec)"],
    marker="o",
    label="Prediction Time (sec)",
)
plt.xlabel("Number of training samples")
plt.ylabel("Time (sec)")
plt.title("OC-SVM Training/Prediction Time vs. Training Samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# %%
# 6. Final One-Class SVM model training and evaluation
print("\nStep 6: Final One-Class SVM model training and evaluation")

sample_size = 10_000
print(f"Using sample size: {sample_size}")

nu = float(best_params["nu"])
kernel = best_params["kernel"]
gamma = best_params["gamma"]
print(f"Using nu={nu}, kernel={kernel}, gamma={gamma} for final One-Class SVM model.")

final_ocsvm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma, cache_size=cache_size)
print("Training One-Class SVM model on BENIGN data...")
X_train_benign = X_train_optimal[y_train == NORMAL_LABEL]

X_train_benign = (
    X_train_benign.sample(n=sample_size, random_state=random_state)
    if len(X_train_benign) > sample_size
    else X_train_benign
)
print(f"Sampled X_train_benign shape: {X_train_benign.shape}")
final_ocsvm.fit(X_train_benign)
print("One-Class SVM training complete.")

evaluate_model(final_ocsvm, X_test_optimal, y_test, with_numpy=False)

# %%
# 7. Evaluate with tcpdump data
print("\nStep 7: Evaluate with tcpdump data")
if os.path.exists(tcpdump_pkl_file_name):
    print(f"Loading dataframe from {tcpdump_pkl_file_name}")
    tcpdump_dataframe = pd.read_pickle(tcpdump_pkl_file_name)
else:
    print(f"Creating dataframe from pcap files and saving to {tcpdump_pkl_file_name}")
    tcpdump_dataframe = DataProcessor.get_dataframe(file_paths=tcpdump_file_paths)
    tcpdump_dataframe.to_pickle(tcpdump_pkl_file_name)

# Add new featueres:
print("Adding new features to the dataframe...")
tcpdump_dataframe = DataProcessor.add_new_features(tcpdump_dataframe)

# Drop object columns and handle categorical data
print("Dropping object columns and handle encoding categorical data...")
tcpdump_df_without_object, available_categorical = DataProcessor.drop_object_columns(
    tcpdump_dataframe, encode_categorical=encode_categorical
)
# Split into features and labels
print("Splitting data into features (X) and labels (y)...")
X_tcpdump, y_tcpdump = DataProcessor.split_to_X_y(tcpdump_df_without_object)

print("Cleaning data...")
DataProcessor.clean_data(X_tcpdump)

print(f"X_tcpdump.shape: {X_tcpdump.shape}")
print(f"y_tcpdump.shape: {y_tcpdump.shape}")

print("Handling categorical encoding...")
print(f"Available categorical features: {available_categorical}")
print(f"Use categorical_encoder: {categorical_encoder}")
X_tcpdump, _, categorical_encoder = DataProcessor.one_hot_encode_categorical(
    X_tcpdump, None, available_categorical, categorical_encoder
)
print(f"X_tcpdump.shape: {X_tcpdump.shape}")

if scaled and scaler is not None:
    print(f"Use MinMaxScaler instance: {scaler}")
    X_tcpdump = pd.DataFrame(
        scaler.transform(X_tcpdump),
        columns=X_tcpdump.columns,
        index=X_tcpdump.index,
    )

y_tcpdump = y_tcpdump.map(lambda x: 1 if x == "benign" else -1)

# Feature selection
print("Feature selection:")
X_tcpdump_optimal = X_tcpdump[optimal_features_list]

print(f"X_tcpdump_optimal.shape: {X_tcpdump_optimal.shape}")
print(f"y_tcpdump.shape: {y_tcpdump.shape}")

evaluate_model(final_ocsvm, X_tcpdump_optimal, y_tcpdump, with_numpy=False)

# %%
# 8. Interpretation with SHAP
if shap_enabled:
    print("\nStep 8: Interpretation with SHAP")
    background_data_summary = shap.kmeans(X_train_optimal, 100)
    explainer = shap.KernelExplainer(final_ocsvm.predict, background_data_summary)
    X_test_optimal_sampled = X_test_optimal.sample(n=2000, random_state=random_state)
    shap_values = explainer.shap_values(X_test_optimal_sampled)
    shap.summary_plot(
        shap_values,
        X_test_optimal_sampled,
        feature_names=optimal_features_list,
        plot_type="bar",
        max_display=30,
    )
    shap.summary_plot(
        shap_values,
        X_test_optimal_sampled,
        feature_names=optimal_features_list,
        max_display=30,
    )

# %%
model_dir = os.path.join(project_root, "models", "ocsvm")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created model directory: {model_dir}")

# save the model
model_file_name = os.path.join(model_dir, "model.pkl")
print(f"Saving the model to {model_file_name}")
joblib.dump(final_ocsvm, model_file_name)
print("Model saved successfully.")

# save the encoder
encoder_file_name = os.path.join(model_dir, "encoder.pkl")
print(f"Saving the encoder to {encoder_file_name}")
joblib.dump(categorical_encoder, encoder_file_name)
print("Encoder saved successfully.")

# save the scaler
scaler_file_name = os.path.join(model_dir, "scaler.pkl")
print(f"Saving the scaler to {scaler_file_name}")
joblib.dump(scaler, scaler_file_name)
print("Scaler saved successfully.")

# save the optimal_features_list
optimal_features_file_name = os.path.join(model_dir, "optimal_features_list.pkl")
print(f"Saving the optimal features list to {optimal_features_file_name}")
joblib.dump(optimal_features_list, optimal_features_file_name)
print("Optimal features list saved successfully.")
