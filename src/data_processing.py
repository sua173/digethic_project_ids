import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from nfstream import NFStreamer

# Feature selection constants
FEATURES_TO_DROP = [
    "id",
    "src_port",
    "bidirectional_first_seen_ms",
    "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms",
    "src2dst_last_seen_ms",
    "dst2src_first_seen_ms",
    "dst2src_last_seen_ms",
]

CATEGORICAL_FEATURES = ["application_name", "application_category_name"]


class DataProcessor:
    """Data processing utilities"""

    @staticmethod
    def get_dataframe(partial=-1, file_paths=None):
        """Load and concatenate CSV files into a dataframe"""
        if file_paths is None:
            print("No file paths provided.")
            return None

        if partial < 0:
            dataframe_list = [
                pd.read_csv(file_path, low_memory=False) for file_path in file_paths
            ]
        else:
            if partial > 4:
                raise ValueError("partial must be between 0 and 4")
            dataframe_list = [pd.read_csv(file_paths[partial], low_memory=False)]

        dataframe = pd.concat(dataframe_list, ignore_index=True)
        dataframe.columns = dataframe.columns.str.strip()

        print(f"Loaded {len(dataframe_list)} files with {len(dataframe)} rows")
        return dataframe

    @staticmethod
    def clean_data(dataframe):
        """Clean data by handling inf and NaN values"""
        numeric_cols = dataframe.select_dtypes(include=["number"]).columns
        has_inf = any(np.isinf(dataframe[col]).any() for col in numeric_cols)
        has_nan = dataframe.isna().any().any()

        # print(f"contains inf/-inf: {has_inf}")
        # print(f"contains NaN: {has_nan}")

        if has_inf:
            inf_counts = pd.Series(
                {col: np.sum(np.isinf(dataframe[col].values)) for col in numeric_cols}
            )
            inf_columns = inf_counts[inf_counts > 0]
            print("\ncolumns with infinit values:")
            print(inf_columns)

            for col in numeric_cols:
                dataframe[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            print("inf/-inf values were replaced with NaN")

        if has_nan:
            nan_counts = dataframe.isna().sum()
            nan_columns = nan_counts[nan_counts > 0]
            print("\ncoluns with NaN values:")
            print(nan_columns)

            dataframe.fillna(0, inplace=True)
            print("NaN values were replaced with 0")

    @staticmethod
    def get_features_to_drop():
        """Get list of features to drop"""
        return FEATURES_TO_DROP

    @staticmethod
    def drop_object_columns(dataframe, encode_categorical=False, with_label=True):
        """Drop object columns from the dataframe"""
        # Remove object columns except label and categorical features to encode
        object_features = dataframe.select_dtypes(include=["object"]).columns.tolist()
        if with_label and "label" in object_features:
            object_features.remove("label")

        # Keep categorical features if encoding is requested
        available_categorical = []
        if encode_categorical:
            print(
                f"Retaining categorical features for encoding: {CATEGORICAL_FEATURES}"
            )
            available_categorical = [
                col for col in CATEGORICAL_FEATURES if col in dataframe.columns
            ]
            for cat_col in available_categorical:
                if cat_col in object_features:
                    object_features.remove(cat_col)

        print(f"Number of columns before dropping object columns: {dataframe.shape[1]}")
        # Drop remaining object columns
        dataframe = dataframe.drop(columns=object_features)
        dataframe = dataframe.dropna(axis=1, how="all")

        if object_features:
            print(f"Dropped object columns ({len(object_features)}): {object_features}")
        print(f"Number of columns after dropping object columns: {dataframe.shape[1]}")

        return dataframe, available_categorical

    @staticmethod
    def split_to_X_y(dataframe, with_label=True):
        if with_label:
            # Remove rows with invalid labels (e.g., only one label present)
            label_counts = dataframe["label"].value_counts()
            valid_labels = label_counts[label_counts > 1].index
            dataframe = dataframe[dataframe["label"].isin(valid_labels)]

            X = dataframe.drop(columns=["label"])
            y = dataframe["label"]
        else:
            X = dataframe
            y = None
        return X, y

    @staticmethod
    def one_hot_encode_categorical(
        X_train, X_test, available_categorical, categorical_encoder=None
    ):
        """One-hot encode categorical features"""
        if available_categorical:
            print(f"Processing categorical features: {available_categorical}")

            # Handle missing/unknown values
            for col in available_categorical:
                X_train[col] = X_train[col].fillna("Unknown")
                X_train[col] = X_train[col].replace(
                    ["", " ", "null", "NULL", "nan"], "Unknown"
                )
                if X_test is not None:
                    X_test[col] = X_test[col].fillna("Unknown")
                    X_test[col] = X_test[col].replace(
                        ["", " ", "null", "NULL", "nan"], "Unknown"
                    )

            # Apply one-hot encoding
            if categorical_encoder is None:
                print("Creating new OneHotEncoder")
                categorical_encoder = OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                )
                encoded_features_train = categorical_encoder.fit_transform(
                    X_train[available_categorical]
                )
                if X_test is not None:
                    encoded_features_test = categorical_encoder.transform(
                        X_test[available_categorical]
                    )
            else:
                print("Using existing OneHotEncoder")
                encoded_features_train = categorical_encoder.transform(
                    X_train[available_categorical]
                )
                if X_test is not None:
                    encoded_features_test = categorical_encoder.transform(
                        X_test[available_categorical]
                    )

            # Create column names for encoded features
            feature_names = categorical_encoder.get_feature_names_out(
                available_categorical
            )

            # Create DataFrame with encoded features
            encoded_df_train = pd.DataFrame(
                encoded_features_train, columns=feature_names, index=X_train.index
            )
            if X_test is not None:
                encoded_df_test = pd.DataFrame(
                    encoded_features_test, columns=feature_names, index=X_test.index
                )

            # Remove original categorical columns and add encoded ones
            X_train = X_train.drop(columns=available_categorical)
            X_train = pd.concat([X_train, encoded_df_train], axis=1)

            if X_test is not None:
                X_test = X_test.drop(columns=available_categorical)
                X_test = pd.concat([X_test, encoded_df_test], axis=1)

            print(f"Added {len(feature_names)} one-hot encoded features")

        return X_train, X_test, categorical_encoder

    @staticmethod
    def add_new_features(dataframe):
        """Add new features to the dataframe"""
        duration_s = dataframe["bidirectional_duration_ms"] / 1000.0
        # avoid division by zero
        duration_s = duration_s.replace(0, 1e-9)

        # flow rate (bytes/sec and packets/sec)
        print("flow rate (bytes/sec and packets/sec)")
        dataframe["flow_bytes_per_sec"] = (
            (dataframe["bidirectional_bytes"] / duration_s)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
        dataframe["flow_packets_psr_sec"] = (
            (dataframe["bidirectional_packets"] / duration_s)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        # packet rate (bytes/sec and packets/sec)
        print("packet rate (bytes/sec and packets/sec)")
        dataframe["src2dst_packets_per_sec"] = (
            (dataframe["src2dst_packets"] / duration_s)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
        dataframe["dst2src_ackets_per_sec"] = (
            (dataframe["dst2src_packets"] / duration_s)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

        # Down/Up Ratio
        print("Down/Up Ratio")
        dataframe["down_up_ratio"] = (
            (dataframe["dst2src_packets"] / dataframe["src2dst_packets"].replace(0, 1))
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
        dataframe["down_up_ratio"] = (
            dataframe["down_up_ratio"].apply(np.floor).astype("int64")
        )

        return dataframe

    @staticmethod
    def pcap2nfstream(
        pcap_file, output_file, idle_timeout=10, active_timeout=120, save_csv=True
    ):
        """Convert pcap file to NFStreamer DataFrame"""
        print("Converting pcap file to NFStreamer DataFrame...")
        my_streamer = NFStreamer(
            source=pcap_file,
            statistical_analysis=True,
            n_dissections=20,
            idle_timeout=idle_timeout,
            active_timeout=active_timeout,
        )
        print("my_streamer created")
        dataframe = my_streamer.to_pandas()

        if save_csv:
            print("DataFrame created from NFStreamer")
            dataframe.to_csv(output_file, index=False)
            print(f"{pcap_file} converted to NFStreamer DataFrame and saved as CSV.")

        print(f"Loaded 1 file with {len(dataframe)} rows")
        return dataframe

    @staticmethod
    def remove_highly_correlated_features(df, threshold=0.95):
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop), to_drop
