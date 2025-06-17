import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display


class DataVisualization:
    """Data visualization utilities"""

    @staticmethod
    def plot_histograms(datasets, label_col="Label"):
        """Plot histograms for multiple datasets"""
        num_datasets = len(datasets)
        cols = 2
        rows = (num_datasets + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        for idx, (name, df) in enumerate(datasets.items()):
            ax = axes[idx]
            counts = df[label_col].value_counts()
            counts.plot(kind="bar", ax=ax)

            for i, count in enumerate(counts):
                ax.text(i, count, str(count), ha="center", va="bottom", fontsize=8)

            ax.set_title(f"{name}")
            ax.set_xlabel("Labels")
            ax.set_ylabel("Counts")
            ax.tick_params(axis="x", rotation=45)

        # Remove unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show(block=False)

    @staticmethod
    def label_table(df, name, label_col="Label"):
        """Display label distribution table"""
        if label_col not in df.columns:
            print(f"'{label_col}' column not found in {name}")
            return

        total = len(df)
        label_counts = df[label_col].value_counts().reset_index()
        label_counts.columns = [label_col, "Count"]
        label_counts["Percentage"] = (label_counts["Count"] / total * 100).round(2)

        # Move BENIGN to top if exists
        if "BENIGN" in label_counts[label_col].values:
            benign_row = label_counts[label_counts[label_col] == "BENIGN"]
            other_rows = label_counts[label_counts[label_col] != "BENIGN"]
            label_counts = pd.concat([benign_row, other_rows], ignore_index=True)

        print(f"--- {name} ---")
        display(label_counts)
        print()

    @staticmethod
    def display_unique_labels(datasets):
        """Display unique labels for all datasets"""
        label_col = "Label"
        for name, df in datasets.items():
            if "label" in df.columns:
                label_col = "label"
                DataVisualization.label_table(df, name, label_col=label_col)
            elif "Label" in df.columns:
                DataVisualization.label_table(df, name)
            else:
                print(f"'label' or 'Label' column not found in {name}")

        DataVisualization.plot_histograms(datasets, label_col=label_col)

    @staticmethod
    def display_ratio_anomalous_benign(datasets):
        """Display ratio of anomalous to benign data"""
        for name, df in datasets.items():
            if "Label" in df.columns:
                label_col = "Label"
            elif "label" in df.columns:
                label_col = "label"
            else:
                print(f"'label' or 'Label' column not found in {name}")
                continue

            total_count = len(df)
            benign_count = np.sum(df[label_col].str.upper() == "BENIGN")
            anomalous_count = np.sum(df[label_col].str.upper() != "BENIGN")

            if total_count > 0:
                benign_ratio = benign_count / total_count * 100
                anomalous_ratio = anomalous_count / total_count * 100
                print(
                    f"{name} - BENIGN Count: {benign_count}, Ratio: {benign_ratio:.2f}%, "
                    f"ANOMALOUS Count: {anomalous_count}, Ratio: {anomalous_ratio:.2f}%"
                )
            else:
                print(f"{name} - No data available.")

        # Total statistics
        total_benign_count = sum(
            np.sum(df[label_col].str.upper() == "BENIGN") for df in datasets.values()
        )
        total_anomalous_count = sum(
            np.sum(df[label_col].str.upper() != "BENIGN") for df in datasets.values()
        )
        total_count = sum(len(df) for df in datasets.values())
        if total_count > 0:
            total_benign_ratio = total_benign_count / total_count * 100
            total_anomalous_ratio = total_anomalous_count / total_count * 100
            print(
                f"Total - BENIGN Count: {total_benign_count}, Ratio: {total_benign_ratio:.2f}%, "
                f"ANOMALOUS Count: {total_anomalous_count}, Ratio: {total_anomalous_ratio:.2f}%"
            )
        else:
            print("Total - No data available.")
