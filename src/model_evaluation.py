import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.model_selection import cross_validate, StratifiedKFold

# Constants
BENIGN_LABEL = 1
ANOMALOUS_LABEL = -1
DEFAULT_CV_SPLITS = 5
RANDOM_STATE = 42


def _calculate_metrics(cm):
    """Calculate metrics from confusion matrix

    Confusion matrix structure with labels=[1, -1]:
         Predicted
    Actual  1   -1
      1   [tn  fp]  <- BENIGN
     -1   [fn  tp]  <- ANOMALOUS (target class)
    """
    # For anomaly detection where -1 (ANOMALOUS) is the positive class
    tn = cm[0, 0]  # True Negative: correctly predicted as BENIGN
    fn = cm[1, 0]  # False Negative: ANOMALOUS predicted as BENIGN
    fp = cm[0, 1]  # False Positive: BENIGN predicted as ANOMALOUS
    tp = cm[1, 1]  # True Positive: correctly predicted as ANOMALOUS

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fprate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnrate = fn / (fn + tp) if (fn + tp) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, fprate, fnrate, f1


def _plot_results(cm, fpr, tpr, roc_auc):
    """Plot confusion matrix and ROC curve"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Confusion Matrix Display
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["BENIGN", "ANOMALOUS"]
    )
    disp.plot(cmap="Reds", ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix")

    # ROC Curve
    axes[1].plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_xlim([-0.1, 1.1])
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show(block=False)


def _calculate_roc_metrics(test_label, pred_scores, model):
    """Calculate ROC metrics for anomaly detection"""
    # Convert labels: ANOMALOUS (-1) becomes positive class (1)
    y_true = (test_label == ANOMALOUS_LABEL).astype(int)

    fpr, tpr, thresholds = roc_curve(y_true, -pred_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def cross_validate_model(model, X, y, split=DEFAULT_CV_SPLITS):
    """Perform cross validation on the model"""
    if split < 2:
        raise ValueError("split must be at least 2")

    cv = StratifiedKFold(n_splits=split, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1"]

    try:
        results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        print("\nCross validation results (training data):")
        for score in scoring:
            print(f"{score}: {results[f'test_{score}'].mean():.4f}")
    except Exception as e:
        print(f"Cross validation failed: {e}")
        raise


def evaluate_model(model, test_data, test_label, with_numpy=True):
    """
    Evaluate model performance on test data.

    Args:
        model: Trained model with predict method
        test_data: Test features
        test_label: Test labels (1 for benign, -1 for anomalous)
        with_numpy: Whether to convert data to numpy arrays

    Returns:
        Tuple of (precision, recall, fprate, fnrate, f1, roc_auc)
    """
    if with_numpy:
        try:
            test_data = test_data.to_numpy()
            test_label = test_label.to_numpy()
        except AttributeError:
            print("Input data is already in numpy format, no conversion needed.")

    pred = model.predict(test_data)
    try:
        pred_scores = model.decision_function(test_data)
        print("using decision_function for prediction scores")
    except AttributeError:
        pred_scores = model.predict_proba(test_data)[:, -1]
        print("using predict_proba for prediction scores")

    # info for test_label
    benign_count = np.sum(test_label == BENIGN_LABEL)
    benign_ratio = benign_count / len(test_label) * 100
    print(f"\nTest Data - BENIGN Count: {benign_count}, Ratio: {benign_ratio:.2f}%")
    anomalous_count = np.sum(test_label == ANOMALOUS_LABEL)
    anomalous_ratio = anomalous_count / len(test_label) * 100
    print(
        f"Test Data - ANOMALOUS Count: {anomalous_count}, Ratio: {anomalous_ratio:.2f}%"
    )

    print("\nTest result:")
    print(
        classification_report(
            test_label,
            pred,
            target_names=["BENIGN", "ANOMALOUS"],
            labels=[BENIGN_LABEL, ANOMALOUS_LABEL],
            zero_division=0,
        )
    )

    cm = confusion_matrix(test_label, pred, labels=[BENIGN_LABEL, ANOMALOUS_LABEL])
    print(cm)

    # Calculate metrics
    precision, recall, fprate, fnrate, f1 = _calculate_metrics(cm)

    # ROC curve
    fpr, tpr, roc_auc = _calculate_roc_metrics(test_label, pred_scores, model)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"False Positive Rate: {fprate:.3f}")
    print(f"False Negative Rate: {fnrate:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Area under the curve: {roc_auc:.3f}")

    # Plot results
    _plot_results(cm, fpr, tpr, roc_auc)
