"""
ForenSightX Central Metrics Module
Iteration 8 Integrity First - Enforcing Global Rule G12

Rule G12: One metrics function; accuracy/precision/recall/F1/Brier derive from
a printed confusion matrix (TP, FP, FN, TN, threshold); a test recomputes every
printed metric from it.
"""

from typing import Dict, Any, Union, Optional
import numpy as np
import math
from sklearn.metrics import roc_auc_score


def derive_metrics_from_confusion_matrix(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    threshold: float = 0.5,
    brier_score: Optional[float] = None,
    auc: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Strict mathematical derivation of classification metrics from integer confusion matrix.
    Guarantees closed-form algebraic consistency across all reports and unit tests.
    """
    total = tp + fp + fn + tn
    if total == 0:
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "total": 0,
            "threshold": threshold,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "brier_score": brier_score if brier_score is not None else 0.0,
            "auc": auc if auc is not None else 0.5,
        }

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total": int(total),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_score": float(brier_score) if brier_score is not None else None,
        "auc": float(auc) if auc is not None else None,
    }


def compute_confusion_and_derived_metrics(
    y_true: Union[np.ndarray, list],
    y_pred_scores: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Sole authoritative evaluation function for binary classification across ForenSightX.
    Computes integer confusion matrix, empirical Brier score, and ROC-AUC.
    """
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    y_scores_arr = np.asarray(y_pred_scores).astype(float).ravel()

    assert len(y_true_arr) == len(y_scores_arr), "Label and score lengths must match"

    # Discrete binary classification at specified threshold
    y_pred_bin = (y_scores_arr >= threshold).astype(int)

    tp = int(np.sum((y_true_arr == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_bin == 0)))

    # Continuous Brier score: mean squared probability error
    brier = float(np.mean((y_scores_arr - y_true_arr) ** 2))

    # ROC-AUC score
    if len(np.unique(y_true_arr)) > 1:
        try:
            auc = float(roc_auc_score(y_true_arr, y_scores_arr))
        except ValueError:
            auc = 0.5
    else:
        auc = 0.5

    metrics = derive_metrics_from_confusion_matrix(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        threshold=threshold,
        brier_score=brier,
        auc=auc,
    )
    return metrics

