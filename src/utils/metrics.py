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


def compute_summary_row_from_seed_runs(
    runs: list,
    ddof: int = 0,
) -> Dict[str, Any]:
    """
    Rule G16: Single authoritative function computing summary row statistics
    (mean and std) strictly from the printed per-seed rows.
    """
    assert len(runs) > 0, "Runs list cannot be empty"

    aucs = [float(r["auc"]) for r in runs]
    accs = [float(r["accuracy"]) for r in runs]
    precs = [float(r["precision"]) for r in runs]
    recs = [float(r["recall"]) for r in runs]
    f1s = [float(r["f1"]) for r in runs]
    briers = [float(r["brier_score"]) for r in runs]

    return {
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs, ddof=ddof)),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=ddof)),
        "mean_precision": float(np.mean(precs)),
        "std_precision": float(np.std(precs, ddof=ddof)),
        "mean_recall": float(np.mean(recs)),
        "std_recall": float(np.std(recs, ddof=ddof)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s, ddof=ddof)),
        "mean_brier_score": float(np.mean(briers)),
        "std_brier_score": float(np.std(briers, ddof=ddof)),
        "count": len(runs),
    }


def find_optimal_threshold(
    y_true: Union[np.ndarray, list],
    y_probs: Union[np.ndarray, list],
    metric: str = "balanced_accuracy",
) -> float:
    """
    Rule G18 & G23: Tunes threshold strictly on validation split per arm per seed.
    Selects optimal threshold maximizing balanced accuracy (or F1 score).
    Balanced accuracy = (TPR + TNR) / 2 = 0.5 * (TP / (TP + FN) + TN / (TN + FP)).
    """
    y_t = np.asarray(y_true).astype(int).ravel()
    y_p = np.asarray(y_probs).astype(float).ravel()

    best_th = 0.5
    best_score = -1.0

    # Grid search across candidate thresholds [0.10, 0.90]
    for th in np.linspace(0.10, 0.90, 81):
        bin_pred = (y_p >= th).astype(int)
        tp = np.sum((y_t == 1) & (bin_pred == 1))
        fp = np.sum((y_t == 0) & (bin_pred == 1))
        fn = np.sum((y_t == 1) & (bin_pred == 0))
        tn = np.sum((y_t == 0) & (bin_pred == 0))

        if metric == "balanced_accuracy":
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = 0.5 * (tpr + tnr)
        else:
            denom = 2.0 * tp + fp + fn
            score = (2.0 * tp / denom) if denom > 0 else 0.0

        if score > best_score:
            best_score = score
            best_th = float(th)

    return round(best_th, 4)


