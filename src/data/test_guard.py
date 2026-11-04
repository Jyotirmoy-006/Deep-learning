"""
ForenSightX Single-Use Test-Split Guard & Integrity Module
Iteration 8 Integrity First - Enforcing Global Rules G13 and G14

Rule G13: Test-split access goes through a single-use guard that logs caller,
count and a hash of the predictions; selection, grid and threshold code only
receives train/val; a test fails if a selection module can import the test loader.

Rule G14: Every N-seed table prints per seed the init-weight hash, final-weight
hash and data-order hash; a test asserts >= N distinct final-weight hashes and
that metrics differ across seeds somewhere.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import inspect
import hashlib
import numpy as np
import torch
import torch.nn as nn
from src.utils.metrics import compute_confusion_and_derived_metrics


class TestLeakageError(RuntimeError):
    """Raised when test split access rules (Rule G13) are violated."""
    __test__ = False


class SingleUseTestGuard:
    """
    Enforces Rule G13: Guards the test split against multiple access,
    logging caller provenance, evaluation counts, and prediction hashes.
    """

    def __init__(
        self,
        X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        split_name: str = "test",
    ):
        if X_test is not None:
            self.X_test = np.asarray(X_test) if isinstance(X_test, (np.ndarray, list)) else X_test.detach().cpu().numpy()
        else:
            self.X_test = None
        if y_test is not None:
            self.y_test = np.asarray(y_test).astype(int).ravel() if isinstance(y_test, (np.ndarray, list)) else y_test.detach().cpu().numpy().astype(int).ravel()
        else:
            self.y_test = None
        self.split_name = split_name
        self._evaluated_seeds: set = set()
        self._access_log: List[Dict[str, Any]] = []

    @property
    def access_log(self) -> List[Dict[str, Any]]:
        return list(self._access_log)

    def get_access_log(self) -> List[Dict[str, Any]]:
        return list(self._access_log)

    @property
    def total_evaluations(self) -> int:
        return len(self._access_log)

    def evaluate(
        self,
        caller_name: str,
        sample_count: int,
        predict_fn_or_probs: Union[Callable[[], np.ndarray], np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Logs caller, sample count, and prediction SHA-256 hash for general test evaluations (Rule G13).
        """
        if callable(predict_fn_or_probs):
            with torch.no_grad():
                preds = predict_fn_or_probs()
        else:
            preds = predict_fn_or_probs

        if isinstance(preds, torch.Tensor):
            preds_np = preds.detach().cpu().numpy().astype(float).ravel()
        else:
            preds_np = np.asarray(preds).astype(float).ravel()

        pred_hash = hashlib.sha256(preds_np.tobytes()).hexdigest()
        self._access_log.append({
            "evaluation_index": len(self._access_log) + 1,
            "caller": caller_name,
            "sample_count": sample_count,
            "prediction_sha256": pred_hash,
        })
        return preds_np

    def evaluate_test(
        self,
        seed: int,
        predict_fn_or_probs: Union[Callable[[np.ndarray], np.ndarray], np.ndarray, torch.Tensor],
        threshold: float = 0.5,
        caller_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single test evaluation strictly guarded per seed.
        Raises TestLeakageError if the seed has already been evaluated or if caller is unauthorized.
        """
        # Inspect caller stack to detect unauthorized selection/grid search routines
        stack = inspect.stack()
        calling_frame = stack[1]
        detected_caller = f"{calling_frame.filename}:{calling_frame.function}:{calling_frame.lineno}"
        active_caller = caller_name or detected_caller

        # Prohibit access from hyperparameter grid or selection functions
        disallowed_keywords = ["grid", "sweep", "select_winner", "tune_threshold", "optimize_hyperparams"]
        for kw in disallowed_keywords:
            if kw in active_caller.lower() or kw in calling_frame.function.lower():
                raise TestLeakageError(
                    f"Rule G13 VIOLATION: Test split cannot be accessed by selection/grid routine: '{active_caller}'"
                )

        # Enforce single-use per seed
        if seed in self._evaluated_seeds:
            raise TestLeakageError(
                f"Rule G13 VIOLATION: Test split already evaluated for seed {seed}. Multiple test evaluations prohibited."
            )

        # Generate predictions
        if callable(predict_fn_or_probs):
            preds = predict_fn_or_probs(self.X_test)
        else:
            preds = predict_fn_or_probs

        if isinstance(preds, torch.Tensor):
            preds_np = preds.detach().cpu().numpy().astype(float).ravel()
        else:
            preds_np = np.asarray(preds).astype(float).ravel()

        assert len(preds_np) == len(self.y_test), (
            f"Prediction count {len(preds_np)} must equal test sample count {len(self.y_test)}"
        )

        # Compute SHA-256 of predictions for immutable reproducibility audit
        pred_bytes = preds_np.tobytes()
        pred_hash = hashlib.sha256(pred_bytes).hexdigest()

        # Compute authoritative confusion matrix & derived metrics
        metrics = compute_confusion_and_derived_metrics(
            y_true=self.y_test,
            y_pred_scores=preds_np,
            threshold=threshold,
        )

        # Record audit log entry
        log_entry = {
            "evaluation_index": len(self._access_log) + 1,
            "seed": seed,
            "caller": active_caller,
            "sample_count": len(self.y_test),
            "prediction_sha256": pred_hash,
            "threshold": threshold,
            "auc": metrics["auc"],
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
        }
        self._access_log.append(log_entry)
        self._evaluated_seeds.add(seed)

        metrics["prediction_sha256"] = pred_hash
        metrics["guard_access_index"] = log_entry["evaluation_index"]
        return metrics


def compute_model_weight_hash(model: nn.Module) -> str:
    """
    Computes deterministic SHA-256 hash of concatenated parameter tensors.
    Used to track init_weight_hash and final_weight_hash under Rule G14.
    """
    param_tensors = []
    for name, p in sorted(model.named_parameters()):
        param_tensors.append(p.detach().cpu().contiguous().view(-1))
    if not param_tensors:
        return hashlib.sha256(b"empty_model").hexdigest()[:16]
    concat_bytes = torch.cat(param_tensors).numpy().tobytes()
    return hashlib.sha256(concat_bytes).hexdigest()[:16]


def compute_data_order_hash(indices_or_ids: Union[List[Any], np.ndarray]) -> str:
    """
    Computes deterministic SHA-256 hash of sample presentation order during training.
    Used to track data_order_hash under Rule G14.
    """
    id_strs = [str(x) for x in indices_or_ids]
    order_bytes = ",".join(id_strs).encode("utf-8")
    return hashlib.sha256(order_bytes).hexdigest()[:16]

