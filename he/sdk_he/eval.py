from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelBinarizer

def _threshold_search_with_objective(y_true: np.ndarray, y_score: np.ndarray,
                                     tmin: float, tmax: float, points: int,
                                     objective: str, beta: float, min_precision: float):
    thresholds = np.linspace(tmin, tmax, points)
    best_t, best_val = thresholds[0], -1.0
    for t in thresholds:
        pred = (y_score > t).astype(int)
        if objective == "recall":
            val = recall_score(y_true, pred, zero_division=0)
        elif objective == "f1":
            val = f1_score(y_true, pred, zero_division=0)
        elif objective == "youden":
            # Youden's J = TPR - FPR
            tp = np.sum((pred == 1) & (y_true == 1))
            fn = np.sum((pred == 0) & (y_true == 1))
            fp = np.sum((pred == 1) & (y_true == 0))
            tn = np.sum((pred == 0) & (y_true == 0))
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            val = tpr - fpr
        else:  # fbeta
            prec = precision_score(y_true, pred, zero_division=0)
            if prec < min_precision:
                val = -1.0
            else:
                val = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if val > best_val:
            best_val, best_t = val, t
    return float(best_t), float(best_val)

def evaluate_classifier(
    X_test: np.ndarray,
    y_test: np.ndarray,
    weights_list: List[np.ndarray],
    classes_: np.ndarray,
    threshold_search: bool,
    threshold_range: Tuple[float, float, int],
    compute_roc: bool,
    use_threshold_for_prediction: bool,
    threshold_objective: str = "fbeta",
    fbeta_beta: float = 2.0,
    min_precision: float = 0.0,
    positive_class: int = 1,
) -> Dict[str, Any]:
    n_classes = len(classes_)
    y_pred_proba = np.zeros((X_test.shape[0], n_classes), dtype=float)

    # 逻辑回归概率（稳定版 sigmoid）
    for i, w in enumerate(weights_list):
        logits = X_test @ w
        logits = np.clip(logits, -50, 50)
        y_pred_proba[:, i] = 1.0 / (1.0 + np.exp(-logits))

    # 默认 argmax
    y_pred = np.argmax(y_pred_proba, axis=1)
    threshold = None

    if n_classes == 2 and threshold_search:
        # 针对指定正类做阈值搜索（使用该类的 logits 更稳定）
        logits_pos = np.clip(X_test @ weights_list[int(positive_class)], -50, 50)
        tmin, tmax, pts = threshold_range
        threshold, _ = _threshold_search_with_objective(
            (y_test == positive_class).astype(int),
            logits_pos,
            tmin, tmax, pts,
            threshold_objective, fbeta_beta, min_precision
        )
        if use_threshold_for_prediction:
            y_pred = (logits_pos > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_dict = classification_report(y_test, y_pred, target_names=classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    roc_info = {}
    if compute_roc:
        lb = LabelBinarizer().fit(classes_)
        y_test_bin = lb.transform(y_test)
        if n_classes == 2:
            y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
        roc_auc = {}
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr_i, tpr_i)
        roc_info = {"auc_by_class": {int(i): float(roc_auc[i]) for i in range(n_classes)}}

    return {
        "accuracy": float(acc),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "threshold": (float(threshold) if threshold is not None else None),
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "roc": roc_info,
    }