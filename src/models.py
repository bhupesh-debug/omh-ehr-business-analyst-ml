"""Model training + evaluation for the three modules.

Module 1 — migration break detection (binary) and break-cause
    (5-class, break rows only): rule baseline | accept-all |
    Logistic Regression | Random Forest. The rule flags are
    deliberately EXCLUDED from the model features — they are the
    break cause by construction, so including them would make the
    classifier a lookup table.
Module 2 — interface nonconformance (binary, HL7 + FHIR rows):
    accept-all baseline | Logistic Regression | Random Forest.
    Precision/recall/F1/ROC-AUC. Plus defect-cause classification
    (nonconformant rows only): majority baseline | Random Forest.
Module 3 — terminology mappability (binary, unmapped codes only):
    majority baseline | Logistic Regression | Random Forest.
    Precision/recall/F1/ROC-AUC. Split is GROUPED by code family
    (source_code prefix) so the same code family never appears in
    both train and hold-out.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler


def _clf_metrics(y_true, y_pred, y_score=None):
    out = {"precision": float(precision_score(y_true, y_pred,
                                              zero_division=0)),
           "recall": float(recall_score(y_true, y_pred, zero_division=0)),
           "f1": float(f1_score(y_true, y_pred, zero_division=0))}
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
    return out


def _multiclass(X_train, y_train, X_test, y_test, label):
    """Majority baseline + Random Forest for a multi-class target."""
    rows = []
    maj = y_train.mode().iloc[0]
    pred = np.full(len(y_test), maj)
    rows.append({"model": "Majority-class baseline",
                 "accuracy": float(accuracy_score(y_test, pred)),
                 "macro_f1": float(f1_score(y_test, pred, average="macro",
                                            zero_division=0))})
    rf = RandomForestClassifier(n_estimators=300,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    rows.append({"model": "Random Forest",
                 "accuracy": float(accuracy_score(y_test, pred)),
                 "macro_f1": float(f1_score(y_test, pred, average="macro",
                                            zero_division=0))})
    imp = pd.Series(rf.feature_importances_,
                    index=X_train.columns).sort_values(ascending=False)
    return rows, imp, rf


def break_cause_models(X_train, y_train, X_test, y_test):
    return _multiclass(X_train, y_train, X_test, y_test, "break")


def defect_cause_models(X_train, y_train, X_test, y_test):
    return _multiclass(X_train, y_train, X_test, y_test, "defect")


def nonconformance_models(X_train, y_train, X_test, y_test):
    """Binary: 1 = nonconformant message/resource."""
    rows = []
    pred = np.zeros(len(y_test), dtype=int)  # accept-all baseline
    rows.append({"model": "Accept-all baseline",
                 **_clf_metrics(y_test, pred, pred.astype(float))})
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42)
    lr.fit(Xtr_s, y_train)
    rows.append({"model": "Logistic Regression",
                 **_clf_metrics(y_test, lr.predict(Xte_s),
                                lr.predict_proba(Xte_s)[:, 1])})
    rf = RandomForestClassifier(n_estimators=300,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rows.append({"model": "Random Forest",
                 **_clf_metrics(y_test, rf.predict(X_test),
                                rf.predict_proba(X_test)[:, 1])})
    imp = pd.Series(rf.feature_importances_,
                    index=X_train.columns).sort_values(ascending=False)
    return rows, imp, rf


def mappability_models(X_train, y_train, X_test, y_test):
    """Binary: 1 = unmapped code is truly mappable (latent truth)."""
    rows = []
    maj = int(y_train.mode().iloc[0])
    pred = np.full(len(y_test), maj)
    rows.append({"model": "Majority-class baseline",
                 **_clf_metrics(y_test, pred, pred.astype(float))})
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42)
    lr.fit(Xtr_s, y_train)
    rows.append({"model": "Logistic Regression",
                 **_clf_metrics(y_test, lr.predict(Xte_s),
                                lr.predict_proba(Xte_s)[:, 1])})
    rf = RandomForestClassifier(n_estimators=300,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rows.append({"model": "Random Forest",
                 **_clf_metrics(y_test, rf.predict(X_test),
                                rf.predict_proba(X_test)[:, 1])})
    imp = pd.Series(rf.feature_importances_,
                    index=X_train.columns).sort_values(ascending=False)
    return rows, imp, rf


def review_queue_metrics(mig_holdout):
    """Top-20 ranked queue: how many of the 20 highest-risk open breaks
    are true missing records (the go-live-critical class), and what share
    of ALL missing records the top 20 captures."""
    q = mig_holdout.sort_values("risk_score", ascending=False).head(20)
    n_miss_top = int((q["break_type"] == "missing_record").sum())
    n_miss_all = int((mig_holdout["break_type"] == "missing_record").sum())
    return {
        "queue_size": 20,
        "missing_records_in_top20": n_miss_top,
        "missing_records_total_holdout": n_miss_all,
        "precision_at_20": n_miss_top / 20,
        "recall_of_missing_records": (n_miss_top / n_miss_all
                                      if n_miss_all else 0.0),
    }
