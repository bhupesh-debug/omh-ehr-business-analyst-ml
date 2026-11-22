"""Orchestrator: generate -> validate -> features -> models -> metrics.

Run from the project root:
    ~/workspace/quant-ipo-engine/.venv/bin/python run_all.py

Deterministic: seed 42 everywhere. Hold-out is the last 61 days
(2025-11-01 to 2025-12-31) for Modules 1 and 2 (time-based). Module 3
uses a grouped random split by code family, because a time split
there would test calendar drift instead of the mapping question
("given this code's description, is it mappable?").
Every metric in the PDF and deck comes from models/*.csv written
here — nothing is hard-coded.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import charts as ch  # noqa: E402
import conformance as cf  # noqa: E402
import data_generation as dg  # noqa: E402
import data_pipeline as dp  # noqa: E402
import feature_engineering as fe  # noqa: E402
import models as md  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

MODELS = os.path.join(ROOT, "models")
CHARTS = os.path.join(ROOT, "reports", "charts")
PROC = os.path.join(ROOT, "data", "processed")
HOLDOUT_START = pd.Timestamp("2025-11-01")


def align(train_f, test_f):
    """Add zero columns so train and test share exactly the same
    feature set (a wave or message type can be absent from one half),
    then sort columns identically — otherwise the numpy arrays the
    models receive would be misaligned."""
    for col in train_f.columns:
        if col not in test_f.columns:
            test_f[col] = 0
    for col in test_f.columns:
        if col not in train_f.columns:
            train_f[col] = 0
    cols = sorted(train_f.columns)
    return train_f[cols], test_f[cols]

os.makedirs(MODELS, exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)


def main():
    print("== 1. generate synthetic data ==")
    dg.main()
    print("== 2. validate + clean ==")
    dp.main()

    mig = pd.read_csv(os.path.join(PROC, "migration.csv"),
                      parse_dates=["migration_date"])
    hl7 = pd.read_csv(os.path.join(PROC, "hl7_messages.csv"),
                      parse_dates=["msg_datetime"])
    fhir = pd.read_csv(os.path.join(PROC, "fhir_resources.csv"),
                       parse_dates=["created_date"])
    tmap = pd.read_csv(os.path.join(PROC, "terminology_map.csv"))

    # ---------------- Module 1: migration reconciliation ----------------
    print("== 3. module 1 — migration reconciliation ==")
    m_train = mig[mig["migration_date"] < HOLDOUT_START].copy()
    m_test = mig[mig["migration_date"] >= HOLDOUT_START].copy()
    # break DETECTION: the deterministic rule (target present + key
    # match + code mapped + not a duplicate) vs statistical models
    # that see only what a BA has BEFORE the checks run (field-match
    # rate, secondary tell, domain, system, wave). The rule misses
    # the silent breaks by design; the models' job is to catch some
    # of those without drowning the queue in false alarms.
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score)
    rule_flag = ((mig["target_present"] == 1) & (mig["key_match"] == 1)
                 & (mig["code_mapped"] == 1)
                 & (mig["is_duplicate"] == 0)).astype(int)
    y_true_break = m_test["has_break"].values
    y_pred_break = 1 - rule_flag.loc[m_test.index].values
    det_rule = {"model": "Rule baseline (target/key/code/duplicate flags)",
                "precision": float(precision_score(
                    y_true_break, y_pred_break, zero_division=0)),
                "recall": float(recall_score(y_true_break, y_pred_break,
                                             zero_division=0)),
                "f1": float(f1_score(y_true_break, y_pred_break,
                                     zero_division=0)),
                "roc_auc": float(roc_auc_score(
                    y_true_break, y_pred_break.astype(float)))}
    f_all_train = fe.build_migration_features(m_train)
    f_all_test = fe.build_migration_features(m_test)
    f_all_train, f_all_test = align(f_all_train, f_all_test)
    det_models, _, rf_det = md.nonconformance_models(
        f_all_train, m_train["has_break"],
        f_all_test, m_test["has_break"])
    det_rows = [det_rule] + det_models
    pd.DataFrame(det_rows).to_csv(
        os.path.join(MODELS, "migration_break_metrics.csv"), index=False)
    # per-row detection predictions, so the silent-break value of the
    # model can be checked directly (the rule catches 0% of them)
    scored_det = pd.DataFrame({
        "y_true": y_true_break,
        "y_pred_rule": y_pred_break,
        "y_pred_rf": rf_det.predict(f_all_test),
        "break_type": m_test["break_type"].values,
        "field_match_rate": m_test["field_match_rate"].values,
    })
    scored_det.to_csv(os.path.join(MODELS, "detection_scored.csv"),
                      index=False)
    print(f"  detection hold-out {len(m_test):,} rows "
          f"({int(m_test['has_break'].sum())} breaks)")
    for r in det_rows:
        print(f"  {r['model']}: P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} AUC={r['roc_auc']:.3f}")

    # break-cause classification on break rows only. The RULE's
    # diagnosis (which flag failed -> which cause) is reported
    # alongside the models: for overt breaks the flag IS the cause,
    # so the rule is near-perfect there and only misses the silent
    # breaks (it calls them "none"). The models use no flag features.
    def _rule_diagnosis(df):
        out = pd.Series("none", index=df.index, dtype=object)
        out[df["target_present"] == 0] = "missing_record"
        out[df["key_match"] == 0] = "key_mismatch"
        out[df["code_mapped"] == 0] = "code_map_failure"
        out[df["is_duplicate"] == 1] = "duplicate"
        return out

    b_train = m_train[m_train["has_break"] == 1].copy()
    b_test = m_test[m_test["has_break"] == 1].copy()
    from sklearn.metrics import accuracy_score
    rule_diag = _rule_diagnosis(b_test)
    rule_cause = {"model": "Rule diagnosis (which flag failed)",
                  "accuracy": float(accuracy_score(
                      b_test["break_type"], rule_diag)),
                  "macro_f1": float(f1_score(
                      b_test["break_type"], rule_diag, average="macro",
                      zero_division=0))}
    f_b_train = fe.build_migration_features(b_train)
    f_b_test = fe.build_migration_features(b_test)
    f_b_train, f_b_test = align(f_b_train, f_b_test)
    m1_models, imp1, _ = md.break_cause_models(
        f_b_train, b_train["break_type"], f_b_test, b_test["break_type"])
    m1 = [rule_cause] + m1_models
    pd.DataFrame(m1).to_csv(
        os.path.join(MODELS, "break_cause_metrics.csv"), index=False)
    imp1.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "break_cause_importance.csv"), index=False)
    print(f"  cause hold-out {len(b_test)} breaks "
          f"({b_test['break_type'].value_counts().to_dict()})")
    for r in m1:
        print(f"  {r['model']}: acc={r['accuracy']:.3f} "
              f"macro-F1={r['macro_f1']:.3f}")

    # ranked review queue on hold-out breaks
    q = md.review_queue_metrics(m_test[m_test["has_break"] == 1])
    pd.DataFrame([q]).to_csv(os.path.join(MODELS, "review_queue.csv"),
                             index=False)
    queue = (m_test[m_test["has_break"] == 1]
             .sort_values("risk_score", ascending=False).head(20)
             [["migration_date", "record_id", "domain", "source_system",
               "break_type", "field_match_rate", "risk_score"]])
    queue.to_csv(os.path.join(MODELS, "review_queue_top20.csv"),
                 index=False)
    print(f"  review queue top-20: {q['missing_records_in_top20']}/20 are "
          f"missing records; recall of missing records "
          f"{q['recall_of_missing_records']:.2%}")

    # ---------------- Module 2: interface conformance ----------------
    print("== 4. module 2 — HL7 v2 / FHIR conformance ==")
    # run the deterministic checker over every message/resource and
    # record which rules failed — the checker is the deliverable, and
    # its agreement with the injected defects is the first metric.
    hl7_flags = hl7["message_text"].apply(
        lambda t: cf.check_hl7(t))
    hl7["checker_conformant"] = [f[0] for f in hl7_flags]
    hl7["checker_failed_rules"] = [";".join(f[1]) for f in hl7_flags]
    fhir_flags = fhir["resource_json"].apply(
        lambda t: cf.check_fhir(json.loads(t)))
    fhir["checker_conformant"] = [f[0] for f in fhir_flags]
    fhir["checker_failed_rules"] = [";".join(f[1]) for f in fhir_flags]
    agree_hl7 = float((hl7["checker_conformant"]
                       == (hl7["nonconformant"] == 0)).mean())
    agree_fhir = float((fhir["checker_conformant"]
                        == (fhir["nonconformant"] == 0)).mean())
    print(f"  checker agreement with injected defects: HL7 {agree_hl7:.3f}, "
          f"FHIR {agree_fhir:.3f}")
    # combined interface frame for the classifier
    h = pd.DataFrame({
        "msg_datetime": hl7["msg_datetime"],
        "msg_type": hl7["msg_type"],
        "source_system": hl7["source_system"],
        "segment_count": hl7["segment_count"],
        "message_length_chars": hl7["message_length_chars"],
        "standard": "HL7v2",
        "nonconformant": hl7["nonconformant"],
        "defect_type": hl7["defect_type"],
        "checker_conformant": hl7["checker_conformant"],
    })
    f = pd.DataFrame({
        "msg_datetime": fhir["created_date"],
        "msg_type": fhir["resource_type"],
        "source_system": "Oracle-Health",
        "segment_count": fhir["resource_json"].apply(
            lambda t: len(json.loads(t))),
        "message_length_chars": fhir["resource_json"].str.len(),
        "standard": "FHIR",
        "nonconformant": fhir["nonconformant"],
        "defect_type": fhir["defect_type"],
        "checker_conformant": fhir["checker_conformant"],
    })
    iface = pd.concat([h, f], ignore_index=True)
    iface.to_csv(os.path.join(PROC, "interface_combined.csv"), index=False)
    i_train = iface[iface["msg_datetime"] < HOLDOUT_START].copy()
    i_test = iface[iface["msg_datetime"] >= HOLDOUT_START].copy()
    f_i_train = fe.build_interface_features(i_train)
    f_i_test = fe.build_interface_features(i_test)
    f_i_train, f_i_test = align(f_i_train, f_i_test)
    m2, imp2, rf2 = md.nonconformance_models(
        f_i_train, i_train["nonconformant"],
        f_i_test, i_test["nonconformant"])
    pd.DataFrame(m2).to_csv(
        os.path.join(MODELS, "nonconformance_metrics.csv"), index=False)
    imp2.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "nonconformance_importance.csv"), index=False)
    print(f"  hold-out {len(i_test):,} messages "
          f"({int(i_test['nonconformant'].sum())} nonconformant)")
    for r in m2:
        print(f"  {r['model']}: P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} AUC={r['roc_auc']:.3f}")
    # defect-cause classification on nonconformant rows only
    d_train = i_train[i_train["nonconformant"] == 1].copy()
    d_test = i_test[i_test["nonconformant"] == 1].copy()
    f_d_train = fe.build_interface_features(d_train)
    f_d_test = fe.build_interface_features(d_test)
    f_d_train, f_d_test = align(f_d_train, f_d_test)
    m2b, imp2b, _ = md.defect_cause_models(
        f_d_train, d_train["defect_type"],
        f_d_test, d_test["defect_type"])
    pd.DataFrame(m2b).to_csv(
        os.path.join(MODELS, "defect_cause_metrics.csv"), index=False)
    imp2b.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "defect_cause_importance.csv"), index=False)
    for r in m2b:
        print(f"  cause {r['model']}: acc={r['accuracy']:.3f} "
              f"macro-F1={r['macro_f1']:.3f}")
    # hold-out scores for the ROC chart
    scored = f_i_test.copy()
    scored["nonconformant"] = i_test["nonconformant"].values
    scored["rf_score"] = rf2.predict_proba(f_i_test)[:, 1]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(f_i_train)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42).fit(sc.transform(f_i_train),
                                                 i_train["nonconformant"])
    scored["lr_score"] = lr.predict_proba(sc.transform(f_i_test))[:, 1]
    scored.to_csv(os.path.join(MODELS, "nonconformance_scored.csv"),
                  index=False)

    # ---------------- Module 3: terminology mapping ----------------
    print("== 5. module 3 — terminology mapping ==")
    cov = (tmap.groupby("target_system")["mapped"]
           .agg(["mean", "count"]).reset_index()
           .rename(columns={"mean": "coverage", "count": "n_codes"}))
    cov.to_csv(os.path.join(MODELS, "coverage.csv"), index=False)
    print(cov.to_string(index=False))
    unmapped = tmap[tmap["mapped"] == 0].copy()
    # grouped split by code family (source_code prefix before '-') so
    # the same code family never appears in both halves
    unmapped["family"] = unmapped["source_code"].str.split("-").str[0] \
        + "-" + unmapped["source_system"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    tr_idx, te_idx = next(gss.split(unmapped, groups=unmapped["family"]))
    split = pd.Series("train", index=unmapped.index)
    split.iloc[te_idx] = "holdout"
    pd.DataFrame({"map_id": unmapped["map_id"], "family": unmapped["family"],
                  "split": split.values}).to_csv(
        os.path.join(MODELS, "mapping_split.csv"), index=False)
    u_train = unmapped.iloc[tr_idx].copy()
    u_test = unmapped.iloc[te_idx].copy()
    f_u_train = fe.build_mapping_features(u_train)
    f_u_test = fe.build_mapping_features(u_test)
    f_u_train, f_u_test = align(f_u_train, f_u_test)
    m3, imp3, _ = md.mappability_models(
        f_u_train, u_train["true_mappable"],
        f_u_test, u_test["true_mappable"])
    pd.DataFrame(m3).to_csv(os.path.join(MODELS, "mapping_metrics.csv"),
                            index=False)
    imp3.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "mapping_importance.csv"), index=False)
    print(f"  unmapped {len(unmapped)} codes; train {len(u_train)}, "
          f"hold-out {len(u_test)} ({int(u_test['true_mappable'].sum())} "
          "truly mappable)")
    for r in m3:
        print(f"  {r['model']}: P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} AUC={r['roc_auc']:.3f}")
    top_unmapped = (unmapped.sort_values("frequency", ascending=False)
                    .head(10)[["source_code", "target_system",
                               "source_description", "frequency",
                               "token_overlap", "true_mappable"]])
    top_unmapped.to_csv(os.path.join(MODELS, "top_unmapped.csv"),
                        index=False)

    # ---------------- Benchmark table ----------------
    print("== 6. benchmark ==")
    # benchmark: best LEARNED model vs the interpretable baseline.
    # The rule rows are deterministic checks, not models — they are
    # reported in the tables below, not in this comparison.
    best_m1 = max([r for r in m1 if not r["model"].startswith("Rule")],
                  key=lambda r: r["accuracy"])
    best_m2 = max(m2, key=lambda r: r["f1"])
    best_m3 = max(m3, key=lambda r: r["f1"])
    bench = pd.DataFrame([
        {"module": "Migration break-cause", "baseline": "Majority-class",
         "baseline_score": next(r["accuracy"] for r in m1
                                if r["model"].startswith("Majority")),
         "best_model": best_m1["model"], "best_score": best_m1["accuracy"],
         "metric": "Accuracy"},
        {"module": "Interface nonconformance", "baseline": "Accept-all",
         "baseline_score": next(r["f1"] for r in m2
                                if r["model"].startswith("Accept-all")),
         "best_model": best_m2["model"], "best_score": best_m2["f1"],
         "metric": "F1"},
        {"module": "Terminology mappability", "baseline": "Majority-class",
         "baseline_score": next(r["f1"] for r in m3
                                if r["model"].startswith("Majority")),
         "best_model": best_m3["model"], "best_score": best_m3["f1"],
         "metric": "F1"},
    ])
    bench.to_csv(os.path.join(MODELS, "benchmark.csv"), index=False)
    print(bench.to_string(index=False))

    # ---------------- Charts ----------------
    print("== 7. charts ==")
    ch.migration_breaks_by_wave(
        os.path.join(PROC, "migration.csv"),
        os.path.join(CHARTS, "migration_breaks_by_wave.png"))
    ch.conformance_by_system(
        os.path.join(PROC, "interface_combined.csv"),
        os.path.join(CHARTS, "conformance_by_system.png"))
    ch.mapping_coverage(os.path.join(PROC, "terminology_map.csv"),
                        os.path.join(CHARTS, "mapping_coverage.png"))
    ch.feature_importance(
        os.path.join(MODELS, "break_cause_importance.csv"),
        os.path.join(CHARTS, "feature_importance.png"),
        "Top drivers of migration break causes (Random Forest)")
    ch.roc_curves(os.path.join(MODELS, "nonconformance_scored.csv"),
                  os.path.join(CHARTS, "roc_curves.png"))
    ch.comparison(os.path.join(MODELS, "nonconformance_metrics.csv"),
                  os.path.join(CHARTS, "comparison.png"),
                  "Interface nonconformance — F1 by model (hold-out)")
    ch.correlation_heatmap(os.path.join(MODELS, "nonconformance_scored.csv"),
                           os.path.join(CHARTS, "correlation_heatmap.png"))

    # ---------------- Model report ----------------
    print("== 8. model report ==")
    rf_bc = next(r for r in m1 if r["model"] == "Random Forest")
    maj_bc = next(r for r in m1 if r["model"].startswith("Majority"))
    rule_bc = next(r for r in m1 if r["model"].startswith("Rule"))
    rf_nc = next(r for r in m2 if r["model"] == "Random Forest")
    acc_nc = next(r for r in m2 if r["model"].startswith("Accept-all"))
    rf_mp = next(r for r in m3 if r["model"] == "Random Forest")
    maj_mp = next(r for r in m3 if r["model"].startswith("Majority"))

    def _f(x, pct=False):
        if isinstance(x, float) and np.isnan(x):
            return "—"
        return f"{x:.2%}" if pct else f"{x:.3f}"

    report = f"""# EHR Data Migration & Interface Analytics — Model Report

**Role:** Business Analyst #26-03354, NY State Office of Mental Health
(OMH), Albany NY — Oracle Health EHR implementation
**Author:** Bhupesh Chandra Dimri — **All data is synthetic. No real
patient data, no real PHI.**
**Hold-out:** Modules 1 & 2 — last 61 days of 2025 (2025-11-01 to
2025-12-31), time-based, no leakage. Module 3 — grouped random 70/30
split by code family (see the Module 3 note below for why).
**Tests:** see `tests/run_tests.py` (run after this script).
**Checker agreement:** the deterministic HL7/FHIR checker agrees with
the injected defects on {agree_hl7:.1%} of HL7 messages and
{agree_fhir:.1%} of FHIR resources — the disagreements are the
deliberately subtle defects: a CPT code where a LOINC is expected
(HL7 wrong_context) and a reference to a Patient id that does not
exist (FHIR stale_reference). The checker validates structure and
code membership, not context or referential integrity — that gap is
reported, not hidden.

## Module 1 — Migration reconciliation (hold-out: {len(m_test):,} records, {int(m_test['has_break'].sum())} true breaks)
Break detection — the deterministic rule catches every overt break
and misses the silent ones by design; the statistical models see
only pre-check features (field-match rate, secondary tell, domain,
system, wave) and earn their place by catching some silent breaks:
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
"""
    for r in det_rows:
        report += (f"| {r['model']} | {_f(r['precision'])} | {_f(r['recall'])} | "
                   f"{_f(r['f1'])} | {_f(r['roc_auc'])} |\n")
    report += f"""
Break-cause diagnosis (hold-out: {len(b_test)} true breaks). The rule
diagnosis reads which flag failed, so it names every overt break and
only misses the silent ones (it calls them "none"). The learned
models use no flag features — that is why their accuracy is modest,
and why it is honest:
| Model | Accuracy | Macro-F1 |
|---|---|---|
"""
    for r in m1:
        report += f"| {r['model']} | {_f(r['accuracy'])} | {_f(r['macro_f1'])} |\n"
    report += f"""
Top drivers: {', '.join(imp1.head(5).index.tolist())}.
Ranked review queue (top 20 by risk score): {q['missing_records_in_top20']}/20 are
true missing records; queue recall of missing records {q['recall_of_missing_records']:.2%}.

## Module 2 — Interface conformance (hold-out: {len(i_test):,} messages/resources, {int(i_test['nonconformant'].sum())} nonconformant)
Conformance rate overall: {(1 - iface['nonconformant'].mean()):.1%} HL7, {(1 - fhir['nonconformant'].mean()):.1%} FHIR (full year).
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
"""
    for r in m2:
        report += (f"| {r['model']} | {_f(r['precision'])} | {_f(r['recall'])} "
                   f"| {_f(r['f1'])} | {_f(r['roc_auc'])} |\n")
    report += """
Defect-cause classification (nonconformant rows only):
| Model | Accuracy | Macro-F1 |
|---|---|---|
"""
    for r in m2b:
        report += f"| {r['model']} | {_f(r['accuracy'])} | {_f(r['macro_f1'])} |\n"
    report += f"""
The metadata-only classifier is honestly weak (best F1
{max(r['f1'] for r in m2 if r['model'] != 'Accept-all baseline'):.3f}):
it sees only message length, segment/field counts, type, source and
month, so content defects — a bad code, a bad ID, a wrong-context
code — are invisible to it. The deterministic checker is the real
Module 2 tool (agreement {agree_hl7:.1%} HL7 / {agree_fhir:.1%}
FHIR); the classifier is a triage aid for the queue, not the
detector. Do not cite the classifier's F1 as the project's
conformance performance — cite the checker's agreement.

## Module 3 — Terminology mapping (unmapped codes: {len(unmapped)}, hold-out: {len(u_test)})
Coverage by target system:
"""
    for _, row in cov.iterrows():
        report += (f"- {row['target_system']}: {row['coverage']:.1%} "
                   f"mapped (n={int(row['n_codes'])})\n")
    report += """
Mappability — can an unmapped code be mapped without manual review?
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
"""
    for r in m3:
        report += (f"| {r['model']} | {_f(r['precision'])} | {_f(r['recall'])} "
                   f"| {_f(r['f1'])} | {_f(r['roc_auc'])} |\n")
    report += f"""
Top unmapped codes by frequency are in models/top_unmapped.csv — the
first thing a BA triages, because one unmapped high-frequency code
breaks more records than fifty rare ones.

## Benchmark (best model vs interpretable baseline)
| Module | Baseline | Best model | Metric |
|---|---|---|---|
"""
    for _, b in bench.iterrows():
        report += (f"| {b['module']} | {b['baseline']} {b['baseline_score']:.3f} | "
                   f"{b['best_model']} {b['best_score']:.3f} | {b['metric']} |\n")
    report += """
## Limits (read before citing any number)
- All data is synthetic: patient names, MRNs, dates, codes, and messages
  are invented. Break rates, defect rates, and coverage are design
  choices, not OMH facts. A model that scores well here proves the
  *pipeline design*, not performance on a real migration.
- Injected defects are deliberately detectable (missing segments, unknown
  codes, bad IDs). Real interface defects are subtler — expect lower
  recall on live feeds and keep a human in the review queue.
- The mappability label is a latent simulation variable, not a clinician's
  judgment; on real unmapped codes a terminologist decides.
- HL7 v2 and FHIR rules implemented here are a subset of the real
  specifications (six HL7 rules, three FHIR rule groups). A production
  conformance suite covers the full IG.
- The candidate does not claim HL7/FHIR experience, healthcare-standards
  expertise, or a BA certification — all are [CONFIRM WITH CANDIDATE]
  items, not facts in this project.
"""
    with open(os.path.join(ROOT, "reports", "model_report.md"), "w") as fh:
        fh.write(report)
    print("  reports/model_report.md written")
    print("DONE. Next: python tests/run_tests.py")


if __name__ == "__main__":
    main()
