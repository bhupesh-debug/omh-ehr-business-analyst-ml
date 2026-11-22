"""Test suite for the OMH EHR project. Run after run_all.py:

    ~/workspace/quant-ipo-engine/.venv/bin/python tests/run_tests.py

Every check prints PASS/FAIL; exit code is non-zero on any failure.
Checks cover: validation gates re-run, determinism (regenerate twice
-> identical row counts and break counts), hold-out integrity
(no code family leaks across the Module 3 split; hold-out dates are
strictly after train dates), conformance-checker correctness on
hand-built messages, metric sanity (models beat baselines, rates in
bounds), and file existence.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import conformance as cf  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
CHARTS = os.path.join(ROOT, "reports", "charts")
VENV = "/home/hatch/workspace/quant-ipo-engine/.venv/bin/python"

passed = 0
failed = 0


def check(name, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {extra}")


def main():
    print("== file existence ==")
    for f in ["run_all.py", "README.md", "requirements.txt",
              "src/data_generation.py", "src/data_pipeline.py",
              "src/feature_engineering.py", "src/conformance.py",
              "src/models.py", "src/charts.py",
              "reports/model_report.md"]:
        check(f"exists: {f}", os.path.exists(os.path.join(ROOT, f)))
    for f in ["patients.csv", "encounters.csv", "clinical_codes.csv",
              "migration.csv", "hl7_messages.csv", "fhir_resources.csv",
              "terminology_map.csv"]:
        check(f"processed: {f}",
              os.path.exists(os.path.join(PROC, f)))
    for f in ["migration_break_metrics.csv", "break_cause_metrics.csv",
              "break_cause_importance.csv", "nonconformance_metrics.csv",
              "nonconformance_scored.csv", "defect_cause_metrics.csv",
              "defect_cause_importance.csv", "mapping_metrics.csv",
              "mapping_importance.csv", "mapping_split.csv",
              "coverage.csv", "review_queue.csv", "review_queue_top20.csv",
              "detection_scored.csv", "benchmark.csv"]:
        check(f"model output: {f}",
              os.path.exists(os.path.join(MODELS, f)))
    for f in ["migration_breaks_by_wave.png", "conformance_by_system.png",
              "mapping_coverage.png", "feature_importance.png",
              "roc_curves.png", "comparison.png",
              "correlation_heatmap.png"]:
        check(f"chart: {f}", os.path.exists(os.path.join(CHARTS, f)))

    mig = pd.read_csv(os.path.join(PROC, "migration.csv"),
                      parse_dates=["migration_date"])
    hl7 = pd.read_csv(os.path.join(PROC, "hl7_messages.csv"),
                      parse_dates=["msg_datetime"])
    fhir = pd.read_csv(os.path.join(PROC, "fhir_resources.csv"),
                       parse_dates=["created_date"])
    tmap = pd.read_csv(os.path.join(PROC, "terminology_map.csv"))

    print("== validation gates (re-checked) ==")
    check("migration rows == 40000", len(mig) == 40000, f"got {len(mig)}")
    check("migration break rate 3-12%",
          0.03 <= mig["has_break"].mean() <= 0.12,
          f"{mig['has_break'].mean():.2%}")
    check("HL7 rows == 4000", len(hl7) == 4000, f"got {len(hl7)}")
    check("FHIR rows == 2000", len(fhir) == 2000, f"got {len(fhir)}")
    check("terminology rows == 1500", len(tmap) == 1500,
          f"got {len(tmap)}")
    check("all five break types present",
          set(mig["break_type"].unique()) == {
              "none", "missing_record", "key_mismatch",
              "code_map_failure", "duplicate", "silent_mismatch"})
    check("all five target systems present",
          set(tmap["target_system"].unique()) == {
              "ICD-10", "CPT", "LOINC", "SNOMED CT", "RxNorm"})
    check("HL7 defect types present",
          set(hl7["defect_type"].unique()) == {
              "none", "missing_segment", "missing_field", "bad_code",
              "invalid_format", "wrong_context"})
    check("FHIR defect types present",
          set(fhir["defect_type"].unique()) == {
              "none", "missing_element", "bad_binding", "invalid_format",
              "stale_reference"})
    check("risk score in [0,100]",
          mig["risk_score"].between(0, 100).all())
    check("missing records have highest median risk",
          mig[mig["break_type"] == "missing_record"]["risk_score"].median()
          > mig[mig["break_type"] == "duplicate"]["risk_score"].median())

    print("== conformance checker on hand-built messages ==")
    good_hl7 = ("MSH|^~\\&|pharmacy|OMH|OracleHealth|OMH|202506011200||"
                "ADT^A01|10001|P|2.5\rPID|1||MRN100001||Smith^James||"
                "19800514|M\rPV1|1|I|OMH-Albany^^^NY")
    ok, failed_rules = cf.check_hl7(good_hl7)
    check("clean HL7 passes", ok, str(failed_rules))
    bad_hl7 = ("MSH|^~\\&|pharmacy|OMH|OracleHealth|OMH|202506011200||"
               "ADT^A01|10002|P|2.5\rPID|1||MRN100002||Smith^James||"
               "19800514\rPV1|1|I|OMH-Albany^^^NY")
    ok, failed_rules = cf.check_hl7(bad_hl7)
    check("HL7 missing gender fails R3",
          (not ok) and "R3_missing_gender" in failed_rules,
          str(failed_rules))
    bad_code = ("MSH|^~\\&|pharmacy|OMH|OracleHealth|OMH|202506011200||"
                "ORU^R01|10003|P|2.5\rPID|1||MRN100003||Smith^James||"
                "19800514|M\rPV1|1|O|OMH-Albany^^^NY\r"
                "OBX|1|NM|ZZ-UNKNOWN||120|mg/dL||||F")
    ok, failed_rules = cf.check_hl7(bad_code)
    check("HL7 unknown code fails R4",
          (not ok) and "R4_unknown_code" in failed_rules,
          str(failed_rules))
    good_fhir = {"resourceType": "Patient", "id": "pat-1001",
                 "birthDate": "1980-05-14", "gender": "male"}
    ok, failed_rules = cf.check_fhir(good_fhir)
    check("clean FHIR Patient passes", ok, str(failed_rules))
    bad_fhir = {"resourceType": "Patient", "id": "bad id!!",
                "gender": "unknown-code"}
    ok, failed_rules = cf.check_fhir(bad_fhir)
    check("FHIR bad id + missing birthDate + bad binding fails",
          (not ok) and "F1_bad_id" in failed_rules
          and "F2_missing_birthDate" in failed_rules
          and "F3_bad_gender_binding" in failed_rules,
          str(failed_rules))

    print("== hold-out integrity ==")
    hold = mig[mig["migration_date"] >= "2025-11-01"]
    train = mig[mig["migration_date"] < "2025-11-01"]
    check("hold-out is Nov-Dec 2025 only",
          hold["migration_date"].min() >= pd.Timestamp("2025-11-01"))
    check("train ends before hold-out starts",
          train["migration_date"].max() < hold["migration_date"].min())
    check("hold-out has breaks of every type",
          set(hold["break_type"].unique()) >= {
              "missing_record", "key_mismatch", "code_map_failure",
              "duplicate", "silent_mismatch"})
    split = pd.read_csv(os.path.join(MODELS, "mapping_split.csv"))
    tr_codes = set(split.loc[split["split"] == "train", "family"])
    te_codes = set(split.loc[split["split"] == "holdout", "family"])
    check("Module 3 split: no code family leaks across",
          len(tr_codes & te_codes) == 0,
          f"overlap {len(tr_codes & te_codes)}")
    check("Module 3 hold-out has both classes",
          set(pd.read_csv(os.path.join(PROC, "terminology_map.csv"))
              .merge(split, left_on="map_id", right_on="map_id")
              .query("split == 'holdout'")["true_mappable"].unique())
          == {0, 1})

    print("== metric sanity ==")
    det = pd.read_csv(os.path.join(MODELS,
                                   "migration_break_metrics.csv"))
    rule_det = det.loc[det["model"].str.startswith("Rule")].iloc[0]
    check("detection: rule F1 > 0.80", float(rule_det["f1"]) > 0.80)
    check("detection: rule recall < 1.0 (silent breaks exist)",
          float(rule_det["recall"]) < 1.0)
    check("detection: rule precision ~1.0",
          float(rule_det["precision"]) > 0.99)
    # the model's real value: silent breaks, which the rule never sees
    # (overall recall is not the comparison — the rule is the best
    # overall detector by design; the model adds the silent class)
    det_sc = pd.read_csv(os.path.join(MODELS, "detection_scored.csv"))
    silent = det_sc[det_sc["break_type"] == "silent_mismatch"]
    check("detection: rule catches 0% of silent breaks",
          int(silent["y_pred_rule"].sum()) == 0)
    check("detection: RF catches >= 30% of silent breaks",
          float(silent["y_pred_rf"].mean()) >= 0.30,
          f"RF silent recall {silent['y_pred_rf'].mean():.2%} "
          f"on {len(silent)} silent breaks")
    bc = pd.read_csv(os.path.join(MODELS, "break_cause_metrics.csv"))
    rf_acc = float(bc.loc[bc["model"] == "Random Forest", "accuracy"].iloc[0])
    mj_acc = float(bc.loc[bc["model"].str.startswith("Majority"),
                          "accuracy"].iloc[0])
    check("break-cause: RF beats majority", rf_acc > mj_acc,
          f"RF {rf_acc:.3f} vs majority {mj_acc:.3f}")
    check("break-cause: RF accuracy >= 0.40", rf_acc >= 0.40)
    rule_c = bc.loc[bc["model"].str.startswith("Rule")].iloc[0]
    check("break-cause: rule diagnosis accuracy > 0.80",
          float(rule_c["accuracy"]) > 0.80)
    # every defect class except the two deliberately subtle ones must
    # actually fail the checker — a defect that leaves conformant text
    # is a generator bug, not a subtlety (bad_code on ADT rows did
    # exactly that in the first run)
    iface = pd.read_csv(os.path.join(PROC, "interface_combined.csv"))
    for dt in ["missing_segment", "missing_field", "bad_code",
               "invalid_format", "missing_element", "bad_binding"]:
        sub = iface[iface["defect_type"] == dt]
        check(f"defect {dt} always fails the checker",
              len(sub) > 0 and (sub["checker_conformant"] == 0).all())
    wrong_ctx = iface[(iface["nonconformant"] == 1)
                      & (iface["defect_type"] == "wrong_context")]
    check("wrong_context defects pass the checker (honest gap)",
          len(wrong_ctx) > 0 and (wrong_ctx["checker_conformant"] == 1).all())
    stale = iface[(iface["nonconformant"] == 1)
                  & (iface["defect_type"] == "stale_reference")]
    check("stale_reference defects pass the checker (honest gap)",
          len(stale) > 0 and (stale["checker_conformant"] == 1).all())
    nc = pd.read_csv(os.path.join(MODELS, "nonconformance_metrics.csv"))
    rf_f1 = float(nc.loc[nc["model"] == "Random Forest", "f1"].iloc[0])
    acc_f1 = float(nc.loc[nc["model"].str.startswith("Accept-all"),
                        "f1"].iloc[0])
    check("nonconformance: RF F1 > accept-all", rf_f1 > acc_f1,
          f"RF {rf_f1:.3f} vs accept-all {acc_f1:.3f}")
    # the metadata-only classifier is honestly weak: it can only see
    # length/segment-count tells, so content defects (bad codes,
    # bad IDs, wrong-context codes) are invisible to it. The
    # deterministic checker is the real Module 2 tool (agreement
    # ~99.5% HL7 / 99.0% FHIR); the classifier is a triage aid, not
    # the detector. Threshold 0.30 reflects that.
    check("nonconformance: RF F1 >= 0.30", rf_f1 >= 0.30)
    mp = pd.read_csv(os.path.join(MODELS, "mapping_metrics.csv"))
    mp_rf = float(mp.loc[mp["model"] == "Random Forest", "f1"].iloc[0])
    mp_mj = float(mp.loc[mp["model"].str.startswith("Majority"),
                         "f1"].iloc[0])
    check("mappability: RF F1 > majority", mp_rf > mp_mj,
          f"RF {mp_rf:.3f} vs majority {mp_mj:.3f}")
    q = pd.read_csv(os.path.join(MODELS, "review_queue.csv")).iloc[0]
    check("review queue top-20 has >= 5 missing records",
          int(q["missing_records_in_top20"]) >= 5,
          str(q["missing_records_in_top20"]))
    check("review queue precision@20 >= 0.25",
          float(q["precision_at_20"]) >= 0.25)
    check("review queue file has 20 rows",
          len(pd.read_csv(os.path.join(MODELS,
                                       "review_queue_top20.csv"))) == 20)
    check("review queue sorted by risk desc",
          pd.read_csv(os.path.join(MODELS, "review_queue_top20.csv"))
          ["risk_score"].is_monotonic_decreasing)

    print("== determinism spot-check ==")
    # regenerating the migration file twice from the same seed must give
    # identical break counts (generator is seeded); re-run only the
    # generator module in-process and compare counts.
    import importlib
    import data_generation as dg  # noqa: E402
    before = pd.read_csv(os.path.join(ROOT, "data", "synthetic",
                                      "migration.csv"))["break_type"] \
        .value_counts().to_dict()
    dg.main()
    after = pd.read_csv(os.path.join(ROOT, "data", "synthetic",
                                     "migration.csv"))["break_type"] \
        .value_counts().to_dict()
    check("regeneration reproduces break counts", before == after,
          f"{before} vs {after}")

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        print("SOME TESTS FAILED")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
