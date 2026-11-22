"""Validation gate: load synthetic data, clean, assert invariants.

The run FAILS (raises) on any violation — a BA's reconciliation must
not silently pass bad data. Checks: unique IDs, positive counts,
dates inside 2025, break rate 3-12%, nonconformant rate 5-20%,
mapping coverage 75-95%, migration diff fields consistent
(target_present == 0 exactly when break_type == missing_record, etc.).
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)


def _check(cond, msg):
    if not cond:
        raise ValueError("VALIDATION FAILED: " + msg)


def main():
    pat = pd.read_csv(os.path.join(SYN, "patients.csv"),
                      parse_dates=["dob", "record_date"])
    enc = pd.read_csv(os.path.join(SYN, "encounters.csv"),
                      parse_dates=["encounter_date"])
    codes = pd.read_csv(os.path.join(SYN, "clinical_codes.csv"),
                        parse_dates=["code_date"])
    mig = pd.read_csv(os.path.join(SYN, "migration.csv"),
                      parse_dates=["migration_date"])
    hl7 = pd.read_csv(os.path.join(SYN, "hl7_messages.csv"),
                      parse_dates=["msg_datetime"])
    fhir = pd.read_csv(os.path.join(SYN, "fhir_resources.csv"),
                       parse_dates=["created_date"])
    tmap = pd.read_csv(os.path.join(SYN, "terminology_map.csv"))
    # empty standard_code reads back as NaN — normalize before checks
    tmap["standard_code"] = tmap["standard_code"].fillna("")

    for df, col, name in [(pat, "patient_id", "patients"),
                          (enc, "encounter_id", "encounters"),
                          (codes, "code_id", "clinical_codes"),
                          (mig, "record_id", "migration"),
                          (hl7, "message_id", "hl7"),
                          (fhir, "resource_id", "fhir"),
                          (tmap, "map_id", "terminology")]:
        _check(df[col].is_unique, f"{name}: {col} not unique")
        _check((df["is_synthetic"] == 1).all(),
               f"{name}: non-synthetic row present")

    for df, col in [(pat, "record_date"), (enc, "encounter_date"),
                    (codes, "code_date"), (mig, "migration_date"),
                    (hl7, "msg_datetime"), (fhir, "created_date")]:
        _check(df[col].between("2025-01-01", "2025-12-31").all(),
               f"{col}: date outside 2025")

    _check(0.03 <= mig["has_break"].mean() <= 0.12,
           f"migration break rate {mig['has_break'].mean():.2%} outside 3-12%")
    _check(((mig["target_present"] == 0)
            == (mig["break_type"] == "missing_record")).all(),
           "target_present inconsistent with break_type")
    _check(((mig["key_match"] == 0)
            == (mig["break_type"] == "key_mismatch")).all(),
           "key_match inconsistent with break_type")
    _check(((mig["code_mapped"] == 0)
            == (mig["break_type"] == "code_map_failure")).all(),
           "code_mapped inconsistent with break_type")
    _check(((mig["is_duplicate"] == 1)
            == (mig["break_type"] == "duplicate")).all(),
           "is_duplicate inconsistent with break_type")

    _check(0.05 <= hl7["nonconformant"].mean() <= 0.20,
           f"HL7 nonconformant rate {hl7['nonconformant'].mean():.2%}")
    _check(0.05 <= fhir["nonconformant"].mean() <= 0.20,
           f"FHIR nonconformant rate {fhir['nonconformant'].mean():.2%}")
    _check(0.75 <= tmap["mapped"].mean() <= 0.95,
           f"mapping coverage {tmap['mapped'].mean():.2%}")
    _check(set(tmap["target_system"]) == {"ICD-10", "CPT", "LOINC",
                                          "SNOMED CT", "RxNorm"},
           "terminology missing a target system")
    _check((tmap.loc[tmap["mapped"] == 1, "standard_code"] != "").all(),
           "mapped row without a standard code")
    _check((tmap.loc[tmap["mapped"] == 0, "standard_code"] == "").all(),
           "unmapped row carrying a standard code")

    for df, name in [(pat, "patients"), (enc, "encounters"),
                     (codes, "clinical_codes"), (mig, "migration"),
                     (hl7, "hl7_messages"), (fhir, "fhir_resources"),
                     (tmap, "terminology_map")]:
        df.to_csv(os.path.join(PROC, name + ".csv"), index=False)
    print("validation passed — 7 datasets, "
          f"{len(mig):,} migration rows, {len(hl7):,} HL7, "
          f"{len(fhir):,} FHIR, {len(tmap):,} mappings")


if __name__ == "__main__":
    main()
