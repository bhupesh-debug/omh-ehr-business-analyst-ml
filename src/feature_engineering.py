"""Feature engineering for the three modules.

All train-fitted statistics come from the TRAINING period only.
Hold-out rows never contribute to their own features.

Module 1 — migration break detection (binary) and break-cause
  classification (5-class, break rows only):
  field_match_rate, secondary_mismatch, days_since_migration_start,
  is_go_live_wave (wave W6), domain / source_system / migration_wave
  one-hots. target_present, key_match, code_mapped, is_duplicate
  and risk_score are DELIBERATELY EXCLUDED — the flags are the break
  cause by construction and the risk score is derived from it.
Module 2 — interface nonconformance (HL7 + FHIR rows):
  msg/resource type one-hots, source system one-hots, segment_count
  (HL7) / field count (FHIR), message_length_chars, is_hl7, month.
Module 3 — terminology mappability (unmapped codes only):
  token_overlap, desc_len_words, log frequency, target_system one-hots.
"""

import numpy as np
import pandas as pd


def build_migration_features(df):
    """Features for break detection / cause classification.

    DELIBERATELY EXCLUDED: target_present, key_match, code_mapped,
    is_duplicate, risk_score — all four flags ARE the break cause by
    construction, and the risk score is derived from it. Including
    them would make the classifier a lookup table (the first run
    scored a perfect 1.000 that way). What remains is what a BA
    actually has before the checks run: the field-match rate, the
    noisy secondary-field tell, the domain, the source system, and
    the wave.
    """
    out = pd.DataFrame(index=df.index)
    out["field_match_rate"] = df["field_match_rate"]
    out["secondary_mismatch"] = df["secondary_mismatch"]
    out["days_since_start"] = (
        df["migration_date"] - pd.Timestamp("2025-01-01")).dt.days
    out["is_go_live_wave"] = (df["migration_wave"] == "W6").astype(int)
    for col, prefix in [("domain", "dom"), ("source_system", "src"),
                        ("migration_wave", "wave")]:
        for val in sorted(df[col].unique()):
            out[f"{prefix}_{val}"] = (df[col] == val).astype(int)
    return out


def build_interface_features(df):
    """df is the combined HL7 + FHIR frame built in run_all.py."""
    out = pd.DataFrame(index=df.index)
    out["is_hl7"] = (df["standard"] == "HL7v2").astype(int)
    out["segment_count"] = df["segment_count"].fillna(0)
    out["message_length_chars"] = df["message_length_chars"].fillna(0)
    out["log_length"] = np.log1p(out["message_length_chars"])
    out["month"] = df["msg_datetime"].dt.month
    for col, prefix in [("msg_type", "type"), ("source_system", "src")]:
        for val in sorted(df[col].unique()):
            out[f"{prefix}_{val}"] = (df[col] == val).astype(int)
    return out


def build_mapping_features(df):
    out = pd.DataFrame(index=df.index)
    out["token_overlap"] = df["token_overlap"]
    out["desc_len_words"] = df["desc_len_words"]
    out["log_frequency"] = np.log1p(df["frequency"])
    for val in sorted(df["target_system"].unique()):
        out[f"sys_{val}"] = (df["target_system"] == val).astype(int)
    return out
