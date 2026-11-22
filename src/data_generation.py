"""Generate ALL synthetic data for the OMH EHR project (seed 42).

NO REAL PATIENT DATA. No real PHI. Every name, MRN, date, code, and
message is invented by this generator. The schema mirrors the JD:
two OMH projects — EHR Data Migration (OMH_PRJ0251616) and EHR
Interface Analysis (OMH_PRJ0251615) — for a 2025 migration from
three legacy source systems into Oracle Health.

Outputs (data/synthetic/):
  patients.csv      6,000 synthetic patients, 3 legacy source systems
  encounters.csv    14,000 synthetic encounters
  clinical_codes.csv 20,000 synthetic coded clinical facts
  migration.csv     40,000 source->target rows (one per source record)
                    with injected breaks: missing_record / key_mismatch /
                    code_map_failure / duplicate
  hl7_messages.csv  4,000 synthetic HL7 v2 messages (ADT/ORM/ORU) with
                    injected conformance defects
  fhir_resources.csv 2,000 synthetic FHIR resources (Patient / Encounter /
                    Observation / MedicationRequest) with injected defects
  terminology_map.csv 1,500 source-code -> standard-code mappings
                    (ICD-10, CPT, LOINC, SNOMED CT, RxNorm) with coverage
                    gaps; unmapped codes carry a latent true_mappable label
"""

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")
os.makedirs(SYN, exist_ok=True)

SEED = 42
START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2025-12-31")

FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
         "Linda", "William", "Elizabeth", "David", "Barbara", "Richard",
         "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

SOURCE_SYSTEMS = ["LEGACY-EHR", "LEGACY-PAT", "LEGACY-ANC"]
DOMAINS = ["demographics", "encounter", "clinical"]

# Local code systems -> standard systems (the JD's terminology list)
STD_CODES = {
    "ICD-10": [("F32.1", "Major depressive disorder single episode moderate"),
               ("I10", "Essential hypertension"),
               ("E11.9", "Type 2 diabetes without complications"),
               ("J45.909", "Unspecified asthma uncomplicated"),
               ("M54.5", "Low back pain"),
               ("N18.3", "Chronic kidney disease stage 3"),
               ("F41.1", "Generalized anxiety disorder"),
               ("G47.33", "Obstructive sleep apnea"),
               ("K21.9", "Gastroesophageal reflux disease"),
               ("I25.10", "Atherosclerotic heart disease")],
    "CPT": [("99213", "Office outpatient visit established patient"),
            ("99214", "Office outpatient visit established patient detailed"),
            ("93000", "Electrocardiogram routine"),
            ("80053", "Comprehensive metabolic panel"),
            ("70553", "MRI brain without contrast"),
            ("97110", "Therapeutic procedure one or more areas"),
            ("90837", "Psychotherapy 60 minutes"),
            ("99285", "Emergency department visit high severity"),
            ("36415", "Collection of venous blood"),
            ("71046", "Chest x-ray two views")],
    "LOINC": [("2339-0", "Glucose in blood"),
              ("2345-7", "Glucose in blood fasting"),
              ("4548-4", "Hemoglobin A1c in blood"),
              ("33747-0", "Blood pressure panel"),
              ("58410-2", "Creatinine in blood"),
              ("2823-3", "Potassium in blood"),
              ("2093-3", "Cholesterol total in blood"),
              ("3094-0", "Urea nitrogen in blood"),
              ("33914-3", "Glomerular filtration rate"),
              ("49765-1", "Calcium in blood")],
    "SNOMED CT": [("38341003", "Hypertensive disorder"),
                  ("44054006", "Diabetes mellitus type 2"),
                  ("73211009", "Diabetes mellitus"),
                  ("195967001", "Asthma"),
                  ("13645005", "Chronic obstructive lung disease"),
                  ("84114007", "Heart failure"),
                  ("36971009", "Sinusitis"),
                  ("386661006", "Fever"),
                  ("35489007", "Depressive disorder"),
                  ("40930008", "Hypothyroidism")],
    "RxNorm": [("860975", "Sertraline 50 MG oral tablet"),
               ("198440", "Metformin 500 MG oral tablet"),
               ("197517", "Lisinopril 10 MG oral tablet"),
               ("207106", "Atorvastatin 20 MG oral tablet"),
               ("856845", "Albuterol inhaler"),
               ("1049502", "Amoxicillin 500 MG oral capsule"),
               ("617311", "Omeprazole 20 MG oral capsule"),
               ("1110621", "Levothyroxine 50 MCG oral tablet"),
               ("1748913", "Gabapentin 300 MG oral capsule"),
               ("1051628", "Amlodipine 5 MG oral tablet")],
}
LOCAL_PREFIX = {"ICD-10": "LEGACY-DX", "CPT": "LEGACY-PROC",
                "LOINC": "LEGACY-LAB", "SNOMED CT": "LEGACY-DX",
                "RxNorm": "LEGACY-RX"}

# Coverage of each standard system in the synthetic mapping (design choice)
COVERAGE = {"ICD-10": 0.92, "CPT": 0.88, "LOINC": 0.90,
            "SNOMED CT": 0.85, "RxNorm": 0.80}


def _wave(dates):
    """Migration wave W1..W6 (bi-monthly) from a date."""
    return "W" + ((dates.dt.month - 1) // 2 + 1).astype(str)


def main():
    rng = np.random.default_rng(SEED)

    # ---------------- patients ----------------
    n_pat = 6000
    pat_ids = np.arange(1, n_pat + 1)
    first = rng.choice(FIRST, n_pat)
    last = rng.choice(LAST, n_pat)
    dob = pd.to_datetime(rng.integers(
        pd.Timestamp("1940-01-01").value // 10 ** 9,
        pd.Timestamp("2010-12-31").value // 10 ** 9, n_pat), unit="s")
    patients = pd.DataFrame({
        "patient_id": pat_ids,
        "mrn": ["MRN" + str(100000 + i) for i in pat_ids],
        "first_name": first,
        "last_name": last,
        "dob": dob,
        "gender": rng.choice(["M", "F"], n_pat, p=[0.48, 0.52]),
        "source_system": rng.choice(SOURCE_SYSTEMS, n_pat,
                                    p=[0.45, 0.35, 0.20]),
        "record_date": pd.to_datetime(rng.integers(
            START.value // 10 ** 9, END.value // 10 ** 9, n_pat), unit="s"),
        "is_synthetic": 1,
    })
    patients.to_csv(os.path.join(SYN, "patients.csv"), index=False)

    # ---------------- encounters ----------------
    n_enc = 14000
    enc = pd.DataFrame({
        "encounter_id": np.arange(1, n_enc + 1),
        "patient_id": rng.choice(pat_ids, n_enc),
        "encounter_date": pd.to_datetime(rng.integers(
            START.value // 10 ** 9, END.value // 10 ** 9, n_enc), unit="s"),
        "encounter_type": rng.choice(
            ["inpatient", "outpatient", "emergency", "telehealth"], n_enc,
            p=[0.25, 0.45, 0.20, 0.10]),
        "facility": rng.choice(
            ["OMH-Albany", "OMH-Buffalo", "OMH-NYC", "OMH-Rochester"], n_enc),
        "source_system": rng.choice(SOURCE_SYSTEMS, n_enc,
                                    p=[0.45, 0.35, 0.20]),
        "is_synthetic": 1,
    })
    enc.to_csv(os.path.join(SYN, "encounters.csv"), index=False)

    # ---------------- clinical codes ----------------
    n_code = 20000
    systems = list(STD_CODES.keys())
    sys_choice = rng.choice(systems, n_code, p=[0.30, 0.20, 0.20, 0.15, 0.15])
    local_code, std_code, desc = [], [], []
    for s in sys_choice:
        code, d = STD_CODES[s][rng.integers(len(STD_CODES[s]))]
        local_code.append(f"{LOCAL_PREFIX[s]}-{rng.integers(1000, 9999)}")
        std_code.append(code)
        desc.append(d)
    codes = pd.DataFrame({
        "code_id": np.arange(1, n_code + 1),
        "patient_id": rng.choice(pat_ids, n_code),
        "encounter_id": rng.choice(enc["encounter_id"], n_code),
        "code_system_local": [LOCAL_PREFIX[s] for s in sys_choice],
        "local_code": local_code,
        "target_system": sys_choice,
        "standard_code": std_code,
        "description": desc,
        "code_date": pd.to_datetime(rng.integers(
            START.value // 10 ** 9, END.value // 10 ** 9, n_code), unit="s"),
        "is_synthetic": 1,
    })
    codes.to_csv(os.path.join(SYN, "clinical_codes.csv"), index=False)

    # ---------------- migration (one row per source record) ----------------
    mig_pat = pd.DataFrame({
        "record_id": ["PAT-" + str(i) for i in pat_ids],
        "domain": "demographics",
        "source_system": patients["source_system"].values,
        "migration_date": patients["record_date"].values,
    })
    mig_enc = pd.DataFrame({
        "record_id": ["ENC-" + str(i) for i in enc["encounter_id"]],
        "domain": "encounter",
        "source_system": enc["source_system"].values,
        "migration_date": enc["encounter_date"].values,
    })
    mig_cli = pd.DataFrame({
        "record_id": ["CLI-" + str(i) for i in codes["code_id"]],
        "domain": "clinical",
        "source_system": rng.choice(SOURCE_SYSTEMS, n_code,
                                    p=[0.45, 0.35, 0.20]),
        "migration_date": codes["code_date"].values,
    })
    mig = pd.concat([mig_pat, mig_enc, mig_cli], ignore_index=True)
    mig["migration_date"] = pd.to_datetime(mig["migration_date"])
    mig["migration_wave"] = _wave(mig["migration_date"])
    n = len(mig)
    # injected breaks (design rates, per domain)
    is_clin = (mig["domain"] == "clinical").values
    is_demo = (mig["domain"] == "demographics").values
    p_missing = np.where(is_demo, 0.015, np.where(is_clin, 0.020, 0.025))
    p_key = np.where(is_demo, 0.025, 0.010)
    p_code = np.where(is_clin, 0.045, 0.005)
    p_dup = np.where(is_demo, 0.008, 0.012)
    u = rng.random(n)
    break_type = np.full(n, "none", dtype=object)
    break_type[(u < p_missing)] = "missing_record"
    break_type[(u >= p_missing) & (u < p_missing + p_key)] = "key_mismatch"
    break_type[(u >= p_missing + p_key)
               & (u < p_missing + p_key + p_code)] = "code_map_failure"
    break_type[(u >= p_missing + p_key + p_code)
               & (u < p_missing + p_key + p_code + p_dup)] = "duplicate"
    # SILENT breaks (~0.6%): every deterministic check passes, but a
    # secondary field disagrees and the field-match rate is depressed.
    # These are the breaks the rule baseline misses by design — the
    # honest reason the rule's recall is below 1.0, and the reason a
    # statistical detector earns its place.
    silent = (break_type == "none") & (rng.random(n) < 0.006)
    break_type[silent] = "silent_mismatch"
    mig["break_type"] = break_type
    mig["has_break"] = (mig["break_type"] != "none").astype(int)
    mig["target_present"] = (mig["break_type"] != "missing_record").astype(int)
    mig["key_match"] = (mig["break_type"] != "key_mismatch").astype(int)
    mig["code_mapped"] = (mig["break_type"] != "code_map_failure").astype(int)
    mig["is_duplicate"] = (mig["break_type"] == "duplicate").astype(int)
    # risk score: the triage number the BA opens each morning. Missing
    # records at go-live are the most severe; duplicates the least.
    # Silent breaks sit in the middle — suspicious, not proven.
    sev = {"none": 0, "duplicate": 25, "code_map_failure": 55,
           "silent_mismatch": 65, "key_mismatch": 75, "missing_record": 95}
    base = mig["break_type"].map(sev).astype(float)
    mig["risk_score"] = (base + rng.normal(0, 8, n)).clip(0, 100).round(1)
    mig.loc[mig["break_type"] == "none", "risk_score"] = rng.uniform(
        0, 10, int((mig["break_type"] == "none").sum())).round(1)
    mig["field_match_rate"] = np.where(
        mig["has_break"] == 1,
        rng.uniform(0.55, 0.95, n).round(3),
        rng.uniform(0.90, 1.0, n).round(3))
    # silent breaks: depressed field-match rate is their tell, but it
    # OVERLAPS the clean range (0.90-0.97) — a threshold alone cannot
    # separate them, which is why the detector is a model, not a rule
    mig.loc[mig["break_type"] == "silent_mismatch", "field_match_rate"] = \
        rng.uniform(0.80, 0.97, int((mig["break_type"]
                                    == "silent_mismatch").sum())).round(3)
    # secondary-field mismatch: noisy tell — 70% of silent breaks, 5%
    # of everything else (a middle-initial or facility disagreement the
    # key checks do not cover)
    mig["secondary_mismatch"] = (
        (rng.random(n) < np.where(mig["break_type"] == "silent_mismatch",
                                  0.70, 0.05)).astype(int))
    mig["is_synthetic"] = 1
    mig.to_csv(os.path.join(SYN, "migration.csv"), index=False)

    # ---------------- HL7 v2 messages ----------------
    n_hl7 = 4000
    msg_types = ["ADT^A01", "ADT^A08", "ORM^O01", "ORU^R01"]
    mt = rng.choice(msg_types, n_hl7, p=[0.30, 0.25, 0.25, 0.20])
    src = rng.choice(["pharmacy", "dietary", "HIE", "LEGACY-EHR"], n_hl7,
                     p=[0.25, 0.15, 0.30, 0.30])
    msg_dt = pd.to_datetime(rng.integers(
        START.value // 10 ** 9, END.value // 10 ** 9, n_hl7), unit="s")
    # defect injection: ~12% nonconformant. wrong_context (a CPT code
    # where a LOINC is expected) is the deliberately subtle defect: the
    # checker validates code membership, not context, so it passes —
    # the honest ~1.5% checker disagreement.
    u2 = rng.random(n_hl7)
    defect = np.full(n_hl7, "none", dtype=object)
    # bad_code / wrong_context corrupt the OBX segment, which only
    # ORM/ORU messages carry — assigning them to ADT rows would label
    # a conformant message nonconformant (the first run did exactly
    # that: only 51.6% of bad_code rows were actually detectable).
    orm = np.isin(mt, ["ORM^O01", "ORU^R01"])
    defect[orm & (u2 < 0.03)] = "missing_segment"
    defect[orm & (u2 >= 0.03) & (u2 < 0.065)] = "missing_field"
    defect[orm & (u2 >= 0.065) & (u2 < 0.09)] = "bad_code"
    defect[orm & (u2 >= 0.09) & (u2 < 0.105)] = "invalid_format"
    defect[orm & (u2 >= 0.105) & (u2 < 0.12)] = "wrong_context"
    defect[(~orm) & (u2 < 0.04)] = "missing_segment"
    defect[(~orm) & (u2 >= 0.04) & (u2 < 0.08)] = "missing_field"
    defect[(~orm) & (u2 >= 0.08) & (u2 < 0.12)] = "invalid_format"
    seg_count = np.where(mt == "ORU^R01", 5,
                         np.where(mt == "ORM^O01", 4, 3))
    seg_count = np.where(defect == "missing_segment", seg_count - 1, seg_count)
    # build the message text (synthetic MSH/PID/PV1/OBX, short form)
    texts = []
    for i in range(n_hl7):
        pid = f"PID|1||MRN{100000 + int(rng.integers(1, 6000))}||" \
              f"{rng.choice(LAST)}^{rng.choice(FIRST)}||" \
              f"{rng.integers(1940, 2010)}{rng.integers(1, 13):02d}" \
              f"{rng.integers(1, 29):02d}|{rng.choice(['M', 'F'])}"
        if defect[i] == "missing_field":
            pid = pid.rsplit("|", 1)[0]  # drop the gender field
        segs = [f"MSH|^~\\&|{src[i]}|OMH|OracleHealth|OMH|"
                f"{msg_dt[i].strftime('%Y%m%d%H%M') if defect[i] != 'invalid_format' else msg_dt[i].strftime('%Y%m%d%H')}|"
                f"|{mt[i]}|{10000 + i}|P|2.5"]
        if defect[i] != "missing_segment":
            segs.append(pid)
            segs.append(f"PV1|1|{'I' if mt[i].startswith('ADT') else 'O'}|"
                        f"OMH-Albany^^^NY")
            if mt[i] in ("ORM^O01", "ORU^R01"):
                if defect[i] == "bad_code":
                    code = "ZZ-UNKNOWN"
                elif defect[i] == "wrong_context":
                    code = "99213"  # CPT procedure code in an OBX that
                    # expects a LOINC lab code — valid code, wrong context
                else:
                    code = rng.choice(["2339-0", "4548-4", "2823-3"])
                segs.append(f"OBX|1|NM|{code}||"
                            f"{rng.integers(50, 200)}|mg/dL||||F")
        texts.append("\r".join(segs))
    # segment count is the ACTUAL segment count of the built text
    seg_count = np.array([t.count("\r") + 1 for t in texts])
    hl7 = pd.DataFrame({
        "message_id": ["HL7-" + str(i) for i in range(1, n_hl7 + 1)],
        "msg_datetime": msg_dt,
        "msg_type": mt,
        "source_system": src,
        "segment_count": seg_count,
        "message_length_chars": [len(t) for t in texts],
        "message_text": texts,
        "defect_type": defect,
        "nonconformant": (defect != "none").astype(int),
        "is_synthetic": 1,
    })
    hl7.to_csv(os.path.join(SYN, "hl7_messages.csv"), index=False)

    # ---------------- FHIR resources ----------------
    n_fhir = 2000
    rtypes = ["Patient", "Encounter", "Observation", "MedicationRequest"]
    rt = rng.choice(rtypes, n_fhir, p=[0.35, 0.25, 0.25, 0.15])
    f_dt = pd.to_datetime(rng.integers(
        START.value // 10 ** 9, END.value // 10 ** 9, n_fhir), unit="s")
    # defect injection: ~9.5% nonconformant. stale_reference (a subject
    # pointing at a Patient id that does not exist) is the deliberately
    # subtle defect: the checker validates structure and bindings, not
    # referential integrity, so it passes — the honest ~1.5% disagreement.
    u3 = rng.random(n_fhir)
    fdef = np.full(n_fhir, "none", dtype=object)
    fdef[(u3 < 0.03)] = "missing_element"
    fdef[(u3 >= 0.03) & (u3 < 0.06)] = "bad_binding"
    fdef[(u3 >= 0.06) & (u3 < 0.08)] = "invalid_format"
    fdef[(u3 >= 0.08) & (u3 < 0.095)] = "stale_reference"
    # Patient resources have no subject to stale-point: fall back to a
    # binding defect instead
    fdef[(fdef == "stale_reference") & (rt == "Patient")] = "bad_binding"
    resources = []
    for i in range(n_fhir):
        if rt[i] == "Patient":
            res = {"resourceType": "Patient",
                   "id": f"pat-{1000 + i}",
                   "identifier": [{"value": f"MRN{100000 + int(rng.integers(1, 6000))}"}],
                   "name": [{"family": str(rng.choice(LAST)),
                             "given": [str(rng.choice(FIRST))]}],
                   "gender": str(rng.choice(["male", "female"])),
                   "birthDate": f"{rng.integers(1940, 2010)}-"
                                f"{rng.integers(1, 13):02d}-"
                                f"{rng.integers(1, 29):02d}"}
            if fdef[i] == "missing_element":
                res.pop("birthDate")
            if fdef[i] == "bad_binding":
                res["gender"] = "unknown-code"
        elif rt[i] == "Encounter":
            res = {"resourceType": "Encounter", "id": f"enc-{1000 + i}",
                   "status": "finished",
                   "class": {"code": "AMB"},
                   "subject": {"reference": f"Patient/pat-{1000 + i}"},
                   "period": {"start": "2025-06-01"}}
            if fdef[i] == "missing_element":
                res.pop("status")
            if fdef[i] == "bad_binding":
                res["class"] = {"code": "NOT-A-CLASS"}
        elif rt[i] == "Observation":
            code, _ = STD_CODES["LOINC"][rng.integers(10)]
            res = {"resourceType": "Observation", "id": f"obs-{1000 + i}",
                   "status": "final",
                   "code": {"coding": [{"system": "http://loinc.org",
                                        "code": code}]},
                   "valueQuantity": {"value": float(rng.integers(50, 200)),
                                     "unit": "mg/dL"}}
            if fdef[i] == "missing_element":
                res.pop("valueQuantity")
            if fdef[i] == "bad_binding":
                res["code"]["coding"][0]["code"] = "99999-9"
        else:
            code, _ = STD_CODES["RxNorm"][rng.integers(10)]
            res = {"resourceType": "MedicationRequest",
                   "id": f"med-{1000 + i}", "status": "active",
                   "intent": "order",
                   "medicationCodeableConcept": {
                       "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                   "code": code}]},
                   "subject": {"reference": f"Patient/pat-{1000 + i}"}}
            if fdef[i] == "missing_element":
                res.pop("intent")
            if fdef[i] == "bad_binding":
                res["medicationCodeableConcept"]["coding"][0]["code"] = "0000000"
        if fdef[i] == "invalid_format":
            res["id"] = f"bad id {i}!!"
        if fdef[i] == "stale_reference":
            # subject points at a Patient id that does not exist —
            # structure is valid, the reference is not
            res["subject"] = {"reference": "Patient/pat-99999"}
        resources.append(json.dumps(res))
    fhir = pd.DataFrame({
        "resource_id": ["FHIR-" + str(i) for i in range(1, n_fhir + 1)],
        "resource_type": rt,
        "created_date": f_dt,
        "resource_json": resources,
        "defect_type": fdef,
        "nonconformant": (fdef != "none").astype(int),
        "is_synthetic": 1,
    })
    fhir.to_csv(os.path.join(SYN, "fhir_resources.csv"), index=False)

    # ---------------- terminology mapping ----------------
    n_map = 1500
    t_systems = list(STD_CODES.keys())
    t_choice = rng.choice(t_systems, n_map, p=[0.30, 0.20, 0.20, 0.15, 0.15])
    rows = []
    for i, s in enumerate(t_choice):
        code, d = STD_CODES[s][rng.integers(len(STD_CODES[s]))]
        mapped = rng.random() < COVERAGE[s]
        local = f"{LOCAL_PREFIX[s]}-{rng.integers(1000, 9999)}"
        if mapped:
            # source description echoes the standard description
            src_desc = d
            overlap = round(float(rng.uniform(0.55, 1.0)), 3)
            std = code
        else:
            # unmapped: description is generic or partial
            if rng.random() < 0.5:
                src_desc = " ".join(d.split()[:2]) + " local variant"
                overlap = round(float(rng.uniform(0.35, 0.65)), 3)
            else:
                src_desc = "Local code — no standard description on file"
                overlap = round(float(rng.uniform(0.0, 0.2)), 3)
            std = ""
        rows.append({
            "map_id": i + 1,
            "source_code": local,
            "source_system": LOCAL_PREFIX[s],
            "source_description": src_desc,
            "target_system": s,
            "standard_code": std,
            "standard_description": d,
            "mapped": int(mapped),
            "token_overlap": overlap,
            "desc_len_words": len(src_desc.split()),
            "frequency": int(rng.integers(1, 500)),
            # latent truth for unmapped codes: mappable if the description
            # still overlaps the standard wording AND the code is used often
            "true_mappable": int((not mapped) and overlap > 0.4
                                 and rng.random() < 0.7) if not mapped else 0,
            "is_synthetic": 1,
        })
    tmap = pd.DataFrame(rows)
    tmap.to_csv(os.path.join(SYN, "terminology_map.csv"), index=False)

    print(f"patients {len(patients):,}  encounters {len(enc):,}  "
          f"codes {len(codes):,}")
    print(f"migration {len(mig):,} rows, break rate "
          f"{mig['has_break'].mean():.2%}")
    print(f"HL7 {len(hl7):,} msgs, nonconformant "
          f"{hl7['nonconformant'].mean():.2%}")
    print(f"FHIR {len(fhir):,} resources, nonconformant "
          f"{fhir['nonconformant'].mean():.2%}")
    print(f"terminology {len(tmap):,} maps, coverage "
          f"{tmap['mapped'].mean():.2%}")


if __name__ == "__main__":
    main()
