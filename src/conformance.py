"""Rule-based HL7 v2 / FHIR conformance checker.

This is the deterministic checker a BA runs against the interface
specification — it parses each synthetic message/resource and reports
exactly which rule failed. It is the baseline every model must beat,
and it is also the deliverable itself: a nonconformant message is
stopped before it reaches Oracle Health.

HL7 v2 rules (per message):
  R1  MSH segment present
  R2  PID segment present (except pure ORM order messages keep PID too —
      the synthetic spec requires PID on every message)
  R3  required fields present (PID-5 name, PID-7 DOB, PID-8 gender)
  R4  OBX observation code belongs to a known value set
      (CPT / LOINC codes used by the generator; "ZZ-*" fails)
  R5  MSH-7 datetime parses as YYYYMMDDHHMM
  R6  PV1 segment present
NOTE: the checker validates structure and code MEMBERSHIP, not
context or referential integrity — a CPT code in an OBX that expects
a LOINC (wrong_context) and a FHIR reference to a nonexistent Patient
(stale_reference) both pass. That gap is deliberate and reported.
FHIR rules (per resource):
  F1  id is a valid token (no spaces / punctuation)
  F2  required elements present per resource type:
      Patient: birthDate; Encounter: status; Observation: valueQuantity;
      MedicationRequest: intent
  F3  code bindings valid: gender in {male, female}; Encounter.class in
      {AMB, IMP, EMER}; Observation.code in the LOINC value set;
      MedicationRequest code in the RxNorm value set

check_hl7(text) and check_fhir(resource_dict) return
( conformant: bool, failed_rules: list[str] ).
"""

import json
import re

LOINC_OK = {"2339-0", "2345-7", "4548-4", "33747-0", "58410-2", "2823-3",
            "2093-3", "3094-0", "33914-3", "49765-1"}
CPT_OK = {"99213", "99214", "93000", "80053", "70553", "97110", "90837",
          "99285", "36415", "71046"}
RXNORM_OK = {"860975", "198440", "197517", "207106", "856845", "1049502",
             "617311", "1110621", "1748913", "1051628"}


def check_hl7(text):
    failed = []
    segs = text.split("\r")
    by_name = {}
    for s in segs:
        parts = s.split("|")
        by_name.setdefault(parts[0], parts)
    if "MSH" not in by_name:
        failed.append("R1_no_MSH")
    else:
        msh = by_name["MSH"]
        dt = msh[6] if len(msh) > 6 else ""
        if not re.fullmatch(r"\d{12}", dt or ""):
            failed.append("R5_bad_datetime")
    if "PID" not in by_name:
        failed.append("R2_no_PID")
    else:
        pid = by_name["PID"]
        # PID-5 name (field index 5), PID-7 DOB (7), PID-8 gender (8)
        if len(pid) <= 5 or not pid[5]:
            failed.append("R3_missing_name")
        if len(pid) <= 7 or not re.fullmatch(r"\d{8}", pid[7] or ""):
            failed.append("R3_missing_dob")
        if len(pid) <= 8 or pid[8] not in ("M", "F"):
            failed.append("R3_missing_gender")
    if "PV1" not in by_name:
        failed.append("R6_no_PV1")
    if "OBX" in by_name:
        obx = by_name["OBX"]
        code = obx[3] if len(obx) > 3 else ""
        if code not in LOINC_OK and code not in CPT_OK:
            failed.append("R4_unknown_code")
    return (len(failed) == 0), failed


def check_fhir(resource):
    if isinstance(resource, str):
        resource = json.loads(resource)
    failed = []
    rid = resource.get("id", "")
    if not re.fullmatch(r"[A-Za-z0-9\-\.]{1,64}", str(rid)):
        failed.append("F1_bad_id")
    rtype = resource.get("resourceType")
    required = {"Patient": ["birthDate"], "Encounter": ["status"],
                "Observation": ["valueQuantity"],
                "MedicationRequest": ["intent"]}
    for field in required.get(rtype, []):
        if field not in resource:
            failed.append(f"F2_missing_{field}")
    if rtype == "Patient":
        if resource.get("gender") not in ("male", "female"):
            failed.append("F3_bad_gender_binding")
    if rtype == "Encounter":
        cls = (resource.get("class") or {}).get("code")
        if cls not in ("AMB", "IMP", "EMER"):
            failed.append("F3_bad_class_binding")
    if rtype == "Observation":
        codings = (resource.get("code") or {}).get("coding") or []
        codes = {c.get("code") for c in codings}
        if not (codes & LOINC_OK):
            failed.append("F3_bad_loinc_binding")
    if rtype == "MedicationRequest":
        codings = ((resource.get("medicationCodeableConcept") or {})
                   .get("coding") or [])
        codes = {c.get("code") for c in codings}
        if not (codes & RXNORM_OK):
            failed.append("F3_bad_rxnorm_binding")
    return (len(failed) == 0), failed
