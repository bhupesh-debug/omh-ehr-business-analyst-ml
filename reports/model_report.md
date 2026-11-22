# EHR Data Migration & Interface Analytics — Model Report

**Role:** Business Analyst #26-03354, NY State Office of Mental Health
(OMH), Albany NY — Oracle Health EHR implementation
**Author:** Bhupesh Chandra Dimri — **All data is synthetic. No real
patient data, no real PHI.**
**Hold-out:** Modules 1 & 2 — last 61 days of 2025 (2025-11-01 to
2025-12-31), time-based, no leakage. Module 3 — grouped random 70/30
split by code family (see the Module 3 note below for why).
**Tests:** see `tests/run_tests.py` (run after this script).
**Checker agreement:** the deterministic HL7/FHIR checker agrees with
the injected defects on 99.5% of HL7 messages and
99.0% of FHIR resources — the disagreements are the
deliberately subtle defects: a CPT code where a LOINC is expected
(HL7 wrong_context) and a reference to a Patient id that does not
exist (FHIR stale_reference). The checker validates structure and
code membership, not context or referential integrity — that gap is
reported, not hidden.

## Module 1 — Migration reconciliation (hold-out: 6,604 records, 484 true breaks)
Break detection — the deterministic rule catches every overt break
and misses the silent ones by design; the statistical models see
only pre-check features (field-match rate, secondary tell, domain,
system, wave) and earn their place by catching some silent breaks:
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Rule baseline (target/key/code/duplicate flags) | 1.000 | 0.913 | 0.955 | 0.957 |
| Accept-all baseline | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.873 | 0.853 | 0.863 | 0.970 |
| Random Forest | 1.000 | 0.857 | 0.923 | 0.961 |

Break-cause diagnosis (hold-out: 484 true breaks). The rule
diagnosis reads which flag failed, so it names every overt break and
only misses the silent ones (it calls them "none"). The learned
models use no flag features — that is why their accuracy is modest,
and why it is honest:
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Rule diagnosis (which flag failed) | 0.913 | 0.667 |
| Majority-class baseline | 0.324 | 0.098 |
| Random Forest | 0.411 | 0.385 |

Top drivers: field_match_rate, days_since_start, secondary_mismatch, dom_clinical, dom_encounter.
Ranked review queue (top 20 by risk score): 20/20 are
true missing records; queue recall of missing records 14.60%.

## Module 2 — Interface conformance (hold-out: 996 messages/resources, 112 nonconformant)
Conformance rate overall: 88.7% HL7, 90.3% FHIR (full year).
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Accept-all baseline | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.468 | 0.464 | 0.466 | 0.729 |
| Random Forest | 0.342 | 0.589 | 0.433 | 0.799 |

Defect-cause classification (nonconformant rows only):
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | 0.250 | 0.050 |
| Random Forest | 0.714 | 0.652 |

The metadata-only classifier is honestly weak (best F1
0.466):
it sees only message length, segment/field counts, type, source and
month, so content defects — a bad code, a bad ID, a wrong-context
code — are invisible to it. The deterministic checker is the real
Module 2 tool (agreement 99.5% HL7 / 99.0%
FHIR); the classifier is a triage aid for the queue, not the
detector. Do not cite the classifier's F1 as the project's
conformance performance — cite the checker's agreement.

## Module 3 — Terminology mapping (unmapped codes: 185, hold-out: 80)
Coverage by target system:
- CPT: 89.6% mapped (n=327)
- ICD-10: 92.3% mapped (n=440)
- LOINC: 89.6% mapped (n=297)
- RxNorm: 78.0% mapped (n=223)
- SNOMED CT: 82.6% mapped (n=213)

Mappability — can an unmapped code be mapped without manual review?
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Majority-class baseline | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.424 | 1.000 | 0.596 | 0.896 |
| Random Forest | 0.500 | 0.929 | 0.650 | 0.913 |

Top unmapped codes by frequency are in models/top_unmapped.csv — the
first thing a BA triages, because one unmapped high-frequency code
breaks more records than fifty rare ones.

## Benchmark (best model vs interpretable baseline)
| Module | Baseline | Best model | Metric |
|---|---|---|---|
| Migration break-cause | Majority-class 0.324 | Random Forest 0.411 | Accuracy |
| Interface nonconformance | Accept-all 0.000 | Logistic Regression 0.466 | F1 |
| Terminology mappability | Majority-class 0.000 | Random Forest 0.650 | F1 |

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
