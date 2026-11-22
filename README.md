# EHR Data Migration & Interface Analytics

**Role:** Business Analyst #26-03354, NY State Office of Mental Health
(OMH), Albany NY — Oracle Health EHR implementation
**Author:** Bhupesh Chandra Dimri — November 2026

**ALL DATA IS SYNTHETIC. No real patient data, no real PHI.** Patient
names, MRNs, dates, codes, and messages are invented with a fixed
seed (42). Break rates, defect rates, and terminology coverage are
design choices, not OMH facts.

## What this project is — and is not

This is a pipeline-design project for the OMH Business Analyst role:
migration reconciliation, HL7 v2 / FHIR conformance checking, and
terminology mapping validation, built the way a BA would actually
work them — deterministic checks first, statistical models only
where they add something the checks cannot do.

It proves **pipeline design, SQL-style profiling, validation logic,
and triage thinking**. It does **not** prove production EHR
experience, HL7/FHIR hands-on work, healthcare-standards expertise,
or a BA certification — the candidate does not claim any of those;
all are **[CONFIRM WITH CANDIDATE]** items, not facts in this
project.

**Match vs the JD: 4/10.** Data profiling, source-to-target mapping,
reconciliation, SQL validation, and test/UAT thinking map directly
to the migration work. HL7 v2 / FHIR interface requirements,
healthcare terminology standards (USCDI, ICD-10, CPT, LOINC,
SNOMED CT, RxNorm), and the required BA certification (CBAP/CCBA/
PMI-PBA) are hard gaps, as is the 84/60/48/36-month healthcare-IS
experience minimum. The project narrows the data/analytics part of
the gap; it does not close the standards or certification parts.

## The three modules

### Module 1 — Migration reconciliation
40,000 synthetic migration rows (6,000 patients, 14,000 encounters,
20,000 clinical codes) across six monthly waves, Jan–Dec 2025.
Break rate 7.41%. Break types: missing_record, key_mismatch,
code_map_failure, duplicate, and **silent_mismatch** — breaks every
deterministic check passes (a secondary field disagrees, the
field-match rate is depressed but overlaps the clean range).

| Detector | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Rule baseline (target/key/code/duplicate flags) | 1.000 | 0.913 | 0.955 | 0.957 |
| Logistic Regression | 0.873 | 0.853 | 0.863 | 0.970 |
| Random Forest | 1.000 | 0.857 | 0.923 | 0.961 |

The rule is the best overall detector by design — it catches every
overt break and misses the silent ones (recall 0.913, not 1.000).
The models' value is the silent class: on the 42 silent breaks in
the hold-out, the rule catches **0%** and the Random Forest catches
**45.2%**, with zero false positives on 6,120 clean rows.

Break-cause diagnosis (hold-out: 484 true breaks): the rule reads
which flag failed — accuracy 0.913, macro-F1 0.667, missing only the
silent breaks. The learned Random Forest uses **no flag features**
(they are the cause by construction — including them scored a
perfect 1.000 in the first run, which was leakage, not learning):
accuracy 0.411, macro-F1 0.385 vs a majority-class baseline of
0.324 / 0.098. Modest and honest: the flags name the cause for
overt breaks; the model's job is triage, not diagnosis.

Ranked review queue (top 20 by risk score): **20/20 are true
missing records**; queue recall of missing records 14.60%.

### Module 2 — HL7 v2 / FHIR conformance
4,000 synthetic HL7 v2 messages (ADT, ORM, ORU; pharmacy, dietary,
HIE, legacy EHR sources) and 2,000 FHIR resources (Patient,
Encounter, Observation, MedicationRequest). Nonconformant: 12.12%
HL7, 9.70% FHIR.

The deterministic checker implements six HL7 rules (R1 MSH present,
R2 PID present, R3 required PID fields, R4 code in a known value
set, R5 MSH-7 datetime parses, R6 PV1 present) and three FHIR rule
groups (F1 id format, F2 required elements, F3 value-set bindings).
**Checker agreement with the injected defects: 99.5% HL7, 99.0%
FHIR.** The disagreements are the two deliberately subtle defect
classes: a CPT code where a LOINC is expected (`wrong_context`,
passes — the checker validates code membership, not context) and a
reference to a Patient id that does not exist (`stale_reference`,
passes — the checker validates structure, not referential
integrity). That gap is reported, not hidden.

The metadata-only classifier (message length, segment/field counts,
type, source, month) is honestly weak — best F1 0.466 (Logistic
Regression); Random Forest F1 0.433. Content defects are invisible
to it. The checker is the real Module 2 tool; the classifier is a
triage aid. Defect-cause classification on nonconformant rows:
Random Forest accuracy 0.714, macro-F1 0.652 vs majority 0.250.

### Module 3 — Terminology mapping validation
1,500 source-to-standard mappings across ICD-10, CPT, LOINC,
SNOMED CT, RxNorm. Overall coverage **87.67%** (ICD-10 92.3%,
CPT 89.6%, LOINC 89.6%, SNOMED CT 82.6%, RxNorm 78.0% — RxNorm is
where the terminologist starts). 185 unmapped codes; the
mappability model (can an unmapped code be mapped without manual
review?) scores Random Forest F1 0.650, ROC-AUC 0.913, vs a
majority-class baseline of 0.000.

## Hold-out design (no leakage)

- **Modules 1 & 2:** time-based. Train = Jan–Oct 2025, hold-out =
  the last 61 days (2025-11-01 to 2025-12-31), including the go-live
  wave W6. Hold-out dates are strictly after train dates.
- **Module 3:** grouped random 70/30 split **by code family**
  (source_code prefix) — the same code family never appears in
  both train and hold-out. A plain random split would leak.
- Module 1 model features deliberately exclude the four rule flags
  and the risk score (both are the label by construction).

## How to run

```bash
pip install -r requirements.txt
python run_all.py            # generate, validate, model, chart, report
python tests/run_tests.py    # 84 checks — must all pass
python make_notebooks.py     # generate the 4 notebooks
python build_pdf.py          # 10-page report -> your_files (run from repo root)
python build_deck.py         # 10-slide deck -> your_files (run from repo root)
```

Python 3.12, scikit-learn >= 1.5, pandas >= 2.0, numpy >= 2.0,
matplotlib >= 3.8. All outputs land in `data/`, `models/`,
`reports/`, `notebooks/`. Full metric tables: `reports/model_report.md`.

## Project structure

```
run_all.py                 orchestrator (generate -> validate -> model -> chart -> report)
build_pdf.py               10-page report builder (reportlab)
build_deck.py              10-slide deck builder (reportlab, landscape)
make_notebooks.py          generates notebooks/01_eda .. 04_evaluation
src/
  data_generation.py       synthetic data, seed 42 (patients, encounters,
                           codes, migration, HL7, FHIR, terminology)
  data_pipeline.py         validation gates — the run FAILS on any violation
  feature_engineering.py   train-only features; flag features excluded (leakage)
  conformance.py           HL7 v2 (R1-R6) and FHIR (F1-F3) checker
  models.py                baselines, Logistic Regression, Random Forest,
                           review-queue metrics
  charts.py                7 charts (breaks by wave, conformance by system,
                           coverage, importance, ROC, comparison, correlation)
tests/run_tests.py         84 checks: file existence, validation gates,
                           checker correctness on hand-built messages,
                           hold-out integrity, metric sanity, determinism
data/synthetic/            generated raw files
data/processed/            validated + cleaned files (incl. interface_combined.csv)
models/                    metric CSVs, review queue, benchmark, scored files
reports/model_report.md    full metric tables + limits
reports/charts/            7 PNG charts
notebooks/                 01_eda, 02_feature_engineering, 03_modeling, 04_evaluation
```

## Design continuity

Built after the `lifesci-clinical-finance` folder in
github.com/bhupesh-debug/Multi-project (risk-adjusted valuation &
budget forecasting for clinical programs — synthetic data, rNPV
regression, Monte Carlo). That project is finance/analytics work,
**not** EHR/HL7 experience, and is never presented as such here.
What carries over is the design discipline: synthetic data stated
plainly, a time-based hold-out, baselines beaten honestly, and
limits on the same page as the results.

## Limits (read before citing any number)

- All data is synthetic; rates are design choices, not OMH facts.
- Injected defects are deliberately detectable; real interface
  defects are subtler — expect lower recall on live feeds and keep
  a human in the review queue.
- The mappability label is a latent simulation variable, not a
  clinician's judgment; on real unmapped codes a terminologist
  decides.
- HL7 v2 / FHIR rules here are a subset of the real specifications
  (six HL7 rules, three FHIR rule groups). A production conformance
  suite covers the full implementation guide.
- The candidate does not claim HL7/FHIR experience,
  healthcare-standards expertise, or a BA certification.

## Publish

Suggested repo (user creates the empty repo first — integrations
cannot create repos): **`omh-ehr-business-analyst-ml`** under
github.com/bhupesh-debug. Push the text files in this directory
(`run_all.py`, `build_pdf.py`, `build_deck.py`,
`make_notebooks.py`, `requirements.txt`, `README.md`, `.gitignore`,
`src/*.py`, `tests/*.py`, `notebooks/*.ipynb`, `reports/*.md`,
`models/*.csv`, `data/processed/*.csv`); regenerate the 7 chart
PNGs with `python run_all.py` after cloning.
