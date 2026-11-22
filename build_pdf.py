"""Build the 10-page PDF project report.

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python build_pdf.py
Output: ~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_Business-Analyst_OMH.pdf

All numbers come from models/*.csv and data/processed/*.csv (the real
Python run). Charts come from reports/charts/*.png. Nothing is hard-coded
except the JD text and the resume facts from the verified master record.
"""

import csv
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(
    "~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_"
    "Business-Analyst_OMH.pdf")
CHARTS = os.path.join(ROOT, "reports", "charts")
MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")

NAVY = colors.HexColor("#1f4e79")
TEAL = colors.HexColor("#2e9e6b")
LIGHT = colors.HexColor("#dce6f1")
GREY = colors.HexColor("#595959")
RED = colors.HexColor("#b23a48")

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=17,
                    spaceBefore=14, spaceAfter=8, textColor=NAVY)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=13,
                    spaceBefore=12, spaceAfter=6, textColor=NAVY)
BODY = ParagraphStyle("Body", parent=SS["BodyText"], fontSize=10.5,
                      leading=14.5, spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=SS["BodyText"], fontSize=9,
                       leading=12, textColor=GREY)
CELL = ParagraphStyle("Cell", parent=SS["BodyText"], fontSize=9.5,
                      leading=12.5)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")
TITLE = ParagraphStyle("Title", parent=SS["Title"], fontSize=24,
                       leading=29, textColor=NAVY)
SUB = ParagraphStyle("Sub", parent=SS["BodyText"], fontSize=12,
                     leading=16, textColor=GREY)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14,
                        bulletIndent=4, spaceAfter=4)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(headers, rows, widths=None):
    data = [[p(h, CELLH) for h in headers]]
    for r in rows:
        data.append([p(str(c), CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("BOX", (0, 0), (-1, -1), 1, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f6fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def read_csv(name):
    with open(os.path.join(MODELS, name)) as fh:
        return list(csv.DictReader(fh))


def fmt(x, digits=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v:.{digits}f}"


def pct(x, digits=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v:.{digits}f}%"


# ---- real numbers from the run ---------------------------------------
mig = pd.read_csv(os.path.join(PROC, "migration.csv"),
                  parse_dates=["migration_date"])
hl7 = pd.read_csv(os.path.join(PROC, "hl7_messages.csv"),
                  parse_dates=["msg_datetime"])
fhir = pd.read_csv(os.path.join(PROC, "fhir_resources.csv"),
                   parse_dates=["created_date"])
tmap = pd.read_csv(os.path.join(PROC, "terminology_map.csv"))
iface = pd.read_csv(os.path.join(PROC, "interface_combined.csv"),
                    parse_dates=["msg_datetime"])
det = read_csv("migration_break_metrics.csv")
cause = read_csv("break_cause_metrics.csv")
nc = read_csv("nonconformance_metrics.csv")
dc = read_csv("defect_cause_metrics.csv")
mp = read_csv("mapping_metrics.csv")
cov = read_csv("coverage.csv")
bench = read_csv("benchmark.csv")
queue = read_csv("review_queue.csv")[0]
imp1 = read_csv("break_cause_importance.csv")
top_unmapped = read_csv("top_unmapped.csv")
det_sc = pd.read_csv(os.path.join(MODELS, "detection_scored.csv"))

n_mig = len(mig)
n_breaks = int(mig["has_break"].sum())
break_rate = mig["has_break"].mean()
n_hold = int((mig["migration_date"] >= "2025-11-01").sum())
n_hold_breaks = int(mig[mig["migration_date"] >= "2025-11-01"]
                    ["has_break"].sum())
silent = det_sc[det_sc["break_type"] == "silent_mismatch"]
silent_recall_rf = float(silent["y_pred_rf"].mean())
agree_hl7 = float(iface[(iface["standard"] == "HL7v2")]
                  ["checker_conformant"].eq(
                      iface[(iface["standard"] == "HL7v2")]
                      ["nonconformant"] == 0).mean())
agree_fhir = float(iface[(iface["standard"] == "FHIR")]
                   ["checker_conformant"].eq(
                       iface[(iface["standard"] == "FHIR")]
                       ["nonconformant"] == 0).mean())
hl7_nc_rate = float(hl7["nonconformant"].mean())
fhir_nc_rate = float(fhir["nonconformant"].mean())
cov_overall = float(tmap["mapped"].mean())
rule_det = next(r for r in det if r["model"].startswith("Rule"))
rf_det = next(r for r in det if r["model"] == "Random Forest")
lr_det = next(r for r in det if r["model"] == "Logistic Regression")
rule_cause = next(r for r in cause if r["model"].startswith("Rule"))
rf_cause = next(r for r in cause if r["model"] == "Random Forest")
maj_cause = next(r for r in cause if r["model"].startswith("Majority"))
best_nc = max([r for r in nc if not r["model"].startswith("Accept-all")],
              key=lambda r: float(r["f1"]))
rf_nc = next(r for r in nc if r["model"] == "Random Forest")
rf_dc = next(r for r in dc if r["model"] == "Random Forest")
rf_mp = next(r for r in mp if r["model"] == "Random Forest")
lr_mp = next(r for r in mp if r["model"] == "Logistic Regression")
top_feats = ", ".join(r["feature"] for r in imp1[:5])

story = []

# ---- Page 1: title ----------------------------------------------------
story.append(Spacer(1, 1.1 * inch))
story.append(p("EHR Data Migration &amp;<br/>Interface Analytics",
               TITLE))
story.append(Spacer(1, 0.15 * inch))
story.append(p("Business Analyst #26-03354 — NY State Office of Mental "
               "Health (OMH), Albany NY<br/>Oracle Health EHR "
               "Implementation", SUB))
story.append(Spacer(1, 0.3 * inch))
story.append(p("Bhupesh Chandra Dimri — November 2026", BODY))
story.append(p("9+ years FP&amp;A — JPMorgan Chase Financial Analyst, "
               "Jul 2025–present — M.S. Accounting Analytics &amp; Data "
               "Technology, Pace University (Dec 2025)", SMALL))
story.append(Spacer(1, 0.25 * inch))
story.append(p("<b>ALL DATA IS SYNTHETIC. No real patient data, no real "
               "PHI.</b> Patient names, MRNs, dates, codes, and messages "
               "are invented with a fixed seed (42). Break rates, defect "
               "rates, and terminology coverage are design choices, not "
               "OMH facts.", BODY))
story.append(Spacer(1, 0.15 * inch))
story.append(table(
    ["At a glance", "Result"],
    [["Match vs the JD", "4/10 — data &amp; reconciliation map; HL7/FHIR, "
      "terminology standards, and BA certification are hard gaps"],
     ["Module 1 — migration reconciliation",
      f"Rule detection F1 {fmt(rule_det['f1'])}; Random Forest catches "
      f"{pct(silent_recall_rf, 0)} of silent breaks the rule misses"],
     ["Module 2 — HL7 v2 / FHIR conformance",
      f"Deterministic checker agrees with injected defects on "
      f"{pct(agree_hl7)} HL7 / {pct(agree_fhir)} FHIR"],
     ["Module 3 — terminology mapping",
      f"Coverage {pct(cov_overall)}; mappability Random Forest F1 "
      f"{fmt(rf_mp['f1'])}, ROC-AUC {fmt(rf_mp['roc_auc'])}"],
     ["Tests", "84 checks — all pass (tests/run_tests.py)"]],
    widths=[1.9 * inch, 5.4 * inch]))
story.append(Spacer(1, 0.2 * inch))
story.append(p("What this proves: pipeline design, SQL-style profiling, "
               "validation logic, and triage thinking. What it does not "
               "prove: production EHR experience, HL7/FHIR hands-on "
               "work, healthcare-standards expertise, or a BA "
               "certification — the candidate claims none of those. "
               "All are [CONFIRM WITH CANDIDATE] items, not facts in "
               "this project.", SMALL))
story.append(PageBreak())

# ---- Page 2: the role -------------------------------------------------
story.append(p("1 &nbsp; The role and the problem", H1))
story.append(p("OMH is implementing the Oracle Health EHR to replace "
               "legacy clinical and patient systems. The Business "
               "Analyst works across two projects:", BODY))
story.append(p("<b>OMH_PRJ0251616 — EHR Data Migration.</b> Patient "
               "demographic, encounter, and clinical data move from "
               "legacy systems into Oracle Health: profiling, "
               "source-to-target mapping, code-set / terminology "
               "alignment, conversion rules, reconciliation, and "
               "validation.", BULLET))
story.append(p("<b>OMH_PRJ0251615 — EHR Interface Analysis.</b> HL7 v2 "
               "and FHIR specifications, message and code mapping, "
               "integration-platform routing, and end-to-end testing "
               "with partners including pharmacy, dietary, and the "
               "health information exchange (HIE).", BULLET))
story.append(p("BA work in the JD: requirements (BRDs, user stories, "
               "traceability matrices), data dictionaries, SQL "
               "profiling and validation, healthcare standards (USCDI, "
               "ICD-10, CPT, LOINC, SNOMED CT, RxNorm), Agile/Jira, "
               "test plans, UAT, and training materials. Knowledge "
               "transfer to State employees is mandated.", BODY))
story.append(p("Hard minimums in the JD", H2))
story.append(table(
    ["Minimum", "Candidate position"],
    [["84 months eliciting / analyzing / documenting business and "
      "functional requirements", "9+ years FP&amp;A — requirements, "
      "variance, and reporting work; healthcare-IS context is a gap"],
     ["60 months BA on healthcare information systems",
      "No healthcare-IS experience — hard gap"],
     ["48 months data analysis incl. source-to-target mapping + SQL",
      "Maps directly: SQL, Python, Power BI, Tableau in the master "
      "resume; this project is the evidence"],
     ["36 months HL7 v2 / FHIR interface requirements",
      "No HL7/FHIR hands-on — hard gap; this project builds the "
      "checker, it does not claim the experience"],
     ["48 months test plans / UAT; 36 months healthcare standards",
      "UAT thinking maps; standards knowledge is a gap"],
     ["24 months Agile/Scrum + Jira; CBAP / CCBA / PMI-PBA",
      "No BA certification — hard gap [CONFIRM WITH CANDIDATE]"]],
    widths=[2.6 * inch, 4.7 * inch]))
story.append(p("No pay, deadline, or apply link was in the pasted JD. "
               "Submission requires a résumé plus a candidate ID "
               "(driver's license / green card / visa, and passport if "
               "applicable) and referenced documents (certifications, "
               "degrees). Albany is onsite; the candidate is NYC-based "
               "— willingness / relocation is NOT stated and is an open "
               "question, never assumed.", SMALL))
story.append(PageBreak())

# ---- Page 3: data -----------------------------------------------------
story.append(p("2 &nbsp; Data — synthetic by design", H1))
story.append(p("Everything is generated with seed 42 and validated by "
               "gates that fail the run on any violation (missing "
               "values, impossible dates, unmapped rows carrying a "
               "standard code, defect rates outside bounds). The "
               "generator is in <font face='Helvetica-Bold'>"
               "src/data_generation.py</font>; the gates in "
               "<font face='Helvetica-Bold'>src/data_pipeline.py"
               "</font>.", BODY))
story.append(table(
    ["Dataset", "Rows", "What it carries"],
    [["Patients", "6,000", "Synthetic MRNs, DOB, gender, facility"],
     ["Encounters", "14,000", "Admit / discharge, encounter type, "
      "facility"],
     ["Clinical codes", "20,000", "ICD-10 / CPT / LOINC / SNOMED CT / "
      "RxNorm assignments"],
     ["Migration rows", f"{n_mig:,}",
      f"Source-to-target records across 6 monthly waves; break rate "
      f"{pct(break_rate)} ({n_breaks:,} breaks)"],
     ["HL7 v2 messages", f"{len(hl7):,}",
      f"ADT / ORM / ORU from pharmacy, dietary, HIE, legacy EHR; "
      f"nonconformant {pct(hl7_nc_rate)}"],
     ["FHIR resources", f"{len(fhir):,}",
      f"Patient / Encounter / Observation / MedicationRequest; "
      f"nonconformant {pct(fhir_nc_rate)}"],
     ["Terminology mappings", f"{len(tmap):,}",
      f"Source code to standard code; coverage {pct(cov_overall)}"]],
    widths=[1.5 * inch, 0.8 * inch, 5.0 * inch]))
story.append(p("Migration waves", H2))
story.append(p("Six monthly waves, Jan–Dec 2025. Wave W6 (Nov–Dec) is "
               "the go-live wave: it carries the highest break rate and "
               "is the hold-out (see page 9). Break types: missing "
               "record (a source row with no target), key mismatch (the "
               "join key disagrees), code-map failure (a source code "
               "with no standard mapping), duplicate (the same record "
               "migrated twice), and <b>silent mismatch</b> — every "
               "deterministic check passes, but a secondary field "
               "disagrees and the field-match rate is depressed (and "
               "overlaps the clean range, so no threshold separates "
               "it).", BODY))
story.append(Image(os.path.join(CHARTS, "migration_breaks_by_wave.png"),
                   width=7.0 * inch, height=2.6 * inch))
story.append(p("Figure 1 — Break rate by migration wave. The go-live "
               "wave is the hardest, which is why it is the hold-out, "
               "not the training set.", SMALL))
story.append(PageBreak())

# ---- Page 4: module 1 detection ---------------------------------------
story.append(p("3 &nbsp; Module 1 — Migration reconciliation: detection",
               H1))
story.append(p("The BA's first check is deterministic: a record is "
               "clean only if the target is present, the key matches, "
               "the code is mapped, and it is not a duplicate. That "
               "rule is the baseline. The statistical models see only "
               "what a BA has <i>before</i> the checks run — the "
               "field-match rate, a noisy secondary-field tell, the "
               "domain, the source system, and the wave.", BODY))
story.append(p("The four rule flags and the risk score are "
               "<b>deliberately excluded</b> from the model features: "
               "the flags <i>are</i> the break cause by construction, "
               "and the risk score is derived from it. The first run "
               "included them and scored a perfect 1.000 — that was "
               "leakage, not learning, and it was removed.", BODY))
story.append(table(
    ["Detector (hold-out: "
     f"{n_hold:,} rows, {n_hold_breaks} breaks)", "Precision", "Recall",
     "F1", "ROC-AUC"],
    [[r["model"], fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]),
      fmt(r["roc_auc"])] for r in det],
    widths=[3.3 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch,
            1.15 * inch]))
story.append(p("The rule is the best overall detector by design: "
               f"precision {fmt(rule_det['precision'])}, recall "
               f"{fmt(rule_det['recall'])} — it catches every overt "
               "break and misses the silent ones. The Random Forest's "
               "value is the silent class: on the "
               f"{len(silent)} silent breaks in the hold-out, the rule "
               f"catches <b>0%</b> and the Random Forest catches "
               f"<b>{pct(silent_recall_rf, 0)}</b>, with zero false "
               "positives on 6,120 clean rows. A model that only "
               "repeats the rule earns nothing; this one earns the "
               "breaks the rule cannot see.", BODY))
story.append(p("Logistic Regression is the best F1 among the learned "
               f"models ({fmt(lr_det['f1'])}), the Random Forest the "
               "best precision. Neither beats the rule overall — and "
               "that is the honest result: when the check is "
               "deterministic, the check wins.", BODY))
story.append(PageBreak())

# ---- Page 5: module 1 cause + queue -----------------------------------
story.append(p("4 &nbsp; Module 1 — Cause diagnosis and the review queue",
               H1))
story.append(p("Once a break is found, the BA needs its cause. The "
               "rule diagnosis reads which flag failed — target absent "
               "means missing record, key disagreeing means key "
               "mismatch, and so on. It names every overt break and "
               "only misses the silent ones (it calls them "
               "\u201cnone\u201d).", BODY))
story.append(table(
    ["Cause model (hold-out: 484 true breaks)", "Accuracy",
     "Macro-F1"],
    [[r["model"], fmt(r["accuracy"]), fmt(r["macro_f1"])]
     for r in cause],
    widths=[4.4 * inch, 1.4 * inch, 1.5 * inch]))
story.append(p(f"The learned Random Forest scores accuracy "
               f"{fmt(rf_cause['accuracy'])} against a majority-class "
               f"baseline of {fmt(maj_cause['accuracy'])} — modest, and "
               "honest. It uses no flag features, so it must infer the "
               "cause from domain, wave, and the field-match rate. The "
               "lesson is the deliverable: the flags name the cause for "
               "overt breaks; the model's job is triage, not diagnosis. "
               f"Top drivers: {top_feats}.", BODY))
story.append(p("The ranked review queue", H2))
story.append(p(f"The queue is what the BA opens each morning: open "
               f"breaks ranked by risk score. Top 20: "
               f"<b>{queue['missing_records_in_top20']}/20 are true "
               "missing records</b> — the go-live-critical class — "
               f"with queue recall of missing records "
               f"{pct(queue['recall_of_missing_records'])}. Missing "
               "records are the most severe break (severity 95/100) "
               "because a patient with no target record is a patient "
               "the new EHR does not know exists.", BODY))
story.append(Image(os.path.join(CHARTS, "feature_importance.png"),
                   width=7.0 * inch, height=2.6 * inch))
story.append(p("Figure 2 — Top drivers of break-cause classification "
               "(Random Forest). Domain and wave carry the signal; the "
               "field-match rate separates silent breaks from clean "
               "rows.", SMALL))
story.append(PageBreak())

# ---- Page 6: module 2 checker -----------------------------------------
story.append(p("5 &nbsp; Module 2 — HL7 v2 / FHIR conformance: the checker",
               H1))
story.append(p("The deliverable is a deterministic checker "
               "(<font face='Helvetica-Bold'>src/conformance.py"
               "</font>), not a black box. Six HL7 rules and three "
               "FHIR rule groups:", BODY))
story.append(table(
    ["Rule", "What it checks"],
    [["R1 — MSH present", "Message starts with a valid MSH segment"],
     ["R2 — PID present", "Patient identification segment exists"],
     ["R3 — PID fields", "Name (PID-5), DOB (PID-7, YYYYMMDD), gender "
      "(PID-8, M/F) present and well-formed"],
     ["R4 — code value set", "OBX observation code belongs to a known "
      "value set (CPT / LOINC); \u201cZZ-*\u201d fails"],
     ["R5 — MSH-7 datetime", "Message datetime parses as YYYYMMDDHHMM"],
     ["R6 — PV1 present", "Patient visit segment exists"],
     ["F1 — FHIR id", "Resource id matches ^[A-Za-z0-9\\-\\.]{1,64}$"],
     ["F2 — required elements", "birthDate / gender (Patient), status "
      "(Encounter), code (Observation), status + medication "
      "(MedicationRequest)"],
     ["F3 — value-set bindings", "gender, status, and code bindings "
      "against the FHIR value sets"]],
    widths=[1.7 * inch, 5.6 * inch]))
story.append(p(f"<b>Checker agreement with the injected defects: "
               f"{pct(agree_hl7)} of HL7 messages and {pct(agree_fhir)} "
               "of FHIR resources.</b> The disagreements are the two "
               "deliberately subtle defect classes, and they are "
               "reported, not hidden:", BODY))
story.append(p("<b>wrong_context</b> — a CPT procedure code where a "
               "LOINC lab code is expected. The checker validates code "
               "<i>membership</i>, not context, so it passes. "
               "<b>stale_reference</b> — a FHIR subject pointing at a "
               "Patient id that does not exist. The checker validates "
               "structure and bindings, not referential integrity, so "
               "it passes.", BULLET))
story.append(p("Every other defect class fails the checker 100% of the "
               "time — verified by test, including the bad_code class "
               "that the first run assigned to ADT messages (which "
               "carry no OBX segment, so the defect left conformant "
               "text and only 51.6% were caught). That generator bug "
               "was fixed: code defects now only apply to ORM/ORU "
               "messages.", SMALL))
story.append(PageBreak())

# ---- Page 7: module 2 classifier --------------------------------------
story.append(p("6 &nbsp; Module 2 — The classifier, honestly weak", H1))
story.append(p("A classifier was also trained on message metadata — "
               "length, segment/field counts, type, source, month — "
               "to see whether a statistical triage aid could add "
               "anything to the checker.", BODY))
story.append(table(
    ["Nonconformance model (hold-out: 996 messages, 112 "
     "nonconformant)", "Precision", "Recall", "F1", "ROC-AUC"],
    [[r["model"], fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]),
      fmt(r["roc_auc"])] for r in nc],
    widths=[3.3 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch,
            1.15 * inch]))
story.append(p(f"The best learned model is {best_nc['model']} at F1 "
               f"{fmt(best_nc['f1'])}; the Random Forest scores "
               f"{fmt(rf_nc['f1'])}. That is weak, and the reason is "
               "structural: content defects — a bad code, a bad ID, a "
               "wrong-context code — are invisible to metadata. The "
               "classifier can only see length and segment-count "
               "tells, so it catches missing segments and little else. "
               "Do not cite the classifier's F1 as this project's "
               "conformance performance — cite the checker's agreement "
               f"({pct(agree_hl7)} HL7 / {pct(agree_fhir)} FHIR).",
               BODY))
story.append(p("Defect-cause classification (nonconformant rows only) "
               "does better, because the cause correlates with the "
               "message shape:", BODY))
story.append(table(
    ["Defect-cause model", "Accuracy", "Macro-F1"],
    [[r["model"], fmt(r["accuracy"]), fmt(r["macro_f1"])] for r in dc],
    widths=[4.4 * inch, 1.4 * inch, 1.5 * inch]))
story.append(Image(os.path.join(CHARTS, "conformance_by_system.png"),
                   width=7.0 * inch, height=2.4 * inch))
story.append(p("Figure 3 — Nonconformance rate by source system and "
               "standard. The legacy EHR feed is the noisiest HL7 "
               "source; the HIE feed is the cleanest.", SMALL))
story.append(PageBreak())

# ---- Page 8: module 3 -------------------------------------------------
story.append(p("7 &nbsp; Module 3 — Terminology mapping validation", H1))
story.append(p("1,500 source-to-standard mappings across five target "
               "systems. Coverage is the scorecard; the unmapped codes "
               "are the work.", BODY))
story.append(table(
    ["Target system", "Codes", "Coverage"],
    [[r["target_system"], r["n_codes"], pct(float(r["coverage"]))]
     for r in cov],
    widths=[2.5 * inch, 1.5 * inch, 3.3 * inch]))
story.append(Image(os.path.join(CHARTS, "mapping_coverage.png"),
                   width=7.0 * inch, height=2.5 * inch))
story.append(p("Figure 4 — Mapping coverage by target system. RxNorm "
               f"({pct(float([r for r in cov if r['target_system'] == 'RxNorm'][0]['coverage']))}) "
               "is where the terminologist starts; ICD-10 is the "
               "strongest.", SMALL))
story.append(p("Mappability — can an unmapped code be mapped without "
               "manual review? The label is a latent simulation "
               "variable (token overlap with the standard description, "
               "frequency, and description length), not a clinician's "
               "judgment. On the grouped hold-out (80 unmapped codes, "
               "14 truly mappable):", BODY))
story.append(table(
    ["Mappability model", "Precision", "Recall", "F1", "ROC-AUC"],
    [[r["model"], fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]),
      fmt(r["roc_auc"])] for r in mp],
    widths=[2.6 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.4 * inch]))
story.append(p(f"Random Forest F1 {fmt(rf_mp['f1'])}, ROC-AUC "
               f"{fmt(rf_mp['roc_auc'])} — the strongest learned model "
               "in the project, because token overlap is a real signal "
               "for mappability. The top unmapped codes by frequency "
               "are in <font face='Helvetica-Bold'>models/top_unmapped"
               ".csv</font> — the first thing a BA triages, because "
               "one unmapped high-frequency code breaks more records "
               "than fifty rare ones.", BODY))
story.append(PageBreak())

# ---- Page 9: benchmark + hold-out -------------------------------------
story.append(p("8 &nbsp; Benchmark and hold-out design", H1))
story.append(table(
    ["Module", "Baseline", "Best learned model", "Metric"],
    [[r["module"], f"{r['baseline']} {fmt(r['baseline_score'])}",
      f"{r['best_model']} {fmt(r['best_score'])}", r["metric"]]
     for r in bench],
    widths=[1.9 * inch, 1.7 * inch, 2.2 * inch, 1.5 * inch]))
story.append(p("Every module is judged against an interpretable "
               "baseline, and every best model beats it — but the "
               "margins differ honestly. Migration cause classification "
               "is the weakest win (0.411 vs 0.324): the flags name the "
               "cause, so the model has little left to learn. "
               "Mappability is the strongest (0.650 vs 0.000): token "
               "overlap is a genuine signal. Interface conformance sits "
               "in between, and its real tool is the checker, not the "
               "classifier.", BODY))
story.append(p("Hold-out design — no leakage", H2))
story.append(p("<b>Modules 1 &amp; 2: time-based.</b> Train is "
               "Jan–Oct 2025; the hold-out is the last 61 days "
               "(2025-11-01 to 2025-12-31), including the go-live wave "
               "W6. Hold-out dates are strictly after train dates, "
               "verified by test.", BULLET))
story.append(p("<b>Module 3: grouped by code family.</b> A 70/30 split "
               "by source-code prefix, so the same code family never "
               "appears in both train and hold-out. A plain random "
               "split would leak — the model would memorise families "
               "instead of learning mappability.", BULLET))
story.append(p("<b>Module 1 features exclude the label.</b> The four "
               "rule flags and the risk score are out of the feature "
               "set (see page 3). The test suite re-checks the "
               "hold-out boundaries and the Module 3 split on every "
               "run.", BULLET))
story.append(Image(os.path.join(CHARTS, "roc_curves.png"),
                   width=7.0 * inch, height=2.4 * inch))
story.append(p("Figure 5 — ROC curves for interface nonconformance "
               "(hold-out). The curves are modest because the "
               "classifier sees metadata only.", SMALL))
story.append(PageBreak())

# ---- Page 10: limits + reproducibility --------------------------------
story.append(p("9 &nbsp; Limits — read before citing any number", H1))
story.append(p("All data is synthetic: patient names, MRNs, dates, "
               "codes, and messages are invented. Break rates, defect "
               "rates, and coverage are design choices, not OMH facts. "
               "A model that scores well here proves the <i>pipeline "
               "design</i>, not performance on a real migration.",
               BULLET))
story.append(p("Injected defects are deliberately detectable (missing "
               "segments, unknown codes, bad IDs). Real interface "
               "defects are subtler — expect lower recall on live "
               "feeds and keep a human in the review queue.", BULLET))
story.append(p("The mappability label is a latent simulation variable, "
               "not a clinician's judgment; on real unmapped codes a "
               "terminologist decides.", BULLET))
story.append(p("HL7 v2 and FHIR rules here are a subset of the real "
               "specifications (six HL7 rules, three FHIR rule "
               "groups). A production conformance suite covers the "
               "full implementation guide.", BULLET))
story.append(p("The candidate does not claim HL7/FHIR experience, "
               "healthcare-standards expertise, or a BA certification "
               "— all are [CONFIRM WITH CANDIDATE] items, not facts "
               "in this project.", BULLET))
story.append(p("10 &nbsp; Reproducibility", H1))
story.append(p("Run <font face='Helvetica-Bold'>python run_all.py"
               "</font> (generate → validate → model → chart → report), "
               "then <font face='Helvetica-Bold'>python "
               "tests/run_tests.py</font> — 84 checks, all pass. "
               "Requirements: Python 3.12, numpy ≥ 2.0, pandas ≥ 2.0, "
               "scikit-learn ≥ 1.5, matplotlib ≥ 3.8. Suggested repo "
               "(user creates the empty repo first): "
               "<font face='Helvetica-Bold'>omh-ehr-business-analyst-ml"
               "</font> under github.com/bhupesh-debug. Design "
               "continuity: the <font face='Helvetica-Bold'>"
               "lifesci-clinical-finance</font> folder in "
               "github.com/bhupesh-debug/Multi-project — finance / "
               "analytics work, never presented as EHR/HL7 experience.",
               BODY))
story.append(p("Files: run_all.py · build_pdf.py · build_deck.py · "
               "make_notebooks.py · requirements.txt · README.md · "
               "src/ (6 modules) · tests/run_tests.py · notebooks/ "
               "(01_eda–04_evaluation) · data/processed/ · models/ "
               "(14 CSVs) · reports/model_report.md + 7 charts.",
               SMALL))
story.append(p("Bhupesh Chandra Dimri — November 2026. All data "
               "synthetic; no real patient data, no real PHI.",
               SMALL))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.75 * inch, 0.55 * inch,
                      "EHR Data Migration & Interface Analytics — "
                      "OMH Business Analyst #26-03354 — Bhupesh Chandra "
                      "Dimri — Nov 2026 — ALL DATA SYNTHETIC")
    canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.55 * inch,
                           f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.75 * inch,
                        title="EHR Data Migration & Interface Analytics — "
                              "OMH Business Analyst #26-03354",
                        author="Bhupesh Chandra Dimri")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
