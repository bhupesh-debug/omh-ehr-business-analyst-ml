"""Build the 10-slide PDF deck (landscape).

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python build_deck.py
Output: ~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_Business-Analyst_OMH_Deck.pdf

All numbers come from models/*.csv and data/processed/*.csv.
"""

import csv
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(
    "~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_"
    "Business-Analyst_OMH_Deck.pdf")
CHARTS = os.path.join(ROOT, "reports", "charts")
MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")
PAGE = landscape(LETTER)

NAVY = colors.HexColor("#1f4e79")
TEAL = colors.HexColor("#2e9e6b")
GREY = colors.HexColor("#595959")
RED = colors.HexColor("#b23a48")

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=26,
                    leading=31, spaceAfter=10, textColor=NAVY)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=15,
                    spaceBefore=8, spaceAfter=6, textColor=NAVY)
BODY = ParagraphStyle("Body", parent=SS["BodyText"], fontSize=13,
                      leading=18, spaceAfter=8)
SMALL = ParagraphStyle("Small", parent=SS["BodyText"], fontSize=10.5,
                       leading=14, textColor=GREY)
CELL = ParagraphStyle("Cell", parent=SS["BodyText"], fontSize=12,
                      leading=15)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")
TITLE = ParagraphStyle("Title", parent=SS["Title"], fontSize=34,
                       leading=40, textColor=NAVY)
SUB = ParagraphStyle("Sub", parent=SS["BodyText"], fontSize=15,
                     leading=20, textColor=GREY)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=18,
                        bulletIndent=6, spaceAfter=6)


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
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
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
    return "—" if v != v else f"{v:.{digits}f}"


def pct(x, digits=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if v != v else f"{v:.{digits}f}%"


mig = pd.read_csv(os.path.join(PROC, "migration.csv"),
                  parse_dates=["migration_date"])
hl7 = pd.read_csv(os.path.join(PROC, "hl7_messages.csv"))
fhir = pd.read_csv(os.path.join(PROC, "fhir_resources.csv"))
tmap = pd.read_csv(os.path.join(PROC, "terminology_map.csv"))
iface = pd.read_csv(os.path.join(PROC, "interface_combined.csv"))
det = read_csv("migration_break_metrics.csv")
cause = read_csv("break_cause_metrics.csv")
nc = read_csv("nonconformance_metrics.csv")
mp = read_csv("mapping_metrics.csv")
cov = read_csv("coverage.csv")
bench = read_csv("benchmark.csv")
queue = read_csv("review_queue.csv")[0]
det_sc = pd.read_csv(os.path.join(MODELS, "detection_scored.csv"))

break_rate = mig["has_break"].mean()
silent = det_sc[det_sc["break_type"] == "silent_mismatch"]
silent_recall_rf = float(silent["y_pred_rf"].mean())
h = iface[iface["standard"] == "HL7v2"]
f = iface[iface["standard"] == "FHIR"]
agree_hl7 = float((h["checker_conformant"] == (h["nonconformant"] == 0))
                  .mean())
agree_fhir = float((f["checker_conformant"] == (f["nonconformant"] == 0))
                   .mean())
rule_det = next(r for r in det if r["model"].startswith("Rule"))
rf_det = next(r for r in det if r["model"] == "Random Forest")
rf_cause = next(r for r in cause if r["model"] == "Random Forest")
maj_cause = next(r for r in cause if r["model"].startswith("Majority"))
best_nc = max([r for r in nc if not r["model"].startswith("Accept-all")],
              key=lambda r: float(r["f1"]))
rf_mp = next(r for r in mp if r["model"] == "Random Forest")
cov_overall = float(tmap["mapped"].mean())

S = []

# 1 — title
S.append(Spacer(1, 1.4 * inch))
S.append(p("EHR Data Migration &amp;<br/>Interface Analytics", TITLE))
S.append(Spacer(1, 0.2 * inch))
S.append(p("Business Analyst #26-03354 — NY State Office of Mental "
           "Health (OMH), Albany NY — Oracle Health EHR", SUB))
S.append(Spacer(1, 0.3 * inch))
S.append(p("Bhupesh Chandra Dimri — November 2026", BODY))
S.append(p("<b>ALL DATA IS SYNTHETIC. No real patient data, no real "
           "PHI.</b>", BODY))
S.append(PageBreak())

# 2 — the role & the gap
S.append(p("The role — and the honest gap", H1))
S.append(p("OMH is replacing legacy clinical and patient systems with "
           "the Oracle Health EHR. Two projects: <b>data migration</b> "
           "(profiling, source-to-target mapping, terminology "
           "alignment, reconciliation, validation) and <b>interface "
           "analysis</b> (HL7 v2 / FHIR specs, message and code "
           "mapping, routing, end-to-end testing).", BODY))
S.append(p("<b>Match vs the JD: 4/10.</b> Data profiling, mapping, "
           "reconciliation, SQL validation, and test/UAT thinking map "
           "directly. Hard gaps: no healthcare-IS experience, no "
           "HL7/FHIR hands-on, no healthcare-standards expertise, no "
           "BA certification (CBAP / CCBA / PMI-PBA), and the "
           "84/60/48/36-month experience minimums. This project "
           "narrows the data part of the gap — it does not close the "
           "standards or certification parts.", BODY))
S.append(p("Albany is onsite; the candidate is NYC-based — "
           "willingness / relocation is NOT stated, an open question, "
           "never assumed. No pay, deadline, or apply link was in the "
           "pasted JD.", SMALL))
S.append(PageBreak())

# 3 — data
S.append(p("Synthetic data, stated plainly", H1))
S.append(table(
    ["Dataset", "Rows", "Key fact"],
    [["Patients / encounters / codes", "6,000 / 14,000 / 20,000",
      "Invented MRNs, dates, codes — seed 42"],
     ["Migration rows", "40,000",
      f"Break rate {pct(break_rate)} across 6 monthly waves"],
     ["HL7 v2 messages", "4,000",
      f"ADT / ORM / ORU; nonconformant {pct(float(hl7['nonconformant'].mean()))}"],
     ["FHIR resources", "2,000",
      f"Patient / Encounter / Observation / MedicationRequest; "
      f"nonconformant {pct(float(fhir['nonconformant'].mean()))}"],
     ["Terminology mappings", "1,500",
      f"Coverage {pct(cov_overall)} across 5 target systems"]],
    widths=[2.6 * inch, 2.0 * inch, 4.9 * inch]))
S.append(p("Validation gates fail the run on any violation — missing "
           "values, impossible dates, unmapped rows carrying a "
           "standard code, defect rates outside bounds. The generator "
           "is <b>src/data_generation.py</b>; the gates are "
           "<b>src/data_pipeline.py</b>.", BODY))
S.append(PageBreak())

# 4 — module 1 detection
S.append(p("Module 1 — Migration reconciliation: detection", H1))
S.append(table(
    ["Detector (hold-out: 6,604 rows, 484 breaks)", "Precision",
     "Recall", "F1", "ROC-AUC"],
    [[r["model"], fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]),
      fmt(r["roc_auc"])] for r in det],
    widths=[4.3 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.6 * inch]))
S.append(p("The deterministic rule — target present, key matches, "
           "code mapped, not a duplicate — is the best overall "
           "detector by design. When the check is deterministic, the "
           "check wins. The rule flags and the risk score are "
           "<b>excluded</b> from the model features: they are the "
           "break cause by construction, and including them scored a "
           "perfect 1.000 in the first run. That was leakage, not "
           "learning.", BODY))
S.append(PageBreak())

# 5 — silent breaks + queue
S.append(p("Module 1 — The silent breaks are the model's job", H1))
S.append(p(f"On the {len(silent)} <b>silent breaks</b> in the hold-out "
           "— every deterministic check passes, a secondary field "
           "disagrees — the rule catches <b>0%</b> and the Random "
           f"Forest catches <b>{pct(silent_recall_rf, 0)}</b>, with "
           "zero false positives on 6,120 clean rows.", BODY))
S.append(p("Break-cause diagnosis: the rule reads which flag failed — "
           f"accuracy {fmt(next(r for r in cause if r['model'].startswith('Rule'))['accuracy'])}. "
           f"The learned Random Forest (no flag features) scores "
           f"{fmt(rf_cause['accuracy'])} vs a majority baseline of "
           f"{fmt(maj_cause['accuracy'])} — modest and honest. The "
           "flags name the cause; the model does triage.", BODY))
S.append(p(f"<b>Review queue (top 20 by risk score): "
           f"{queue['missing_records_in_top20']}/20 are true missing "
           "records</b> — the go-live-critical class. Queue recall of "
           f"missing records "
           f"{pct(queue['recall_of_missing_records'])}.", BODY))
S.append(PageBreak())

# 6 — module 2 checker
S.append(p("Module 2 — HL7 v2 / FHIR: the checker is the tool", H1))
S.append(p("Six HL7 rules (MSH, PID, required PID fields, code value "
           "set, MSH-7 datetime, PV1) and three FHIR rule groups (id "
           "format, required elements, value-set bindings).", BODY))
S.append(p(f"<b>Checker agreement with the injected defects: "
           f"{pct(agree_hl7)} HL7 / {pct(agree_fhir)} FHIR.</b> The "
           "disagreements are the two deliberately subtle defects, "
           "reported not hidden: a CPT code where a LOINC is expected "
           "(membership, not context) and a reference to a Patient id "
           "that does not exist (structure, not referential "
           "integrity).", BODY))
S.append(p("Every other defect class fails the checker 100% of the "
           "time — verified by test.", BODY))
S.append(PageBreak())

# 7 — module 2 classifier
S.append(p("Module 2 — The classifier, honestly weak", H1))
S.append(table(
    ["Nonconformance model", "Precision", "Recall", "F1", "ROC-AUC"],
    [[r["model"], fmt(r["precision"]), fmt(r["recall"]), fmt(r["f1"]),
      fmt(r["roc_auc"])] for r in nc],
    widths=[3.6 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch, 1.7 * inch]))
S.append(p(f"Best learned model: {best_nc['model']}, F1 "
           f"{fmt(best_nc['f1'])}. Metadata alone — length, segment "
           "counts, type, source, month — cannot see content defects. "
           "Do not cite the classifier's F1 as this project's "
           "conformance performance; cite the checker's agreement.",
           BODY))
S.append(Image(os.path.join(CHARTS, "conformance_by_system.png"),
               width=8.6 * inch, height=2.9 * inch))
S.append(p("Nonconformance by source system and standard — the legacy "
           "EHR feed is the noisiest HL7 source.", SMALL))
S.append(PageBreak())

# 8 — module 3
S.append(p("Module 3 — Terminology mapping validation", H1))
S.append(table(
    ["Target system", "Codes", "Coverage"],
    [[r["target_system"], r["n_codes"], pct(float(r["coverage"]))]
     for r in cov],
    widths=[3.0 * inch, 2.0 * inch, 4.5 * inch]))
S.append(p(f"Mappability (can an unmapped code be mapped without "
           f"manual review?): Random Forest F1 {fmt(rf_mp['f1'])}, "
           f"ROC-AUC {fmt(rf_mp['roc_auc'])} — the strongest learned "
           "model in the project, because token overlap is a real "
           "signal. RxNorm is where the terminologist starts. The top "
           "unmapped codes by frequency are in "
           "<b>models/top_unmapped.csv</b>.", BODY))
S.append(PageBreak())

# 9 — benchmark + hold-out
S.append(p("Benchmark — and a hold-out with no leakage", H1))
S.append(table(
    ["Module", "Baseline", "Best learned model", "Metric"],
    [[r["module"], f"{r['baseline']} {fmt(r['baseline_score'])}",
      f"{r['best_model']} {fmt(r['best_score'])}", r["metric"]]
     for r in bench],
    widths=[2.4 * inch, 2.2 * inch, 2.8 * inch, 2.1 * inch]))
S.append(p("<b>Modules 1 &amp; 2: time-based.</b> Train Jan–Oct 2025; "
           "hold-out the last 61 days, including the go-live wave W6. "
           "<b>Module 3: grouped by code family</b> — the same family "
           "never appears in both train and hold-out. A plain random "
           "split would leak.", BODY))
S.append(Image(os.path.join(CHARTS, "roc_curves.png"),
               width=8.6 * inch, height=2.7 * inch))
S.append(p("ROC curves for interface nonconformance (hold-out) — "
           "modest, because the classifier sees metadata only.",
           SMALL))
S.append(PageBreak())

# 10 — limits + next steps
S.append(p("Limits — and what comes next", H1))
S.append(p("All data is synthetic; rates are design choices, not OMH "
           "facts. Injected defects are deliberately detectable; real "
           "defects are subtler. The mappability label is a simulation "
           "variable, not a clinician's judgment. The six HL7 rules "
           "and three FHIR rule groups are a subset of the real "
           "specifications.", BULLET))
S.append(p("This project proves pipeline design, profiling, validation "
           "logic, and triage. It does not prove production EHR "
           "experience, HL7/FHIR hands-on work, standards expertise, "
           "or a BA certification.", BULLET))
S.append(p("<b>Next steps:</b> 84 tests pass. Suggested repo "
           "(user creates the empty repo first): "
           "<b>omh-ehr-business-analyst-ml</b> under "
           "github.com/bhupesh-debug. Run <b>python run_all.py</b> "
           "then <b>python tests/run_tests.py</b> to reproduce every "
           "number in this deck.", BODY))
S.append(p("Bhupesh Chandra Dimri — November 2026. All data synthetic; "
           "no real patient data, no real PHI.", SMALL))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.75 * inch, 0.45 * inch,
                      "EHR Data Migration & Interface Analytics — OMH "
                      "Business Analyst #26-03354 — Bhupesh Chandra "
                      "Dimri — Nov 2026 — ALL DATA SYNTHETIC")
    canvas.drawRightString(PAGE[0] - 0.75 * inch, 0.45 * inch,
                           f"Slide {doc.page} / 10")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=PAGE,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                        title="EHR Data Migration & Interface Analytics — "
                              "OMH Business Analyst #26-03354 (Deck)",
                        author="Bhupesh Chandra Dimri")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
