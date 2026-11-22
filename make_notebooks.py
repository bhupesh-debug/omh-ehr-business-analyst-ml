"""Generate the 4 notebooks (01_eda, 02_feature_engineering,
03_modeling, 04_evaluation) as .ipynb JSON from code strings.
Run after run_all.py. The notebooks re-run the pipeline pieces
interactively; they do not re-train from scratch on open.
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(ROOT, "notebooks")
os.makedirs(NB, exist_ok=True)

HDR = ("# EHR Data Migration & Interface Analytics — {title}\n"
       "# Role: Business Analyst #26-03354, NY State Office of Mental "
       "Health (OMH), Albany NY\n"
       "# ALL DATA IS SYNTHETIC. No real patient data, no real PHI.\n"
       "# Author: Bhupesh Chandra Dimri — November 2026\n")


def nb(title, cells):
    out = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}},
           "nbformat": 4, "nbformat_minor": 5}
    out["cells"].append({"cell_type": "markdown", "metadata": {},
                         "source": [HDR.format(title=title)]})
    for c in cells:
        if c.startswith("# "):
            out["cells"].append({"cell_type": "markdown", "metadata": {},
                                 "source": [c]})
        else:
            out["cells"].append({"cell_type": "code",
                                 "execution_count": None, "metadata": {},
                                 "outputs": [], "source": [c]})
    return out


def write(name, data):
    p = os.path.join(NB, name)
    with open(p, "w") as fh:
        json.dump(data, fh, indent=1)
    print("wrote", p)


write("01_eda.ipynb", nb("EDA", [
    "# Load the synthetic data and check the shape of each domain",
    "import pandas as pd\n"
    "mig = pd.read_csv('data/processed/migration.csv', parse_dates=['migration_date'])\n"
    "hl7 = pd.read_csv('data/processed/hl7_messages.csv', parse_dates=['msg_datetime'])\n"
    "fhir = pd.read_csv('data/processed/fhir_resources.csv', parse_dates=['created_date'])\n"
    "tmap = pd.read_csv('data/processed/terminology_map.csv')\n"
    "print(mig.shape, hl7.shape, fhir.shape, tmap.shape)\n"
    "print(mig['break_type'].value_counts(normalize=True).round(4))\n"
    "print(hl7['defect_type'].value_counts(normalize=True).round(4))\n"
    "print(tmap.groupby('target_system')['mapped'].mean().round(3))",
    "# Break rate by migration wave — does the go-live wave get worse?",
    "print(mig.groupby('migration_wave')['has_break'].mean().round(4))",
    "# Conformance by source system and standard",
    "iface = pd.read_csv('data/processed/interface_combined.csv')\n"
    "print(iface.groupby(['standard', 'source_system'])['nonconformant'].mean().round(4))",
]))

write("02_feature_engineering.ipynb", nb("Feature engineering", [
    "# Features are fitted on the training period only — the hold-out never contributes",
    "import sys\n"
    "sys.path.insert(0, 'src')\n"
    "import feature_engineering as fe\n"
    "import pandas as pd\n"
    "mig = pd.read_csv('data/processed/migration.csv', parse_dates=['migration_date'])\n"
    "br = mig[mig['has_break'] == 1]\n"
    "F = fe.build_migration_features(br)\n"
    "print(F.shape)\n"
    "print(F.head(3).T)",
    "# Interface features: message type, source system, length, segment count",
    "iface = pd.read_csv('data/processed/interface_combined.csv', parse_dates=['msg_datetime'])\n"
    "G = fe.build_interface_features(iface)\n"
    "print(G.shape)\n"
    "print(sorted(G.columns.tolist()))",
]))

write("03_modeling.ipynb", nb("Modeling", [
    "# Rule baselines run first — a model must beat them to earn its place",
    "import pandas as pd\n"
    "det = pd.read_csv('models/migration_break_metrics.csv')\n"
    "print(det.to_string(index=False))\n"
    "bc = pd.read_csv('models/break_cause_metrics.csv')\n"
    "print(bc.to_string(index=False))\n"
    "nc = pd.read_csv('models/nonconformance_metrics.csv')\n"
    "print(nc.to_string(index=False))\n"
    "mp = pd.read_csv('models/mapping_metrics.csv')\n"
    "print(mp.to_string(index=False))",
    "# The review queue: the deliverable the BA opens each morning",
    "q = pd.read_csv('models/review_queue_top20.csv')\n"
    "print(q.to_string(index=False))",
]))

write("04_evaluation.ipynb", nb("Evaluation", [
    "# Benchmark: best model vs interpretable baseline, per module",
    "import pandas as pd\n"
    "print(pd.read_csv('models/benchmark.csv').to_string(index=False))",
    "# Coverage scorecard — which standard system needs the terminologist first?",
    "print(pd.read_csv('models/coverage.csv').to_string(index=False))\n"
    "print(pd.read_csv('models/top_unmapped.csv').head(10).to_string(index=False))",
    "# Limits, stated on the same page as the results",
    "print(open('reports/model_report.md').read().split('## Limits')[1][:1500])",
]))
print("done")
