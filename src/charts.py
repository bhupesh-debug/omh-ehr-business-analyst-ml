"""Charts for the OMH EHR project (all PNG, all labelled SYNTHETIC)."""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "figure.dpi": 120})
NAVY = "#1f4e79"
TEAL = "#2e9e6b"
RED = "#b23a48"
GREY = "#595959"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def migration_breaks_by_wave(mig_csv, out):
    mig = pd.read_csv(mig_csv, parse_dates=["migration_date"])
    g = mig.groupby("migration_wave").agg(
        rows=("record_id", "count"),
        breaks=("has_break", "sum"))
    g["rate"] = g["breaks"] / g["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(g.index, g["rate"] * 100, color=NAVY)
    for i, v in enumerate(g["rate"] * 100):
        ax.text(i, v + 0.15, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Break rate (%)")
    ax.set_xlabel("Migration wave (bi-monthly, 2025)")
    ax.set_title("Migration break rate by wave (synthetic)")
    ax.set_ylim(0, max(g["rate"] * 100) * 1.25)
    _save(fig, out)


def conformance_by_system(iface_csv, out):
    df = pd.read_csv(iface_csv)
    g = df.groupby(["standard", "source_system"]).agg(
        rows=("nonconformant", "size"),
        bad=("nonconformant", "sum"))
    g["rate"] = 1 - g["bad"] / g["rows"]
    piv = g["rate"].unstack("standard") * 100
    fig, ax = plt.subplots(figsize=(7, 3.4))
    piv.plot(kind="bar", ax=ax, color=[NAVY, TEAL], width=0.7)
    ax.set_ylabel("Conformant (%)")
    ax.set_xlabel("Source system")
    ax.set_title("Interface conformance by source system and standard "
                 "(synthetic)")
    ax.set_ylim(85, 100)
    ax.legend(title="Standard")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def mapping_coverage(tmap_csv, out):
    tmap = pd.read_csv(tmap_csv)
    g = tmap.groupby("target_system")["mapped"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(g.index, g["mean"] * 100, color=TEAL)
    for i, (v, n) in enumerate(zip(g["mean"] * 100, g["count"])):
        ax.text(v + 0.5, i, f"{v:.1f}%  (n={n})", va="center", fontsize=9)
    ax.set_xlabel("Mapped (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Terminology mapping coverage by target system (synthetic)")
    _save(fig, out)


def feature_importance(imp_csv, out, title):
    imp = pd.read_csv(imp_csv).head(10)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color=NAVY)
    ax.set_xlabel("Random Forest importance")
    ax.set_title(title)
    _save(fig, out)


def roc_curves(scored_csv, out):
    df = pd.read_csv(scored_csv)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for col, label, color in [("rf_score", "Random Forest", NAVY),
                              ("lr_score", "Logistic Regression", TEAL)]:
        fpr, tpr, _ = roc_curve(df["nonconformant"], df[col])
        ax.plot(fpr, tpr, label=label, color=color)
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Interface nonconformance — ROC (hold-out, synthetic)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out)


def comparison(metrics_csv, out, title, metric="f1"):
    m = pd.read_csv(metrics_csv)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(m["model"], m[metric], color=[GREY, TEAL, NAVY][:len(m)])
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(m[metric]):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    _save(fig, out)


def correlation_heatmap(scored_csv, out):
    df = pd.read_csv(scored_csv)
    num = df.select_dtypes(include=[np.number])
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(corr.values, cmap="Blues", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Interface feature correlations (hold-out, synthetic)")
    _save(fig, out)
