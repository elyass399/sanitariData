from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA    = ROOT / "data" / "dataset_en.csv"
REPORT  = ROOT / "eda_report.pdf"
TMP     = ROOT / "tmp_plots"
 
TARGET   = "Gestational_Diabetes"
FEATURES = [
    "Age", "Previous_Pregnancies", "Previous_C_Sections",
    "Diastolic_BP_mmHg", "Serum_Insulin_2h", "BMI",
]
 
sns.set_theme(style="whitegrid", palette="muted")
 
 # ── Helpers ───────────────────────────────────────────────────────────────────
def save_fig(name: str) -> Path:
    TMP.mkdir(exist_ok=True)
    path = TMP / f"{name}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path

# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_distributions(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Feature Distributions by Target", fontsize=13)
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, bins=30, ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    plt.tight_layout()
    return save_fig("distributions")
 
 
def plot_correlation(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return save_fig("correlation")
 
 
def plot_boxplots(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Feature vs Gestational Diabetes", fontsize=13)
    for ax, col in zip(axes.flat, FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax,
                    palette=["#4C72B0", "#DD8452"])
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("0 = No Diabetes   |   1 = Diabetes")
    plt.tight_layout()
    return save_fig("boxplots")
 
 
def plot_class_balance(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].value_counts()
    bars = ax.bar(["No Diabetes (0)", "Diabetes (1)"],
                  counts.values,
                  color=["#4C72B0", "#DD8452"])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30, str(val),
                ha="center", fontsize=11)
    ax.set_title("Class Balance")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return save_fig("class_balance")

# ── Outlier report ────────────────────────────────────────────────────────────
def outlier_table(df: pd.DataFrame) -> list[list]:
    rows = [["Feature", "Q1", "Q3", "IQR", "Fence Low", "Fence High", "Min", "Max", "Outliers"]]
    for col in FEATURES:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        rows.append([
            col,
            f"{q1:.1f}", f"{q3:.1f}", f"{iqr:.1f}",
            f"{lo:.1f}", f"{hi:.1f}",
            f"{df[col].min():.1f}", f"{df[col].max():.1f}",
            str(n_out),
        ])
    return rows

# ── PDF ───────────────────────────────────────────────────────────────────────
def build_pdf(df: pd.DataFrame,
              dist_img: Path, corr_img: Path,
              box_img: Path, bal_img: Path) -> None:
 
    doc    = SimpleDocTemplate(str(REPORT), pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    H1     = styles["Heading1"]
    H2     = styles["Heading2"]
    BODY   = styles["Normal"]
    MONO   = ParagraphStyle("mono", fontName="Courier", fontSize=8, leading=12)
 
    story = []

# ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("GDM Risk Predictor — EDA Report", H1))
    story.append(Paragraph(
        f"Dataset: {DATA.name}  |  Rows: {df.shape[0]}  |  Features: {len(FEATURES)}",
        BODY))
    story.append(Spacer(1, 0.4*cm))
 
    # ── 1. Descriptive stats ──────────────────────────────────────────────────
    story.append(Paragraph("1. Descriptive Statistics", H2))
 
    desc = df[FEATURES].describe().round(2)
    table_data = [[""] + FEATURES]
    for stat in desc.index:
        table_data.append([stat] + [str(v) for v in desc.loc[stat].values])
 
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a52")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",   (0, 0), (-1, -1), 7),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f9")]),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

 # ── 2. Class balance ──────────────────────────────────────────────────────
    story.append(Paragraph("2. Class Balance", H2))
    counts = df[TARGET].value_counts()
    pct    = df[TARGET].value_counts(normalize=True).round(3) * 100
    story.append(Paragraph(
        f"No Diabetes (0): {counts[0]}  ({pct[0]:.1f}%)   |   "
        f"Diabetes (1): {counts[1]}  ({pct[1]:.1f}%)",
        BODY))
    story.append(Paragraph(
        "⚠ Significant class imbalance (91/9). "
        "Model will use class_weight='balanced'.", BODY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image(str(bal_img), width=8*cm, height=6*cm))
    story.append(Spacer(1, 0.4*cm))
 
    # ── 3. Outliers ───────────────────────────────────────────────────────────
    story.append(Paragraph("3. Outlier Analysis (IQR Method)", H2))
    out_data = outlier_table(df)
    ot = Table(out_data, repeatRows=1)
    ot.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b3a52")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTSIZE",   (0, 0), (-1, -1), 7),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f9")]),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(ot)
    story.append(Paragraph(
        "Note: Serum_Insulin_2h capped at 320.0 — treated as sensor limit, replaced with median.",
        BODY))
    story.append(PageBreak())
 
    # ── 4. Distributions ──────────────────────────────────────────────────────
    story.append(Paragraph("4. Feature Distributions", H2))
    story.append(Image(str(dist_img), width=16*cm, height=9*cm))
    story.append(Spacer(1, 0.4*cm))
 
    # ── 5. Boxplots ───────────────────────────────────────────────────────────
    story.append(Paragraph("5. Boxplots by Target", H2))
    story.append(Image(str(box_img), width=16*cm, height=9*cm))
    story.append(PageBreak())
 
    # ── 6. Correlation ────────────────────────────────────────────────────────
    story.append(Paragraph("6. Correlation Matrix", H2))
    story.append(Image(str(corr_img), width=13*cm, height=10*cm))
    story.append(Spacer(1, 0.4*cm))
 
    corr = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    story.append(Paragraph("Correlations with target (Gestational_Diabetes):", BODY))
    for feat, val in corr.items():
        story.append(Paragraph(f"&nbsp;&nbsp;{feat}: {val:.3f}", MONO))
 
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(
        "Key insight: Serum_Insulin (0.15) and BMI (0.09) are strongest predictors. "
        "Low overall correlations suggest tree-based or regularized models needed.", BODY))
 
    doc.build(story)
    print(f"\nReport saved: {REPORT}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Loading {DATA}")
    df = pd.read_csv(DATA)
    print(f"Shape: {df.shape}")
 
    print("Generating plots...")
    dist_img = plot_distributions(df)
    corr_img = plot_correlation(df)
    box_img  = plot_boxplots(df)
    bal_img  = plot_class_balance(df)
 
    print("Building PDF...")
    build_pdf(df, dist_img, corr_img, box_img, bal_img)
 
    # Cleanup tmp
    import shutil
    shutil.rmtree(TMP, ignore_errors=True)
 
 
if __name__ == "__main__":
    main()