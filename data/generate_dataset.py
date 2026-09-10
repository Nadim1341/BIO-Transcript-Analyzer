"""
Synthetic Cleft Therapy Transcriptomic Dataset Generator.

Generates realistic differential gene expression statistical tables
for cleft lip and palate (CL/P) therapy research across multiple database
sources (RefSeq, ENSEMBL, lncRNAWiki, Ace View, miTranscriptome, UCSC Genes).
Produces both datasets.csv and datasets.pdf in data/.
"""

import os
from pathlib import Path
import random
import numpy as np
import pandas as pd

# Core cleft palate and craniofacial development genes
CLEFT_GENES = [
    ("IRF6", "RefSeq", "NM_006147.4", 2.85, 9.42, 1.2e-15, 2.4e-13),
    ("MSX1", "RefSeq", "NM_002448.4", -2.14, -7.12, 3.4e-11, 4.1e-9),
    ("TP63", "RefSeq", "NM_003722.5", 1.95, 6.84, 8.9e-10, 8.2e-8),
    ("TGFB3", "RefSeq", "NM_003239.5", -2.76, -8.91, 4.5e-14, 7.3e-12),
    ("BMP4", "RefSeq", "NM_001202.6", 1.62, 5.34, 4.2e-7, 2.1e-5),
    ("FGF8", "RefSeq", "NM_033163.4", 2.21, 7.82, 1.1e-12, 1.5e-10),
    ("WNT5A", "RefSeq", "NM_003392.5", -1.88, -6.15, 6.7e-9, 5.2e-7),
    ("SHH", "RefSeq", "NM_000193.4", 1.45, 4.88, 2.1e-6, 8.9e-5),
    ("SATB2", "RefSeq", "NM_015265.4", -2.35, -7.95, 8.2e-13, 1.2e-10),
    ("TBX22", "RefSeq", "NM_016951.3", -1.98, -6.42, 2.8e-9, 2.4e-7),
    ("PVRL1", "RefSeq", "NM_002855.5", 1.35, 4.25, 1.5e-5, 5.3e-4),
    ("MEIS2", "RefSeq", "NM_170674.3", 1.12, 3.82, 9.5e-5, 2.6e-3),
    ("PAX9", "RefSeq", "NM_006194.4", -1.72, -5.71, 1.8e-8, 1.3e-6),
    ("ARHGAP29", "RefSeq", "NM_004815.4", 1.52, 5.12, 8.3e-7, 3.8e-5),
    ("CRISPLD2", "RefSeq", "NM_031476.3", -1.41, -4.68, 5.2e-6, 1.9e-4),
    ("FOXL2", "ENSEMBL", "ENST00000305886", 1.78, 5.92, 1.2e-8, 9.4e-7),
    ("TFAP2A", "ENSEMBL", "ENST00000375929", -2.05, -6.78, 9.8e-10, 8.8e-8),
    ("JAG1", "ENSEMBL", "ENST00000254958", 1.28, 4.15, 2.3e-5, 7.8e-4),
    ("GLI3", "ENSEMBL", "ENST00000378684", -1.64, -5.45, 4.1e-8, 2.7e-6),
    ("MAFB", "ENSEMBL", "ENST00000300854", 1.89, 6.31, 3.7e-9, 3.1e-7),
    ("FGFR1", "ENSEMBL", "ENST00000397091", -1.38, -4.52, 9.1e-6, 3.1e-4),
    ("FGFR2", "ENSEMBL", "ENST00000358487", 2.15, 7.42, 6.3e-12, 8.9e-10),
    ("SNAI1", "ENSEMBL", "ENST00000244050", -2.42, -8.15, 4.1e-13, 6.5e-11),
    ("TWIST1", "ENSEMBL", "ENST00000234435", 1.74, 5.81, 1.9e-8, 1.4e-6),
    ("ZEB1", "ENSEMBL", "ENST00000383891", -1.55, -5.19, 6.5e-7, 3.1e-5),
    ("lnc-CLEFT1", "lncRNAWiki", "LNC_CLEFT0012", 2.45, 8.11, 4.8e-13, 7.1e-11),
    ("lnc-IRF6-AS1", "lncRNAWiki", "LNC_IRF6AS_01", 1.84, 6.12, 7.2e-9, 5.5e-7),
    ("NONCODE0029", "lncRNAWiki", "NONCODEG00291", -2.10, -6.95, 5.4e-10, 5.1e-8),
    ("LINC00152", "lncRNAWiki", "LINC00152_v1", 1.68, 5.54, 3.1e-8, 2.1e-6),
    ("lnc-MSX1-IT1", "lncRNAWiki", "LNC_MSX1IT_03", -1.92, -6.28, 4.5e-9, 3.6e-7),
    ("NONCODE0144", "lncRNAWiki", "NONCODEG01443", -1.45, -4.81, 2.9e-6, 1.1e-4),
    ("lnc-WNT5A-OT", "lncRNAWiki", "LNC_WNT5A_02", 1.32, 4.31, 1.2e-5, 4.4e-4),
    ("NONCODE0581", "lncRNAWiki", "NONCODEG05819", -1.77, -5.88, 1.4e-8, 1.1e-6),
    ("AV_TGFB3_a", "Ace View", "AV_TGFB3_01", -2.08, -6.82, 8.4e-10, 7.7e-8),
    ("AV_IRF6_b", "Ace View", "AV_IRF6_02", 1.71, 5.68, 2.1e-8, 1.5e-6),
    ("AV_BMP4_c", "Ace View", "AV_BMP4_03", 1.48, 4.95, 1.8e-6, 7.5e-5),
    ("AV_SATB2_d", "Ace View", "AV_SATB2_04", -1.82, -6.01, 1.1e-8, 8.7e-7),
    ("miT-miR-140", "miTranscriptome", "MINT004210", -2.31, -7.68, 2.2e-12, 3.2e-10),
    ("miT-miR-200b", "miTranscriptome", "MINT008912", 1.98, 6.55, 1.8e-9, 1.6e-7),
    ("miT-miR-451a", "miTranscriptome", "MINT012455", -1.61, -5.32, 5.5e-8, 3.5e-6),
    ("uc001abc.1", "UCSC Genes", "uc001abc.1", 1.54, 5.15, 7.8e-7, 3.6e-5),
    ("uc002def.2", "UCSC Genes", "uc002def.2", -1.85, -6.08, 9.2e-9, 7.4e-7),
    ("uc003ghi.1", "UCSC Genes", "uc003ghi.1", 1.21, 3.98, 4.5e-5, 1.4e-3),
]


def generate_dataset(num_samples: int = 500) -> pd.DataFrame:
    """Generate synthetic differential expression table with realistic biological parameters."""
    np.random.seed(42)
    random.seed(42)

    rows = []

    # 1. Add known benchmark cleft genes
    for gene_name, db, tid, logfc, t_val, pval, padj in CLEFT_GENES:
        rows.append({
            "transcript_id": tid,
            "logFC": logfc,
            "t": t_val,
            "P.Value": pval,
            "adj.P.Val": padj,
            "database_source": db
        })

    # 2. Add randomized background transcripts across all 6 databases
    db_choices = ["RefSeq", "ENSEMBL", "lncRNAWiki", "Ace View", "miTranscriptome", "UCSC Genes"]
    db_weights = [0.35, 0.30, 0.15, 0.08, 0.06, 0.06]

    remaining = num_samples - len(rows)

    for i in range(remaining):
        db = random.choices(db_choices, weights=db_weights)[0]
        idx = i + 1000

        if db == "RefSeq":
            prefix = random.choice(["NM_", "NR_", "XM_"])
            tid = f"{prefix}{idx:06d}.1"
        elif db == "ENSEMBL":
            tid = f"ENST{idx:011d}"
        elif db == "lncRNAWiki":
            tid = f"NONCODEG{idx:06d}"
        elif db == "Ace View":
            tid = f"AV_{idx:05d}"
        elif db == "miTranscriptome":
            tid = f"MINT{idx:06d}"
        else:
            tid = f"uc{idx:03d}xyz.1"

        # Categorize randomly: 15% Up, 15% Down, 70% Neutral
        cat = random.choices(["Up", "Down", "Neutral"], weights=[0.15, 0.15, 0.70])[0]

        if cat == "Up":
            logfc = float(np.random.normal(loc=1.8, scale=0.6))
            logfc = max(1.01, logfc)
            p_val = float(10 ** -np.random.uniform(2.5, 12.0))
            adj_p = min(0.045, p_val * np.random.uniform(1.2, 5.0))
            t_val = logfc * float(np.random.uniform(2.8, 3.8))
        elif cat == "Down":
            logfc = float(np.random.normal(loc=-1.8, scale=0.6))
            logfc = min(-1.01, logfc)
            p_val = float(10 ** -np.random.uniform(2.5, 12.0))
            adj_p = min(0.045, p_val * np.random.uniform(1.2, 5.0))
            t_val = logfc * float(np.random.uniform(2.8, 3.8))
        else:
            logfc = float(np.random.normal(loc=0.0, scale=0.45))
            # Keep within (-1, 1) mostly
            if abs(logfc) >= 1.0:
                # Force high p-value (insignificant)
                p_val = float(np.random.uniform(0.08, 0.95))
                adj_p = min(1.0, p_val * float(np.random.uniform(1.1, 1.5)))
            else:
                p_val = float(np.random.uniform(0.001, 0.95))
                adj_p = min(1.0, max(0.051, p_val * float(np.random.uniform(1.1, 2.0))))
            t_val = logfc * float(np.random.uniform(1.8, 3.0))

        rows.append({
            "transcript_id": tid,
            "logFC": round(logfc, 4),
            "t": round(t_val, 4),
            "P.Value": p_val,
            "adj.P.Val": adj_p,
            "database_source": db
        })

    df = pd.DataFrame(rows)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def create_pdf(df: pd.DataFrame, output_path: Path):
    """
    Generate clean PDF representation using ReportLab.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontSize=14,
        leading=16,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569")
    )

    elements = []
    elements.append(Paragraph("Cleft Therapy Differential Transcriptomic Expression Dataset", title_style))
    elements.append(Paragraph("Target: Cleft Palate & Craniofacial Tissue Models • Statistical Analysis Table (Limma/Voom)", body_style))
    elements.append(Spacer(1, 10))

    # Prepare table data (limit to first 120 rows for clean multi-page PDF generation)
    subset_df = df.head(120).copy()
    headers = ["Transcript ID", "logFC", "t-statistic", "P.Value", "adj.P.Val", "Database Source"]
    
    table_data = [headers]
    for _, r in subset_df.iterrows():
        table_data.append([
            str(r["transcript_id"]),
            f"{r['logFC']:.3f}",
            f"{r['t']:.3f}",
            f"{r['P.Value']:.2e}",
            f"{r['adj.P.Val']:.2e}",
            str(r["database_source"])
        ])

    t = Table(table_data, colWidths=[110, 60, 70, 75, 75, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
    ]))

    elements.append(t)
    doc.build(elements)


def main():
    data_dir = Path(__file__).resolve().parent
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "datasets.csv"
    pdf_path = data_dir / "datasets.pdf"

    print(f"Generating synthetic cleft therapy dataset with 500 transcripts...")
    df = generate_dataset(500)

    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    try:
        create_pdf(df, pdf_path)
        print(f"Saved PDF: {pdf_path}")
    except Exception as e:
        print(f"ReportLab PDF generation note: {e}")


if __name__ == "__main__":
    main()
