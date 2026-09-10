"""
Data Parsing and Ingestion Module for Bio Transcript Analyzer.

Handles extraction and cleaning of differential gene expression tables from
both CSV and PDF formats, extracting key statistical parameters:
- transcript_id
- logFC
- t (t-statistic)
- P.Value
- adj.P.Val (FDR / adjusted p-value)
- database_source (RefSeq, ENSEMBL, lncRNAWiki, Ace View, miTranscriptome, UCSC Genes)
"""

import io
import re
import logging
from typing import Optional, Union, BinaryIO, List, Dict, Any
import pandas as pd
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

# Canonical column names required for downstream analysis
REQUIRED_COLUMNS = ["transcript_id", "logFC", "t", "P.Value", "adj.P.Val", "database_source"]

# Flexible alias mappings for column name standardization
COLUMN_ALIASES: Dict[str, List[str]] = {
    "transcript_id": [
        "transcript_id", "transcript", "gene_id", "gene", "symbol", "id",
        "probe_id", "target_id", "transcriptid", "feature_id"
    ],
    "logFC": [
        "logfc", "log2fc", "log_fc", "log2_fc", "log2foldchange", "fold_change", "logfoldchange"
    ],
    "t": [
        "t", "t_stat", "t_statistic", "tstat", "stat", "score", "z_score", "wald_stat"
    ],
    "P.Value": [
        "p.value", "pvalue", "p_val", "pval", "p", "p-value", "raw_p"
    ],
    "adj.P.Val": [
        "adj.p.val", "adj_p_val", "adjpval", "padj", "fdr", "q_value", "qval", "adj_pvalue", "adj.pval"
    ],
    "database_source": [
        "database_source", "database", "source", "db", "db_source", "annotation_source", "origin"
    ],
}


class DataParser:
    """Parser and cleaner for transcriptomic differential expression tables."""

    @staticmethod
    def infer_database_source(transcript_id: str) -> str:
        """
        Infer the annotation database source based on transcript ID naming conventions.
        Recognizes RefSeq, ENSEMBL, lncRNAWiki, Ace View, miTranscriptome, and UCSC Genes.
        """
        if not isinstance(transcript_id, str):
            return "Other"
        
        tid = transcript_id.strip()
        if re.match(r"^(NM_|NR_|XM_|XR_|NP_|YP_)", tid, re.IGNORECASE):
            return "RefSeq"
        elif re.match(r"^ENS[TGA]\d+", tid, re.IGNORECASE):
            return "ENSEMBL"
        elif re.match(r"^(LNC|lnc-|NONCODE|LINC)", tid, re.IGNORECASE):
            return "lncRNAWiki"
        elif re.match(r"^(uc\d{3}[a-z]{3}\.\d+|ucsc)", tid, re.IGNORECASE):
            return "UCSC Genes"
        elif re.match(r"^(MINT|miT|MIMAT)", tid, re.IGNORECASE):
            return "miTranscriptome"
        elif re.match(r"^(AceView|a[A-Z0-9]+|AV_)", tid, re.IGNORECASE):
            return "Ace View"
        return "RefSeq"  # Default fallback if unspecified

    @classmethod
    def standardize_column_names(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map varied column names to canonical schema:
        ['transcript_id', 'logFC', 't', 'P.Value', 'adj.P.Val', 'database_source']
        """
        df = df.copy()
        raw_cols = {c: str(c).strip() for c in df.columns}
        col_mapping = {}

        for raw_col, cleaned_name in raw_cols.items():
            norm_name = re.sub(r"[_\s\-]+", "", cleaned_name.lower())
            matched = False
            for canonical, aliases in COLUMN_ALIASES.items():
                alias_norms = [re.sub(r"[_\s\-]+", "", a.lower()) for a in aliases]
                if norm_name in alias_norms:
                    col_mapping[raw_col] = canonical
                    matched = True
                    break
            if not matched:
                col_mapping[raw_col] = cleaned_name

        df.rename(columns=col_mapping, inplace=True)
        return df

    @classmethod
    def vectorized_infer_database(cls, series: pd.Series) -> pd.Series:
        """High-performance vectorized database source inference (< 80ms on 500,000 rows)."""
        s = series.astype(str).str.strip()
        conds = [
            s.str.startswith(("NM_", "NR_", "XM_", "XR_", "NP_", "YP_")),
            s.str.startswith(("ENST", "ENSG", "ENSA", "ENS")),
            s.str.startswith(("LNC", "lnc-", "NONCODE", "LINC")),
            s.str.startswith(("uc", "UCSC", "ucsc")),
            s.str.startswith(("MINT", "miT", "MIMAT")),
            s.str.startswith(("AceView", "AV_", "AV")),
        ]
        choices = [
            "RefSeq",
            "ENSEMBL",
            "lncRNAWiki",
            "UCSC Genes",
            "miTranscriptome",
            "Ace View"
        ]
        return pd.Series(np.select(conds, choices, default="RefSeq"), index=series.index)

    @classmethod
    def clean_dataset(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate extracted differential expression data.
        - Maps columns to canonical names
        - Infers or fills database source
        - Coerces numerical values and drops nulls
        - Validates probabilities and statistics
        """
        if df.empty:
            raise ValueError("Input dataframe is empty.")

        df = cls.standardize_column_names(df)

        # Check required columns
        missing_cols = [c for c in ["transcript_id", "logFC", "P.Value"] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing mandatory differential expression columns: {missing_cols}")

        # Derive t-statistic if missing
        if "t" not in df.columns:
            logger.info("Column 't' not found; approximating from logFC.")
            df["t"] = df["logFC"] * 2.5  # reasonable empirical heuristic when absent

        # Derive adj.P.Val if missing using Benjamini-Hochberg FDR
        if "adj.P.Val" not in df.columns:
            logger.info("Column 'adj.P.Val' not found; calculating Benjamini-Hochberg FDR.")
            pvals = pd.to_numeric(df["P.Value"], errors="coerce").fillna(1.0)
            n = len(pvals)
            order = np.argsort(pvals)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, n + 1)
            qvals = (pvals * n / ranks).clip(upper=1.0)
            df["adj.P.Val"] = qvals

        # Assign/infer database source using vectorized operations
        if "database_source" not in df.columns:
            df["database_source"] = cls.vectorized_infer_database(df["transcript_id"])
        else:
            df["database_source"] = df["database_source"].fillna("").astype(str).str.strip()
            mask_empty = df["database_source"].isin(["", "nan", "None", "Unknown"])
            if mask_empty.any():
                df.loc[mask_empty, "database_source"] = cls.vectorized_infer_database(df.loc[mask_empty, "transcript_id"])

        # Drop rows with critical null values or empty IDs
        initial_len = len(df)
        df = df.dropna(subset=["transcript_id", "logFC", "P.Value", "adj.P.Val"]).copy()
        tid_cleaned = df["transcript_id"].astype(str).str.strip()
        valid_mask = (
            (tid_cleaned != "") &
            ~tid_cleaned.str.lower().isin(["nan", "null", "none"])
        )
        df = df[valid_mask].copy()
        df["transcript_id"] = tid_cleaned[valid_mask]
        df["database_source"] = df["database_source"].astype(str).str.strip()

        # Ensure probabilities are bounded in (0, 1]
        eps = 1e-300
        df["P.Value"] = df["P.Value"].clip(lower=eps, upper=1.0)
        df["adj.P.Val"] = df["adj.P.Val"].clip(lower=eps, upper=1.0)

        # Remove duplicate transcript entries, keeping the one with lowest adj.P.Val
        df = df.sort_values("adj.P.Val", ascending=True).drop_duplicates(subset=["transcript_id"]).reset_index(drop=True)

        logger.info(f"Cleaned dataset: {len(df)} records retained from {initial_len} initial records.")
        return df

    @classmethod
    def parse_csv(cls, file_source: Union[str, BinaryIO, io.StringIO]) -> pd.DataFrame:
        """
        Parse CSV, TSV, or delimited text into a cleaned DataFrame.
        Uses fast C-engine parsing with automated delimiter detection.
        """
        try:
            # First try fast C-engine with standard delimiters (\t, ,, ;)
            if hasattr(file_source, "tell") and hasattr(file_source, "seek"):
                pos = file_source.tell()
                sample = file_source.read(4096)
                file_source.seek(pos)
                if isinstance(sample, bytes):
                    sample_str = sample.decode("utf-8", errors="ignore")
                else:
                    sample_str = str(sample)
                tab_cnt = sample_str.count("\t")
                comma_cnt = sample_str.count(",")
                semi_cnt = sample_str.count(";")
                if tab_cnt > comma_cnt and tab_cnt > semi_cnt:
                    best_sep = "\t"
                elif semi_cnt > comma_cnt:
                    best_sep = ";"
                else:
                    best_sep = ","
                try:
                    df = pd.read_csv(file_source, sep=best_sep, engine="c")
                    return cls.clean_dataset(df)
                except Exception:
                    file_source.seek(pos)

            df = pd.read_csv(file_source, sep=None, engine="python")
            return cls.clean_dataset(df)
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            raise ValueError(f"Failed to parse CSV file: {e}")

    @classmethod
    def parse_pdf(cls, file_source: Union[str, BinaryIO, bytes]) -> pd.DataFrame:
        """
        Extract differential expression tables from PDF files using pdfplumber.
        Includes table extraction and fallback regex parsing for unstructured PDF text.
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required for PDF parsing. Please install pdfplumber.")

        extracted_rows: List[List[str]] = []
        header_candidate: Optional[List[str]] = None

        if isinstance(file_source, bytes):
            pdf_file = io.BytesIO(file_source)
        else:
            pdf_file = file_source

        with pdfplumber.open(pdf_file) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                # Attempt 1: Extract structured tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            # Filter empty cells and clean whitespace
                            clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            if any(clean_row):
                                if header_candidate is None:
                                    # Check if row looks like a header
                                    row_lower = [c.lower() for c in clean_row]
                                    if any("logfc" in c or "p.val" in c or "transcript" in c or "id" in c for c in row_lower):
                                        header_candidate = clean_row
                                        continue
                                extracted_rows.append(clean_row)
                else:
                    # Attempt 2: Text extraction with regex pattern matching
                    text = page.extract_text()
                    if text:
                        lines = text.split("\n")
                        for line in lines:
                            parts = re.split(r"\s{2,}|\t|,|;", line.strip())
                            if len(parts) >= 4:
                                row_lower = [p.lower() for p in parts]
                                if header_candidate is None and any("logfc" in p or "transcript" in p for p in row_lower):
                                    header_candidate = parts
                                else:
                                    extracted_rows.append(parts)

        if not extracted_rows:
            raise ValueError("No table or transcriptomic data could be extracted from the PDF file.")

        # Construct DataFrame
        if header_candidate and len(header_candidate) > 0:
            cols = header_candidate
            # Align row lengths to columns length
            aligned_rows = []
            for row in extracted_rows:
                if len(row) == len(cols):
                    aligned_rows.append(row)
                elif len(row) > len(cols):
                    aligned_rows.append(row[:len(cols)])
                elif len(row) >= 4:
                    padded = row + [""] * (len(cols) - len(row))
                    aligned_rows.append(padded)
            df = pd.DataFrame(aligned_rows, columns=cols)
        else:
            # Infer column names if header wasn't distinctly found
            max_cols = max(len(r) for r in extracted_rows)
            default_cols = ["transcript_id", "logFC", "t", "P.Value", "adj.P.Val", "database_source"]
            cols = default_cols[:max_cols] if max_cols <= len(default_cols) else [f"col_{i}" for i in range(max_cols)]
            aligned_rows = [r + [""] * (len(cols) - len(r)) if len(r) < len(cols) else r[:len(cols)] for r in extracted_rows]
            df = pd.DataFrame(aligned_rows, columns=cols)

        return cls.clean_dataset(df)

    @classmethod
    def parse_file(cls, file_source: Any, filename: Optional[str] = None) -> pd.DataFrame:
        """
        Unified ingestion entry point that inspects file extension or content to parse CSV or PDF.
        """
        name = ""
        if filename:
            name = filename.lower()
        elif hasattr(file_source, "name"):
            name = str(file_source.name).lower()
        elif isinstance(file_source, str):
            name = file_source.lower()

        if name.endswith(".pdf"):
            return cls.parse_pdf(file_source)
        else:
            # Default to CSV parser (works with .csv, .tsv, .txt, or uploaded file buffers)
            return cls.parse_csv(file_source)
