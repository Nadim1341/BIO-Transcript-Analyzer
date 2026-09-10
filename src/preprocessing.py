"""
Feature Engineering and Biological Rule-Based Labeling Module.

Implements:
- 3-class biological regulation labeling:
    * Up-Regulated: logFC >= 1.0 AND adj.P.Val < 0.05
    * Down-Regulated: logFC <= -1.0 AND adj.P.Val < 0.05
    * Neutral: Otherwise
- Signal amplification via -log10(P.Value) and -log10(adj.P.Val)
- One-hot encoding of annotation database sources
- Feature standardization using scikit-learn's StandardScaler
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Recognized annotation database categories
CANONICAL_DATABASES: List[str] = [
    "RefSeq",
    "ENSEMBL",
    "lncRNAWiki",
    "Ace View",
    "miTranscriptome",
    "UCSC Genes",
]

# Label string constants
CLASS_UP = "Up-Regulated"
CLASS_DOWN = "Down-Regulated"
CLASS_NEUTRAL = "Neutral"
LABEL_ORDER = [CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP]


class TranscriptPreprocessor:
    """
    Feature engineering, rule-based labeling, and standardization pipeline.
    """

    def __init__(
        self,
        logfc_up_threshold: float = 1.0,
        logfc_down_threshold: float = -1.0,
        adjp_threshold: float = 0.05,
    ):
        """
        Initialize preprocessor with dynamic biological significance thresholds.

        :param logfc_up_threshold: Minimal logFC for Up-Regulation (default: 1.0)
        :param logfc_down_threshold: Maximal logFC for Down-Regulation (default: -1.0)
        :param adjp_threshold: Significance threshold for FDR / adj.P.Val (default: 0.05)
        """
        self.logfc_up_threshold = float(logfc_up_threshold)
        self.logfc_down_threshold = float(logfc_down_threshold)
        self.adjp_threshold = float(adjp_threshold)

        self.scaler: Optional[StandardScaler] = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(LABEL_ORDER)
        self.feature_names: List[str] = []
        self.numerical_cols: List[str] = []
        self.db_cols: List[str] = []

    def assign_biological_labels(
        self,
        df: pd.DataFrame,
        logfc_up: Optional[float] = None,
        logfc_down: Optional[float] = None,
        adjp_thresh: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Categorize biological transcripts into 3 target classes:
        - Up-Regulated: logFC >= logfc_up AND adj.P.Val < adjp_thresh
        - Down-Regulated: logFC <= logfc_down AND adj.P.Val < adjp_thresh
        - Neutral: Otherwise
        """
        up_val = self.logfc_up_threshold if logfc_up is None else float(logfc_up)
        down_val = self.logfc_down_threshold if logfc_down is None else float(logfc_down)
        adjp_val = self.adjp_threshold if adjp_thresh is None else float(adjp_thresh)

        df = df.copy()

        # Initialize as Neutral
        df["target_label"] = CLASS_NEUTRAL

        # Apply biological rules
        up_mask = (df["logFC"] >= up_val) & (df["adj.P.Val"] < adjp_val)
        down_mask = (df["logFC"] <= down_val) & (df["adj.P.Val"] < adjp_val)

        df.loc[up_mask, "target_label"] = CLASS_UP
        df.loc[down_mask, "target_label"] = CLASS_DOWN

        # Encode integer target class: 0: Down-Regulated, 1: Neutral, 2: Up-Regulated
        df["target_class"] = self.label_encoder.transform(df["target_label"])

        return df

    def compute_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute signal amplification features:
        - neg_log10_pvalue = -log10(P.Value)
        - neg_log10_adjpval = -log10(adj.P.Val)
        - abs_logFC = |logFC|
        """
        df = df.copy()
        eps = 1e-300  # Avoid log10(0) division errors

        # Signal amplification
        df["neg_log10_pvalue"] = -np.log10(df["P.Value"].clip(lower=eps))
        df["neg_log10_adjpval"] = -np.log10(df["adj.P.Val"].clip(lower=eps))
        df["abs_logFC"] = np.abs(df["logFC"])

        return df

    def encode_database_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-Hot Encode annotation database sources.
        Ensures canonical database categories are consistently represented.
        """
        df = df.copy()
        db_series = df["database_source"].fillna("Other").astype(str)

        for db in CANONICAL_DATABASES:
            col_name = f"db_{db.replace(' ', '_')}"
            df[col_name] = (db_series.str.lower() == db.lower()).astype(float)

        # Catch-all for non-canonical databases
        other_mask = ~db_series.str.lower().isin([d.lower() for d in CANONICAL_DATABASES])
        df["db_Other"] = other_mask.astype(float)

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        is_train: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Full feature engineering pipeline:
        1. Rule-based labeling
        2. Signal amplification (-log10 P.Value)
        3. One-hot encoding of databases
        4. Standard scaling of continuous features

        :param df: Cleaned input DataFrame
        :param is_train: If True, fits StandardScaler; if False, uses existing scaler
        :return: (X_scaled_df, y_series, processed_full_df)
        """
        # Step 1: Rule-based labeling
        df_proc = self.assign_biological_labels(df)

        # Step 2: Signal amplification
        df_proc = self.compute_engineered_features(df_proc)

        # Step 3: One-hot encode databases
        df_proc = self.encode_database_sources(df_proc)

        # Define feature groups (exclude neg_log10_adjpval to avoid direct target label leakage)
        self.numerical_cols = ["logFC", "t", "neg_log10_pvalue", "abs_logFC"]
        self.db_cols = [f"db_{db.replace(' ', '_')}" for db in CANONICAL_DATABASES] + ["db_Other"]
        self.feature_names = self.numerical_cols + self.db_cols

        # Step 4: Scale numerical features
        if is_train or self.scaler is None:
            self.scaler = StandardScaler()
            scaled_num = self.scaler.fit_transform(df_proc[self.numerical_cols])
        else:
            scaled_num = self.scaler.transform(df_proc[self.numerical_cols])

        # Assertions after scaling
        if scaled_num.shape[0] != len(df_proc):
            raise ValueError("Scaling resulted in row count mismatch.")
        if np.isnan(scaled_num).any():
            raise ValueError("Scaling produced NaN values.")

        # Combine scaled numerical features with binary one-hot database features
        scaled_num_df = pd.DataFrame(scaled_num, columns=self.numerical_cols, index=df_proc.index)
        db_df = df_proc[self.db_cols].copy()

        X = pd.concat([scaled_num_df, db_df], axis=1)
        y = df_proc["target_class"]
        # Guard against NaNs in the final feature matrix
        if X.isnull().any().any():
            raise ValueError("NaN values detected in feature matrix after scaling.")
        # Ensure target_label contains exactly the three expected classes
        expected_labels = {CLASS_UP, CLASS_DOWN, CLASS_NEUTRAL}
        actual_labels = set(df_proc["target_label"].unique())
        missing = expected_labels - actual_labels
        if missing:
            raise ValueError(f"Missing expected target labels after labeling: {missing}")
        return X, y, df_proc
    def get_class_names(self) -> List[str]:
        """Return human-readable class names in numerical order [0, 1, 2]."""
        return list(self.label_encoder.classes_)
