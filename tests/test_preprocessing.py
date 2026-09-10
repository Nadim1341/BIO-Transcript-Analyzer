"""
Unit tests for feature engineering and rule-based labeling (src/preprocessing.py).
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    TranscriptPreprocessor,
    CLASS_UP,
    CLASS_DOWN,
    CLASS_NEUTRAL,
    CANONICAL_DATABASES
)


@pytest.fixture
def mock_dataset():
    """Mock dataset spanning edge cases for biological labeling."""
    return pd.DataFrame({
        "transcript_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "logFC": [
            1.5,    # Up: logFC >= 1.0, adj.P.Val < 0.05
            -1.8,   # Down: logFC <= -1.0, adj.P.Val < 0.05
            0.2,    # Neutral: low logFC, high pval
            2.5,    # Neutral: high logFC but high pval (not significant)
            -2.0,   # Neutral: low logFC but high pval
            1.0     # Up: exact boundary logFC == 1.0, adj.P.Val < 0.05
        ],
        "t": [4.5, -5.2, 0.4, 1.2, -1.5, 3.2],
        "P.Value": [1e-5, 2e-6, 0.45, 0.12, 0.25, 0.001],
        "adj.P.Val": [0.002, 0.001, 0.65, 0.15, 0.30, 0.049],
        "database_source": ["RefSeq", "ENSEMBL", "lncRNAWiki", "Ace View", "miTranscriptome", "UCSC Genes"]
    })


def test_biological_labeling_rules(mock_dataset):
    """Test 3-class biological rule categorization."""
    preprocessor = TranscriptPreprocessor(
        logfc_up_threshold=1.0,
        logfc_down_threshold=-1.0,
        adjp_threshold=0.05
    )
    labeled_df = preprocessor.assign_biological_labels(mock_dataset)

    labels = labeled_df.set_index("transcript_id")["target_label"].to_dict()

    assert labels["T1"] == CLASS_UP
    assert labels["T2"] == CLASS_DOWN
    assert labels["T3"] == CLASS_NEUTRAL
    assert labels["T4"] == CLASS_NEUTRAL  # High fold change but non-significant p-val
    assert labels["T5"] == CLASS_NEUTRAL  # Low fold change but non-significant p-val
    assert labels["T6"] == CLASS_UP       # Boundary condition logFC=1.0, adj.P.Val=0.049


def test_custom_thresholds(mock_dataset):
    """Test dynamic adjustments to thresholds."""
    # Stricter thresholds: logFC >= 2.0
    preprocessor = TranscriptPreprocessor(
        logfc_up_threshold=2.0,
        logfc_down_threshold=-2.0,
        adjp_threshold=0.01
    )
    labeled_df = preprocessor.assign_biological_labels(mock_dataset)
    labels = labeled_df.set_index("transcript_id")["target_label"].to_dict()

    # T1 had logFC=1.5 which is < 2.0, so under strict threshold it should be Neutral
    assert labels["T1"] == CLASS_NEUTRAL
    assert labels["T2"] == CLASS_NEUTRAL  # logFC=-1.8 which is > -2.0


def test_feature_engineering(mock_dataset):
    """Test signal amplification feature calculations."""
    preprocessor = TranscriptPreprocessor()
    fe_df = preprocessor.compute_engineered_features(mock_dataset)

    assert "neg_log10_pvalue" in fe_df.columns
    assert "neg_log10_adjpval" in fe_df.columns
    assert "abs_logFC" in fe_df.columns

    # -log10(1e-5) == 5.0
    assert np.isclose(fe_df.loc[0, "neg_log10_pvalue"], 5.0, atol=1e-3)
    assert np.isclose(fe_df.loc[0, "abs_logFC"], 1.5, atol=1e-3)


def test_one_hot_encoding(mock_dataset):
    """Test database source one-hot encoding includes all canonical databases."""
    preprocessor = TranscriptPreprocessor()
    encoded_df = preprocessor.encode_database_sources(mock_dataset)

    for db in CANONICAL_DATABASES:
        col = f"db_{db.replace(' ', '_')}"
        assert col in encoded_df.columns
        assert encoded_df[col].isin([0.0, 1.0]).all()

    assert "db_Other" in encoded_df.columns


def test_standard_scaling_pipeline(mock_dataset):
    """Test end-to-end prepare_features produces standardized features."""
    preprocessor = TranscriptPreprocessor()
    X, y, df_proc = preprocessor.prepare_features(mock_dataset, is_train=True)

    assert len(X) == len(mock_dataset)
    assert len(y) == len(mock_dataset)
    assert not X.isnull().values.any()

    # Numerical columns should be standardized (mean ~ 0)
    for col in ["logFC", "t", "neg_log10_pvalue"]:
        mean_val = X[col].mean()
        assert np.isclose(mean_val, 0.0, atol=1e-7)
