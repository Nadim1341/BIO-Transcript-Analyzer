import pytest
import pandas as pd
import numpy as np
from src.preprocessing import TranscriptPreprocessor

def test_prepare_features_nan_detection():
    """Verify that if scaling results in NaN or if raw features have unhandled NaNs, ValueError is raised."""
    prep = TranscriptPreprocessor()
    df = pd.DataFrame({
        "transcript_id": ["TX_1", "TX_2", "TX_3"],
        "logFC": [1.5, -1.5, np.nan],
        "t": [2.0, -2.0, 0.0],
        "P.Value": [0.01, 0.01, 0.5],
        "adj.P.Val": [0.02, 0.02, 0.6],
        "database_source": ["RefSeq", "ENSEMBL", "lncRNAWiki"]
    })
    
    with pytest.raises(Exception):
        prep.prepare_features(df, is_train=True)


def test_prepare_features_missing_labels():
    """Verify that if a dataset doesn't contain all 3 expected classes, an error is raised."""
    prep = TranscriptPreprocessor()
    # All rows neutral
    df = pd.DataFrame({
        "transcript_id": [f"TX_{i}" for i in range(10)],
        "logFC": [0.1] * 10,
        "t": [0.1] * 10,
        "P.Value": [0.8] * 10,
        "adj.P.Val": [0.9] * 10,
        "database_source": ["RefSeq"] * 10
    })
    
    with pytest.raises(ValueError, match="Missing expected target labels"):
        prep.prepare_features(df, is_train=True)
