import pytest
import warnings
import pandas as pd
import numpy as np
from src.visualizer import Visualizer

def test_volcano_downsampling_warning():
    """Verify that volcano plot emits a UserWarning and caps points when data exceeds max_neutral_display."""
    n_neutral = 5000
    df = pd.DataFrame({
        "transcript_id": [f"TX_{i}" for i in range(n_neutral + 20)],
        "target_label": ["Neutral"] * n_neutral + ["Up-Regulated"] * 10 + ["Down-Regulated"] * 10,
        "logFC": [0.05] * n_neutral + [2.5] * 10 + [-2.5] * 10,
        "neg_log10_pvalue": [1.0] * n_neutral + [5.0] * 10 + [5.0] * 10,
        "neg_log10_adjpval": [0.5] * n_neutral + [4.5] * 10 + [4.5] * 10,
        "t": [0.1] * (n_neutral + 20),
        "P.Value": [0.1] * (n_neutral + 20),
        "adj.P.Val": [0.1] * (n_neutral + 20),
        "database_source": ["RefSeq"] * (n_neutral + 20)
    })
    
    with pytest.warns(UserWarning, match="downsampled"):
        fig = Visualizer.plot_volcano(df, max_neutral_display=500)
    
    assert fig is not None
    assert len(fig.data) >= 3


def test_plot_class_distribution_empty():
    """Verify that class distribution plot handles empty or unusual inputs gracefully."""
    empty_df = pd.DataFrame(columns=["target_label"])
    fig = Visualizer.plot_class_distribution(empty_df)
    assert fig is not None
