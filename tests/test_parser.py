"""
Unit tests for data parsing and ingestion module (src/data_parser.py).
"""

import io
import pytest
import pandas as pd
import numpy as np
from src.data_parser import DataParser, REQUIRED_COLUMNS


@pytest.fixture
def sample_csv_data():
    """Sample CSV string with canonical column names."""
    return io.StringIO(
        "transcript_id,logFC,t,P.Value,adj.P.Val,database_source\n"
        "NM_006147.4,2.85,9.42,1.2e-15,2.4e-13,RefSeq\n"
        "ENST00000305886,-2.14,-7.12,3.4e-11,4.1e-9,ENSEMBL\n"
        "NONCODEG00291,0.12,0.45,0.65,0.78,lncRNAWiki\n"
        "AV_TGFB3_01,-1.95,-6.2,8.9e-10,8.2e-8,Ace View\n"
        "MINT004210,-2.31,-7.68,2.2e-12,3.2e-10,miTranscriptome\n"
        "uc001abc.1,1.54,5.15,7.8e-7,3.6e-5,UCSC Genes\n"
    )


@pytest.fixture
def aliased_csv_data():
    """Sample CSV with alternative column header names."""
    return io.StringIO(
        "probe_id,log2FC,t_stat,p_val,FDR\n"
        "NM_001202.6,1.62,5.34,4.2e-7,2.1e-5\n"
        "ENST00000375929,-2.05,-6.78,9.8e-10,8.8e-8\n"
        "LNC_CLEFT0012,2.45,8.11,4.8e-13,7.1e-11\n"
    )


def test_parse_csv_canonical(sample_csv_data):
    """Test parsing standard CSV data with canonical headers."""
    df = DataParser.parse_csv(sample_csv_data)
    assert not df.empty
    assert len(df) == 6
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert df["logFC"].dtype in [np.float64, float]
    assert df["adj.P.Val"].dtype in [np.float64, float]


def test_parse_csv_aliased_and_inferred_db(aliased_csv_data):
    """Test column renaming and database inference for non-standard CSV headers."""
    df = DataParser.parse_csv(aliased_csv_data)
    assert not df.empty
    assert len(df) == 3
    assert "transcript_id" in df.columns
    assert "logFC" in df.columns
    assert "P.Value" in df.columns
    assert "adj.P.Val" in df.columns
    assert "database_source" in df.columns

    # Test automatic database inference
    sources = df.set_index("transcript_id")["database_source"].to_dict()
    assert sources["NM_001202.6"] == "RefSeq"
    assert sources["ENST00000375929"] == "ENSEMBL"
    assert sources["LNC_CLEFT0012"] == "lncRNAWiki"


def test_infer_database_source():
    """Verify database identification regex patterns."""
    assert DataParser.infer_database_source("NM_003239.5") == "RefSeq"
    assert DataParser.infer_database_source("NR_123456") == "RefSeq"
    assert DataParser.infer_database_source("ENST00000254958") == "ENSEMBL"
    assert DataParser.infer_database_source("LNC_IRF6AS_01") == "lncRNAWiki"
    assert DataParser.infer_database_source("NONCODE001") == "lncRNAWiki"
    assert DataParser.infer_database_source("AV_BMP4_03") == "Ace View"
    assert DataParser.infer_database_source("MINT008912") == "miTranscriptome"
    assert DataParser.infer_database_source("uc002def.2") == "UCSC Genes"


def test_cleaning_nulls_and_bounds():
    """Test data cleaning removes null rows and clips invalid probabilities."""
    dirty_data = io.StringIO(
        "transcript_id,logFC,t,P.Value,adj.P.Val,database_source\n"
        "NM_VALID,1.5,4.0,0.01,0.02,RefSeq\n"
        ",2.0,5.0,0.001,0.01,RefSeq\n"  # Missing transcript ID
        "NM_NULL_FC,,4.0,0.01,0.02,RefSeq\n"  # Missing logFC
        "NM_BOUNDS,0.5,1.2,0.0,-0.05,RefSeq\n"  # Non-positive p-values
    )
    df = DataParser.parse_csv(dirty_data)
    assert len(df) == 2
    assert "NM_VALID" in df["transcript_id"].values
    assert "NM_BOUNDS" in df["transcript_id"].values
    
    # Check probability clipping
    bounds_row = df[df["transcript_id"] == "NM_BOUNDS"].iloc[0]
    assert bounds_row["P.Value"] > 0
    assert bounds_row["adj.P.Val"] > 0


def test_empty_csv_raises():
    """Empty CSV should raise ValueError."""
    with pytest.raises(ValueError):
        DataParser.parse_csv(io.StringIO(""))
