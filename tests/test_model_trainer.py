import pytest
import numpy as np
import pandas as pd
from src.model_trainer import ModelTrainer


def test_split_data_shape_and_alignment():
    trainer = ModelTrainer(random_state=42)
    n_samples = 200
    X = pd.DataFrame(np.random.randn(n_samples, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.choice([0, 1, 2], size=n_samples, p=[0.1, 0.8, 0.1]))

    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == n_samples
    assert set(y_train.unique()) == {0, 1, 2}
    assert set(y_test.unique()) == {0, 1, 2}


def test_model_training_and_balanced_evaluation():
    trainer = ModelTrainer(random_state=42)
    cols = ["logFC", "t", "neg_log10_pvalue", "abs_logFC", "db_RefSeq", "db_ENSEMBL", "db_Other"]
    n_samples = 300
    X_data = np.random.randn(n_samples, len(cols))
    y_data = np.ones(n_samples, dtype=int)

    # Class 0: Down (strong negative logFC)
    y_data[:15] = 0
    X_data[:15, 0] = -3.5
    X_data[:15, 1] = -4.0

    # Class 2: Up (strong positive logFC)
    y_data[15:30] = 2
    X_data[15:30, 0] = 3.5
    X_data[15:30, 1] = 4.0

    X = pd.DataFrame(X_data, columns=cols)
    y = pd.Series(y_data)

    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    comp_df, results, best_name, best_clf = trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test)

    assert len(comp_df) == 4
    expected_metrics = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for m in expected_metrics:
        assert m in comp_df.columns

    # Verify MLP did not collapse to 0.333
    mlp_row = comp_df[comp_df["Model"] == "MLP"].iloc[0]
    assert mlp_row["Balanced Accuracy"] > 0.80
