"""
Automated End-to-End Benchmark & Stress-Testing Script for Bio Transcript Analyzer.

Performs:
- Synthetic dataset generation of arbitrary scale (e.g., 50k, 500k rows)
- Strict pipeline validation (Preprocessing -> Stratified Split -> Multiclass Training)
- Performance & Timing metrics output
- Quality gate: asserts Balanced Accuracy >= threshold
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Set up project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.preprocessing import TranscriptPreprocessor
from src.model_trainer import ModelTrainer


def generate_benchmark_data(n_rows: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic cleft therapy transcriptomic dataset."""
    rng = np.random.RandomState(random_state)
    
    # 2% Up, 2% Down, 96% Neutral to mirror biological transcriptomes
    n_up = int(n_rows * 0.02)
    n_down = int(n_rows * 0.02)
    n_neutral = n_rows - n_up - n_down
    
    up_logfc = rng.normal(loc=2.2, scale=0.5, size=n_up)
    up_pval = rng.uniform(1e-12, 1e-4, size=n_up)
    
    down_logfc = rng.normal(loc=-2.2, scale=0.5, size=n_down)
    down_pval = rng.uniform(1e-12, 1e-4, size=n_down)
    
    neutral_logfc = rng.normal(loc=0.0, scale=0.4, size=n_neutral)
    neutral_pval = rng.uniform(0.06, 0.99, size=n_neutral)
    
    logfc = np.concatenate([up_logfc, down_logfc, neutral_logfc])
    pval = np.concatenate([up_pval, down_pval, neutral_pval])
    t_stat = logfc * rng.uniform(2.0, 3.5, size=n_rows)
    adjp = np.clip(pval * 1.5, 1e-300, 1.0)
    
    dbs = ["RefSeq", "ENSEMBL", "lncRNAWiki", "Ace View", "miTranscriptome", "UCSC Genes"]
    db_sources = rng.choice(dbs, size=n_rows)
    ids = [f"BENCH_TX_{i:07d}" for i in range(n_rows)]
    
    # Shuffle
    perm = rng.permutation(n_rows)
    return pd.DataFrame({
        "transcript_id": np.array(ids)[perm],
        "logFC": logfc[perm],
        "t": t_stat[perm],
        "P.Value": pval[perm],
        "adj.P.Val": adjp[perm],
        "database_source": db_sources[perm]
    })


def run_benchmark(n_rows: int = 50000, min_bal_acc: float = 0.70) -> int:
    print(f"=== Running Bio Transcript Analyzer Benchmark ({n_rows:,} records) ===")
    t_start = time.time()
    
    print("1. Generating synthetic transcriptomics data...")
    t0 = time.time()
    df = generate_benchmark_data(n_rows=n_rows)
    print(f"   -> Done in {time.time() - t0:.2f}s")
    
    print("2. Running Preprocessing & Feature Engineering...")
    t0 = time.time()
    preprocessor = TranscriptPreprocessor(
        logfc_up_threshold=1.0,
        logfc_down_threshold=-1.0,
        adjp_threshold=0.05
    )
    X, y, df_proc = preprocessor.prepare_features(df, is_train=True)
    print(f"   -> Preprocessed {len(X):,} rows x {X.shape[1]} features in {time.time() - t0:.2f}s")
    
    print("3. Performing Stratified Downsampled Split...")
    t0 = time.time()
    trainer = ModelTrainer(random_state=42)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    print(f"   -> Train size: {len(X_train):,}, Test size: {len(X_test):,} in {time.time() - t0:.2f}s")
    
    print("4. Training and Evaluating Classifiers...")
    t0 = time.time()
    comp_df, results, best_name, best_model = trainer.train_and_evaluate_all(
        X_train, X_test, y_train, y_test, class_names=preprocessor.get_class_names()
    )
    train_time = time.time() - t0
    print(f"   -> Finished model evaluation in {train_time:.2f}s")
    
    print("\n=== Model Benchmark Leaderboard ===")
    print(comp_df.to_string(index=False))
    
    total_time = time.time() - t_start
    print(f"\nTotal Pipeline Runtime: {total_time:.2f}s")
    print(f"Best Performing Model: {best_name}")
    
    best_bal_acc = comp_df.iloc[0]["Balanced Accuracy"]
    if best_bal_acc < min_bal_acc:
        print(f"\nFAILED: Best Balanced Accuracy ({best_bal_acc:.4f}) is below minimum threshold ({min_bal_acc:.4f})!")
        return 1
    
    print(f"\nSUCCESS: Pipeline passed quality criteria with Balanced Accuracy {best_bal_acc:.4f} >= {min_bal_acc:.4f}.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full end-to-end benchmark on Bio Transcript Analyzer.")
    parser.add_argument("--size", type=int, default=20000, help="Number of synthetic transcripts to simulate.")
    parser.add_argument("--min_acc", type=float, default=0.70, help="Minimum acceptable balanced accuracy threshold.")
    args = parser.parse_args()
    
    sys.exit(run_benchmark(n_rows=args.size, min_bal_acc=args.min_acc))
