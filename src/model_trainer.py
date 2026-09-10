"""
Machine Learning Pipeline Module.

Implements:
- 80% Training and 20% Testing stratified split
- Training and evaluation of 4 ML classifiers:
    1. Random Forest (RF)
    2. XGBoost (XGB)
    3. Support Vector Machine (SVM)
    4. Multi-Layer Perceptron (MLP)
- Metrics calculation: Accuracy, Precision, Recall, F1-Score, ROC-AUC (OvR)
- Model comparison and selection of best-performing model
"""

import logging
import warnings
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Automated Machine Learning benchmark pipeline for transcript classification.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.class_names: List[str] = []

    def get_default_classifiers(self) -> Dict[str, Any]:
        """Initialize the 4 specified machine learning models with robust, high-performance defaults."""
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=4,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="mlogloss",
                n_jobs=-1
            ),
            "SVM": SVC(
                kernel="rbf",
                C=1.5,
                gamma="scale",
                probability=True,
                max_iter=2500,
                class_weight="balanced",
                random_state=self.random_state
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=0.001,
                learning_rate_init=0.01,
                max_iter=120,
                early_stopping=False,
                random_state=self.random_state
            )
        }

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        max_majority_train: int = 3000,
        max_majority_test: int = 1000
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Stratified biological split preserving 100% of all rare biomarker classes (Up and Down).
        Only downsamples the overwhelming majority class (Neutral) to ensure representative,
        sub-second training and proper multi-class evaluation.
        """
        class_counts = y.value_counts()
        majority_class = class_counts.idxmax()

        is_maj = (y == majority_class)
        X_min, y_min = X[~is_maj], y[~is_maj]
        X_maj, y_maj = X[is_maj], y[is_maj]

        # 1. Split minority classes preserving 100% of them
        if len(y_min) >= 2:
            strat_min = y_min if (y_min.value_counts().min() >= 2) else None
            X_min_tr, X_min_te, y_min_tr, y_min_te = train_test_split(
                X_min, y_min, test_size=test_size, random_state=self.random_state, stratify=strat_min
            )
        else:
            X_min_tr, X_min_te, y_min_tr, y_min_te = X_min, X_min.iloc[0:0], y_min, y_min.iloc[0:0]

        # 2. Split majority class (downsample if massive to keep training sub-second)
        total_maj_cap = max_majority_train + max_majority_test
        if len(X_maj) > total_maj_cap:
            logger.info(f"Subsampling majority neutral class from {len(X_maj):,} to {total_maj_cap:,} while keeping 100% of rare regulated transcripts.")
            sampled_maj = X_maj.sample(n=total_maj_cap, random_state=self.random_state)
            test_ratio = max_majority_test / total_maj_cap
            X_maj_tr, X_maj_te = train_test_split(
                sampled_maj, test_size=test_ratio, random_state=self.random_state
            )
            y_maj_tr = y.loc[X_maj_tr.index]
            y_maj_te = y.loc[X_maj_te.index]
        else:
            X_maj_tr, X_maj_te, y_maj_tr, y_maj_te = train_test_split(
                X_maj, y_maj, test_size=test_size, random_state=self.random_state
            )

        # 3. Combine synchronously to guarantee identical row counts and clean reset indices
        X_train_raw = pd.concat([X_min_tr, X_maj_tr], axis=0)
        y_train_raw = pd.concat([y_min_tr, y_maj_tr], axis=0)
        shuffle_tr = np.random.RandomState(self.random_state).permutation(len(X_train_raw))
        X_train = X_train_raw.iloc[shuffle_tr].reset_index(drop=True)
        y_train = y_train_raw.iloc[shuffle_tr].reset_index(drop=True)

        X_test_raw = pd.concat([X_min_te, X_maj_te], axis=0)
        y_test_raw = pd.concat([y_min_te, y_maj_te], axis=0)
        shuffle_te = np.random.RandomState(self.random_state).permutation(len(X_test_raw))
        X_test = X_test_raw.iloc[shuffle_te].reset_index(drop=True)
        y_test = y_test_raw.iloc[shuffle_te].reset_index(drop=True)

        return X_train, X_test, y_train, y_test

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        num_classes: int
    ) -> Dict[str, Any]:
        """
        Evaluate trained classifier across Accuracy, Balanced Accuracy, Macro/Weighted F1, Precision, Recall, and ROC-AUC.
        """
        from sklearn.metrics import balanced_accuracy_score

        y_pred = model.predict(X_test)

        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Multiclass ROC-AUC (One-vs-Rest, macro average)
        roc_auc = 0.0
        roc_curves_data: Dict[int, Dict[str, Any]] = {}

        if y_prob is not None and num_classes > 1:
            try:
                present_classes = np.unique(y_test)
                if len(present_classes) == y_prob.shape[1]:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
                else:
                    roc_auc = 0.5

                for c_idx in range(y_prob.shape[1]):
                    y_binary = (y_test == c_idx).astype(int)
                    if len(np.unique(y_binary)) > 1:
                        fpr, tpr, _ = roc_curve(y_binary, y_prob[:, c_idx])
                        c_auc = auc(fpr, tpr)
                        roc_curves_data[c_idx] = {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "auc": float(c_auc)
                        }
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                roc_auc = 0.5

        cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

        return {
            "Accuracy": float(acc),
            "Balanced Accuracy": float(bal_acc),
            "Precision": float(prec_macro),
            "Recall": float(rec_macro),
            "F1-Score": float(f1_macro),
            "Weighted F1": float(f1_weighted),
            "Weighted Precision": float(prec_weighted),
            "Weighted Recall": float(rec_weighted),
            "ROC-AUC": float(roc_auc),
            "confusion_matrix": cm,
            "roc_curves": roc_curves_data,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def train_and_evaluate_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        class_names: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]], str, Any]:
        """
        Train all 4 classifiers, compute metrics, and select best model based on Balanced Accuracy and Macro F1.
        """
        self.class_names = class_names or ["Down-Regulated", "Neutral", "Up-Regulated"]
        num_classes = len(self.class_names)
        classifiers = self.get_default_classifiers()

        self.models = {}
        self.results = {}
        comparison_rows = []

        # Compute balanced class sample weights to counteract severe class imbalance
        sample_weights = compute_sample_weight("balanced", y_train)
        best_score = -1.0

        for name, clf in classifiers.items():
            logger.info(f"Training {name} classifier...")
            if name in ("MLP", "XGBoost"):
                clf.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                clf.fit(X_train, y_train)
            self.models[name] = clf

            eval_res = self.evaluate_model(clf, X_test, y_test, num_classes)
            self.results[name] = eval_res

            comparison_rows.append({
                "Model": name,
                "Accuracy": eval_res["Accuracy"],
                "Balanced Accuracy": eval_res["Balanced Accuracy"],
                "Precision": eval_res["Precision"],
                "Recall": eval_res["Recall"],
                "F1-Score": eval_res["F1-Score"],
                "ROC-AUC": eval_res["ROC-AUC"]
            })

            # Selection criterion: Combined Macro F1-Score and Balanced Accuracy
            combined_score = eval_res["F1-Score"] * 0.6 + eval_res["Balanced Accuracy"] * 0.4
            if combined_score > best_score:
                best_score = combined_score
                self.best_model_name = name
                self.best_model = clf

        comparison_df = pd.DataFrame(comparison_rows).sort_values("Balanced Accuracy", ascending=False).reset_index(drop=True)
        logger.info(f"Best model determined: {self.best_model_name} with score {best_score:.4f}")

        return comparison_df, self.results, self.best_model_name, self.best_model
