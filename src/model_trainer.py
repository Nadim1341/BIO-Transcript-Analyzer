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
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.001,
                max_iter=200,
                early_stopping=True,
                n_iter_no_change=6,
                random_state=self.random_state
            )
        }

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        max_train_samples: int = 4000,
        max_test_samples: int = 1500
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into 80% Training and 20% Testing sets with stratification.
        Automatically sub-samples massive datasets to keep benchmark training fast and responsive.
        """
        # Fall back to unstratified split if any class has fewer than 2 instances
        class_counts = y.value_counts()
        stratify_param = y if (class_counts.min() >= 2) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        # Scale down training set if dataset is massive to maintain sub-second responsiveness
        if len(X_train) > max_train_samples:
            logger.info(f"Dataset is large ({len(X_train)} train rows). Subsampling to {max_train_samples} for responsive training.")
            strat_train = y_train if (y_train.value_counts().min() >= 2) else None
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train,
                train_size=max_train_samples,
                random_state=self.random_state,
                stratify=strat_train
            )

        if len(X_test) > max_test_samples:
            strat_test = y_test if (y_test.value_counts().min() >= 2) else None
            X_test, _, y_test, _ = train_test_split(
                X_test, y_test,
                train_size=max_test_samples,
                random_state=self.random_state,
                stratify=strat_test
            )

        return X_train, X_test, y_train, y_test

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        num_classes: int
    ) -> Dict[str, Any]:
        """
        Evaluate trained classifier across Accuracy, Precision, Recall, F1, and ROC-AUC.
        """
        y_pred = model.predict(X_test)

        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Multiclass ROC-AUC (One-vs-Rest)
        roc_auc = 0.0
        roc_curves_data: Dict[int, Dict[str, Any]] = {}

        if y_prob is not None and num_classes > 1:
            try:
                # If only a subset of classes is present in y_test, handle carefully
                present_classes = np.unique(y_test)
                if len(present_classes) == y_prob.shape[1]:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                else:
                    roc_auc = 0.5

                # Compute ROC curves for individual classes
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
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
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
        Train all 4 classifiers, compute metrics, and select best model based on F1-Score.
        """
        self.class_names = class_names or ["Down-Regulated", "Neutral", "Up-Regulated"]
        num_classes = len(self.class_names)
        classifiers = self.get_default_classifiers()

        self.models = {}
        self.results = {}
        comparison_rows = []

        best_score = -1.0

        for name, clf in classifiers.items():
            logger.info(f"Training {name} classifier...")
            clf.fit(X_train, y_train)
            self.models[name] = clf

            eval_res = self.evaluate_model(clf, X_test, y_test, num_classes)
            self.results[name] = eval_res

            comparison_rows.append({
                "Model": name,
                "Accuracy": eval_res["Accuracy"],
                "Precision": eval_res["Precision"],
                "Recall": eval_res["Recall"],
                "F1-Score": eval_res["F1-Score"],
                "ROC-AUC": eval_res["ROC-AUC"]
            })

            # Selection criterion: Combined F1-Score and Accuracy
            combined_score = eval_res["F1-Score"] * 0.7 + eval_res["Accuracy"] * 0.3
            if combined_score > best_score:
                best_score = combined_score
                self.best_model_name = name
                self.best_model = clf

        comparison_df = pd.DataFrame(comparison_rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
        logger.info(f"Best model determined: {self.best_model_name} with score {best_score:.4f}")

        return comparison_df, self.results, self.best_model_name, self.best_model
