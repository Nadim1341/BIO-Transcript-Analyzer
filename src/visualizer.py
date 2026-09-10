"""
Interactive Visualization Engine using Plotly for Bio Transcript Analyzer.

Implements:
- Publication-quality Interactive Volcano Plot (Red=Up, Blue=Down, Grey=Neutral)
- Multi-class Confusion Matrix Heatmap
- Multi-class One-vs-Rest ROC Curves
- Model Benchmark Performance Comparison Chart
- SHAP Feature Importance Attribution Bar Plot
- Class Distribution Donut Chart
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Theme Colors tailored for modern bioinformatics dashboard
COLOR_UP = "#EF4444"       # Vibrant Red
COLOR_DOWN = "#3B82F6"     # Bright Blue
COLOR_NEUTRAL = "#94A3B8"  # Slate Grey
COLOR_BG = "rgba(0,0,0,0)"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"


class Visualizer:
    """Interactive Plotly visual generator for bioinformatics and ML outputs."""

    @staticmethod
    def plot_volcano(
        df: pd.DataFrame,
        logfc_up: float = 1.0,
        logfc_down: float = -1.0,
        adjp_thresh: float = 0.05,
        y_axis_col: str = "neg_log10_pvalue",
        title: str = "Differential Expression Volcano Plot"
    ) -> go.Figure:
        """
        Interactive Volcano Plot.
        X-axis: logFC
        Y-axis: -log10(P.Value) or -log10(adj.P.Val)
        Color-coded: Red=Up, Blue=Down, Grey=Neutral
        """
        plot_df = df
        if "neg_log10_pvalue" not in plot_df.columns or "neg_log10_adjpval" not in plot_df.columns:
            plot_df = plot_df.copy()
            if "neg_log10_pvalue" not in plot_df.columns:
                plot_df["neg_log10_pvalue"] = -np.log10(plot_df["P.Value"].clip(lower=1e-300))
            if "neg_log10_adjpval" not in plot_df.columns:
                plot_df["neg_log10_adjpval"] = -np.log10(plot_df["adj.P.Val"].clip(lower=1e-300))
        y_col = y_axis_col if y_axis_col in plot_df.columns else "neg_log10_pvalue"
        y_label = "-log10(P.Value)" if y_col == "neg_log10_pvalue" else "-log10(adj.P.Val)"

        fig = go.Figure()

        # Define category display specs
        categories = [
            ("Neutral", COLOR_NEUTRAL, 4, 0.4),
            ("Down-Regulated", COLOR_DOWN, 7, 0.9),
            ("Up-Regulated", COLOR_UP, 7, 0.9),
        ]

        # For large datasets, cap neutral background cloud at 3,000 points (preserves 100% of shape while dropping JSON size 95%)
        max_neutral_display = 3000
        labels_arr = plot_df["target_label"].values

        for cat_name, color, size, opacity in categories:
            match_indices = np.flatnonzero(labels_arr == cat_name)
            total_cat_count = len(match_indices)
            if total_cat_count == 0:
                continue

            # Keep 100% of Up and Down regulated genes; only sample the neutral cloud if needed
            if cat_name == "Neutral" and total_cat_count > max_neutral_display:
                sampled_indices = np.random.RandomState(42).choice(match_indices, size=max_neutral_display, replace=False)
                subset = plot_df.iloc[sampled_indices]
                trace_name = f"Neutral ({total_cat_count:,} total, {max_neutral_display:,} cloud)"
            else:
                subset = plot_df.iloc[match_indices]
                trace_name = f"{cat_name} ({total_cat_count:,})"

            # Vectorized hover metadata
            custom_cols = ["transcript_id", "database_source", "t", "P.Value", "adj.P.Val"]
            custom_matrix = subset[custom_cols].fillna(0.0).values

            hovertemplate = (
                "<b>ID:</b> %{customdata[0]}<br>"
                "<b>Database:</b> %{customdata[1]}<br>"
                f"<b>Regulation:</b> {cat_name}<br>"
                "<b>logFC:</b> %{x:.3f}<br>"
                "<b>t-statistic:</b> %{customdata[2]:.3f}<br>"
                "<b>P.Value:</b> %{customdata[3]:.2e}<br>"
                "<b>adj.P.Val:</b> %{customdata[4]:.2e}<extra></extra>"
            )

            fig.add_trace(
                go.Scattergl(
                    x=subset["logFC"],
                    y=subset[y_col],
                    mode="markers",
                    name=trace_name,
                    marker=dict(
                        color=color,
                        size=size,
                        opacity=opacity,
                    ),
                    customdata=custom_matrix,
                    hovertemplate=hovertemplate
                )
            )

        # Threshold lines
        p_threshold_y = -np.log10(adjp_thresh)

        # Vertical line for Down cutoff
        fig.add_vline(
            x=logfc_down,
            line=dict(color="#64748b", width=1.5, dash="dash"),
            annotation_text=f"logFC {logfc_down}",
            annotation_position="bottom left"
        )
        # Vertical line for Up cutoff
        fig.add_vline(
            x=logfc_up,
            line=dict(color="#64748b", width=1.5, dash="dash"),
            annotation_text=f"logFC {logfc_up}",
            annotation_position="bottom right"
        )
        # Horizontal line for p-value cutoff
        fig.add_hline(
            y=p_threshold_y,
            line=dict(color="#64748b", width=1.5, dash="dash"),
            annotation_text=f"p={adjp_thresh}",
            annotation_position="top left"
        )

        fig.update_layout(
            title=dict(text=title, font=dict(family=FONT_FAMILY, size=18, color="#0f172a")),
            xaxis=dict(
                title="log2 Fold Change (logFC)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="#cbd5e1",
                gridcolor="#f1f5f9"
            ),
            yaxis=dict(
                title=y_label,
                gridcolor="#f1f5f9"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family=FONT_FAMILY, size=12)
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor=COLOR_BG,
            hovermode="closest",
            margin=dict(l=60, r=40, t=80, b=60),
            height=580
        )
        return fig

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        title: str = "Confusion Matrix"
    ) -> go.Figure:
        """
        Interactive heatmap representation of the multi-class confusion matrix.
        """
        if normalize:
            cm_sum = cm.sum(axis=1, keepdims=True)
            cm_display = np.divide(cm.astype("float"), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
            text_values = [[f"{val:.2%}<br>({cm[i, j]})" for j, val in enumerate(row)] for i, row in enumerate(cm_display)]
            z_val = cm_display
        else:
            text_values = [[str(val) for val in row] for row in cm]
            z_val = cm

        fig = go.Figure(
            data=go.Heatmap(
                z=z_val,
                x=class_names,
                y=class_names,
                text=text_values,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Proportion" if normalize else "Count")
            )
        )

        fig.update_layout(
            title=dict(text=title, font=dict(family=FONT_FAMILY, size=16)),
            xaxis=dict(title="Predicted Label", tickangle=-20),
            yaxis=dict(title="True Label", autorange="reversed"),
            plot_bgcolor=COLOR_BG,
            paper_bgcolor=COLOR_BG,
            margin=dict(l=70, r=50, t=70, b=70),
            height=450
        )
        return fig

    @staticmethod
    def plot_roc_curves(
        roc_curves_data: Dict[int, Dict[str, Any]],
        class_names: List[str],
        title: str = "Multiclass One-vs-Rest ROC Curves"
    ) -> go.Figure:
        """
        Plotly One-vs-Rest ROC curve comparison.
        """
        fig = go.Figure()

        # Random chance baseline
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (AUC = 0.500)",
                line=dict(color="#94a3b8", dash="dash", width=1.5)
            )
        )

        palette = ["#3B82F6", "#10B981", "#EF4444", "#8B5CF6", "#F59E0B"]

        for idx, (c_idx, data) in enumerate(roc_curves_data.items()):
            c_name = class_names[c_idx] if c_idx < len(class_names) else f"Class {c_idx}"
            auc_val = data.get("auc", 0.0)
            color = palette[idx % len(palette)]

            fig.add_trace(
                go.Scatter(
                    x=data["fpr"],
                    y=data["tpr"],
                    mode="lines",
                    name=f"{c_name} (AUC = {auc_val:.3f})",
                    line=dict(color=color, width=2.5)
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(family=FONT_FAMILY, size=16)),
            xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02], gridcolor="#f1f5f9"),
            yaxis=dict(title="True Positive Rate (Sensitivity)", range=[-0.02, 1.02], gridcolor="#f1f5f9"),
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=0.05,
                xanchor="right",
                x=0.98,
                bgcolor="rgba(255,255,255,0.85)"
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor=COLOR_BG,
            margin=dict(l=60, r=40, t=60, b=60),
            height=460
        )
        return fig

    @staticmethod
    def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
        """
        Grouped bar chart for benchmark comparison across the 4 ML classifiers.
        """
        metrics = ["Balanced Accuracy", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        melted_df = comparison_df.melt(
            id_vars=["Model"],
            value_vars=[m for m in metrics if m in comparison_df.columns],
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            melted_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            text_auto=".3f",
            color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
        )

        fig.update_layout(
            title=dict(text="Model Benchmark Performance Comparison", font=dict(family=FONT_FAMILY, size=16)),
            yaxis=dict(range=[0, 1.15], title="Score", gridcolor="#f1f5f9"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="#ffffff",
            paper_bgcolor=COLOR_BG,
            margin=dict(l=50, r=40, t=70, b=50),
            height=420
        )
        return fig

    @staticmethod
    def plot_shap_summary(feature_importance_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
        """
        Horizontal bar plot displaying top mean absolute SHAP values.
        """
        top_df = feature_importance_df.head(top_n).sort_values("Mean_Abs_SHAP", ascending=True)

        fig = go.Figure(
            go.Bar(
                x=top_df["Mean_Abs_SHAP"],
                y=top_df["Feature"],
                orientation="h",
                marker=dict(
                    color=top_df["Mean_Abs_SHAP"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="|SHAP|")
                ),
                text=[f"{v:.4f}" for v in top_df["Mean_Abs_SHAP"]],
                textposition="outside"
            )
        )

        fig.update_layout(
            title=dict(text=f"Top {top_n} Features by Mean |SHAP| Impact", font=dict(family=FONT_FAMILY, size=16)),
            xaxis=dict(title="Mean Absolute SHAP Value", gridcolor="#f1f5f9"),
            yaxis=dict(title="Feature"),
            plot_bgcolor="#ffffff",
            paper_bgcolor=COLOR_BG,
            margin=dict(l=140, r=60, t=60, b=50),
            height=450
        )
        return fig

    @staticmethod
    def plot_class_distribution(df_or_counts: Any) -> go.Figure:
        """
        Donut chart showing class distribution among Up, Down, and Neutral transcripts.
        Supports both raw DataFrames or precomputed count dictionaries for instant rendering.
        """
        if isinstance(df_or_counts, dict):
            counts = pd.Series(df_or_counts)
        elif hasattr(df_or_counts, "value_counts"):
            counts = df_or_counts.value_counts()
        else:
            counts = df_or_counts["target_label"].value_counts()

        colors = {
            "Up-Regulated": COLOR_UP,
            "Down-Regulated": COLOR_DOWN,
            "Neutral": COLOR_NEUTRAL
        }
        color_seq = [colors.get(c, "#64748b") for c in counts.index]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(counts.index),
                    values=list(counts.values),
                    hole=0.55,
                    marker=dict(colors=color_seq, line=dict(color="#ffffff", width=2)),
                    textinfo="label+percent",
                    hoverinfo="label+value+percent"
                )
            ]
        )

        fig.update_layout(
            title=dict(text="Biological Regulation Distribution", font=dict(family=FONT_FAMILY, size=16)),
            showlegend=False,
            paper_bgcolor=COLOR_BG,
            plot_bgcolor=COLOR_BG,
            margin=dict(l=30, r=30, t=60, b=30),
            height=320
        )
        return fig
