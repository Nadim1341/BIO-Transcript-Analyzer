"""
Bio Transcript Analyzer - High-Performance Interactive Streamlit Web Interface.

Provides:
- Blazing-fast responsive sidebar controls for dynamic logFC and adj.P.Val adjustments
- Hardware-accelerated (WebGL Scattergl) Volcano Plot rendering 50,000+ points smoothly
- Asynchronous and cached ML Benchmark pipeline with stratified sub-sampling
- Fast SHAP (SHapley Additive exPlanations) Explainability and Top Biomarker Rankings
- Universal File Uploader for custom CSV/PDF gene expression tables with memory caching
"""

import os
import sys
import io
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Setup system path to ensure clean modular imports from src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_parser import DataParser
from src.preprocessing import TranscriptPreprocessor
from src.model_trainer import ModelTrainer
from src.xai_engine import XAIEngine
from src.visualizer import Visualizer

# Configure page layout and branding
st.set_page_config(
    page_title="Bio Transcript Analyzer | Cleft Therapy Transcriptomics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern scientific aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    """Load default cleft therapy transcriptomic dataset from data/ directory."""
    pdf_path = project_root / "data" / "datasets.pdf"
    csv_path = project_root / "data" / "datasets.csv"

    if pdf_path.exists():
        try:
            return DataParser.parse_pdf(str(pdf_path))
        except Exception as e:
            logger.warning(f"Failed to read default PDF ({e}), attempting CSV fallback.")
    
    if csv_path.exists():
        return DataParser.parse_csv(str(csv_path))
    
    raise FileNotFoundError("Default dataset not found in data/ folder. Please upload a dataset.")


@st.cache_data(show_spinner="Parsing uploaded gene expression dataset...")
def parse_uploaded_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Cache parsed uploaded file content by its byte digest to avoid re-parsing on UI events."""
    return DataParser.parse_file(io.BytesIO(file_bytes), filename)


@st.cache_data(show_spinner=False)
def compute_biological_labels(
    df: pd.DataFrame,
    logfc_up: float,
    logfc_down: float,
    adjp_val: float
) -> pd.DataFrame:
    """Fast, vectorized rule-based labeling and signal amplification (< 10ms)."""
    preprocessor = TranscriptPreprocessor(
        logfc_up_threshold=logfc_up,
        logfc_down_threshold=logfc_down,
        adjp_threshold=adjp_val
    )
    df_labeled = preprocessor.assign_biological_labels(df)
    return preprocessor.compute_engineered_features(df_labeled)


@st.cache_data(show_spinner="Training machine learning classifiers (RF, XGBoost, SVM, MLP)...")
def run_ml_pipeline(
    df: pd.DataFrame,
    logfc_up: float,
    logfc_down: float,
    adjp_thresh: float
):
    """
    Train and benchmark the 4 ML models. Optimized with stratified subsampling
    to finish in ~1 second even for datasets with 50,000+ transcripts.
    """
    preprocessor = TranscriptPreprocessor(
        logfc_up_threshold=logfc_up,
        logfc_down_threshold=logfc_down,
        adjp_threshold=adjp_thresh
    )
    X, y, df_processed = preprocessor.prepare_features(df, is_train=True)
    class_names = preprocessor.get_class_names()

    trainer = ModelTrainer(random_state=42)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)

    comp_df, results, best_name, best_model = trainer.train_and_evaluate_all(
        X_train, X_test, y_train, y_test, class_names=class_names
    )

    return {
        "preprocessor": preprocessor,
        "trainer": trainer,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "df_processed": df_processed,
        "class_names": class_names,
        "comparison_df": comp_df,
        "results": results,
        "best_name": best_name,
        "best_model": best_model,
        "feature_names": preprocessor.feature_names
    }


@st.cache_data(show_spinner="Computing SHAP Explainability attributions...")
def run_xai_pipeline(_best_model, best_name: str, feature_names: list, _X_test: pd.DataFrame, _df_processed: pd.DataFrame):
    """Compute representative SHAP values and biomarker rankings in milliseconds."""
    xai = XAIEngine(_best_model, best_name, feature_names)
    shap_vals, shap_df = xai.compute_shap_values(_X_test)
    overall_top, db_rankings = xai.rank_top_biomarkers(_df_processed, _X_test, top_n=10)
    return shap_df, overall_top, db_rankings


def main():
    st.markdown('<div class="main-title">🧬 Bio Transcript Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">High-Performance Cleft Therapy Transcriptomics • Machine Learning Benchmarking • Explainable AI (SHAP)</div>',
        unsafe_allow_html=True
    )

    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.header("⚙️ Biological Thresholds")
    
    # Mode toggle: Form (instant, zero lag on large datasets) vs Continuous Live Drag
    use_live_drag = st.sidebar.toggle(
        "Continuous Live Sliders",
        value=False,
        help="When enabled, sliders trigger immediate reruns on every mouse movement. When disabled (recommended for 100k+ rows), thresholds are applied cleanly without drag stutter."
    )

    if use_live_drag:
        logfc_up = st.sidebar.slider(
            "Up-Regulation logFC Threshold (≥)",
            min_value=0.2,
            max_value=3.0,
            value=st.session_state.get("saved_logfc_up", 1.0),
            step=0.1,
            key="live_logfc_up"
        )
        logfc_down = st.sidebar.slider(
            "Down-Regulation logFC Threshold (≤)",
            min_value=-3.0,
            max_value=-0.2,
            value=st.session_state.get("saved_logfc_down", -1.0),
            step=0.1,
            key="live_logfc_down"
        )
        adjp_val = st.sidebar.slider(
            "Significance adj.P.Val / FDR (<)",
            min_value=0.001,
            max_value=0.200,
            value=st.session_state.get("saved_adjp_val", 0.050),
            step=0.005,
            format="%.3f",
            key="live_adjp_val"
        )
    else:
        with st.sidebar.form("threshold_form"):
            logfc_up = st.slider(
                "Up-Regulation logFC (≥)",
                min_value=0.2,
                max_value=3.0,
                value=st.session_state.get("saved_logfc_up", 1.0),
                step=0.1,
                help="Transcripts with logFC ≥ this value and adj.P.Val < threshold are classified as Up-Regulated."
            )
            logfc_down = st.slider(
                "Down-Regulation logFC (≤)",
                min_value=-3.0,
                max_value=-0.2,
                value=st.session_state.get("saved_logfc_down", -1.0),
                step=0.1,
                help="Transcripts with logFC ≤ this value and adj.P.Val < threshold are classified as Down-Regulated."
            )
            adjp_val = st.slider(
                "Significance adj.P.Val / FDR (<)",
                min_value=0.001,
                max_value=0.200,
                value=st.session_state.get("saved_adjp_val", 0.050),
                step=0.005,
                format="%.3f",
                help="Adjusted P-Value (Benjamini-Hochberg FDR) significance cutoff."
            )
            apply_submitted = st.form_submit_button(
                "⚡ Apply Thresholds",
                type="primary",
                use_container_width=True
            )
            if apply_submitted:
                st.session_state["saved_logfc_up"] = logfc_up
                st.session_state["saved_logfc_down"] = logfc_down
                st.session_state["saved_adjp_val"] = adjp_val

    st.sidebar.markdown("---")
    st.sidebar.header("📁 Dataset Selection")
    data_source_mode = st.sidebar.radio(
        "Data Source",
        ["Default Cleft Therapy Dataset", "Upload Custom File (CSV/PDF)"],
        index=0
    )

    raw_df: pd.DataFrame = pd.DataFrame()

    if data_source_mode == "Upload Custom File (CSV/PDF)":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Gene Expression Table",
            type=["csv", "pdf", "tsv", "txt"],
            help="Upload a differential expression statistical table."
        )
        if uploaded_file is not None:
            try:
                # Cache uploaded file bytes to prevent re-parsing on every slider interaction
                file_bytes = uploaded_file.getvalue()
                raw_df = parse_uploaded_file(file_bytes, uploaded_file.name)
                st.sidebar.success(f"Loaded: {uploaded_file.name} ({len(raw_df):,} records)")
            except Exception as e:
                st.sidebar.error(f"Error parsing file: {e}")
                return
        else:
            st.info("👈 Please upload a CSV or PDF dataset from the sidebar to begin.")
            return
    else:
        try:
            raw_df = load_default_data()
        except Exception as e:
            st.error(f"Could not load default dataset: {e}. Please use the upload feature.")
            return

    # Database filter in sidebar
    available_dbs = sorted(list(raw_df["database_source"].unique()))
    selected_dbs = st.sidebar.multiselect(
        "Filter Database Sources",
        options=available_dbs,
        default=available_dbs,
        help="Filter visible transcripts by source annotation database."
    )

    if not selected_dbs:
        st.warning("Please select at least one database source in the sidebar.")
        return

    filtered_df = raw_df[raw_df["database_source"].isin(selected_dbs)].copy()

    # ------------------ FAST BIOLOGICAL PREPROCESSING (<10ms) ------------------
    df_proc = compute_biological_labels(filtered_df, logfc_up, logfc_down, adjp_val)

    # Compute regulation summary metrics
    total_transcripts = len(df_proc)
    up_count = int((df_proc["target_label"] == "Up-Regulated").sum())
    down_count = int((df_proc["target_label"] == "Down-Regulated").sum())
    neutral_count = int((df_proc["target_label"] == "Neutral").sum())

    # ------------------ TOP METRIC CARDS ------------------
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div style="color:#64748b;font-size:0.85rem;">TOTAL TRANSCRIPTS</div>'
            f'<div class="metric-val" style="color:#1e293b;">{total_transcripts:,}</div></div>',
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div style="color:#dc2626;font-size:0.85rem;">UP-REGULATED</div>'
            f'<div class="metric-val" style="color:#dc2626;">{up_count:,}</div></div>',
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div style="color:#2563eb;font-size:0.85rem;">DOWN-REGULATED</div>'
            f'<div class="metric-val" style="color:#2563eb;">{down_count:,}</div></div>',
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><div style="color:#64748b;font-size:0.85rem;">NEUTRAL</div>'
            f'<div class="metric-val" style="color:#64748b;">{neutral_count:,}</div></div>',
            unsafe_allow_html=True
        )
    with m5:
        st.markdown(
            f'<div class="metric-card"><div style="color:#059669;font-size:0.85rem;">DATASET STATUS</div>'
            f'<div class="metric-val" style="color:#059669;font-size:1.35rem;">⚡ Realtime Ready</div></div>',
            unsafe_allow_html=True
        )

    st.write("")

    # ------------------ MAIN TABS ------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌋 Volcano Plot & Data Explorer",
        "📊 ML Benchmark & Evaluation",
        "🧠 Explainable AI & Biomarkers (SHAP)",
        "📄 Raw Dataset & Schema"
    ])

    # ==================== TAB 1: VOLCANO PLOT ====================
    with tab1:
        col_plot, col_dist = st.columns([3, 1])

        with col_plot:
            p_metric = st.radio(
                "Y-Axis Metric:",
                ["-log10(P.Value)", "-log10(adj.P.Val)"],
                horizontal=True
            )
            y_col = "neg_log10_pvalue" if p_metric == "-log10(P.Value)" else "neg_log10_adjpval"

            # Uses WebGL Scattergl for smooth 60fps rendering of 50k+ points
            volcano_fig = Visualizer.plot_volcano(
                df_proc,
                logfc_up=logfc_up,
                logfc_down=logfc_down,
                adjp_thresh=adjp_val,
                y_axis_col=y_col,
                title=f"Differential Expression Volcano Plot ({len(df_proc):,} Transcripts)"
            )
            st.plotly_chart(volcano_fig, use_container_width=True)

        with col_dist:
            dist_fig = Visualizer.plot_class_distribution(df_proc)
            st.plotly_chart(dist_fig, use_container_width=True)

            st.markdown("#### Regulation Breakdown")
            st.markdown(f"- **Up-Regulated:** `{up_count}` ({up_count/max(1, total_transcripts):.1%})")
            st.markdown(f"- **Down-Regulated:** `{down_count}` ({down_count/max(1, total_transcripts):.1%})")
            st.markdown(f"- **Neutral:** `{neutral_count}` ({neutral_count/max(1, total_transcripts):.1%})")

        st.markdown("### 🔍 Filtered Transcript Browser")
        search_kw = st.text_input("Search Transcript ID or Keyword:", placeholder="e.g. NM_000123, ENSG, IRF6...")
        
        # Filter option for significant transcripts
        filter_sig = st.checkbox(
            "Show only statistically significant transcripts (Up/Down)",
            value=(total_transcripts > 5000),
            help="Filter table to only Up-Regulated and Down-Regulated transcripts for instant responsiveness."
        )

        display_df = df_proc[[
            "transcript_id", "database_source", "target_label", "logFC", "t", "P.Value", "adj.P.Val"
        ]].copy()

        if filter_sig:
            display_df = display_df[display_df["target_label"].isin(["Up-Regulated", "Down-Regulated"])]

        if search_kw:
            display_df = display_df[display_df["transcript_id"].str.contains(search_kw, case=False, na=False)]

        preview_limit = 500
        if len(display_df) > preview_limit:
            st.caption(f"Showing first {preview_limit:,} of {len(display_df):,} records for optimal browser performance. Use CSV download for complete dataset.")
            df_to_show = display_df.iloc[:preview_limit]
        else:
            df_to_show = display_df

        st.dataframe(
            df_to_show.style.format({
                "logFC": "{:.3f}",
                "t": "{:.3f}",
                "P.Value": "{:.2e}",
                "adj.P.Val": "{:.2e}"
            }),
            use_container_width=True,
            height=300
        )

        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Transcripts (CSV)",
            data=csv_data,
            file_name="filtered_transcripts.csv",
            mime="text/csv"
        )

    # ==================== TAB 2: MODEL EVALUATION ====================
    with tab2:
        st.markdown("### 🏆 4-Classifier Benchmark Comparison")
        st.caption("Random Forest, XGBoost, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP).")

        current_params_key = f"{len(filtered_df)}_{logfc_up}_{logfc_down}_{adjp_val}"
        is_trained = (
            "ml_results" in st.session_state
            and st.session_state.get("ml_params_key") == current_params_key
        )

        col_train_btn, col_train_status = st.columns([1, 2])
        with col_train_btn:
            train_clicked = st.button("🚀 Train & Benchmark 4 Models", type="primary", use_container_width=True)

        with col_train_status:
            if is_trained:
                st.success("✅ Models trained and up-to-date for current biological thresholds.")
            else:
                st.info("💡 Thresholds updated. Click the button to run the 4-classifier benchmark.")

        # Train models only when requested or if auto-training is enabled
        if train_clicked or (not is_trained and total_transcripts <= 2000):
            with st.spinner("Training Random Forest, XGBoost, SVM, and MLP classifiers..."):
                pipeline_output = run_ml_pipeline(filtered_df, logfc_up, logfc_down, adjp_val)
                st.session_state["ml_results"] = pipeline_output
                st.session_state["ml_params_key"] = current_params_key
                is_trained = True

        if is_trained and "ml_results" in st.session_state:
            ml_out = st.session_state["ml_results"]
            comp_df = ml_out["comparison_df"]
            results = ml_out["results"]
            best_name = ml_out["best_name"]
            class_names = ml_out["class_names"]

            # Benchmark Comparison Table
            with st.expander("ℹ️ Why are benchmark scores so high (~1.000)?", expanded=False):
                st.markdown(r"""
                **Scientific Context of Model Scores**:
                - **Deterministic Boundary**: The ground-truth biological classes (*Up-Regulated*, *Down-Regulated*, *Neutral*) are created by exact mathematical criteria ($\text{logFC} \ge 1.0 \land \text{adj.P.Val} < 0.05$).
                - **Feature Alignment**: Because classifiers receive the statistical metrics ($\text{logFC}, t\text{-statistic}, -\log_{10}(\text{P.Value})$), all 4 models (RF, XGBoost, SVM, MLP) readily learn this decision boundary with near-perfect separation.
                - **Class-Preserving Stratification**: The pipeline preserves 100% of the rare biomarker instances in the holdout test set while downsampling the neutral background.
                - **Balanced Accuracy & Macro F1**: Shows the unweighted average performance across all 3 biological classes equally.
                """)

            formatted_comp = comp_df.copy()
            for col in ["Balanced Accuracy", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
                if col in formatted_comp.columns:
                    formatted_comp[col] = formatted_comp[col].apply(lambda x: f"{x:.4f}")

            st.dataframe(formatted_comp, use_container_width=True)

            # Bar chart comparison
            bar_fig = Visualizer.plot_model_comparison(comp_df)
            st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔍 Model Diagnostic Curves")

            selected_model_name = st.selectbox(
                "Select Classifier for Detailed Diagnostics:",
                options=list(results.keys()),
                index=list(results.keys()).index(best_name)
            )

            model_res = results[selected_model_name]
            col_cm, col_roc = st.columns(2)

            with col_cm:
                normalize_cm = st.checkbox("Normalize Confusion Matrix", value=True)
                cm_fig = Visualizer.plot_confusion_matrix(
                    model_res["confusion_matrix"],
                    class_names=class_names,
                    normalize=normalize_cm,
                    title=f"{selected_model_name} Confusion Matrix"
                )
                st.plotly_chart(cm_fig, use_container_width=True)

            with col_roc:
                roc_fig = Visualizer.plot_roc_curves(
                    model_res["roc_curves"],
                    class_names=class_names,
                    title=f"{selected_model_name} Multiclass ROC Curves (One-vs-Rest)"
                )
                st.plotly_chart(roc_fig, use_container_width=True)

    # ==================== TAB 3: EXPLAINABLE AI & BIOMARKERS ====================
    with tab3:
        st.markdown("### 🧠 Explainable AI & Biomarker Discovery")
        st.caption("SHAP feature importance ranking & multi-database biomarker candidate prioritization.")

        # Compute Top Biomarkers directly from dataset (instantaneous, no training delay)
        xai_dummy = XAIEngine(None, "Benchmark", [])
        overall_top, db_rankings = xai_dummy.rank_top_biomarkers(df_proc, top_n=10)

        # SHAP section if model is trained
        has_trained_model = "ml_results" in st.session_state and st.session_state.get("ml_params_key") == current_params_key

        try:
            if has_trained_model:
                ml_out = st.session_state["ml_results"]
                best_name = ml_out["best_name"]
                best_model = ml_out["best_model"]
                feature_names = ml_out["feature_names"]
                X_test = ml_out["X_test"]

                shap_df, _, _ = run_xai_pipeline(best_model, best_name, feature_names, X_test, df_proc)

                col_shap, col_bio = st.columns([1, 1])
                with col_shap:
                    st.markdown(f"#### Global Feature Importance (SHAP) - {best_name}")
                    shap_fig = Visualizer.plot_shap_summary(shap_df, top_n=10)
                    st.plotly_chart(shap_fig, use_container_width=True)

                with col_bio:
                    st.markdown("#### 🌟 Overall Top 10 Biomarker Transcripts")
                    st.dataframe(
                        overall_top.style.format({
                            "logFC": "{:.3f}",
                            "t": "{:.3f}",
                            "P.Value": "{:.2e}",
                            "adj.P.Val": "{:.2e}",
                            "Biomarker_Score": "{:.2f}"
                        }),
                        use_container_width=True,
                        height=450
                    )
            else:
                st.info("💡 Train models in Tab 2 to view SHAP attribution summary plots. Biomarker tables below are computed in real-time:")
                st.markdown("#### 🌟 Overall Top 10 Biomarker Transcripts")
                st.dataframe(
                    overall_top.style.format({
                        "logFC": "{:.3f}",
                        "t": "{:.3f}",
                        "P.Value": "{:.2e}",
                        "adj.P.Val": "{:.2e}",
                        "Biomarker_Score": "{:.2f}"
                    }),
                    use_container_width=True,
                    height=350
                )

            st.markdown("---")
            st.markdown("### 🧬 Top 10 Biomarkers by Database Annotation Source")
            st.caption("Stratified analysis highlighting candidate transcripts across RefSeq, ENSEMBL, and lncRNAWiki.")

            db_tabs = st.tabs(["RefSeq", "ENSEMBL", "lncRNAWiki", "Ace View", "miTranscriptome", "UCSC Genes"])
            db_names = ["RefSeq", "ENSEMBL", "lncRNAWiki", "Ace View", "miTranscriptome", "UCSC Genes"]

            for i, db_name in enumerate(db_names):
                with db_tabs[i]:
                    ranking = db_rankings.get(db_name, pd.DataFrame())
                    if not ranking.empty:
                        st.dataframe(
                            ranking.style.format({
                                "logFC": "{:.3f}",
                                "t": "{:.3f}",
                                "P.Value": "{:.2e}",
                                "adj.P.Val": "{:.2e}",
                                "Biomarker_Score": "{:.2f}"
                            }),
                            use_container_width=True
                        )
                        csv_ranking = ranking.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label=f"📥 Download {db_name} Top Biomarkers (CSV)",
                            data=csv_ranking,
                            file_name=f"{db_name.lower()}_top_biomarkers.csv",
                            mime="text/csv",
                            key=f"dl_{db_name}"
                        )
                    else:
                        st.info(f"No transcripts found for database: {db_name}")

        except Exception as e:
            st.error(f"Error computing Explainable AI values: {e}")

    # ==================== TAB 4: RAW DATASET & SCHEMA ====================
    with tab4:
        st.markdown("### 📄 Dataset Inspection & Schema Details")
        st.markdown(f"- **Total Rows:** {len(raw_df):,}")
        st.markdown(f"- **Detected Columns:** `{list(raw_df.columns)}`")
        st.markdown(f"- **Database Sources:** `{list(raw_df['database_source'].value_counts().to_dict().items())}`")

        # Show first 500 rows for high responsiveness
        st.dataframe(raw_df.head(500), use_container_width=True)

        full_csv = raw_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Cleaned Dataset (CSV)",
            data=full_csv,
            file_name="clean_transcriptomic_dataset.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
