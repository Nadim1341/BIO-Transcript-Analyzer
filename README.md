# 🧬 Bio Transcript Analyzer

An end-to-end bioinformatics and machine learning software pipeline with an interactive Streamlit web dashboard for analyzing differential gene expression tables and explainable AI in cleft therapy transcriptomics.

---

## 📌 Overview & Domain

In cleft lip and palate (CL/P) therapy research, differential gene expression profiling across craniofacial tissues uncovers essential developmental biomarkers and therapeutic targets. This application processes statistical tables produced by differential expression workflows (such as Limma/Voom or DESeq2) across multiple biological database sources:
- **RefSeq** (NCBI)
- **ENSEMBL** (EBI)
- **lncRNAWiki** (Long non-coding RNAs)
- **Ace View** (NCBI alternative splicing annotations)
- **miTranscriptome** (microRNA & non-coding transcripts)
- **UCSC Genes**

The system classifies transcripts into biological regulation states, trains and benchmarks 4 machine learning models (**Random Forest, XGBoost, Support Vector Machine, and Multi-Layer Perceptron**), and provides explainable AI through **SHAP** (SHapley Additive exPlanations) values to rank biomarker candidates.

---

## 🏗️ Project Architecture

```
bio_transcript_analyzer/
├── data/
│   ├── datasets.pdf                   # Raw dataset in PDF format
│   ├── datasets.csv                   # Raw dataset in CSV format
│   └── generate_dataset.py            # Synthetic dataset generator for craniofacial transcripts
├── src/
│   ├── __init__.py
│   ├── data_parser.py                 # Multi-format PDF and CSV table parser & data cleaning
│   ├── preprocessing.py               # Biological rule labeling, signal amplification, OHE, StandardScaler
│   ├── model_trainer.py               # RF, XGBoost, SVM, MLP benchmark pipeline & metrics
│   ├── xai_engine.py                  # SHAP calculation & top biomarker ranking across databases
│   └── visualizer.py                  # Interactive Plotly volcano plots, confusion matrix, ROC curves
├── app/
│   └── main_dashboard.py              # Interactive Streamlit Web Interface
├── tests/
│   ├── __init__.py
│   ├── test_parser.py                 # Pytest for data parsing and validation
│   └── test_preprocessing.py          # Pytest for biological labeling and feature scaling
├── requirements.txt                   # Pinned project dependencies
└── README.md                          # Complete system documentation
```

---

## ⚙️ Core Functionalities

### 1. Data Parsing & Ingestion (`src/data_parser.py`)
- Ingests structured differential expression statistical tables from **PDF** (via `pdfplumber`) and **CSV/TSV** files.
- Extracts standard parameters: `transcript_id`, `logFC`, `t`, `P.Value`, `adj.P.Val`, and `database_source`.
- Automatically maps varied column aliases (`log2FC`, `p_val`, `FDR`, `gene_id`, etc.) to a canonical schema.
- Cleans missing values, bounds probabilities in $(0, 1]$, and infers database annotations when missing.

### 2. Feature Engineering & Rule-Based Labeling (`src/preprocessing.py`)
- **Biological Regulation Labeling Rules**:
  - **Up-Regulated**: $\text{logFC} \ge 1.0 \land \text{adj.P.Val} < 0.05$
  - **Down-Regulated**: $\text{logFC} \le -1.0 \land \text{adj.P.Val} < 0.05$
  - **Neutral**: Otherwise
- **Signal Amplification**: Computes $-\log_{10}(\text{P.Value})$ and $-\log_{10}(\text{adj.P.Val})$.
- **Database One-Hot Encoding**: Encodes RefSeq, ENSEMBL, lncRNAWiki, Ace View, miTranscriptome, and UCSC Genes.
- **Normalization**: Standardizes numerical features using scikit-learn's `StandardScaler`.

### 3. Machine Learning Pipeline (`src/model_trainer.py`)
- Splits data into an **80% Training / 20% Testing** stratified partition.
- Trains and benchmarks 4 classifiers:
  1. **Random Forest** (RF)
  2. **XGBoost** (XGB)
  3. **Support Vector Machine** (SVM)
  4. **Multi-Layer Perceptron** (MLP)
- Calculates comprehensive performance metrics: **Accuracy, Precision, Recall, F1-Score, and ROC-AUC (OvR)**.
- Computes multi-class Confusion Matrices and One-vs-Rest ROC curves.

### 4. Explainable AI & Biomarker Discovery (`src/xai_engine.py`)
- Calculates SHAP values for the top-performing model.
- Evaluates global and class-specific feature attributions.
- Computes a composite **Biomarker Score**:
  $$\text{Biomarker Score} = (|\text{logFC}| \times -\log_{10}(\text{adj.P.Val})) \times \text{Regulation Factor}$$
- Identifies and ranks the **Top 10 Biomarker Transcripts** across RefSeq, ENSEMBL, and lncRNAWiki.

### 5. Interactive Streamlit Web Interface (`app/main_dashboard.py`)
- **Dynamic Sidebar Controls**: Real-time interactive sliders for $\text{logFC}$ thresholds and $\text{adj.P.Val}$ cutoffs.
- **Interactive Volcano Plot**: Responsive Plotly scatter plot with red (Up), blue (Down), and grey (Neutral) color coding.
- **Model Benchmark Tab**: Comparative table, performance bar chart, interactive confusion matrix, and multi-class ROC curves.
- **Explainable AI Tab**: SHAP feature importance summary plot, overall top biomarkers, and database-specific tabs with CSV export.
- **Universal File Uploader**: Ingest custom CSV or PDF transcriptomic tables for universal cross-disease research.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone & Navigate
```bash
git clone <repository_url>
cd bio_transcript_analyzer
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 System Execution

### Launch the Streamlit Web Dashboard
From the `bio_transcript_analyzer` directory:
```bash
streamlit run app/main_dashboard.py
```
*The web interface will automatically open in your browser at `http://localhost:8501`.*

### Generate Synthetic Dataset
To regenerate the sample craniofacial differential expression tables (CSV and PDF):
```bash
python data/generate_dataset.py
```

### Run Unit Tests
To execute the comprehensive unit test suite:
```bash
pytest tests/ -v
```

---

## 🧪 Dependencies

| Package | Purpose |
| :--- | :--- |
| `pandas` | Tabular data manipulation and statistical filtering |
| `numpy` | Vectorized numerical computations and logarithmic transformations |
| `scikit-learn` | Preprocessing (`StandardScaler`), model evaluation, RF, SVM, MLP |
| `xgboost` | Gradient boosted tree classification |
| `shap` | Explainable AI (Shapley Additive exPlanations) |
| `streamlit` | Reactive web interface and dashboard controls |
| `plotly` | Interactive publication-grade data visualizations |
| `pdfplumber` | PDF table and text extraction |
| `reportlab` | Publication-style PDF dataset generation |
| `pytest` | Unit testing and automated verification |

---

## 📄 License
MIT License. Built for bioinformatics research and computational biology pipelines.
