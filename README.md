# 🚀 Performance Regression Detection Benchmark (ICPE)

This project implements a **Machine Learning pipeline** designed to automate the triage of software performance alerts (Performance Regression Detection).

It predicts whether a performance alert triggered by CI/CD systems represents a **real software bug** or **noise** (false positive). The system operates in a strict **"Realistic Mode,"** meaning all "future" data (human notes, triage tags, manual classifications) is rigorously stripped to prevent Data Leakage, ensuring the benchmark reflects real-time decision-making capabilities.

## 🧠 Core Architecture

The model utilizes a Hybrid Architecture combining three distinct signal sources:

1.  **Multi-Scale Time Series Analysis:** Parallel extraction of statistical features (slope, z-score, step-change) over multiple window sizes (Short/Medium/Long) using `multiprocessing`.
2.  **Contextual NLP Embeddings:** Vectorization of technical context (Repository + Framework + Test Suite names) using **FastText** (compressed) to capture semantic relationships between test suites.
3.  **Metadata & Heuristics:** Processing of platform info, architecture (ARM/x86), and historical backfill data.

The classifier is built on **CatBoost**, optimized via **Optuna**.

## 📂 Project Structure

```text
.
├── benchmark.py           # 🏁 Main Entry Point: Runs the full pipeline (Load -> Train -> Eval -> Report)
├── cat_boost_best.py      # 🧪 Optimization Script: Hyperparameter tuning via Optuna
├── config.py              # ⚙️ Configuration: Path definitions and constants
├── data_loader.py         # 📥 Data Ingestion: Loads raw CSVs and aggregates alerts
├── model_utils.py         # 🛠️ Utilities: PCA, Data Leakage prevention, CatBoost Pool creation
├── preprocessing.py       # 🧹 Feature Engineering: NLP (FastText), Complex String/JSON parsing
├── timeseries_multi.py    # 📈 Time Series Engine: Parallel extraction of history signals
└── benchmark_results/     # 📊 Output: Scientific plots, error analysis logs (Auto-generated)
```

## 🛠️ Installation & Requirements
Prerequisites
Python 3.9+

Internet access (required to download the compressed FastText model on the first run).

~16GB RAM recommended (for processing Time Series in parallel).

Dependencies
Installs the required packages:

``` text 
pip install pandas numpy matplotlib seaborn scikit-learn catboost cleanlab optuna psutil compress-fasttext tqdm pyarrow
```

## 💾 Data Layout
The project expects a specific directory structure relative to the code location. By default, it looks for a folder named icpe_data located two levels up from the script (see config.py).

Expected Hierarchy:

``` text 
/icpe_data/                <-- Root Data Directory
    ├── alerts_data.csv    # Raw alerts export
    ├── bugs_data.csv      # Bug tracker export (Labels)
    └── timeseries-data/   # Directory containing history CSVs per signature
          ├── repo_name/
          │    └── 123456_timeseries_data.csv
          └── ...
```

## 🚀 Usage
1. Run the Standard Benchmark
To run the full training, evaluation, and reporting pipeline:

```text
python benchmark.py
```

What this does:

Loading: Loads and aggregates alert data.

TS Extraction: Extracts Time Series features (cached in ./derived_features as parquet).

NLP: Generates embeddings for test contexts using FastText.

Training: Trains the CatBoost model.

Evaluation: Calculates AUPRC, Precision@K.

Analysis: Runs Cleanlab to detect potential labeling errors in the ground truth.

Reporting: Exports scientific graphs and an error analysis CSV to benchmark_results/.

2. Hyperparameter Optimization
To re-optimize the CatBoost parameters using Optuna:

``` text 
python cat_boost_best.py
```

Note: This utilizes SQLite for storage and runs 500 trials with Repeated Stratified K-Fold validation.

## 📊 Outputs & Reporting
The benchmark.py script automatically generates a benchmark_results/ folder containing:

1. **scientific_feature_importance.png:** Grouped importance of Signals (TS) vs. Context (NLP) vs. Metadata.

2. **scientific_pr_curve.png:** Precision-Recall curve.

3. **scientific_latency_breakdown.png:** Waterfall chart of pipeline latency per commit (for production feasibility analysis).

4. **benchmark_errors.csv:** A detailed audit file containing the "Top Misses" (High confidence False Positives/Negatives) for manual review.


