## Performance Alert Bug Prediction (ICPE 2026)
This repository contains the source code and benchmarking pipelines for the paper "Automated Bug Prediction from Performance Alerts using Multi-Modal Learning" (Target: ICPE 2026).

The project implements a machine learning pipeline (CatBoost) to predict whether a performance regression alert in a CI/CD environment (Mozilla) will result in a valid bug report. It compares three distinct approaches:

Static Baseline: Metadata only.

Time Series Enhanced: Metadata + Statistical signal features (Slope, Z-Score, etc.).

Hybrid (NLP): Metadata + Time Series + Textual Embeddings (FastText) from triage notes.

# Project Structure

```text


├── benchmark.py                # Main Pipeline: Static + Time Series features (No NLP)
├── benchmark_tabular.py        # Baseline Pipeline: Static Metadata only
├── benchmark_with_notes.py     # Hybrid Pipeline: Static + Time Series + NLP (FastText)
│
├── cat_boost_best.py           # Optuna Optimization for Main Pipeline
├── cat_boost_best_with_nlp.py  # Optuna Optimization for Hybrid Pipeline
├── cat_boost_best_static.py    # Optuna Optimization for Static Pipeline
│
├── data_loader.py              # Loads CSVs and aggregates raw alerts
├── preprocessing.py            # Feature engineering, split, and FastText loading
├── timeseries_multi.py         # Parallel extraction of Time Series features
├── model_utils.py              # PCA, CatBoost Pool helpers, and Metrics
├── config.py                   # Paths and global constants
└── requirements.txt            # Python dependencies

```

# Getting Started
Prerequisites

Python 3.10+

RAM: >16GB recommended (for Time Series processing)

CPU: Multi-core recommended (pipeline uses ProcessPoolExecutor)

Installation
Install the required dependencies:

Bash
```text
pip install pandas numpy scikit-learn catboost optuna seaborn matplotlib tqdm psutil compress-fasttext pyarrow
```

OR 

```text
pip install -r requirements.txt
```
Data Setup
The code expects the following directory structure relative to the project root (defined in config.py):

```

../icpe_data/
    ├── alerts_data.csv       # Raw alert data
    ├── bugs_data.csv         # Bug tracker metadata
    └── timeseries-data/      # Folder containing per-signature .csv files
```

# Running the Benchmarks
To reproduce the tables and figures from the paper, run the three benchmark scripts. Each script generates a specific output folder containing Precision-Recall curves, Confusion Matrices, and Feature Importance plots.

1. **Static Baseline (Blind)**
Uses only categorical and numerical metadata (e.g., platform, suite, regression type). It strictly ignores Time Series signals to serve as a naive baseline.

```text
python benchmark_tabular.py
```
Outputs to: benchmark_results_static_only/
2. Time Series Pipeline (Standard)
Enriches the static data with statistical features computed from the raw time series history (window size 5, 20, 50). This includes signal-to-noise ratio, slope changes, and step detection.

```text
python benchmark.py

```

Outputs to: benchmark_results_no_nlp/
ncludes resource monitoring (RAM/CPU usage tracking)
3. Hybrid Pipeline (With NLP)
Adds semantic understanding by embedding triage notes using a compressed FastText model (cc.en.300.compressed.bin). PCA is applied to reduce embedding dimensionality before training.

```text
python benchmark_with_notes.py

```
Outputs to: benchmark_results_with_notes/
# Hyperparameter Optimization
The hyperparameters in the benchmark_\*.py files are fixed based on previous optimization runs. To re-run the optimization using Optuna, use the following scripts. These will create SQLite databases (optuna_*.db) to store trial results.

Static Only: python cat_boost_best_static.py

Time Series: python cat_boost_best.py

Hybrid (NLP): python cat_boost_best_with_nlp.py

# Methodology & Key Features
Weighted Loss: The model uses sample_weight derived from Bug Priority (P1/P2/P3). Critical bugs are weighted higher (x10) than minor issues to prioritize recall on severe regressions.

Time Series Feature Extraction (timeseries_multi.py):

Uses concurrent.futures to process thousands of time series files in parallel.

Computes multi-scale features: delta, z-score, slope_change, std_ratio over short, medium, and long windows.

Blind Mode: In the static benchmark, features that might leak the magnitude of the regression (like z-score or pct_change) are explicitly banned to ensure a fair comparison with the signal-processing approach.

# Outputs
Each benchmark generates:

pr_curve.png: Precision-Recall curve with AUPRC score.

feature_importance.png: Top contributing features (aggregated by type: Static vs. TS vs. NLP).

confusion_matrix.png: Normalized prediction accuracy.

runtime_metrics_full.csv (Only for standard benchmark): Detailed breakdown of CPU/RAM usage per stage.