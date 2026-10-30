# E-commerce Sentiment Analysis and Sales Insights

[![Python](https://img.shields.io/badge/Python-3.8%2B%20%7C%203.14-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.3.3-150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.5-013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.8-11557c.svg?style=flat)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-3776AB.svg?style=flat)](https://seaborn.pydata.org/)
[![Joblib](https://img.shields.io/badge/Joblib-1.5.3-4B8BBE.svg?style=flat)](https://joblib.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--981--92--5407--1-blue.svg?style=flat&logo=doi&logoColor=white)](https://doi.org/10.1007/978-981-92-5407-1){:target="_blank"}
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

---

## Associated Publication

This research and codebase accompanies the following academic publication:

- **Book:** *AI Regulation in Industry 5.0*
- **DOI:** [`10.1007/978-981-92-5407-1`](https://doi.org/10.1007/978-981-92-5407-1){:target="_blank"}
- **Chapter Title:** *AI-Driven E-Commerce in Industry 5.0: Ethical, Regulatory, and Predictive Insights for India*
- **Authors:** *Runu Patgiri, Vanlalruata Hnamte, Gurram Ramakrishna*
- **Publisher:** Springer Nature (*Artificial Intelligence and the Rule of Law* series)

### Citation (BibTeX)

```bibtex
@incollection{ecommerce_industry5_2026,
  title     = {AI-Driven E-Commerce in Industry 5.0: Ethical, Regulatory, and Predictive Insights for India},
  authors   = {Runu Patgiri, Vanlalruata Hnamte, Gurram Ramakrishna},
  booktitle = {AI Regulation in Industry 5.0},
  publisher = {Springer},
  doi       = {10.1007/978-981-92-5407-1},
  year      = {2026}
}
```

---

## Overview

This project delivers an end‑to‑end, reproducible analytical and predictive machine learning pipeline designed to investigate consumer sentiment, pricing dispersion, and multi-channel sales dynamics across Indian e-commerce marketplaces:

- **Consumer Sentiment Classification:** Trains and validates an NLP text sentiment classifier on Amazon customer product reviews using TF‑IDF feature extraction with Logistic Regression across repeated stratified trials.
- **Multi-Channel Sales Analytics:** Ingests and standardizes heterogeneous e-commerce sales datasets, computing sales distributions, top revenue-generating SKUs, geographic demand concentrations across Indian states/regions, and marketplace profitability.
- **Cross-Platform MRP Dispersion:** Analyzes price variance and Maximum Retail Price (MRP) dispersion across key Indian e-commerce channels (Amazon, Flipkart, Myntra, Ajio, Limeroad, Paytm, Snapdeal).
- **ASIN–SKU Cross-Domain Integration:** Maps product reviews and sentiment polarity aggregates directly to inventory SKUs/ASINs to deliver unified catalog intelligence.

The main execution pipeline is orchestrated by [`amazon_ecom_analysis_integrated_full.py`](amazon_ecom_analysis_integrated_full.py), generating standardized datasets, trained model weights, publication-ready vector plots (`.pdf`/`.svg`), and full quantitative summary metrics under `outputs_integrated_full/`.

---

## Software Stack & Dependencies

The pipeline is tested and validated on **Python 3.8+** (including **Python 3.14**). Below are the core software components and library versions utilized:

| Component / Library | Tested Version | Purpose / Role |
| :--- | :--- | :--- |
| **Python** | `3.8+` (tested on `3.14.9`) | Base runtime execution environment |
| **scikit-learn** | `1.8.0` | TF-IDF vectorization, Logistic Regression classifier, train-test splitting, metrics |
| **pandas** | `2.3.3` | High-performance CSV parsing, multi-channel schema standardization, data aggregation |
| **NumPy** | `2.3.5` | Numerical computations, array operations, and descriptive statistical metrics |
| **matplotlib** | `3.10.8` | Base plotting engine for publication figures, distributions, and confusion matrices |
| **seaborn** | `0.13.2` | Statistical data visualization, heatmaps, and palette styling |
| **joblib** | `1.5.3` | High-efficiency model serialization and vectorizer artifact caching |
| **Jupyter** | `Notebook` | Interactive exploratory analysis, plotting workflows, and gradient chart generation |

---

## Codebase Details & Architecture

The repository is modularly structured into pipeline orchestrators, dedicated NLP classifiers, data inspectors, and reporting notebooks:

```
ecommerce_sentimental_analysis_prediction/
├── amazon_ecom_analysis_integrated_full.py   # Primary end-to-end pipeline (Reviews + Sales + MRP + Integration)
├── tfidf_lr_sentiment.py                     # Standalone TF-IDF + Logistic Regression sentiment module
├── amazon_ecom_analysis.py                   # Core sales analysis and channel profitability analyzer
├── dataset_inspector.py                      # Pre-flight diagnostic tool for dataset schemas and missingness
├── plotting.ipynb                            # Interactive visualization notebook for publication-grade charts
├── mrp_dispersion_box.{pdf,png,svg}          # MRP dispersion figures across e-commerce platforms
├── top_10_skus_green_gradient.pdf            # High-resolution top SKUs revenue chart
├── top_10_regions_green_gradient.pdf         # Regional revenue distribution chart
├── outputs_integrated_full/                  # Generated CSV tables, metrics reports, figures, and model weights
└── LICENSE                                   # MIT License
```

### Module Descriptions

1. **`amazon_ecom_analysis_integrated_full.py` (Main Pipeline)**
   - Orchestrates the full lifecycle: data loading, automated header standardization, date parsing, text preprocessing, multi-run sentiment experimentation, MRP dispersion calculation, sales region breakdown, and ASIN↔SKU integration.
   - Saves all artifacts directly into `outputs_integrated_full/`.

2. **`tfidf_lr_sentiment.py` (NLP Sentiment Classifier)**
   - Specialized script focused solely on text sentiment extraction.
   - Cleans review text, discards neutral feedback (`Score == 3`), builds an n-gram TF-IDF matrix (up to 50,000 features), fits an L2-regularized Logistic Regression model, outputs confusion matrices, classification reports, and serializes `tfidf_vectorizer.joblib` and `tfidf_lr_model.joblib`.

3. **`amazon_ecom_analysis.py` (Sales & Channel Profitability)**
   - Ingests sales reports and assesses logistics channel profitability (e.g., Shiprocket, INCREFF).
   - Generates top SKU performance tables and exploratory MRP variance charts.

4. **`dataset_inspector.py` (Data Diagnostics)**
   - Fast diagnostic script to inspect CSV headers, estimate row/column counts, compute null rates, and verify input schemas before launching long compute jobs.

5. **`plotting.ipynb` (Visualization Studio)**
   - Jupyter Notebook for generating custom publication-ready gradient plots, styled boxplots, and figures matching book and journal requirements.

---

## Empirical Results & Pipeline Summary

Based on empirical runs on the complete benchmark datasets (`outputs_integrated_full/pipeline_summary.json`):

- **Dataset Scale:**
  - **525,814** preprocessed customer reviews.
  - **178,405** standardized sales transactions across channels.
- **Sentiment Model Performance (5-Run Stratified Mean ± Std):**
  - **Accuracy:** `95.63%` ($\pm 0.10\%$)
  - **Macro Precision:** `93.88%` ($\pm 0.16\%$)
  - **Macro Recall:** `88.99%` ($\pm 0.26\%$)
  - **Macro F1-Score:** `91.21%` ($\pm 0.21\%$)
- **Per-Class Metrics:**
  - **Positive Class:** Precision `96.27%`, Recall `98.64%`, F1 `97.44%` (Support: ~66,567 per test split)
  - **Negative Class:** Precision `91.49%`, Recall `79.35%`, F1 `84.99%` (Support: ~12,306 per test split)
- **Multi-Channel Pricing:** Tracks MRP dispersion across 10 marketplace tiers: *Amazon, Amazon FBA, Flipkart, Myntra, Ajio, Limeroad, Paytm, Snapdeal, MRP Old, Final MRP Old*.

---

## Quick Start

### 1. Installation

Clone the repository and install the dependencies:

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/vanlalruata/ecommerce_sentimental_analysis_prediction.git
cd ecommerce_sentimental_analysis_prediction

pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 2. Dataset Setup

Download the required datasets:

1. [Amazon Product Reviews Dataset (Kaggle)](https://www.kaggle.com/code/tunahandemir173/sentiment-analysis-on-amazon-product-reviews/input)
2. [E-commerce Sales Report Dataset (Kaggle)](https://www.kaggle.com/code/kartikmanekar007/e-commerce-sales-dataset/input)

### 3. Run Integrated Pipeline

Execute the complete integrated pipeline:

```bash
python amazon_ecom_analysis_integrated_full.py \
  --reviews "path/to/Reviews.csv" \
  --sales_folder "path/to/sales_csv_folder" \
  --sentiment_runs 5 \
  --max_tfidf_features 50000 \
  --topn_skus 20
```

Minimal required command:

```bash
python amazon_ecom_analysis_integrated_full.py \
  --reviews "path/to/Reviews.csv" \
  --sales_folder "path/to/sales_csv_folder"
```

---

## Command-Line Arguments

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--reviews` | `str` | *(Required)* | Filepath to Amazon `Reviews.csv` |
| `--sales_folder` | `str` | *(Required)* | Directory containing one or more e-commerce sales CSVs |
| `--sentiment_runs` | `int` | `5` | Number of repeated train/test runs for empirical stability |
| `--max_tfidf_features` | `int` | `50000` | Maximum vocabulary size for TF-IDF vectorization |
| `--topn_skus` | `int` | `20` | Number of top revenue SKUs to extract for detailed per-order statistics |

---

## Output Artifacts Catalog

Executing the integrated pipeline outputs all artifacts to `outputs_integrated_full/`:

### Processed Datasets & Statistical Tables

- `merged_sales.csv`: Concatenated, standardized multi-channel sales transactions.
- `reviews_preprocessed.csv`: Cleaned customer reviews with sentiment labels and normalized text.
- `asin_sku_sentiment_summary.csv`: Joined ASIN↔SKU cross-reference table with aggregated review sentiment scores.
- `sentiment_classification_report.csv`: Detailed classification precision/recall/F1 metrics for the final run.
- `sentiment_experiment_stats.csv`: Mean and standard deviation across repeated randomized splits.
- `sentiment_experiment_per_class_stats.csv`: Breakdown of per-class stability across experiment iterations.
- `channel_profitability.csv`: Mean profitability metrics across logistics and store channels.
- `mrp_dispersion_stats.csv`: Descriptive statistics (mean, std, quartiles) of pricing across marketplaces.
- `top_skus.csv` & `top_skus_stats.csv`: Top SKUs ranked by revenue with per-order quantity and revenue stats.
- `sales_by_region.csv` & `sales_region_stats.csv`: Revenue, order volume, and variance aggregated across Indian states.
- `pipeline_summary.json`: Comprehensive machine-readable summary of the entire execution.

### Visualizations & Saved Models

- `confusion_matrix.pdf` / `.svg`: Heatmap of true vs. predicted sentiment classes.
- `channel_profitability.pdf` / `.svg`: Comparative bar plot of channel profit metrics.
- `mrp_dispersion.pdf` / `.svg`: Multi-channel boxplots illustrating price distribution and outlier spreads.
- `top_skus.pdf` / `.svg`: Bar chart of top revenue-driving SKUs.
- `tfidf_vectorizer.joblib`: Serialized TF-IDF feature transformer.
- `tfidf_lr_model.joblib`: Serialized Logistic Regression sentiment classifier.

---

## Model Inference Example

To score new, unseen customer reviews using the trained artifacts:

### Processed Datasets & Statistical Tables

- `merged_sales.csv`: Concatenated, standardized multi-channel sales transactions.
- `reviews_preprocessed.csv`: Cleaned customer reviews with sentiment labels and normalized text.
- `asin_sku_sentiment_summary.csv`: Joined ASIN↔SKU cross-reference table with aggregated review sentiment scores.
- `sentiment_classification_report.csv`: Detailed classification precision/recall/F1 metrics for the final run.
- `sentiment_experiment_stats.csv`: Mean and standard deviation across repeated randomized splits.
- `sentiment_experiment_per_class_stats.csv`: Breakdown of per-class stability across experiment iterations.
- `channel_profitability.csv`: Mean profitability metrics across logistics and store channels.
- `mrp_dispersion_stats.csv`: Descriptive statistics (mean, std, quartiles) of pricing across marketplaces.
- `top_skus.csv` & `top_skus_stats.csv`: Top SKUs ranked by revenue with per-order quantity and revenue stats.
- `sales_by_region.csv` & `sales_region_stats.csv`: Revenue, order volume, and variance aggregated across Indian states.
- `pipeline_summary.json`: Comprehensive machine-readable summary of the entire execution.

### Visualizations & Saved Models

- `confusion_matrix.pdf` / `.svg`: Heatmap of true vs. predicted sentiment classes.
- `channel_profitability.pdf` / `.svg`: Comparative bar plot of channel profit metrics.
- `mrp_dispersion.pdf` / `.svg`: Multi-channel boxplots illustrating price distribution and outlier spreads.
- `top_skus.pdf` / `.svg`: Bar chart of top revenue-driving SKUs.
- `tfidf_vectorizer.joblib`: Serialized TF-IDF feature transformer.
- `tfidf_lr_model.joblib`: Serialized Logistic Regression sentiment classifier.

---

## Model Inference Example

To score new, unseen customer reviews using the trained artifacts:

```python
import joblib
from pathlib import Path

# Load serialized pipeline artifacts
model_dir = Path("outputs_integrated_full")
vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
classifier = joblib.load(model_dir / "tfidf_lr_model.joblib")

# Sample review inputs
reviews = [
    "Outstanding quality, fast delivery, and fits perfectly!",
    "Poor material, arrived damaged, and customer service refused return.",
    "Decent product for the price, works as expected."
]

# Transform and predict
X_transformed = vectorizer.transform(reviews)
predictions = classifier.predict(X_transformed)

for review, sentiment in zip(reviews, predictions):
    print(f"[{sentiment.upper()}] {review}")
```

---

## License

This project is open-source under the [MIT License](LICENSE).  
Copyright (c) 2026 Vanlalruata Hnamte.

---

## Acknowledgments

- Development supported by Python scientific libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `joblib`.
- Amazon Product Reviews and Sales Report datasets courtesy of community contributors on Kaggle.
- Theoretical, ethical, and regulatory framing developed in conjunction with *AI Regulation in Industry 5.0* (Springer).
