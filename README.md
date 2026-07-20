# E-commerce Sentiment Analysis and Sales Insights

## Overview
This project provides an end‑to‑end, reproducible pipeline to:
- Train and evaluate a text‑based sentiment classifier on Amazon product reviews (TF‑IDF + Logistic Regression).
- Load and standardize multiple e‑commerce sales CSV exports, then generate exploratory analytics and visualizations (top SKUs, region revenue, MRP dispersion, channel profitability).
- Integrate sales with review aggregates by ASIN to create SKU/ASIN insight summaries.

The primary entry point is `ecommerce_sentimental_analysis_prediction/amazon_ecom_analysis_integrated_full.py`. Running it produces cleaned datasets, plots, reports, and a JSON pipeline summary under `outputs_integrated_full/`.

## Key Features
- Robust CSV loading with safe fallbacks; automatic date parsing.
- Review preprocessing with neutral review filtering and label generation (positive/negative).
- Repeatable sentiment experiments across multiple random splits with mean ± std metrics saved to disk.
- Sales CSV auto‑standardization for common fields (SKU, ASIN, Amount, Qty, regions).
- Channel profitability comparison and MRP dispersion visualization across marketplaces.
- Top SKU and regional revenue breakdowns with per‑order stats.
- Integrated review–sales join on ASIN with SKU mapping.

## Requirements
- Python 3.8+
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

Install via pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Project Structure (relevant parts)
- `ecommerce_sentimental_analysis_prediction/`
  - `amazon_ecom_analysis_integrated_full.py` — main integrated pipeline (sentiment + sales + integration).
  - Other helpers and notebooks may exist (e.g., `tfidf_lr_sentiment.py`, `dataset_inspector.py`, `plotting.ipynb`) but are not required for the main run.

## Input Data
- Reviews CSV: Expected Amazon review schema (typical columns include `Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text`).
  - The script tolerates subsets; key columns used are `Text`, `Score`, `Time`.
  - Unix epoch seconds in `Time` are auto‑parsed when numeric.
- Sales folder: A directory containing one or more CSV files. The script auto‑detects columns and standardizes names.
  - Heuristics map columns to: `SKU`, `ASIN`, `Amount`, `Qty`, `Date`, shipping `region` (derived from `ship-state` or `ship-country`).
  - Amount/QTY columns are cleaned to numeric (handles currency symbols, commas, etc.).

## Outputs
All artifacts are written to `outputs_integrated_full/`:
- Data/export CSVs
  - `merged_sales.csv` — concatenated, standardized sales data.
  - `reviews_preprocessed.csv` — cleaned reviews with labels and normalized text.
  - `sentiment_classification_report.csv` — detailed classification report (last run).
  - `sentiment_experiment_stats.csv` — mean ± std for macro metrics across runs.
  - `sentiment_experiment_per_class_stats.csv` — per‑class aggregated stats.
  - `channel_profitability.csv` — mean of detected profitability‑like columns.
  - `mrp_dispersion_stats.csv` — descriptive stats of MRP‑like fields by channel/store.
  - `top_skus.csv` — top SKUs by revenue.
  - `top_skus_stats.csv` — per‑order revenue/qty stats for top SKUs.
  - `sales_by_region.csv` — revenue aggregated by region.
  - `sales_region_stats.csv` — per‑region mean/std and counts.
  - `sales_overall_stats.csv` — overall amount/qty stats.
  - `asin_sku_sentiment_summary.csv` — ASIN↔SKU map joined with review aggregates.
  - `pipeline_summary.json` — compact summary of the entire run.
- Plots and reports
  - `confusion_matrix.pdf` — confusion matrix for the last sentiment run.
  - `channel_profitability.pdf` — bar chart of mean profitability across detected channels.
  - `mrp_dispersion.pdf` — boxplot of MRP dispersion across channels.
  - `top_skus.pdf` — bar chart of top SKUs by revenue.
  - `sales_experiment_report.txt` — text summary of core sales stats.
  - `sentiment_experiment_report.txt` — text summary with mean ± std metrics and per‑class table.

Note: Some comments in the script mention `.svg`; the current implementation saves `.pdf` for figures.

## How It Works (high level)
1. Reviews
   - `load_reviews()` reads and parses review timestamps.
   - `preprocess_reviews()` normalizes text, filters out neutral scores (Score == 3), and creates `sent_label` (positive ≥4, negative ≤2).
   - `train_sentiment_tfidf_lr_runs()` runs N experiments (default 5): TF‑IDF (1–2 grams, up to 50k features) + Logistic Regression. It records accuracy, precision, recall, f1 (macro) and per‑class stats; saves a confusion matrix and model artifacts (`tfidf_vectorizer.joblib`, `tfidf_lr_model.joblib`) from the last run.
2. Sales
   - `load_and_standardize_sales()` loads all CSVs in the sales folder, auto‑maps common column names, parses dates, cleans `Amount` and `Qty`, derives `region`, and saves `merged_sales.csv`.
   - `compute_channel_profitability()` scans for columns named like `shiprocket`/`increff` and computes mean values, saving CSV and a bar chart if found.
   - `mrp_dispersion()` identifies MRP‑like columns (mentions of `mrp`, `amazon mrp`, or marketplace names) and produces a boxplot and stats.
   - `top_skus_and_region()` exports top SKUs by revenue and region revenue charts.
   - `sales_experiments()` writes overall amount/qty stats, per‑region stats, top‑SKU per‑order stats, and a textual summary.
3. Integration
   - `integrate_reviews_sales_sentiment()` joins review aggregates (count, mean score) with sales ASIN↔SKU mapping and exports `asin_sku_sentiment_summary.csv`.
4. Summary
   - `pipeline_summary.json` aggregates key counts and experiment summaries.

## Quick Start
Example command (Windows paths shown):
```bash
python ecommerce_sentimental_analysis_prediction/amazon_ecom_analysis_integrated_full.py \
  --reviews "H:/Datasets/ecommerce/ecommerce_review/Reviews.csv" \
  --sales_folder "H:/Datasets/ecommerce/ecommerce_sale" \
  --sentiment_runs 5 \
  --max_tfidf_features 50000 \
  --topn_skus 20
```
Minimal required arguments:
```bash
python ecommerce_sentimental_analysis_prediction/amazon_ecom_analysis_integrated_full.py \
  --reviews <path/to/Reviews.csv> \
  --sales_folder <path/to/sales_csv_folder>
```

## Command‑line Arguments
- `--reviews` (str, required): Path to Amazon `Reviews.csv`.
- `--sales_folder` (str, required): Folder containing one or more sales CSV files.
- `--sentiment_runs` (int, default=5): Number of repeated train/test runs for sentiment evaluation.
- `--max_tfidf_features` (int, default=50000): TF‑IDF vocabulary cap.
- `--topn_skus` (int, default=20): Number of top SKUs to include in per‑SKU stats.

## Data Assumptions and Heuristics
- Column auto‑mapping is heuristic. If your CSVs use unusual headers, consider renaming columns to include recognizable tokens (e.g., `sku`, `asin`, `amount`, `qty`, `date`, `ship-state`, `ship-country`).
- Amount and MRP fields are cleaned by removing non‑numeric characters and coercing to numeric.
- Regions are derived from shipping state or country; if both are absent, region aggregation will be skipped.
- For sentiment, neutral reviews (Score == 3) are dropped to create a binary problem (positive vs negative).

## Reusing the Trained Model
After a run, you can load the vectorizer and model to score new text:
```python
import joblib
from pathlib import Path

out = Path("outputs_integrated_full")
vect = joblib.load(out/"tfidf_vectorizer.joblib")
clf = joblib.load(out/"tfidf_lr_model.joblib")

texts = [
    "Great fit and quality, highly recommend!",
    "Terrible stitching and arrived broken.",
]
X = vect.transform(texts)
preds = clf.predict(X)
print(list(zip(texts, preds)))
```

## Troubleshooting
- Empty outputs or many NaNs:
  - Check that `--sales_folder` contains CSV files and columns map correctly. Inspect `merged_sales.csv` to confirm mappings.
- No channel profitability output:
  - Ensure your dataset contains columns with names including `shiprocket` or `increff`.
- No MRP plots:
  - Ensure sales CSVs have MRP‑like columns (`mrp`, marketplace names like `flipkart`, `myntra`, `amazon mrp`, etc.).
- Poor sentiment accuracy:
  - Increase `--max_tfidf_features`, try more runs via `--sentiment_runs`, or inspect `reviews_preprocessed.csv` for class balance and text quality.

## Reproducibility Notes
- Random seeds for train/test splits are deterministic per run index (base 42 + i), enabling comparable repeated runs while still sampling different splits.
- Figures are saved in `.pdf` format for quality and portability.

## License
Specify your project license here (e.g., MIT, Apache‑2.0). If omitted, all rights reserved by default.

## Acknowledgments
- Built with `scikit-learn`, `pandas`, `numpy`, `seaborn`, and `matplotlib`.
- Amazon Reviews dataset format inspired by common Kaggle distributions of `Reviews.csv`.

## Maintainers
- Add maintainer names/contacts here.

## Changelog
- Initial integrated pipeline: sentiment + sales standardization + integration + reporting (current version).
