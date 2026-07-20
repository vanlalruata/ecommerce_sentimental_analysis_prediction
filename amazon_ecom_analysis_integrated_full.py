#!/usr/bin/env python3

# amazon_ecom_analysis_integrated_full.py
#
# Full integrated analysis pipeline for the provided e-commerce datasets.
#
# Usage:
# python amazon_ecom_analysis_integrated_full.py --reviews "path/to/Reviews.csv" --sales_folder "path/to/sales_folder/"
#
# In my case
# python amazon_ecom_analysis_integrated_full.py --reviews "H:/Datasets/ecommerce/ecommerce_review/Reviews.csv" --sales_folder "H:/Datasets/ecommerce/ecommerce_sale"
# Outputs:
# - outputs_integrated_full/
#     - merged_sales.csv
#     - reviews_preprocessed.csv
#     - sentiment_classification_report.csv
#     - confusion_matrix.svg
#     - tfidf_vectorizer.joblib
#     - tfidf_lr_model.joblib
#     - sentiment_experiment_stats.csv
#     - sentiment_experiment_report.txt
#     - channel_profitability.csv
#     - channel_profitability.svg
#     - mrp_dispersion.svg
#     - mrp_dispersion_stats.csv
#     - top_skus.csv
#     - top_skus.svg
#     - top_skus_stats.csv
#     - sales_by_region.csv
#     - sales_region_stats.csv
#     - sales_overall_stats.csv
#     - pipeline_summary.json
#     - asin_sku_sentiment_summary.csv

import argparse, json, re
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib
sns.set(style="whitegrid")

OUT = Path("outputs_integrated_full")
OUT.mkdir(exist_ok=True)

def safe_read_csv(path, **kwargs):
    path = Path(path)
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return pd.DataFrame()

def detect_and_parse_date(df, candidates):
    for c in candidates:
        if c in df.columns:
            try:
                df['date_parsed'] = pd.to_datetime(df[c], errors='coerce', dayfirst=False)
                if df['date_parsed'].isna().mean() > 0.3:
                    df['date_parsed'] = pd.to_datetime(df[c], errors='coerce', dayfirst=True)
                return df
            except Exception:
                continue
    df['date_parsed'] = pd.NaT
    return df

def clean_amount_series(s):

    # Cleans an amount-like column or multiple columns merged under 'Amount'.
    # Handles both Series and DataFrame inputs, flattening nested columns if needed.

    import pandas as pd
    import numpy as np

    # Case 1: If input is a DataFrame (multiple "Amount" columns)
    if isinstance(s, pd.DataFrame):
        all_cols = []
        for col in s.columns:
            col_data = s[col]
            # If this column is itself a DataFrame (nested), flatten it
            if isinstance(col_data, pd.DataFrame):
                for subcol in col_data.columns:
                    sub_data = pd.to_numeric(
                        col_data[subcol].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True),
                        errors='coerce'
                    )
                    all_cols.append(sub_data)
            else:
                sub_data = pd.to_numeric(
                    col_data.astype(str).str.replace(r'[^0-9\.\-]', '', regex=True),
                    errors='coerce'
                )
                all_cols.append(sub_data)
        if not all_cols:
            return pd.Series(dtype=float)
        # Combine all numeric columns by row-wise mean, ignoring NaN
        combined = pd.concat(all_cols, axis=1)
        return combined.mean(axis=1, skipna=True)

    # Case 2: If it's a Series
    elif isinstance(s, pd.Series):
        if s.dtype == object:
            s = s.astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        return pd.to_numeric(s, errors='coerce')

    # Case 3: Unexpected type (return empty numeric)
    else:
        return pd.Series(dtype=float)

def load_reviews(path):
    df = safe_read_csv(path)
    if df.empty:
        return df
    expected = ['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time','Summary','Text']
    df = df[[c for c in expected if c in df.columns]]
    if 'Time' in df.columns and np.issubdtype(df['Time'].dtype, np.number):
        try:
            df['time_parsed'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
        except:
            df['time_parsed'] = pd.to_datetime(df['Time'], errors='coerce')
    else:
        df['time_parsed'] = pd.to_datetime(df['Time'], errors='coerce') if 'Time' in df.columns else pd.NaT
    return df

def preprocess_reviews(df, drop_neutral=True):
    df = df.dropna(subset=['Text','Score']).copy()
    df['text_clean'] = df['Text'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
    if drop_neutral and 'Score' in df.columns:
        df = df[df['Score'] != 3]
    df['sent_label'] = df['Score'].apply(lambda x: 'positive' if x >=4 else ('negative' if x<=2 else 'neutral'))
    df = df[df['sent_label']!='neutral'].sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(OUT/"reviews_preprocessed.csv", index=False)
    return df

def train_sentiment_tfidf_lr_runs(df, runs=5, max_features=50000):

    # Train TF-IDF + Logistic Regression multiple times with different train/test splits
    # and report mean ± std for accuracy, precision, recall, f1 (macro) and per-class metrics aggregated.
    # Saves:
    #   - sentiment_experiment_stats.csv
    #   - sentiment_experiment_report.txt
    #   - sentiment_classification_report.csv (from last run)
    #   - confusion_matrix.svg (from last run)
    #   - tfidf_vectorizer.joblib (from last run)
    #   - tfidf_lr_model.joblib (from last run)

    if df.empty:
        print("No review data provided to train_sentiment_tfidf_lr_runs.")
        return {}

    X = df['text_clean']
    y = df['sent_label']

    # containers for aggregated metrics
    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': []
    }
    # also store per-class metrics if needed
    per_class = {}

    for i in range(runs):
        rs = 42 + i  # deterministic seeds
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=rs, stratify=y)
        vect = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
        Xtr_tfidf = vect.fit_transform(Xtr)
        Xte_tfidf = vect.transform(Xte)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr_tfidf, ytr)
        preds = clf.predict(Xte_tfidf)

        # compute core metrics
        acc = accuracy_score(yte, preds)
        precision, recall, f1, support = precision_recall_fscore_support(yte, preds, average='macro', zero_division=0)
        metrics['accuracy'].append(acc)
        metrics['precision_macro'].append(precision)
        metrics['recall_macro'].append(recall)
        metrics['f1_macro'].append(f1)

        # per-class metrics for this run
        for cls in ['positive', 'negative']:  # force consistent order
            if cls not in yte.values and cls not in preds:
                # class absent in both actual and predicted sets
                continue
            try:
                p, r, f, s = precision_recall_fscore_support(
                    yte, preds, labels=[cls], average=None, zero_division=0
                )
                per_class.setdefault(cls, {'precision': [], 'recall': [], 'f1': [], 'support': []})
                per_class[cls]['precision'].append(float(p[0]) if len(p) > 0 else 0.0)
                per_class[cls]['recall'].append(float(r[0]) if len(r) > 0 else 0.0)
                per_class[cls]['f1'].append(float(f[0]) if len(f) > 0 else 0.0)
                per_class[cls]['support'].append(int(s[0]) if len(s) > 0 else 0)
            except Exception:
                # handle cases where class doesn't exist in this run
                per_class.setdefault(cls, {'precision': [], 'recall': [], 'f1': [], 'support': []})
                per_class[cls]['precision'].append(0.0)
                per_class[cls]['recall'].append(0.0)
                per_class[cls]['f1'].append(0.0)
                per_class[cls]['support'].append(0)

        # If last run, save artifacts for reuse
        if i == runs - 1:
            # detailed classification report
            report = classification_report(yte, preds, output_dict=True, zero_division=0)
            pd.DataFrame(report).transpose().to_csv(OUT/"sentiment_classification_report.csv")
            # confusion matrix plot
            cm = confusion_matrix(yte, preds, labels=sorted(list(set(y))))
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=sorted(list(set(y))), yticklabels=sorted(list(set(y))), ax=ax)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title('Confusion Matrix (TF-IDF + LR) - last run')
            fig.savefig(OUT/"confusion_matrix.pdf", bbox_inches='tight'); plt.close(fig)
            # save vectorizer and model from last run
            joblib.dump(vect, OUT/"tfidf_vectorizer.joblib")
            joblib.dump(clf, OUT/"tfidf_lr_model.joblib")

    # compute mean ± std for macro metrics
    stats = []
    for mname, values in metrics.items():
        arr = np.array(values)
        stats.append({'metric': mname, 'mean': float(arr.mean()), 'std': float(arr.std()), 'values': values})

    # prepare per-class aggregated stats
    per_class_stats = []
    for cls, d in per_class.items():
        per_class_stats.append({
            'class': cls,
            'precision_mean': float(np.mean(d['precision'])),
            'precision_std': float(np.std(d['precision'])),
            'recall_mean': float(np.mean(d['recall'])),
            'recall_std': float(np.std(d['recall'])),
            'f1_mean': float(np.mean(d['f1'])),
            'f1_std': float(np.std(d['f1'])),
            'support_mean': float(np.mean(d['support'])),
            'support_std': float(np.std(d['support'])),
        })

    # Save results to CSV and text report
    stats_df = pd.DataFrame(stats).set_index('metric')
    stats_df.to_csv(OUT/"sentiment_experiment_stats.csv")

    per_class_df = pd.DataFrame(per_class_stats).set_index('class')
    per_class_df.to_csv(OUT/"sentiment_experiment_per_class_stats.csv")

    # write human readable report
    report_lines = []
    report_lines.append(f"Sentiment experiment summary (runs={runs}):\n")
    for _, row in stats_df.reset_index().iterrows():
        report_lines.append(f"{row['metric']}: {row['mean']:.6f} ± {row['std']:.6f}")
    report_lines.append("\nPer-class stats:\n")
    report_lines.append(per_class_df.to_string())
    with open(OUT/"sentiment_experiment_report.txt", "w") as fh:
        fh.write("\n".join(report_lines))

    print("Saved sentiment experiment stats and artifacts to", OUT)

    # return a compact summary dict
    summary = {r['metric']: {'mean': r['mean'], 'std': r['std']} for r in stats}
    # summary['per_class'] = {row['class']: {k: row[k] for k in row if k!='class'} for _, row in per_class_df.reset_index().iterrows()}
    summary['per_class'] = {}
    for _, row in per_class_df.reset_index().iterrows():
        # handle both 'class' column and unnamed index cases
        cls_name = row.get('class', row.get(per_class_df.index.name, None))
        if cls_name is None:
            continue
        summary['per_class'][cls_name] = {k: row[k] for k in row.index if k not in ['class', per_class_df.index.name]}

    return summary

def load_and_standardize_sales(sales_folder):
    p = Path(sales_folder)
    files = {f.name.lower(): str(f) for f in p.glob("*.csv")}
    print("Detected sales files:", list(files.keys()))
    df_list = []
    for name, path in files.items():
        df = safe_read_csv(path)
        if df.empty:
            continue
        df['source_file'] = name
        df_list.append(df)
    if not df_list:
        return pd.DataFrame()
    big = pd.concat(df_list, ignore_index=True, sort=False)
    colmap = {}
    for c in big.columns:
        lc = c.lower().strip()
        if lc in ['sku','sku code','sku_code'] or 'sku code' in lc or lc.startswith('sku '):
            colmap[c] = 'SKU'
        if 'asin' in lc:
            colmap[c] = 'ASIN'
        if 'amount' == lc or lc.startswith('amount') or 'gross' in lc or 'amt' in lc or 'rate' in lc:
            colmap[c] = 'Amount'
        if 'qty' in lc or 'pcs' in lc:
            colmap[c] = 'Qty'
        if 'ship-state' in lc or 'ship_state' in lc or 'ship state' in lc:
            colmap[c] = 'ship-state'
        if 'ship-country' in lc or lc=='country' or 'ship country' in lc:
            colmap[c] = 'ship-country'
        if 'date' in lc or 'time' in lc:
            colmap[c] = 'Date'
    big = big.rename(columns=colmap)
    big = detect_and_parse_date(big, ['Date','DATE','date_parsed'])
    # Remove duplicate column names (e.g. multiple 'Amount' or 'Qty')
    if big.columns.duplicated().any():
        big = big.loc[:, ~big.columns.duplicated()]
    if 'Amount' in big.columns:
        big['Amount'] = clean_amount_series(big['Amount'])
    if 'Qty' in big.columns:
        big['Qty'] = pd.to_numeric(big['Qty'], errors='coerce')
    if 'ship-state' in big.columns:
        big['region'] = big['ship-state'].astype(str)
    elif 'ship-country' in big.columns:
        big['region'] = big['ship-country'].astype(str)
    else:
        big['region'] = None
    big.to_csv(OUT/"merged_sales.csv", index=False)
    return big

def compute_channel_profitability(big):
    profit_cols = [c for c in big.columns if 'shiprocket' in c.lower() or 'increff' in c.lower() or 'shiprocket'==c.lower() or 'increff'==c.lower()]
    results = {}
    for c in profit_cols:
        try:
            s = pd.to_numeric(big[c].astype(str).str.replace(r'[^0-9\\.-]', '', regex=True), errors='coerce')
            results[c] = float(s.mean())
        except Exception:
            continue
    if results:
        pd.DataFrame.from_dict(results, orient='index', columns=['mean_profit']).to_csv(OUT/"channel_profitability.csv")
        fig, ax = plt.subplots(figsize=(6,3))
        pd.Series(results).sort_values().plot(kind='bar', ax=ax)
        ax.set_ylabel('Mean profit'); ax.set_title('Channel mean profit comparison')
        fig.savefig(OUT/"channel_profitability.pdf", bbox_inches='tight'); plt.close(fig)
    else:
        print("No channel profit columns detected in merged data.")
    return results

def mrp_dispersion(big):
    mrp_cols = [c for c in big.columns if 'mrp' in c.lower() or 'amazon mrp' in c.lower() or 'ajio' in c.lower() or 'flipkart' in c.lower() or 'myntra' in c.lower() or 'paytm' in c.lower() or 'limeroad' in c.lower()]
    if not mrp_cols:
        print("No MRP-like columns detected.")
        return None
    tmp = big[mrp_cols].copy()
    for c in mrp_cols:
        tmp[c] = pd.to_numeric(tmp[c].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True), errors='coerce')
    tmp = tmp.melt(var_name='store', value_name='mrp').dropna()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=tmp, x='store', y='mrp', ax=ax)
    ax.set_title('MRP dispersion across channels')
    fig.savefig(OUT/"mrp_dispersion.pdf", bbox_inches='tight'); plt.close(fig)
    tmp.groupby('store')['mrp'].describe().to_csv(OUT/"mrp_dispersion_stats.csv")
    return mrp_cols

def top_skus_and_region(big, topn=20):
    if 'SKU' in big.columns and 'Amount' in big.columns:
        skuagg = big.groupby('SKU').agg(total_revenue=('Amount','sum'), total_qty=('Qty','sum')).reset_index().sort_values('total_revenue', ascending=False)
        skuagg.head(topn).to_csv(OUT/"top_skus.csv", index=False)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(data=skuagg.head(topn), x='total_revenue', y='SKU', ax=ax)
        ax.set_title("Top SKUs by revenue"); fig.savefig(OUT/"top_skus.pdf", bbox_inches='tight'); plt.close(fig)
    if 'region' in big.columns and big['region'].notna().sum()>0:
        agg = big.groupby('region').agg(revenue=('Amount','sum')).reset_index().sort_values('revenue', ascending=False)
        agg.to_csv(OUT/"sales_by_region.csv", index=False)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(data=agg.head(20), x='revenue', y='region', ax=ax)
        ax.set_title('Top 20 regions by revenue'); fig.savefig(OUT/"sales_by_region.pdf", bbox_inches='tight'); plt.close(fig)
    return True

def sales_experiments(big, topn=20):
    """
    Compute mean ± std experiments on sales data:
      - overall Amount and Qty mean±std
      - per-region revenue mean±std (for regions with enough data)
      - top-N SKU revenue & qty stats
    Saves CSVs and a text report.
    """
    if big is None or big.empty:
        print("No sales data for experiments.")
        return {}

    reports = {}

    # Overall stats
    overall = {}
    if 'Amount' in big.columns:
        overall['amount_mean'] = float(big['Amount'].mean())
        overall['amount_std'] = float(big['Amount'].std())
    if 'Qty' in big.columns:
        overall['qty_mean'] = float(big['Qty'].mean())
        overall['qty_std'] = float(big['Qty'].std())

    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(OUT/"sales_overall_stats.csv", index=False)
    reports['overall'] = overall

    # Region stats
    region_stats = {}
    if 'region' in big.columns and big['region'].notna().sum() > 0:
        region_grp = big.groupby('region').agg(revenue=('Amount','sum'), count_orders=('Amount','count')).reset_index()
        # For per-region revenue mean and std across orders in each region, compute grouped stats
        def per_region_stats(g):
            return pd.Series({
                'revenue_sum': g['Amount'].sum(),
                'revenue_mean': g['Amount'].mean(),
                'revenue_std': g['Amount'].std(),
                'qty_mean': g['Qty'].mean() if 'Qty' in g else np.nan,
                'qty_std': g['Qty'].std() if 'Qty' in g else np.nan,
                'n_orders': len(g)
            })
        reg_stats_df = big.groupby('region').apply(per_region_stats).reset_index().sort_values('revenue_sum', ascending=False)
        reg_stats_df.to_csv(OUT/"sales_region_stats.csv", index=False)
        reports['region_stats'] = reg_stats_df.to_dict(orient='records')
    else:
        print("No region info for sales experiments.")

    # Top SKUs stats
    top_skus_stats = {}
    if 'SKU' in big.columns and 'Amount' in big.columns:
        skuagg = big.groupby('SKU').agg(total_revenue=('Amount','sum'), total_qty=('Qty','sum'), orders=('Amount','count')).reset_index().sort_values('total_revenue', ascending=False)
        top_n = skuagg.head(topn).copy()
        # Calculate mean ± std of revenue per SKU across its orders (if we want per-order stats)
        per_sku_order_stats = []
        for sku in top_n['SKU'].tolist():
            g = big[big['SKU'] == sku]
            revenue_mean = float(g['Amount'].mean())
            revenue_std = float(g['Amount'].std())
            qty_mean = float(g['Qty'].mean()) if 'Qty' in g else np.nan
            qty_std = float(g['Qty'].std()) if 'Qty' in g else np.nan
            per_sku_order_stats.append({
                'SKU': sku,
                'total_revenue': float(top_n[top_n['SKU']==sku]['total_revenue'].values[0]),
                'orders': int(top_n[top_n['SKU']==sku]['orders'].values[0]),
                'revenue_mean_per_order': revenue_mean,
                'revenue_std_per_order': revenue_std,
                'qty_mean_per_order': qty_mean,
                'qty_std_per_order': qty_std
            })
        top_skus_df = pd.DataFrame(per_sku_order_stats)
        top_skus_df.to_csv(OUT/"top_skus_stats.csv", index=False)
        reports['top_skus_stats'] = per_sku_order_stats

    # Human-readable sales experiment report
    lines = []
    lines.append("Sales experiment summary\n")
    lines.append("Overall stats:")
    for k,v in overall.items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append("\nTop SKU stats (saved to top_skus_stats.csv)\n")
    if 'region_stats' in reports:
        lines.append("Per-region stats saved to sales_region_stats.csv (sample):")
        lines.append(reg_stats_df.head(10).to_string(index=False))
    with open(OUT/"sales_experiment_report.txt", "w") as fh:
        fh.write("\n".join(lines))

    print("Saved sales experiment reports to", OUT)
    return reports

def integrate_reviews_sales_sentiment(reviews_df, big):
    if 'ASIN' in big.columns:
        map_asin_sku = big[['ASIN','SKU']].dropna().drop_duplicates()
        agg_rev = reviews_df.groupby('ProductId').agg(review_count=('Id','count'), mean_score=('Score','mean')).reset_index().rename(columns={'ProductId':'ASIN'})
        merged = pd.merge(map_asin_sku, agg_rev, on='ASIN', how='left')
        merged.to_csv(OUT/"asin_sku_sentiment_summary.csv", index=False)
        return merged
    else:
        print("ASIN column not found in merged sales; cannot join reviews by ASIN.")
        return None

def main(args):
    reviews = load_reviews(args.reviews)
    print("Reviews loaded:", reviews.shape)
    reviews_pp = preprocess_reviews(reviews, drop_neutral=True)
    print("Reviews post-processed:", reviews_pp.shape)

    # Run sentiment experiments (multiple runs) and save stats
    sentiment_summary = train_sentiment_tfidf_lr_runs(reviews_pp, runs=args.sentiment_runs, max_features=args.max_tfidf_features)
    print("Sentiment experiments completed.")

    big = load_and_standardize_sales(args.sales_folder)
    print("Merged sales rows:", big.shape[0])

    ch = compute_channel_profitability(big)
    mrp_cols = mrp_dispersion(big)
    top_skus_and_region(big, topn=20)

    # Run sales experiments
    sales_experiment_summary = sales_experiments(big, topn=args.topn_skus)

    merged_asin = integrate_reviews_sales_sentiment(reviews_pp, big)

    # Compose pipeline summary including experiment results (summary values only)
    summary = {
        "reviews_preprocessed": int(reviews_pp.shape[0]),
        "merged_sales_rows": int(big.shape[0]) if hasattr(big, 'shape') else 0,
        "channel_profit": ch,
        "mrp_cols": mrp_cols if mrp_cols else [],
        "sentiment_experiment_summary": sentiment_summary,
        "sales_experiment_summary_overview": {
            "overall": sales_experiment_summary.get('overall', {}),
            "top_skus_count": len(sales_experiment_summary.get('top_skus_stats', [])) if 'top_skus_stats' in sales_experiment_summary else 0
        }
    }
    with open(OUT/"pipeline_summary.json","w") as fh:
        json.dump(summary, fh, indent=2)
    print("Outputs written to", OUT)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, help="path to Reviews.csv")
    parser.add_argument("--sales_folder", required=True, help="path to folder containing sales CSV files")
    parser.add_argument("--sentiment_runs", type=int, default=5, help="number of repeated runs for sentiment experiment (default 5)")
    parser.add_argument("--max_tfidf_features", type=int, default=50000, help="max features for TF-IDF vectorizer")
    parser.add_argument("--topn_skus", type=int, default=20, help="top N SKUs to include in SKU experiments")
    args = parser.parse_args()
    main(args)
