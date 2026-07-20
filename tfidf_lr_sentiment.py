#!/usr/bin/env python3
"""
tfidf_lr_sentiment.py
Term Frequency-Inverse Document Frequency (TF-IDF)
TF–IDF + Logistic Regression sentiment pipeline:
- Loads Reviews.csv, preprocesses text and labels
- Trains TF–IDF vectorizer + Logistic Regression classifier
- Evaluates, saves classification report and confusion matrix (SVG)
- Saves model and vectorizer (joblib)
- Produces predictions CSV and optional ASIN-level sentiment aggregates (joined with merged_sales.csv)

Usage:
python tfidf_lr_sentiment.py --reviews "path/to/Reviews.csv" --output "outputs_tfidf" --merged_sales "path/to/merged_sales.csv"

Requirements:
pip install pandas numpy scikit-learn matplotlib seaborn joblib
"""
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set(style="whitegrid")
RANDOM_SEED = 42

def load_reviews(path):
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)
    return df

def preprocess_reviews(df, drop_neutral=True):
    # Drop rows missing core fields
    if 'Text' not in df.columns or 'Score' not in df.columns:
        raise ValueError("Reviews CSV must contain 'Text' and 'Score' columns.")
    df = df.dropna(subset=['Text', 'Score']).copy()
    # Basic text normalization
    df['text_clean'] = df['Text'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
    # Map numeric score to sentiment label
    df['label'] = df['Score'].apply(lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
    if drop_neutral:
        df = df[df['label'] != 'neutral']
    # Shuffle reproducibly
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return df

def train_tfidf_lr(df, max_features=50000, ngram_range=(1,2), test_size=0.15, class_weight=None):
    X = df['text_clean'].values
    y = df['label'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)
    vect = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    Xtr_tfidf = vect.fit_transform(Xtr)
    Xte_tfidf = vect.transform(Xte)
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight)
    clf.fit(Xtr_tfidf, ytr)
    preds = clf.predict(Xte_tfidf)
    probs = clf.predict_proba(Xte_tfidf) if hasattr(clf, "predict_proba") else None
    report = classification_report(yte, preds, output_dict=True)
    cm = confusion_matrix(yte, preds, labels=['positive','negative'])
    return {
        "vect": vect, "clf": clf,
        "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte,
        "preds": preds, "probs": probs,
        "report": report, "confusion_matrix": cm
    }

def save_report_csv(report_dict, out_csv):
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(out_csv, index=True)

def plot_confusion_matrix_svg(cm, labels, out_svg):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)

def predict_full_and_save(df, vect, clf, out_csv):
    texts = df['text_clean'].astype(str).values
    X_tfidf = vect.transform(texts)
    preds = clf.predict(X_tfidf)
    probs = clf.predict_proba(X_tfidf) if hasattr(clf, "predict_proba") else None
    out_df = df.copy()
    out_df['pred_label'] = preds
    if probs is not None:
        # find index of positive class
        classes = list(clf.classes_)
        if 'positive' in classes:
            idx = classes.index('positive')
            out_df['pred_prob_positive'] = probs[:, idx]
        else:
            # store probabilities for first class as fallback
            out_df['pred_prob_0'] = probs[:, 0]
    out_df.to_csv(out_csv, index=False)
    return out_df

def aggregate_asin(predictions_csv, merged_sales_csv=None, out_csv=None):
    df = pd.read_csv(predictions_csv, low_memory=False)
    # require ProductId/ASIN column
    if 'ProductId' not in df.columns:
        raise ValueError("Predictions file must contain 'ProductId' column (ASIN).")
    # ensure pred_prob_positive exists (if not, compute approximate)
    if 'pred_prob_positive' not in df.columns and 'pred_label' in df.columns:
        df['pred_prob_positive'] = (df['pred_label'] == 'positive').astype(float)
    agg = df.groupby('ProductId').agg(
        review_count=('ProductId','count'),
        mean_prob_positive=('pred_prob_positive','mean'),
        positive_share=('pred_label', lambda s: (s=='positive').mean())
    ).reset_index().rename(columns={'ProductId':'ASIN'})
    if merged_sales_csv and Path(merged_sales_csv).exists():
        ms = pd.read_csv(merged_sales_csv, low_memory=False)
        if 'ASIN' in ms.columns and 'SKU' in ms.columns:
            map_df = ms[['ASIN','SKU']].dropna().drop_duplicates()
            agg = agg.merge(map_df, left_on='ASIN', right_on='ASIN', how='left')
    if out_csv:
        agg.to_csv(out_csv, index=False)
    return agg

def save_artifacts(vect, clf, outdir):
    joblib.dump(vect, outdir/"tfidf_vectorizer.joblib")
    joblib.dump(clf, outdir/"tfidf_lr_model.joblib")

def main(args):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    # Load and preprocess
    df_raw = load_reviews(args.reviews)
    print(f"Loaded reviews: {df_raw.shape}")
    df_pp = preprocess_reviews(df_raw, drop_neutral=not args.keep_neutral)
    print(f"Preprocessed reviews: {df_pp.shape}")
    # Train
    res = train_tfidf_lr(df_pp, max_features=args.max_features, ngram_range=(1,2), test_size=args.test_size, class_weight=None)
    # Save artifacts
    save_artifacts(res['vect'], res['clf'], outdir)
    # Save report and confusion matrix
    save_report_csv(res['report'], outdir/"sentiment_classification_report.csv")
    plot_confusion_matrix_svg(res['confusion_matrix'], ['positive','negative'], outdir/"confusion_matrix.pdf")
    # Full predictions and save
    preds_df = predict_full_and_save(df_pp, res['vect'], res['clf'], outdir/"predictions.csv")
    # Aggregate per ASIN and optionally join with merged sales
    agg = aggregate_asin(outdir/"predictions.csv", merged_sales_csv=args.merged_sales, out_csv=(outdir/"asin_sentiment_agg.csv"))
    # Save summary
    summary = {
        "num_reviews": int(df_pp.shape[0]),
        "num_asin_aggregates": int(agg.shape[0]) if agg is not None else 0
    }
    Path(outdir/"pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    print("Outputs saved to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, help="Path to Reviews.csv")
    parser.add_argument("--output", default="outputs_tfidf", help="Output folder")
    parser.add_argument("--merged_sales", default=None, help="Path to merged_sales.csv for ASIN->SKU join (optional)")
    parser.add_argument("--max_features", type=int, default=50000, help="TF-IDF max features")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--keep_neutral", action="store_true", help="Keep neutral (score==3) reviews")
    args = parser.parse_args()
    main(args)
