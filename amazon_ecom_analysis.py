#!/usr/bin/env python3

#amazon_ecom_analysis.py
#Run: python amazon_ecom_analysis.py --reviews path/to/amazon_reviews.csv --sales path/to/Sale\ Report.csv
#Outputs saved to `outputs/`:
# - preprocessed CSVs
# - sentiment model metrics CSV
# - confusion matrix SVG
# - top SKU and channel profitability CSVs and plots

import argparse, os, json
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

def load_reviews(path):
    df = pd.read_csv(path, low_memory=False)
    # basic sanity
    print("Reviews columns:", df.columns.tolist())
    return df

def preprocess_reviews(df, score_positive=4, score_negative=2, drop_neutral=True):
    df = df.dropna(subset=['Text','Score'])
    # simple cleanup
    df['text_clean'] = df['Text'].astype(str).str.replace(r'\s+',' ', regex=True).str.strip().str.lower()
    if drop_neutral:
        df = df[df['Score'] != 3]
    df['label'] = np.where(df['Score'] >= score_positive, 'positive', 'negative')
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def train_tfidf_lr(df):
    X = df['text_clean']
    y = df['label']
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.15,random_state=42,stratify=y)
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    Xtr_tfidf = vect.fit_transform(Xtr)
    Xte_tfidf = vect.transform(Xte)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr_tfidf, ytr)
    preds = clf.predict(Xte_tfidf)
    report = classification_report(yte, preds, output_dict=True)
    cm = confusion_matrix(yte, preds, labels=['positive','negative'])
    # save metrics
    pd.DataFrame(report).transpose().to_csv(os.path.join(OUT,"sentiment_classification_report.csv"))
    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive','negative'], yticklabels=['positive','negative'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    fig.savefig(os.path.join(OUT,"confusion_matrix.pdf"), bbox_inches='tight')
    plt.close(fig)
    # save vectorizer and model optionally via joblib
    try:
        import joblib
        joblib.dump(vect, os.path.join(OUT,"tfidf_vectorizer.joblib"))
        joblib.dump(clf, os.path.join(OUT,"tfidf_lr_model.joblib"))
    except Exception as e:
        print("joblib not available or save failed:", e)
    return report

def load_sales(path):
    df = pd.read_csv(path, low_memory=False)
    print("Sales columns:", df.columns.tolist())
    return df

def preprocess_sales(df):
    # parse date
    for c in ['Date','DATE','date']:
        if c in df.columns:
            df['date_parsed'] = pd.to_datetime(df[c], errors='coerce')
            break
    # numeric normalization for amount and qty
    for col in ['Amount','GROSS AMT','Gross Amt','gross_amt','AmountUSD','Amount']:
        if col in df.columns:
            df['amount_num'] = pd.to_numeric(df[col], errors='coerce')
            break
    for col in ['Qty','PCS','quantity','Quantity']:
        if col in df.columns:
            df['qty_num'] = pd.to_numeric(df[col], errors='coerce')
            break
    return df

def channel_profitability(df):
    # assume columns 'Shiprocket' and 'INCREFF' available numerics
    channels = [c for c in ['Shiprocket','INCREFF'] if c in df.columns]
    out = {}
    for c in channels:
        out[c] = df[c].dropna().mean()
    # Save summary
    pd.DataFrame.from_dict(out, orient='index', columns=['mean_profit']).to_csv(os.path.join(OUT,"channel_profitability.csv"))
    return out

def sku_toplist(df, topn=10):
    if 'SKU' in df.columns and 'amount_num' in df.columns:
        skuagg = df.groupby('SKU').agg(total_revenue=('amount_num','sum'), total_qty=('qty_num','sum')).reset_index()
        skuagg = skuagg.sort_values('total_revenue', ascending=False).head(topn)
        skuagg.to_csv(os.path.join(OUT,"top_skus.csv"), index=False)
        # plot top skus
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=skuagg, x='total_revenue', y='SKU', ax=ax)
        ax.set_title("Top SKUs by revenue")
        fig.savefig(os.path.join(OUT,"top_skus.pdf"), bbox_inches='tight')
        plt.close(fig)
        return skuagg
    return None

def geo_sales_map(df):
    # simple aggregate by 'State' or 'state' column if present
    col = None
    for c in ['State','state','STATE','country','Country']:
        if c in df.columns:
            col = c; break
    if col is None:
        print("No geography column detected; skipping map")
        return None
    agg = df.groupby(col).agg(revenue=('amount_num','sum')).reset_index()
    agg.to_csv(os.path.join(OUT,"sales_by_region.csv"), index=False)
    # plot as horizontal bar for now
    fig, ax = plt.subplots(figsize=(8,6))
    top = agg.sort_values('revenue', ascending=False).head(20)
    sns.barplot(data=top, x='revenue', y=col, ax=ax)
    ax.set_title("Top 20 regions by revenue")
    fig.savefig(os.path.join(OUT,"sales_by_region.pdf"), bbox_inches='tight')
    plt.close(fig)
    return agg

def main(args):
    reviews = load_reviews(args.reviews)
    reviews_pp = preprocess_reviews(reviews, drop_neutral=True)
    reviews_pp.to_csv(os.path.join(OUT,"reviews_preprocessed.csv"), index=False)
    print("Preprocessed reviews:", reviews_pp.shape)
    rep = train_tfidf_lr(reviews_pp)
    with open(os.path.join(OUT,"sentiment_summary.json"),"w") as f:
        json.dump(rep, f, indent=2)
    sales = load_sales(args.sales)
    sales_pp = preprocess_sales(sales)
    sales_pp.to_csv(os.path.join(OUT,"sales_preprocessed.csv"), index=False)
    ch = channel_profitability(sales_pp)
    sku = sku_toplist(sales_pp)
    geo = geo_sales_map(sales_pp)
    print("Done. Outputs in", OUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, help="Path to Amazon reviews CSV")
    parser.add_argument("--sales", required=True, help="Path to e-commerce sales CSV (Sale Report)")
    args = parser.parse_args()
    main(args)
