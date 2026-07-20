#!/usr/bin/env python3

#dataset_inspector.py
#Run locally in the folder where your datasets exist or provide full paths.

#How to run:
#python dataset_inspector.py --sales_folder "H:\\Datasets\\ecommerce\\ecommerce_sale" --reviews "H:\\Datasets\\ecommerce\\ecommerce_review\\Reviews.csv"

#Output: inspection_outputs/summary.json plus several CSVs and small SVG preview plots.

import os, json, argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("inspection_outputs")
OUT.mkdir(exist_ok=True)

def safe_read_csv(path, nrows=None):
    path = Path(path)
    try:
        if nrows:
            return pd.read_csv(path, nrows=nrows, low_memory=False)
        else:
            return pd.read_csv(path, low_memory=False)
    except Exception as e:
        return {"error": str(e)}

def inspect_file(path):
    info = {"path": str(path)}
    data = safe_read_csv(path, nrows=5)
    if isinstance(data, dict) and data.get("error"):
        info["error"] = data["error"]
        return info
    df_head = safe_read_csv(path, nrows=5)
    df_full = None
    try:
        df_full = safe_read_csv(path, nrows=1000)  # small sample for quick stats
    except:
        pass
    # basic header summary
    info["columns"] = list(df_head.columns)
    info["sample_rows"] = df_head.fillna("").to_dict(orient="records")
    # try to get counts using a fast dtype pass
    try:
        full = pd.read_csv(path, low_memory=False)
        info["num_rows_est"] = full.shape[0]
        info["num_cols"] = full.shape[1]
        # top-10 null counts (fast)
        na = full.isna().sum().sort_values(ascending=False).head(20).to_dict()
        info["top_na_counts"] = na
        # detect date-like, geo-like, amount-like columns
        cols = list(full.columns)
        info["date_like"] = [c for c in cols if any(k in c.lower() for k in ["date","time","month"])]
        info["geo_like"] = [c for c in cols if any(k in c.lower() for k in ["state","country","city","ship","address"])]
        info["amount_like"] = [c for c in cols if any(k in c.lower() for k in ["amount","gross","mrp","price","amt","rate","value"])]
        info["qty_like"] = [c for c in cols if any(k in c.lower() for k in ["qty","pcs","quantity"])]
        # basic dtypes summary
        info["dtypes"] = {c: str(full[c].dtype) for c in cols[:50]}
        # value counts for key columns (small sample to avoid memory)
        for key in ["SKU","ASIN","ProductId","Score","Category","Fulfilment","ship-state","ship-country","country","Order ID","OrderID"]:
            if key in full.columns:
                info.setdefault("value_samples",{})[key] = full[key].dropna().astype(str).unique()[:10].tolist()
        # save a tiny head sample CSV
        full.head(50).to_csv(OUT / f"sample_{path.name}", index=False)
        # quick plots for amount-like and qty-like if present
        if info["amount_like"]:
            col = info["amount_like"][0]
            s = pd.to_numeric(full[col].astype(str).str.replace(r'[^0-9\.-]', '', regex=True), errors='coerce')
            s.dropna(inplace=True)
            if not s.empty:
                s.sample(min(1000,len(s))).hist(bins=40)
                plt.title(f"Distribution of {col}")
                plt.savefig(OUT / f"dist_{path.name}_{col}.svg", bbox_inches='tight')
                plt.clf()
        if info["qty_like"]:
            col = info["qty_like"][0]
            s = pd.to_numeric(full[col].astype(str).str.replace(r'[^0-9\.-]', '', regex=True), errors='coerce')
            s.dropna(inplace=True)
            if not s.empty:
                s.value_counts().head(20).plot(kind='bar')
                plt.title(f"Top values in {col}")
                plt.savefig(OUT / f"topvals_{path.name}_{col}.svg", bbox_inches='tight')
                plt.clf()
    except Exception as e:
        info["error_summary"] = str(e)
    return info

def main(args):
    report = {}
    # inspect reviews
    if args.reviews:
        p = Path(args.reviews)
        report['reviews'] = inspect_file(p)
    # inspect sales folder
    if args.sales_folder:
        sf = Path(args.sales_folder)
        report['sales_files'] = {}
        for f in sf.glob("*.csv"):
            report['sales_files'][f.name] = inspect_file(f)
    # write summary
    with open(OUT / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("Inspection written to", OUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True, help="path to Reviews.csv")
    parser.add_argument("--sales_folder", required=True, help="path to sales folder containing the 7 CSVs")
    args = parser.parse_args()
    main(args)
