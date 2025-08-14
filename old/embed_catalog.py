import os, sys, json, time, math, pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# --- Config you can tweak ---
EMBED_MODEL = "text-embedding-3-small"  # swap to text-embedding-3-large for highest quality
TEXT_COL_CANDIDATES = ["transcript_text", "transcript", "text"]
PATH_COL_CANDIDATES = ["transcript_path", "path"]
ID_COL_CANDIDATES = ["id", "video_id", "uid"]
BATCH_SIZE = 96  # safe for current token limits with short texts; lower if you see 400 errors
OUTPUT = "embeddings.parquet"

def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_text(row, text_col, path_col, base_dir=None):
    if text_col and isinstance(row[text_col], str) and row[text_col].strip():
        return row[text_col]
    if path_col and isinstance(row[path_col], str) and row[path_col].strip():
        p = row[path_col]
        if base_dir and not os.path.isabs(p):
            p = os.path.join(base_dir, p)
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""
    return ""

def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if len(sys.argv) < 2:
        print("Usage: python embed_catalog.py <catalog.csv|xlsx> [base_dir_for_paths]")
        sys.exit(1)

    in_path = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Read CSV or Excel
    if in_path.lower().endswith(".xlsx") or in_path.lower().endswith(".xls"):
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)

    id_col = pick_column(df, ID_COL_CANDIDATES) or df.columns[0]
    text_col = pick_column(df, TEXT_COL_CANDIDATES)
    path_col = pick_column(df, PATH_COL_CANDIDATES)

    # Build a working text column
    texts = []
    for _, row in df.iterrows():
        texts.append(load_text(row, text_col, path_col, base_dir))
    df["_text_for_embedding"] = texts

    # Drop empties
    df = df[df["_text_for_embedding"].astype(str).str.strip() != ""].copy()
    if df.empty:
        print("No rows with text to embed. Check your columns or paths.")
        sys.exit(1)

    # Batch embeddings
    all_embeddings = []
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding"):
        batch = df["_text_for_embedding"].iloc[i:i+BATCH_SIZE].tolist()
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            embs = [d.embedding for d in resp.data]
            all_embeddings.extend(embs)
        except Exception as e:
            print("Error on batch", i, "->", e)
            # simple backoff + retry once
            time.sleep(3)
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            embs = [d.embedding for d in resp.data]
            all_embeddings.extend(embs)

    df = df.iloc[:len(all_embeddings)].copy()
    df["_embedding"] = all_embeddings
    # Save as Parquet for compactness & speed
    df.to_parquet(OUTPUT, index=False)
    print(f"Saved {len(df)} embeddings to {OUTPUT}")
    print("Columns:", ", ".join(df.columns))

if __name__ == "__main__":
    main()


