from __future__ import annotations

import numpy as np
import pandas as pd
from ..config import DATA_DIR

def save_dataset_files(df: pd.DataFrame, X: np.ndarray, uid: str):
    raw_name = f"raw_{uid}.parquet"
    emb_name = f"emb_{uid}.npy"
    df.to_parquet(DATA_DIR / raw_name, index=False)
    np.save(DATA_DIR / emb_name, X)
    return raw_name, emb_name
