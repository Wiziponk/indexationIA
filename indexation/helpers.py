from __future__ import annotations

"""Utility helper functions."""

from typing import Any

from .config import client


def ensure_list(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return [val]


def get_nested_value(d: Any, path: str) -> Any:
    for part in path.split('.'):
        if isinstance(d, dict):
            d = d.get(part, "")
        else:
            return ""
    return d


def suggest_cluster_names(df, title_col, cluster_col="_cluster", max_titles=5):
    """Use chat completion to suggest a short name for each cluster."""
    names = {}
    if title_col not in df.columns:
        return names
    for cl, sub in df.groupby(cluster_col):
        titles = (
            sub[title_col]
            .dropna()
            .astype(str)
            .head(max_titles)
            .tolist()
        )
        if not titles:
            continue
        prompt = (
            "Voici quelques titres d’éléments appartenant au même cluster : "
            + "; ".join(titles)
            + ". Donne un nom de thème très court en français."
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tu aides à nommer des clusters."},
                    {"role": "user", "content": prompt},
                ],
            )
            names[int(cl)] = resp.choices[0].message.content.strip()
        except Exception:
            continue
    return names
