import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning: cast some object columns to category when reasonable."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() < (len(df) / 2):
            try:
                df.loc[:, c] = df[c].astype("category")
            except Exception:
                pass
    return df


def recode_color_to_target(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Map color strings to numeric target values if color exists."""
    df = df.copy()
    if "color" in df.columns:
        df["target_num"] = df["color"].astype(str).str.lower().map(mapping)
    return df


def add_accuracy_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add column acc = 1 if choice == target_num else 0 (if both exist)."""
    df = df.copy()
    if "choice" in df.columns and "target_num" in df.columns:
        df["acc"] = (
            pd.to_numeric(df["choice"], errors="coerce")
            == pd.to_numeric(df["target_num"], errors="coerce")
        ).astype(int)
    return df
