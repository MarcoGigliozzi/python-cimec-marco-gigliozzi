from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sheets(xlsx_path: str | Path, cfg: dict) -> dict[str, pd.DataFrame]:
    """
    Load Excel sheets into a dict of DataFrames.
    Try to standardize keys: look for 'trial' and 'session'.
    """
    xlsx_path = Path(xlsx_path)
    xl = pd.ExcelFile(xlsx_path)

    sheets = {}
    for name in xl.sheet_names:
        df = xl.parse(name)
        lname = name.strip().lower()
        if "trial" in lname:
            sheets["trial"] = df
        elif "session" in lname:
            sheets["session"] = df
        else:
            sheets[lname] = df
    return sheets
