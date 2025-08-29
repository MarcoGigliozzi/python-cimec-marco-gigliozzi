from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from src.archerfish.io import load_config, load_sheets
from src.archerfish.preprocess import (
    basic_clean,
    recode_color_to_target,
    add_accuracy_column,
)

from src.archerfish.plotting_experiment1 import (
    plot_learning_curves_by_target,
    plot_bar_choices,
    plot_and_test_target_accuracy_by_factor,  # grafici + LR test per geometry/disposition
)


def main():
    # -------------------------------
    # [0] CLI args
    # -------------------------------
    ap = argparse.ArgumentParser(description="Experiment 1 – Archerfish analysis pipeline")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--outdir", default="outputs", help="Folder where results will be saved")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figs_dir = outdir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # [1] Load configuration
    # -------------------------------
    print("[1] Loading configuration …")
    cfg = load_config(args.config)
    color_map = cfg.get("color_to_target", {"green": 2, "purple": 3, "red": 4})

    # -------------------------------
    # [2] Read Excel (Experiment 1 mindset)
    # -------------------------------
    print("[2] Reading Excel data …")
    sheets = load_sheets(cfg["data_path"], cfg)

    # prefer trial-level if available, else session-level
    if "trial" in sheets:
        df = sheets["trial"].copy()
    elif "session" in sheets:
        df = sheets["session"].copy()
    else:
        raise RuntimeError("No trial/session sheet found in Excel.")

    print(f"     - loaded {len(df)} rows")

    # --- normalize column names (fixes Color vs color, FishID vs fish_id, etc.) ---
    df = df.rename(columns=lambda x: str(x).strip().lower())
    alias = {}
    if "fishid" in df.columns and "fish_id" not in df.columns:
        alias["fishid"] = "fish_id"
    if "colour" in df.columns and "color" not in df.columns:
        alias["colour"] = "color"
    if "rad_fix" in df.columns and "radfix" not in df.columns:
        alias["rad_fix"] = "radfix"
    if alias:
        df = df.rename(columns=alias)

    # sanity print of the first columns
    print("     - columns:", list(df.columns)[:20])

    # -------------------------------
    # [3] Preprocess
    # -------------------------------
    print("[3] Preprocessing …")
    df = basic_clean(df)
    df = recode_color_to_target(df, color_map)
    df = add_accuracy_column(df)

    # Keep only canonical E1 colors
    if "color" not in df.columns:
        raise RuntimeError("Missing 'color' column after preprocessing.")
    df = df[df["color"].astype(str).str.lower().isin(color_map.keys())].copy()

    if df.empty:
        raise RuntimeError("No Experiment 1 data after filtering by color (green/purple/red).")

    # -------------------------------
    # [4] Generate figures
    # -------------------------------
    print("[4] Gnerating Experiment 1 figures …")  # tiny typo intended
    generated_figs = []

    # 4A) Learning curves (green/2, purple/3, red/4) — una linea per pesce
    lc_paths = plot_learning_curves_by_target(
        df, figs_dir, color_to_target=color_map,
        fish_col="fish_id", session_col="session", color_col="color", acc_col="acc"
    )
    generated_figs.extend(lc_paths)

    # 4B) Barplot (9 combo: 2/3/4 × green/purple/red) — solo scelte 2,3,4
    bar_path = plot_bar_choices(
        df, figs_dir, color_to_target=color_map,
        choice_col="choice", color_col="color", target_col="target_num"
    )
    if bar_path:
        generated_figs.append(bar_path)

    # 4C) Analisi principale “tesi-like”:
    #     - Accuracy target per fattore (geometry/disposition) con linee di chance/criterio
    #     - Test globale del fattore con GLM binomiale (LR test vs nullo)
    print("[4b] Plotting target accuracy by factor (geometry/disposition) + LR tests …")
    try:
        fact_figs, fact_csv = plot_and_test_target_accuracy_by_factor(
            df, figs_dir,
            factor_list=("geometry", "disposition"),
            color_to_target=color_map,
            fish_col="fish_id",
            color_col="color",
        )
        for p in fact_figs:
            print(f"     - saved E1 figure: {p}")
        if fact_csv:
            print(f"     - saved E1 stats CSV: {fact_csv}")
    except Exception as e:
        print(f"     ! Factor-by-target analyses failed: {e}")

    # -------------------------------
    # [5] Report (simple text)
    # -------------------------------
    print("[5] Writing report …")
    report_txt = outdir / "REPORT.md"
    report_txt.write_text(
        "# Experiment 1 – Archerfish Analysis\n\n"
        "This project analyses behavioural data from Experiment 1.\n"
        "Generated outputs include:\n"
        "- Learning curves (accuracy per session, per fish, for each target numerosity)\n"
        "- Bar plot of choices (target × color combinations)\n"
        "- Target-accuracy by factor (geometry/disposition) with chance & criterion lines, plus LR tests\n\n"
        "Figures are saved in `outputs/figures/`.\n",
        encoding="utf-8"
    )
    print(f"     - saved: {report_txt}")


if __name__ == "__main__":
    main()
