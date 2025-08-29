import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Iterable, Optional, Dict

import statsmodels.formula.api as smf
import statsmodels.api as sm


# ---------------------------------------------
# Utility
# ---------------------------------------------
def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------
# 1) Learning curves per pesce per target
# ---------------------------------------------
def plot_learning_curves_by_target(
    df: pd.DataFrame,
    outdir: str | Path,
    color_to_target: Optional[Dict[str, int]] = None,
    fish_col: str = "fish_id",
    session_col: str = "session",
    color_col: str = "color",
    acc_col: str = "acc",
) -> list[Path]:
    outdir = _ensure_dir(outdir)
    if color_to_target is None:
        color_to_target = {"green": 2, "purple": 3, "red": 4}

    fig_paths: list[Path] = []

    for clr, tgt in color_to_target.items():
        sub = df[df[color_col].astype(str).str.lower() == clr].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(7, 4))
        sns.lineplot(
            data=sub,
            x=session_col,
            y=acc_col,
            hue=fish_col,
            estimator="mean",
            errorbar=None,
            lw=1.5,
        )
        plt.axhline(1/3, ls="--", lw=1, color="gray", label="chance (0.333)")
        plt.axhline(0.459, ls=":", lw=1, color="gray", label="criterion (0.459)")
        plt.ylim(0, 1)
        plt.title(f"Learning curve – {clr} (target={tgt})")
        plt.ylabel("Accuracy")
        plt.xlabel("Session")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()

        p = Path(outdir) / f"learning_curve_target{tgt}_{clr}.png"
        plt.savefig(p, dpi=300)
        plt.close()
        fig_paths.append(p)

    return fig_paths


# ---------------------------------------------
# 2) Bar plot scelte 2/3/4 × colore
# ---------------------------------------------
def plot_bar_choices(
    df: pd.DataFrame,
    outdir: str | Path,
    color_to_target: Optional[Dict[str, int]] = None,
    choice_col: str = "choice",
    color_col: str = "color",
    target_col: str = "target_num",
) -> Optional[Path]:
    outdir = _ensure_dir(outdir)
    df = df.copy()

    if color_to_target is None:
        color_to_target = {"green": 2, "purple": 3, "red": 4}

    if color_col not in df.columns or choice_col not in df.columns:
        print("[WARN] plot_bar_choices: missing columns for bar plot")
        return None

    if target_col not in df.columns:
        df[target_col] = df[color_col].astype(str).str.lower().map(color_to_target)

    # keep only valid choices (2,3,4)
    df["choice_num"] = pd.to_numeric(df[choice_col], errors="coerce")
    df = df[df["choice_num"].isin([2, 3, 4])].copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # order of x-axis
    order = [
        (2, "green"), (3, "green"), (4, "green"),
        (2, "purple"), (3, "purple"), (4, "purple"),
        (2, "red"), (3, "red"), (4, "red"),
    ]
    df["cond_x"] = [
        f"{int(tn)}-{clr}"
        for tn, clr in zip(df[target_col], df[color_col].astype(str).str.lower())
    ]
    valid_labels = [f"{n}-{c}" for (n, c) in order]
    df = df[df["cond_x"].isin(valid_labels)].copy()

    prop = (
        df.groupby(["cond_x", "choice_num"], observed=True)
          .size()
          .reset_index(name="n")
    )
    totals = prop.groupby("cond_x", observed=True)["n"].sum().rename("totals")
    prop = prop.merge(totals, on="cond_x", how="left")
    prop["prop"] = prop["n"] / prop["totals"]

    plt.figure(figsize=(10, 4))
    sns.barplot(
        data=prop,
        x="cond_x",
        y="prop",
        hue="choice_num",
        order=valid_labels,
        dodge=True,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.xlabel("Target × Color condition")
    plt.ylabel("Choice proportion")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Chosen numerosity", frameon=False)

    plt.tight_layout()
    p = Path(outdir) / "choices_bar_by_color_target.png"
    plt.savefig(p, dpi=300)
    plt.close()
    return p


# ---------------------------------------------
# 3) Analisi predittori non numerici (fixata)
# ---------------------------------------------
def analyze_non_numeric_predictors(
    df: pd.DataFrame,
    outdir: str | Path,
    predictors: Iterable[str] = ("geometry", "disposition"),
    fish_col: str = "fish_id",
    acc_col: str = "acc",
) -> tuple[Optional[Path], Optional[Path]]:
    outdir = _ensure_dir(outdir)
    df = df.copy()

    if acc_col not in df.columns and {"choice", "target_num"} <= set(df.columns):
        df[acc_col] = (pd.to_numeric(df["choice"], errors="coerce") ==
                       pd.to_numeric(df["target_num"], errors="coerce")).astype(float)

    rows = []
    for pred in predictors:
        if pred not in df.columns:
            continue

        sub = df.dropna(subset=[pred, acc_col]).copy()
        if sub.empty or sub[pred].nunique() <= 1:
            continue

        formula = f"{acc_col} ~ C({pred})"
        try:
            if fish_col in sub.columns:
                model = smf.gee(
                    formula=formula,
                    groups=sub[fish_col],
                    data=sub,
                    family=sm.families.Binomial(),
                ).fit()
            else:
                model = smf.glm(
                    formula=formula,
                    data=sub,
                    family=sm.families.Binomial(),
                ).fit()

            for name, coef in model.params.items():
                if "Intercept" in name:
                    continue
                pval = model.pvalues.get(name, float("nan"))
                rows.append(
                    {
                        "predictor": pred,
                        "term": name,
                        "coef_logit": coef,
                        "p_value": pval,
                        "n": len(sub),
                        "levels": int(sub[pred].nunique()),
                    }
                )
        except Exception as e:
            print(f"[WARN] GEE/GLM failed for {pred}: {e}")
            continue

    if not rows:
        print("[INFO] No coefficients extracted; are predictors present or varying?")
        return None, None

    out_df = pd.DataFrame(rows).sort_values(["predictor", "term"]).reset_index(drop=True)
    csv_path = Path(outdir) / "non_numeric_predictors_effects.csv"
    out_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(7, 4))
    y = range(len(out_df))
    plt.scatter(out_df["coef_logit"], y)
    plt.axvline(0, color="gray", ls="--", lw=1)
    plt.yticks(y, out_df["term"], fontsize=8)
    plt.xlabel("Log-odds (coefficient)")
    plt.title("Non-numeric predictors (exploratory)")
    plt.tight_layout()
    fig_path = Path(outdir) / "non_numeric_predictors_forest.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return csv_path, fig_path


# ---------------------------------------------
# 4) Analisi stile tesi: accuracy target × geometry/disposition
# ---------------------------------------------
def _bar_accuracy_by_factor_for_subset(
    df: pd.DataFrame,
    factor: str,
    outdir: Path,
    title: str,
    fname: str,
) -> Path:
    g = (
        df.groupby(factor)["acc"]
          .agg(["mean", "count", "std"])
          .reset_index()
          .rename(columns={"mean": "acc_mean", "std": "acc_sd"})
    )
    g["acc_sem"] = g["acc_sd"] / g["count"].clip(lower=1) ** 0.5

    levels = list(g[factor].astype(str))

    plt.figure(figsize=(6.8, 4.0))
    plt.bar(levels, g["acc_mean"], yerr=g["acc_sem"], capsize=3,
            edgecolor="black", linewidth=0.6)
    plt.axhline(1/3, ls="--", lw=1, color="gray", label="chance (0.333)")
    plt.axhline(0.459, ls=":", lw=1, color="gray", label="criterion (0.459)")
    plt.ylim(0, 1)
    plt.xlabel(factor.capitalize())
    plt.ylabel("Accuracy")
    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    p = outdir / fname
    plt.savefig(p, dpi=300)
    plt.close()
    return p


def plot_and_test_target_accuracy_by_factor(
    df: pd.DataFrame,
    outdir: str | Path,
    factor_list: Iterable[str] = ("geometry", "disposition"),
    color_to_target: Optional[Dict[str, int]] = None,
    fish_col: str = "fish_id",
    color_col: str = "color",
) -> tuple[list[Path], Optional[Path]]:
    outdir = _ensure_dir(outdir)
    if color_to_target is None:
        color_to_target = {"green": 2, "purple": 3, "red": 4}

    df = df.copy()
    if "target_num" not in df.columns and color_col in df.columns:
        df["target_num"] = df[color_col].astype(str).str.lower().map(color_to_target)
    if "acc" not in df.columns and {"choice", "target_num"} <= set(df.columns):
        df["acc"] = (pd.to_numeric(df["choice"], errors="coerce") ==
                     pd.to_numeric(df["target_num"], errors="coerce")).astype(float)

    rows = []
    fig_paths: list[Path] = []

    for clr, tgt in [("green", 2), ("purple", 3), ("red", 4)]:
        sub = df[
            (df[color_col].astype(str).str.lower() == clr) &
            (pd.to_numeric(df["target_num"], errors="coerce") == tgt)
        ].copy()
        if sub.empty:
            continue

        for factor in factor_list:
            if factor not in sub.columns:
                continue

            title = f"Accuracy by {factor} – {clr} (target={tgt})"
            fname = f"E1_target{tgt}_{clr}_by_{factor}.png"
            try:
                fig_paths.append(
                    _bar_accuracy_by_factor_for_subset(sub, factor, Path(outdir), title, fname)
                )
            except Exception as e:
                print(f"[WARN] Plot by factor failed for {clr}/{factor}: {e}")

            try:
                sub2 = sub.dropna(subset=[factor, "acc"]).copy()
                if sub2.empty or sub2[factor].nunique() <= 1:
                    rows.append({"color": clr, "target": tgt, "factor": factor,
                                 "n": len(sub2), "levels": int(sub2[factor].nunique()),
                                 "LR": float("nan"), "df": 0, "p_value": float("nan")})
                    continue

                full = smf.glm(f"acc ~ C({factor})", data=sub2,
                               family=sm.families.Binomial()).fit()
                null = smf.glm("acc ~ 1", data=sub2,
                               family=sm.families.Binomial()).fit()

                LR = 2 * (full.llf - null.llf)
                df_diff = full.df_model - null.df_model
                from scipy.stats import chi2
                p = chi2.sf(LR, df_diff) if df_diff > 0 else float("nan")

                rows.append({"color": clr, "target": tgt, "factor": factor,
                             "n": len(sub2), "levels": int(sub2[factor].nunique()),
                             "LR": LR, "df": df_diff, "p_value": p})
            except Exception as e:
                print(f"[WARN] LR test failed for {clr}/{factor}: {e}")
                rows.append({"color": clr, "target": tgt, "factor": factor,
                             "n": len(sub), "levels": int(sub[factor].nunique()) if factor in sub.columns else 0,
                             "LR": float("nan"), "df": 0, "p_value": float("nan")})

    csv_path = None
    if rows:
        out = pd.DataFrame(rows)
        csv_path = Path(outdir) / "E1_factor_effects_LRtests.csv"
        out.to_csv(csv_path, index=False)

    return fig_paths, csv_path
