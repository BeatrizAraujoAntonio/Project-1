"""NHANES activity threshold analysis entry point.

Place downloaded NHANES XPT files in data/raw, then extend this script to
merge, clean, analyze, and chart the processed dataset.
"""

import os
from pathlib import Path

import pandas as pd
import requests
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.smoothers_lowess import lowess

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
CHARTS_DIR = PROJECT_DIR / "outputs" / "charts"

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"
NHANES_FILES = {
    "PAQ_G.XPT": BASE_URL + "2011/DataFiles/PAQ_G.xpt",
    "PAQ_H.XPT": BASE_URL + "2013/DataFiles/PAQ_H.xpt",
    "CFQ_G.XPT": BASE_URL + "2011/DataFiles/CFQ_G.xpt",
    "CFQ_H.XPT": BASE_URL + "2013/DataFiles/CFQ_H.xpt",
}


def is_xport_file(path: Path) -> bool:
    """Return whether a file appears to be a SAS XPORT file."""
    if not path.exists():
        return False

    return path.read_bytes().startswith(b"HEADER RECORD*******")


def download_raw_files() -> None:
    """Download required NHANES XPT files into data/raw."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in NHANES_FILES.items():
        destination = RAW_DATA_DIR / filename
        if is_xport_file(destination):
            print(f"{filename} already exists.")
            continue

        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)

    print("All files ready.")


def load_and_merge_data() -> pd.DataFrame:
    """Load PAQ and CFQ XPT files, then merge them by participant ID."""
    paq_g = pd.read_sas(RAW_DATA_DIR / "PAQ_G.XPT", format="xport", encoding="utf-8")
    paq_h = pd.read_sas(RAW_DATA_DIR / "PAQ_H.XPT", format="xport", encoding="utf-8")
    paq = pd.concat([paq_g, paq_h], ignore_index=True)

    cfq_g = pd.read_sas(RAW_DATA_DIR / "CFQ_G.XPT", format="xport", encoding="utf-8")
    cfq_h = pd.read_sas(RAW_DATA_DIR / "CFQ_H.XPT", format="xport", encoding="utf-8")
    cfq = pd.concat([cfq_g, cfq_h], ignore_index=True)

    return pd.merge(
        paq[["SEQN", "PAD615", "PAD630"]],
        cfq[["SEQN", "CFDDS"]],
        on="SEQN",
        how="inner",
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged NHANES data and compute weekly activity minutes."""
    df = df.copy()
    df.replace([7777, 9999], np.nan, inplace=True)
    df.dropna(subset=["PAD615", "PAD630", "CFDDS"], inplace=True)

    df["weekly_activity_min"] = df["PAD615"] + 2 * df["PAD630"]
    cap = df["weekly_activity_min"].quantile(0.99)
    df["weekly_activity_min"] = df["weekly_activity_min"].clip(upper=cap)
    df.rename(columns={"CFDDS": "cog_score"}, inplace=True)

    return df


def save_processed_data(df: pd.DataFrame) -> Path:
    """Save cleaned data for analysis."""
    output_path = PROCESSED_DATA_DIR / "clean.csv"
    df[["weekly_activity_min", "cog_score"]].to_csv(output_path, index=False)
    return output_path


def smooth_activity_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit a LOWESS curve for cognitive score by weekly activity minutes."""
    x = df["weekly_activity_min"].values
    y = df["cog_score"].values

    smoothed = lowess(y, x, frac=0.3, it=3, return_sorted=True)
    x_smooth = smoothed[:, 0]
    y_smooth = smoothed[:, 1]

    return x_smooth, y_smooth


def segmented_rss(bp: float, x: np.ndarray, y: np.ndarray) -> float:
    """Compute RSS for a two-segment linear fit at breakpoint bp."""
    left = x <= bp
    right = x > bp
    rss = 0.0

    for mask in [left, right]:
        if mask.sum() < 3:
            return np.inf
        x_segment = sm.add_constant(x[mask])
        model = sm.OLS(y[mask], x_segment).fit()
        rss += model.ssr

    return rss


def find_activity_breakpoint(df: pd.DataFrame) -> float:
    """Find the activity threshold that minimizes two-segment RSS."""
    x = df["weekly_activity_min"].values
    y = df["cog_score"].values
    lo, hi = np.percentile(x, [10, 90])

    result = minimize_scalar(
        segmented_rss,
        bounds=(lo, hi),
        method="bounded",
        args=(x, y),
    )
    return result.x


def fit_segment(x_segment: np.ndarray, y_segment: np.ndarray) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit a linear model for one segment."""
    x_with_constant = sm.add_constant(x_segment)
    return sm.OLS(y_segment, x_with_constant).fit()


def fit_threshold_segments(
    df: pd.DataFrame,
    breakpoint: float,
) -> tuple[
    sm.regression.linear_model.RegressionResultsWrapper,
    sm.regression.linear_model.RegressionResultsWrapper,
]:
    """Fit linear models below and above the activity threshold."""
    x = df["weekly_activity_min"].values
    y = df["cog_score"].values
    left_mask = x <= breakpoint
    right_mask = x > breakpoint

    left_model = fit_segment(x[left_mask], y[left_mask])
    right_model = fit_segment(x[right_mask], y[right_mask])

    return left_model, right_model


def save_activity_threshold_chart(
    df: pd.DataFrame,
    x_smooth: np.ndarray,
    y_smooth: np.ndarray,
    breakpoint: float,
    left_model: sm.regression.linear_model.RegressionResultsWrapper,
    right_model: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[Path, Path]:
    """Save scatter, LOWESS, segmented regression, and threshold chart."""
    x = df["weekly_activity_min"].values
    y = df["cog_score"].values
    left_mask = x <= breakpoint
    right_mask = x > breakpoint

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    ax.scatter(x, y, alpha=0.12, s=14, color="#888888", label="Participants")
    ax.plot(
        x_smooth,
        y_smooth,
        color="#1565C0",
        lw=2.5,
        label="LOWESS Smooth",
        zorder=3,
    )

    x_left = np.linspace(x[left_mask].min(), breakpoint, 200)
    x_right = np.linspace(breakpoint, x[right_mask].max(), 200)
    y_left = left_model.params[0] + left_model.params[1] * x_left
    y_right = right_model.params[0] + right_model.params[1] * x_right

    ax.plot(x_left, y_left, "--", color="#E65100", lw=2, label="Seg. Regression")
    ax.plot(x_right, y_right, "--", color="#E65100", lw=2)

    ax.axvline(
        breakpoint,
        color="#B8860B",
        lw=2,
        linestyle=":",
        label=f"Threshold approx {breakpoint:.0f} min/wk",
    )
    ax.axvspan(0, breakpoint, alpha=0.06, color="#B8860B")

    y_top = ax.get_ylim()[1]
    ax.annotate(
        f"Inflection\n{breakpoint:.0f} min/wk",
        xy=(breakpoint, y_top * 0.92),
        xytext=(breakpoint + 30, y_top * 0.92),
        fontsize=9,
        color="#B8860B",
        arrowprops={"arrowstyle": "->", "color": "#B8860B", "lw": 1.2},
    )

    ax.set_xlabel("Weekly Physical Activity (MET-equivalent minutes)", fontsize=11)
    ax.set_ylabel("Digit Symbol Substitution Score (Cognitive Proxy)", fontsize=11)
    ax.set_title(
        "Physical Activity Threshold for Cognitive Protection\n"
        "NHANES 2011-2014 | Segmented Regression + LOWESS",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    fig.text(
        0.5,
        0.01,
        "Beatriz A Antonio",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    ax.legend(framealpha=0.85, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    png_path = CHARTS_DIR / "activity_threshold.png"
    svg_path = CHARTS_DIR / "activity_threshold.svg"
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, svg_path


def main() -> None:
    """Run the activity threshold analysis."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    download_raw_files()

    raw_files = sorted(RAW_DATA_DIR.glob("*.XPT"))
    print(f"Found {len(raw_files)} NHANES XPT file(s) in {RAW_DATA_DIR}")

    df = load_and_merge_data()
    print(f"Merged dataset: {len(df)} participants")

    clean_df = clean_data(df)
    output_path = save_processed_data(clean_df)
    print(f"Clean dataset: {len(clean_df)} rows")
    print(f"Saved processed data to {output_path}")

    x_smooth, y_smooth = smooth_activity_curve(clean_df)
    print(f"LOWESS curve: {len(x_smooth)} smoothed points")

    breakpoint = find_activity_breakpoint(clean_df)
    print(f"Inflection point: {breakpoint:.1f} min/week")

    left_model, right_model = fit_threshold_segments(clean_df, breakpoint)
    print("--- Left segment (below threshold) ---")
    print(f"  Slope: {left_model.params[1]:.4f}  p={left_model.pvalues[1]:.4f}")
    print("--- Right segment (above threshold) ---")
    print(f"  Slope: {right_model.params[1]:.4f}  p={right_model.pvalues[1]:.4f}")

    png_path, svg_path = save_activity_threshold_chart(
        clean_df,
        x_smooth,
        y_smooth,
        breakpoint,
        left_model,
        right_model,
    )
    print(f"Chart saved to {png_path}")
    print(f"Chart saved to {svg_path}")


if __name__ == "__main__":
    main()
