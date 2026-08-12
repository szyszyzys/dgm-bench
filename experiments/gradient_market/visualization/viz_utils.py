"""
Unified Visualization Utilities
================================
Single source of truth for:
  - Publication styling (fonts, sizes, grid)
  - Defense/attack naming and ordering
  - Color palettes (auto-extends for new methods)
  - Data loading from experiment results
  - Common plot helpers (bar annotations, legends, saving)

Usage:
    from viz_utils import *
    setup_style()
    df = load_step_results("./results", step_prefix="step12")
    fig, axes = create_metric_row(df, metrics=["acc", "asr", "bsr", "msr"])
    save_figure(fig, "figures/my_plot.pdf")
"""

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# 1. DEFENSE & ATTACK REGISTRY
#    Add new methods here — every script picks them up automatically.
# =============================================================================

# Canonical display names (code-name → paper-name)
DEFENSE_NAMES = {
    "fedavg":        "FedAvg",
    "fltrust":       "FLTrust",
    "martfl":        "MartFL",
    "skymask":       "SkyMask",
    "skymask_small": "SkyMask-S",
    "trimmed_mean":  "Trim-Mean",
    "multi_krum":    "Multi-Krum",
    "rflpa":         "RFLPA",
    "spmc":          "SPMC",
    "flame":         "FLAME",
    "deepsight":     "DeepSight",
    "bulyan":        "Bulyan",
    "foolsgold":     "FoolsGold",
}

ATTACK_NAMES = {
    "backdoor":       "Backdoor",
    "image_backdoor": "Backdoor",
    "text_backdoor":  "Backdoor",
    "tabular_backdoor": "Backdoor",
    "labelflip":      "Label Flip",
    "label_flip":     "Label Flip",
    "min_max":        "Min-Max",
    "min_sum":        "Min-Sum",
    "fang_krum":      "Fang-Krum",
    "fang_trim":      "Fang-Trim",
    "scaling":        "Scaling",
    "dba":            "DBA",
    "badnet":         "BadNet",
    "pivot":          "Targeted Pivot",
    "drowning":       "Drowning",
    "alie":           "ALIE",
    "none":           "No Attack",
    "no_attack":      "No Attack",
}

DATASET_NAMES = {
    "cifar100": "CIFAR-100",
    "cifar10":  "CIFAR-10",
    "femnist":  "FEMNIST",
    "texas100": "Texas-100",
    "purchase100": "Purchase-100",
    "trec":     "TREC",
}

SYBIL_NAMES = {
    "no_sybil":      "No Sybil",
    "mimic":         "Mimic",
    "pivot":         "Pivot",
    "knock_out":     "Knock-Out",
    "oracle_blend":  "Oracle Blend",
    "drowning":      "Drowning",
    "alie":          "ALIE",
}

# Display order (determines x-axis ordering in plots)
DEFENSE_ORDER = [
    "FedAvg", "FLTrust", "MartFL", "SkyMask", "SkyMask-S",
    "Trim-Mean", "Multi-Krum", "RFLPA", "SPMC",
    "FLAME", "DeepSight", "Bulyan", "FoolsGold",
]

# =============================================================================
# 2. COLOR PALETTES
#    Designed for colorblind-friendliness and consistent identity across figures.
# =============================================================================

DEFENSE_COLORS = {
    "FedAvg":     "#7f8c8d",   # Grey     (baseline)
    "FLTrust":    "#3498db",   # Blue
    "MartFL":     "#2ecc71",   # Green
    "SkyMask":    "#e74c3c",   # Red
    "SkyMask-S":  "#e67e22",   # Orange
    "Trim-Mean":  "#9b59b6",   # Purple
    "Multi-Krum": "#1abc9c",   # Teal
    "RFLPA":      "#f39c12",   # Amber
    "SPMC":       "#2c3e50",   # Dark Navy
    "FLAME":      "#fd79a8",   # Light Pink
    "DeepSight":  "#6c5ce7",   # Indigo
    "Bulyan":     "#00b894",   # Mint
    "FoolsGold":  "#fdcb6e",   # Gold
}

# Metric colors (for grouped bar charts)
METRIC_COLORS = {
    "Accuracy": "#2ca02c",   # Green
    "ASR":      "#d62728",   # Red
    "BSR":      "#1f77b4",   # Blue
    "MSR":      "#ff7f0e",   # Orange
}

# Valuation dual-stack colors
VALUATION_COLORS = {
    "benign_paid":      "#2ca02c",  # Green
    "benign_discarded": "#bbbbbb",  # Grey
    "adv_paid":         "#d62728",  # Red
    "adv_blocked":      "#2c3e50",  # Dark navy
}

# Marker styles for line plots (cycle through these)
DEFENSE_MARKERS = {
    "FedAvg":     "o",
    "FLTrust":    "s",
    "MartFL":     "D",
    "SkyMask":    "^",
    "SkyMask-S":  "v",
    "Trim-Mean":  "P",
    "Multi-Krum": "X",
    "RFLPA":      "*",
    "SPMC":       "h",
    "FLAME":      "H",
    "DeepSight":  "d",
    "Bulyan":     "8",
    "FoolsGold":  ">",
}


def get_color(defense: str) -> str:
    """Get color for a defense, auto-generating if unknown."""
    if defense in DEFENSE_COLORS:
        return DEFENSE_COLORS[defense]
    # Auto-generate from a large palette for unknown defenses
    extra_colors = list(mcolors.TABLEAU_COLORS.values())
    idx = hash(defense) % len(extra_colors)
    return extra_colors[idx]


def get_marker(defense: str) -> str:
    """Get marker for a defense."""
    return DEFENSE_MARKERS.get(defense, "o")


def get_palette(defenses: List[str]) -> Dict[str, str]:
    """Build a color palette dict for a list of defenses."""
    return {d: get_color(d) for d in defenses}


def order_defenses(defenses: List[str]) -> List[str]:
    """Sort defenses by canonical order, unknowns at the end."""
    known = [d for d in DEFENSE_ORDER if d in defenses]
    unknown = sorted(set(defenses) - set(DEFENSE_ORDER))
    return known + unknown


# =============================================================================
# 3. NAME FORMATTING
# =============================================================================

def fmt(name: str) -> str:
    """Universal name formatter: code-name → paper-name."""
    if not isinstance(name, str):
        return str(name)
    key = name.lower().strip()
    for registry in [DEFENSE_NAMES, ATTACK_NAMES, DATASET_NAMES, SYBIL_NAMES]:
        if key in registry:
            return registry[key]
    return name.replace("_", " ").title()


# Backward-compatible aliases
format_label = fmt
PRETTY_NAMES = {**DEFENSE_NAMES, **ATTACK_NAMES, **DATASET_NAMES, **SYBIL_NAMES}


# =============================================================================
# 4. PUBLICATION STYLE
# =============================================================================

def setup_style():
    """Apply publication-quality matplotlib/seaborn style globally."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.8)

    plt.rcParams.update({
        # Fonts
        "font.family":          "sans-serif",
        "font.weight":          "bold",
        "axes.labelweight":     "bold",
        "axes.titleweight":     "bold",

        # Sizes
        "axes.titlesize":       24,
        "axes.labelsize":       20,
        "xtick.labelsize":      18,
        "ytick.labelsize":      18,
        "legend.fontsize":      18,
        "legend.title_fontsize": 20,

        # Lines & borders
        "axes.linewidth":       2.0,
        "axes.edgecolor":       "#333333",
        "lines.linewidth":      3.0,
        "lines.markersize":     10,

        # Figure defaults
        "figure.figsize":       (10, 6),
        "figure.dpi":           100,

        # PDF text (editable in Illustrator/Inkscape)
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
    })


# Backward-compatible alias
set_publication_style = setup_style


# =============================================================================
# 5. DATA LOADING
# =============================================================================

def load_run_metrics(run_dir: Path) -> Dict[str, Any]:
    """
    Load all metrics from a single experiment run directory.
    Combines final_metrics.json + marketplace_report.json.
    """
    data = {}

    # --- final_metrics.json ---
    metrics_file = run_dir / "final_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                m = json.load(f)
            data["acc"] = m.get("acc", m.get("test_accuracy", m.get("accuracy", 0)))
            data["asr"] = m.get("asr", m.get("test_asr", 0))
            data["completed_rounds"] = m.get("completed_rounds", 0)
            data["total_upload_mb"] = m.get("total_upload_mb_accepted", m.get("total_upload_mb", 0))
        except (json.JSONDecodeError, Exception):
            pass

    # --- marketplace_report.json ---
    report_file = run_dir / "marketplace_report.json"
    if report_file.exists():
        try:
            with open(report_file) as f:
                report = json.load(f)
            sellers = list(report.get("seller_summaries", {}).values())
            adv = [s for s in sellers if s.get("type") == "adversary"]
            ben = [s for s in sellers if s.get("type") == "benign"]

            data["bsr"] = np.mean([s["selection_rate"] for s in ben]) if ben else 0.0
            data["msr"] = np.mean([s["selection_rate"] for s in adv]) if adv else 0.0
            data["n_sellers"] = len(sellers)
            data["n_adversaries"] = len(adv)
        except Exception:
            pass

    return data


def _extract_from_path(run_dir: Path, step_prefix: str) -> Dict[str, str]:
    """
    Extract defense, dataset, attack, and other metadata from directory path.
    Uses the path components and any config.yaml found nearby.
    """
    info = {}
    full_path = str(run_dir).lower().replace("\\", "/")

    # Defense
    for code, pretty in DEFENSE_NAMES.items():
        # Match as path component (e.g., /agg-martfl/ or _martfl_)
        if f"agg-{code}" in full_path or f"_{code}_" in full_path or f"/{code}/" in full_path:
            info["defense"] = pretty
            break

    # Dataset
    for code, pretty in DATASET_NAMES.items():
        if code in full_path:
            info["dataset"] = pretty
            break

    # Attack
    for code, pretty in ATTACK_NAMES.items():
        if code in full_path and code not in ("none", "no_attack"):
            info["attack"] = pretty
            break

    # Numeric params from path (adv_rate, poison_rate, n_sellers, etc.)
    adv_match = re.search(r"adv[_-]?(0[_.]?\d+)", full_path)
    if adv_match:
        info["adv_rate"] = float(adv_match.group(1).replace("_", ".").replace("p", "."))

    poison_match = re.search(r"poison[_-]?(0[_.]?\d+|1[_.]?0)", full_path)
    if poison_match:
        info["poison_rate"] = float(poison_match.group(1).replace("_", ".").replace("p", "."))

    nsellers_match = re.search(r"sellers[_-]?(\d+)", full_path)
    if nsellers_match:
        info["n_sellers"] = int(nsellers_match.group(1))

    alpha_match = re.search(r"alpha[_-]?([\d]+[_.]?[\d]*)", full_path)
    if alpha_match:
        info["dirichlet_alpha"] = float(alpha_match.group(1).replace("_", ".").replace("p", "."))

    # Sybil strategy
    for code, pretty in SYBIL_NAMES.items():
        if f"sybil-{code}" in full_path or f"sybil_{code}" in full_path:
            info["sybil_strategy"] = pretty
            break

    return info


def load_step_results(
    results_dir: str,
    step_prefix: str,
    require_success: bool = True,
) -> pd.DataFrame:
    """
    Scan a results directory for all runs matching a step prefix.

    Args:
        results_dir: Root results directory (e.g., "./results")
        step_prefix: Filter to directories starting with this (e.g., "step12", "step5")
        require_success: Only include runs with .success marker

    Returns:
        DataFrame with columns: defense, dataset, attack, acc, asr, bsr, msr, ...
    """
    base = Path(results_dir)
    records = []

    # Find step directories
    step_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith(step_prefix)])
    if not step_dirs:
        # Try glob for nested structures
        step_dirs = sorted(base.glob(f"{step_prefix}*"))

    for step_dir in step_dirs:
        # Find all run directories (run_*_seed_*)
        run_dirs = list(step_dir.rglob("run_*_seed_*"))
        if not run_dirs:
            # Fallback: directories containing final_metrics.json
            run_dirs = [p.parent for p in step_dir.rglob("final_metrics.json")]

        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            if require_success and not (run_dir / ".success").exists():
                continue

            metrics = load_run_metrics(run_dir)
            if not metrics:
                continue

            path_info = _extract_from_path(run_dir, step_prefix)

            # Also try loading config.yaml for accurate metadata
            config_info = _load_config_metadata(run_dir)
            # Config overrides path-based extraction
            merged = {**path_info, **config_info, **metrics, "run_dir": str(run_dir)}
            records.append(merged)

    df = pd.DataFrame(records)

    if not df.empty:
        # Normalize metrics to percentages
        for col in ["acc", "asr", "bsr", "msr"]:
            if col in df.columns:
                mask = df[col] <= 1.0
                df.loc[mask, col] = df.loc[mask, col] * 100

        # Apply display names
        if "defense" in df.columns:
            df["defense"] = df["defense"].apply(lambda x: fmt(x) if isinstance(x, str) else x)
        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].apply(lambda x: fmt(x) if isinstance(x, str) else x)
        if "attack" in df.columns:
            df["attack"] = df["attack"].apply(lambda x: fmt(x) if isinstance(x, str) else x)

    return df


def _load_config_metadata(run_dir: Path) -> Dict[str, str]:
    """Try to load metadata from a nearby config.yaml."""
    info = {}
    # Walk up to find config.yaml (usually 2-3 levels up from run dir)
    current = run_dir
    for _ in range(4):
        config_path = current / "config.yaml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg:
                    exp = cfg.get("experiment", {})
                    agg = cfg.get("aggregation", {})
                    if exp.get("dataset_name"):
                        info["dataset"] = fmt(exp["dataset_name"])
                    if agg.get("method"):
                        info["defense"] = fmt(agg["method"])
                    if exp.get("adv_rate") is not None:
                        info["adv_rate"] = float(exp["adv_rate"])
                    if exp.get("n_sellers") is not None:
                        info["n_sellers"] = int(exp["n_sellers"])
                    poison_cfg = cfg.get("adversary_seller_config", {}).get("poisoning", {})
                    if poison_cfg.get("type") and poison_cfg["type"] != "none":
                        info["attack"] = fmt(poison_cfg["type"])
                    if poison_cfg.get("poison_rate") is not None:
                        info["poison_rate"] = float(poison_cfg["poison_rate"])
                    sybil_cfg = cfg.get("adversary_seller_config", {}).get("sybil", {})
                    if sybil_cfg.get("is_sybil"):
                        mode = sybil_cfg.get("gradient_default_mode", "unknown")
                        info["sybil_strategy"] = fmt(mode)
                break
            except Exception:
                pass
        current = current.parent
    return info


# =============================================================================
# 6. PLOT HELPERS
# =============================================================================

def save_figure(fig, path: str, dpi: int = 300):
    """Save figure as PDF (and optionally PNG preview)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(p, bbox_inches="tight", dpi=dpi, format="pdf")
    # Also save a PNG preview
    png_path = p.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")


def annotate_bars(ax, fmt_str: str = "{:.0f}", fontsize: int = 14, min_height: float = 2.0):
    """Add value labels on top of bar chart bars."""
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if h > min_height:
                ax.annotate(
                    fmt_str.format(h),
                    (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom",
                    fontsize=fontsize, fontweight="bold",
                )


def add_shared_legend(fig, axes, ncol: int = None, loc: str = "upper center"):
    """
    Add a single shared legend above the figure.
    Deduplicates handles across all axes.
    """
    handles, labels = [], []
    seen = set()
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)
        leg = ax.get_legend()
        if leg:
            leg.remove()

    if not handles:
        return

    if ncol is None:
        ncol = len(handles)

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        frameon=False,
        fontsize=18,
        columnspacing=1.5,
        handletextpad=0.5,
    )


def create_metric_row(
    df: pd.DataFrame,
    metrics: List[str] = None,
    x: str = "defense",
    hue: str = None,
    kind: str = "bar",
    figsize: Tuple[float, float] = None,
    titles: List[str] = None,
    ylabels: List[str] = None,
    sharey: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a 1×N row of subplots, one per metric — the workhorse layout.

    Args:
        df: DataFrame with columns for x, hue, and each metric
        metrics: Column names to plot (default: acc, asr, bsr, msr)
        x: Column for x-axis categories
        hue: Column for color grouping (optional)
        kind: "bar" or "line"
        figsize: Override figure size (default: auto-scaled)
        titles: Per-subplot titles
        ylabels: Per-subplot y-axis labels
        sharey: Share y-axis range across subplots

    Returns:
        (fig, axes) tuple
    """
    if metrics is None:
        metrics = ["acc", "asr", "bsr", "msr"]

    metric_display = {
        "acc": "Accuracy (%)", "asr": "ASR (%)",
        "bsr": "BSR (%)",      "msr": "MSR (%)",
    }

    n = len(metrics)
    if figsize is None:
        figsize = (7 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=sharey, constrained_layout=True)
    if n == 1:
        axes = np.array([axes])

    # Determine defense ordering from data
    if x == "defense" and "defense" in df.columns:
        present = df["defense"].unique().tolist()
        x_order = order_defenses(present)
    else:
        x_order = None

    palette = get_palette(x_order) if x == "defense" and x_order else None

    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        plot_df = df.dropna(subset=[metric])
        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        if kind == "bar":
            sns.barplot(
                data=plot_df, x=x, y=metric, hue=hue,
                order=x_order, palette=palette if hue is None else None,
                edgecolor="white", linewidth=1.5, ax=ax,
                errorbar="sd", capsize=0.1,
            )
            annotate_bars(ax)
        elif kind == "line":
            for defense in (x_order or plot_df[hue or x].unique()):
                sub = plot_df[plot_df[hue or x] == defense]
                if sub.empty:
                    continue
                ax.plot(
                    sub[x], sub[metric],
                    color=get_color(defense), marker=get_marker(defense),
                    label=defense, linewidth=3, markersize=10,
                )

        # Labels
        title = titles[i] if titles and i < len(titles) else metric_display.get(metric, metric)
        ylabel = ylabels[i] if ylabels and i < len(ylabels) else "(%)"
        ax.set_title(title, fontsize=22, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_xlabel("")
        ax.set_ylim(bottom=0)

        # Style tweaks
        sns.despine(ax=ax, left=False, bottom=False, right=True, top=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=30)

    return fig, axes


def plot_grouped_metrics(
    df: pd.DataFrame,
    group_col: str = "defense",
    metrics: List[str] = None,
    title: str = "",
    figsize: Tuple[float, float] = (14, 7),
) -> plt.Figure:
    """
    Grouped bar chart: one group per defense, one bar per metric.
    Classic 4-bar comparison (Accuracy / ASR / BSR / MSR).
    """
    if metrics is None:
        metrics = ["acc", "asr", "bsr", "msr"]

    metric_labels = {"acc": "Accuracy", "asr": "ASR", "bsr": "BSR", "msr": "MSR"}
    available = [m for m in metrics if m in df.columns]

    # Aggregate across seeds
    agg = df.groupby(group_col)[available].mean().reset_index()
    present = order_defenses(agg[group_col].unique().tolist())
    agg = agg[agg[group_col].isin(present)]

    # Melt for seaborn
    melted = agg.melt(id_vars=[group_col], value_vars=available,
                      var_name="Metric", value_name="Percentage")
    melted["Metric"] = melted["Metric"].map(metric_labels)

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=melted, x=group_col, y="Percentage", hue="Metric",
        order=present, palette=METRIC_COLORS,
        edgecolor="white", linewidth=1.5, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Percentage (%)", fontsize=20)
    ax.set_ylim(0, 119)
    if title:
        ax.set_title(title, fontsize=24, fontweight="bold")

    annotate_bars(ax, fontsize=14)

    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(available), frameon=False, fontsize=18,
    )
    sns.despine(ax=ax, right=True, top=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    return fig


def plot_line_sweep(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str = "defense",
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    figsize: Tuple[float, float] = (10, 6),
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Line plot with markers for a parameter sweep (e.g., acc vs adv_rate per defense).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    defenses = order_defenses(df[hue_col].unique().tolist())

    for defense in defenses:
        sub = df[df[hue_col] == defense].sort_values(x_col)
        if sub.empty:
            continue
        # Aggregate over seeds
        agg = sub.groupby(x_col)[y_col].agg(["mean", "std"]).reset_index()
        ax.plot(
            agg[x_col], agg["mean"],
            color=get_color(defense), marker=get_marker(defense),
            label=defense, linewidth=3, markersize=10,
        )
        ax.fill_between(
            agg[x_col],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=get_color(defense), alpha=0.15,
        )

    ax.set_xlabel(xlabel or x_col, fontsize=20)
    ax.set_ylabel(ylabel or y_col, fontsize=20)
    if title:
        ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, right=True, top=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    return ax


def plot_dual_stack(
    df: pd.DataFrame,
    metric_col: str = "selection_rate",
    title: str = "",
    figsize: Tuple[float, float] = (12, 6),
) -> plt.Figure:
    """
    Dual stacked bar chart: benign vs adversary breakdown.
    Expects columns: defense, Benign_Paid, Benign_Discarded, Adv_Paid, Adv_Discarded
    """
    df = df.copy()
    df["Total_Benign"] = (df["Benign_Paid"] + df["Benign_Discarded"]).replace(0, 1)
    df["Total_Adv"] = (df["Adv_Paid"] + df["Adv_Discarded"]).replace(0, 1)

    for prefix, total in [("Benign", "Total_Benign"), ("Adv", "Total_Adv")]:
        df[f"Pct_{prefix}_Paid"] = df[f"{prefix}_Paid"] / df[total] * 100
        df[f"Pct_{prefix}_Disc"] = df[f"{prefix}_Discarded"] / df[total] * 100

    present = order_defenses(df["defense"].unique().tolist())
    df = df.set_index("defense").loc[[d for d in present if d in df.index]]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))
    w = 0.35

    vc = VALUATION_COLORS
    ax.bar(x - w/2, df["Pct_Benign_Paid"], w, label="Benign: Paid",
           color=vc["benign_paid"], edgecolor="black", linewidth=1.2)
    ax.bar(x - w/2, df["Pct_Benign_Disc"], w, bottom=df["Pct_Benign_Paid"],
           label="Benign: Discarded", color=vc["benign_discarded"],
           edgecolor="black", linewidth=1.2, hatch="//")
    ax.bar(x + w/2, df["Pct_Adv_Paid"], w, label="Adversary: Paid",
           color=vc["adv_paid"], edgecolor="black", linewidth=1.2)
    ax.bar(x + w/2, df["Pct_Adv_Disc"], w, bottom=df["Pct_Adv_Paid"],
           label="Adversary: Blocked", color=vc["adv_blocked"],
           edgecolor="black", linewidth=1.2, hatch="..")

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=20)
    ax.set_ylim(0, 115)
    if title:
        ax.set_title(title, fontsize=22, fontweight="bold")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, frameon=False, fontsize=16)
    sns.despine(ax=ax, right=True, top=True)

    return fig


# =============================================================================
# 7. CONVENIENCE: Quick full-step visualization
# =============================================================================

def quick_benchmark_plots(
    results_dir: str,
    step_prefix: str,
    output_dir: str,
    dataset_filter: str = None,
):
    """
    One-liner to generate standard benchmark plots for any step.
    Produces: grouped metrics bar chart + metric row per dataset.
    """
    setup_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_step_results(results_dir, step_prefix)
    if df.empty:
        print(f"No results found for {step_prefix} in {results_dir}")
        return

    print(f"Loaded {len(df)} runs for {step_prefix}")
    print(f"  Defenses: {df.get('defense', pd.Series()).unique().tolist()}")
    print(f"  Datasets: {df.get('dataset', pd.Series()).unique().tolist()}")

    datasets = [dataset_filter] if dataset_filter else df.get("dataset", pd.Series()).unique()

    for ds in datasets:
        if pd.isna(ds):
            continue
        sub = df[df["dataset"] == ds] if "dataset" in df.columns else df
        if sub.empty:
            continue

        # 1. Grouped metrics bar
        fig = plot_grouped_metrics(sub, title=f"{step_prefix} — {ds}")
        save_figure(fig, str(out / f"{step_prefix}_grouped_{ds}.pdf"))

        # 2. Metric row
        fig, axes = create_metric_row(sub)
        add_shared_legend(fig, axes)
        save_figure(fig, str(out / f"{step_prefix}_metric_row_{ds}.pdf"))

    print(f"Done — figures saved to {out}")
