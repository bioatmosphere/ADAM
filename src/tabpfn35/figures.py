"""Figures for the TabPFN-3.5 BNPP-fraction product.

Design rules followed throughout: magnitude is encoded by a single-hue sequential
ramp (never a rainbow), the two map layers use different hues so a reader never
confuses the estimate with its uncertainty, cells outside the applicability
domain are drawn in neutral grey rather than given a colour they do not deserve,
and graticules/coastlines stay recessive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from . import config

# Ink and surface tokens (kept in one place so every figure matches).
INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
SURFACE = "#ffffff"
NO_DATA = "#e6e6e6"
ACCENT = "#2a6f4e"          # same hue family as the magnitude ramp
ACCENT_SECONDARY = "#5b4b8a"

ESTIMATE_CMAP = "Greens"     # single hue, light -> dark
UNCERTAINTY_CMAP = "Purples"
# The first steps of these ramps are almost white, which reads as "no data" next
# to the unpredicted cells. Start the ramp past them so every painted cell is
# visibly painted.
RAMP_START = 0.18


def ramp(name: str, start: float = RAMP_START):
    """A sequential colormap with its near-white head trimmed off."""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_trimmed", base(np.linspace(start, 1.0, 256))
    )


def _style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_MUTED,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _map_axes(fig, position=111):
    """A Robinson map axis if cartopy is available, otherwise a plain lat/lon axis."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax = fig.add_subplot(position, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor=NO_DATA, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=INK_MUTED, zorder=3)
        return ax, ccrs.PlateCarree()
    except ImportError:
        ax = fig.add_subplot(position)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_facecolor(NO_DATA)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return ax, None


def _draw_layer(ax, transform, field: xr.DataArray, cmap: str, vmin: float, vmax: float):
    kwargs = {
        "cmap": ramp(cmap) if isinstance(cmap, str) else cmap,
        "vmin": vmin, "vmax": vmax, "shading": "auto", "zorder": 2,
    }
    if transform is not None:
        kwargs["transform"] = transform
    return ax.pcolormesh(field.lon, field.lat, np.ma.masked_invalid(field.values), **kwargs)


def map_figure(
    field: xr.DataArray,
    title: str,
    label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    save_path,
    note: str | None = None,
) -> None:
    _style()
    fig = plt.figure(figsize=(9.5, 5.2))
    ax, transform = _map_axes(fig)
    mesh = _draw_layer(ax, transform, field, cmap, vmin, vmax)
    ax.set_title(title, pad=12, color=INK)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, shrink=0.6, aspect=38)
    cbar.set_label(label)
    cbar.outline.set_edgecolor(INK_MUTED)
    cbar.outline.set_linewidth(0.5)

    caption = "Grey: land outside the model's applicability domain or without predictors."
    if note:
        caption = f"{caption}  {note}"
    fig.text(0.5, 0.045, caption, ha="center", va="top", fontsize=8.5, color=INK_MUTED)

    save_path = _ensure_parent(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


def prediction_maps(predictions: xr.Dataset | None = None) -> None:
    predictions = predictions or xr.open_dataset(config.PREDICTION_NC)
    estimate = predictions["bnpp_fraction"]
    finite = estimate.values[np.isfinite(estimate.values)]
    vmin, vmax = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) \
        if finite.size else (0.0, 1.0)

    map_figure(
        estimate,
        "Belowground share of net primary productivity, predicted by TabPFN-3.5",
        "BNPP / TNPP (posterior median)",
        ESTIMATE_CMAP,
        vmin,
        vmax,
        config.FIGURE_DIR / "global_bnpp_fraction_tabpfn35.png",
        note=f"Colour range clipped to the 2nd-98th percentile ({vmin:.2f}-{vmax:.2f}).",
    )

    if "bnpp_fraction_interval_width" in predictions:
        width = predictions["bnpp_fraction_interval_width"]
        w = width.values[np.isfinite(width.values)]
        map_figure(
            width,
            "Predictive uncertainty: width of the 80% interval",
            "q90 - q10 (BNPP / TNPP)",
            UNCERTAINTY_CMAP,
            float(np.percentile(w, 2)) if w.size else 0.0,
            float(np.percentile(w, 98)) if w.size else 1.0,
            config.FIGURE_DIR / "global_bnpp_fraction_uncertainty.png",
            note="Wider intervals mark cells the training data constrain least.",
        )


def latitudinal_profile(predictions: xr.Dataset | None = None, save_path=None) -> None:
    """Area-weighted latitudinal mean of the estimate, with the 10-90 band."""
    predictions = predictions or xr.open_dataset(config.PREDICTION_NC)
    weights = np.cos(np.deg2rad(predictions.lat))
    # Latitude bands holding only a handful of predicted cells (the few Antarctic
    # fragments that pass the domain test) would add noise, not signal.
    counts = predictions["bnpp_fraction"].notnull().sum(dim="lon")
    enough = counts >= 20

    def profile(name):
        if name not in predictions:
            return None
        field = predictions[name]
        cell_weights = weights.broadcast_like(field).where(field.notnull()).fillna(0.0)
        return field.weighted(cell_weights).mean(dim="lon", skipna=True).where(enough)

    median = profile("bnpp_fraction")
    low, high = profile("bnpp_fraction_q10"), profile("bnpp_fraction_q90")

    _style()
    fig, ax = plt.subplots(figsize=(5.4, 6.2))
    if low is not None and high is not None:
        ax.fill_betweenx(predictions.lat, low, high, color=to_rgba(ACCENT, 0.18), linewidth=0,
                         label="80% predictive interval")
    ax.plot(median, predictions.lat, color=ACCENT, linewidth=2, label="Posterior median")
    ax.set_xlabel("BNPP / TNPP")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Latitudinal profile of the belowground share", pad=10)
    ax.grid(True, axis="x", color=NO_DATA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(0.98, 0.02, "Latitudes with fewer than 20 predicted cells omitted",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK_MUTED)

    save_path = _ensure_parent(save_path or config.FIGURE_DIR / "latitudinal_profile.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


SCHEME_LABELS = {
    "random": "Random 80/20 split\n(replicate sites on both sides)",
    "site": "Grouped by coordinate\n(unseen site)",
    "spatial": "Blocked 5 deg spatial CV\n(unseen region)",
    "site_mean": "Blocked spatial CV,\none record per coordinate",
}


def validation_figure(
    predictions: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    save_path=None,
) -> None:
    """Observed vs predicted for each validation scheme, one panel per scheme."""
    predictions = predictions if predictions is not None else \
        pd.read_csv(config.OUTPUT_DIR / "validation_predictions.csv")
    scores = scores if scores is not None else \
        pd.read_csv(config.OUTPUT_DIR / "validation_scores.csv")

    schemes = [s for s in ("random", "site", "spatial", "site_mean")
               if s in set(predictions["scheme"])]
    _style()
    fig, axes = plt.subplots(1, len(schemes), figsize=(4.1 * len(schemes), 4.4), sharex=True,
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, scheme in zip(axes, schemes):
        panel = predictions[predictions["scheme"] == scheme]
        row = scores[scores["scheme"] == scheme].iloc[0]
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        # Interval whiskers are informative at a few hundred points and pure ink
        # at a few thousand, where the coverage number carries the same message.
        if ({"q10", "q90"}.issubset(panel.columns) and panel["q10"].notna().any()
                and len(panel) <= 1500):
            ax.vlines(panel["observed"], panel["q10"], panel["q90"],
                      color=to_rgba(ACCENT, 0.12), linewidth=0.8, zorder=2)
        alpha = 0.55 if len(panel) <= 1500 else 0.25
        ax.scatter(panel["observed"], panel["predicted"], s=9,
                   facecolor=to_rgba(ACCENT, alpha), edgecolor="none", zorder=3)
        ax.set_title(SCHEME_LABELS.get(scheme, scheme), fontsize=10.5, pad=8)
        ax.text(0.04, 0.96,
                f"R² {row['r2']:.2f}\nRMSE {row['rmse']:.3f}\n80% coverage "
                f"{row['interval_coverage']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, color=NO_DATA, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Observed BNPP / TNPP")
    axes[0].set_ylabel("Predicted BNPP / TNPP")
    fig.suptitle("TabPFN-3.5 validation: the split decides the score", y=1.02, fontsize=13)

    save_path = _ensure_parent(save_path or config.FIGURE_DIR / "validation_scatter.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


def domain_figure(predictions: xr.Dataset | None = None, save_path=None) -> None:
    """Where the model may be trusted: the applicability domain as a map."""
    predictions = predictions or xr.open_dataset(config.PREDICTION_NC)
    if "domain_distance" not in predictions:
        return
    field = predictions["domain_distance"]
    threshold = float(field.attrs.get("threshold", np.nan))
    map_figure(
        field,
        "Distance to the nearest training site in predictor space",
        f"Standardised distance (in domain <= {threshold:.2f})",
        UNCERTAINTY_CMAP,
        0.0,
        float(np.nanpercentile(field.values, 98)),
        _ensure_parent(save_path or config.FIGURE_DIR / "applicability_domain.png"),
        note="Large distances are extrapolation; those cells are left unpredicted.",
    )


def _ensure_parent(path):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def all_figures() -> None:
    print("Figures:")
    if config.PREDICTION_NC.exists():
        predictions = xr.open_dataset(config.PREDICTION_NC)
        prediction_maps(predictions)
        latitudinal_profile(predictions)
        domain_figure(predictions)
    else:
        print(f"  {config.PREDICTION_NC} not found, skipping maps")
    if (config.OUTPUT_DIR / "validation_predictions.csv").exists():
        validation_figure()
    else:
        print("  validation outputs not found, skipping validation figure")
