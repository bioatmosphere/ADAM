"""Training data loading, grouping for honest cross-validation, applicability domain.

Two properties of the ADAM training table drive the design here:

1. Missing predictor values are *kept*.  TabPFN handles NaN natively, so the
   ~20 % of records with incomplete soil coverage are passed through as-is
   instead of being filled with a column median (which invents structure).
2. The 5 837 records with a BNPP fraction sit at only 530 distinct coordinates,
   and every predictor is a coordinate-level climatology.  Records sharing a
   coordinate therefore share an identical feature vector.  A random train/test
   split puts replicates of the same coordinate on both sides and measures
   memorisation of the site mean rather than the ability to predict a new
   place, so grouped and spatially blocked splits are the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def load_training_frame(path=None) -> pd.DataFrame:
    """Load the aggregated table, keeping only records with a usable target."""
    path = path or config.TRAINING_CSV
    df = pd.read_csv(path)

    missing = [c for c in config.FEATURES + [config.TARGET, "lat", "lon"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[df[config.TARGET].notna()].copy()
    # A fraction outside (0, 1] cannot be a BNPP/TNPP ratio.
    bad = (df[config.TARGET] <= 0) | (df[config.TARGET] > 1)
    if bad.any():
        print(f"  dropping {int(bad.sum())} records with {config.TARGET} outside (0, 1]")
        df = df[~bad].copy()

    df = df.reset_index(drop=True)
    return df


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix (NaN preserved) and target vector."""
    return df[config.FEATURES].copy(), df[config.TARGET].copy()


def site_groups(df: pd.DataFrame) -> np.ndarray:
    """One group per distinct coordinate.

    ``Site_ID`` is present for only 798 of the records, so the coordinate is the
    reliable site key -- and it is also the right one, because the predictors are
    constant within a coordinate.
    """
    key = df["lat"].round(4).astype(str) + "_" + df["lon"].round(4).astype(str)
    return key.to_numpy()


def spatial_block_groups(df: pd.DataFrame, block_degrees: float | None = None) -> np.ndarray:
    """One group per ``block_degrees`` x ``block_degrees`` lat/lon block."""
    size = block_degrees or config.SPATIAL_BLOCK_DEGREES
    lat_block = np.floor(df["lat"].to_numpy() / size).astype(int)
    lon_block = np.floor(df["lon"].to_numpy() / size).astype(int)
    return np.array([f"{a}_{b}" for a, b in zip(lat_block, lon_block)])


def describe_replication(df: pd.DataFrame) -> dict:
    """Summary of the replication structure that motivates grouped CV."""
    sites = site_groups(df)
    blocks = spatial_block_groups(df)
    counts = pd.Series(sites).value_counts()
    return {
        "n_records": int(len(df)),
        "n_sites": int(counts.size),
        "n_spatial_blocks": int(pd.Series(blocks).nunique()),
        "records_per_site_median": float(counts.median()),
        "records_per_site_max": int(counts.max()),
        "largest_site_share": float(counts.max() / len(df)),
    }


@dataclass
class ApplicabilityDomain:
    """Nearest-neighbour applicability domain in standardised feature space.

    Follows the area-of-applicability idea (Meyer & Pebesma 2021): a prediction
    location is in-domain when it is no further from the training data than the
    training data are from each other.

    Missing predictors are handled explicitly rather than by imputing the column
    mean.  A cell is compared to the training set only over the predictors it
    actually has, and the resulting squared distance is rescaled by
    ``n_features / n_available``.  Without that rescaling an ice cell with no
    soil data would land exactly on the training mean in nine of seventeen
    dimensions and be declared in-domain.
    """

    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    train_points: np.ndarray
    threshold: float
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def fit(cls, X: pd.DataFrame, percentile: float | None = None) -> "ApplicabilityDomain":
        percentile = percentile if percentile is not None else config.AOA_PERCENTILE
        values = X.to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        standardised = (values - mean) / scale
        # Reference points carry the column mean where a predictor is missing; the
        # query side is what gets the availability rescaling.
        reference = np.nan_to_num(standardised, nan=0.0, posinf=0.0, neginf=0.0)
        # Records sharing a coordinate have identical feature vectors, so the
        # distance distribution has to be built from the distinct ones -- on the
        # raw records every nearest-neighbour distance would be 0.
        unique_reference, index = np.unique(reference, axis=0, return_index=True)
        d = _nearest_distance(standardised[index], unique_reference, exclude_self=True)
        threshold = float(np.percentile(d, percentile))
        return cls(
            list(X.columns),
            mean,
            scale,
            unique_reference,
            threshold,
            lower=np.nanmin(values, axis=0),
            upper=np.nanmax(values, axis=0),
        )

    def _standardise(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.columns].to_numpy(dtype=float)
        return (values - self.mean) / self.scale

    def distance(self, X: pd.DataFrame) -> np.ndarray:
        """Distance from each row of ``X`` to the nearest training point."""
        return _nearest_distance(self._standardise(X), self.train_points, exclude_self=False)

    def in_range(self, X: pd.DataFrame) -> np.ndarray:
        """True where every observed predictor lies within the training range.

        A univariate screen on top of the distance test.  It is what rules out
        ice sheets and hyper-arid deserts, whose climate sits outside the
        sampled range and where no soil data exist to pull the distance up.
        """
        values = X[self.columns].to_numpy(dtype=float)
        observed = np.isfinite(values)
        below = observed & (values < self.lower)
        above = observed & (values > self.upper)
        return ~(below | above).any(axis=1)

    def inside(self, X: pd.DataFrame, min_observed: int = 6) -> np.ndarray:
        """In-domain = near the training data, inside its range, and not too sparse."""
        values = X[self.columns].to_numpy(dtype=float)
        enough = np.isfinite(values).sum(axis=1) >= min_observed
        return (self.distance(X) <= self.threshold) & self.in_range(X) & enough


def _nearest_distance(
    query: np.ndarray, reference: np.ndarray, exclude_self: bool, chunk: int = 512
) -> np.ndarray:
    """Distance from each (possibly incomplete) query row to the nearest reference row.

    ``query`` may contain NaN; those dimensions are skipped and the squared
    distance is scaled by ``n_features / n_available`` so that a row with few
    observed predictors is not automatically close to everything.
    """
    n_features = query.shape[1]
    out = np.empty(len(query), dtype=float)
    for start in range(0, len(query), chunk):
        stop = min(start + chunk, len(query))
        block = query[start:stop]
        available = np.isfinite(block)
        n_available = available.sum(axis=1)
        filled = np.where(available, block, 0.0)
        diff = filled[:, None, :] - reference[None, :, :]
        diff *= available[:, None, :]
        sq = (diff ** 2).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sq *= np.where(n_available > 0, n_features / np.maximum(n_available, 1), np.inf)[:, None]
        if exclude_self:
            sq[np.arange(stop - start), np.arange(start, stop)] = np.inf
        out[start:stop] = np.sqrt(sq.min(axis=1))
    return out


def site_mean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One record per coordinate, with the mean target.

    Every predictor is constant within a coordinate, so this is the dataset
    without replication: 529 rows, each a distinct feature vector.  Scoring on it
    removes the weighting artefact by which a handful of intensively sampled
    sites dominate record-level metrics.
    """
    columns = config.FEATURES + ["lat", "lon"]
    aggregated = (
        df.groupby(["lat", "lon"], as_index=False)
        .agg({config.TARGET: "mean", **{c: "first" for c in config.FEATURES}})
    )
    counts = df.groupby(["lat", "lon"], as_index=False).size().rename(columns={"size": "n_records"})
    return aggregated.merge(counts, on=["lat", "lon"])
