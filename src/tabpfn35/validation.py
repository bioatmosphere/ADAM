"""Validation of TabPFN-3.5 on the ADAM BNPP-fraction data.

Three splitting schemes are run, in increasing honesty for the task the global
map represents (predicting a place the model has never seen):

``random``
    Plain 80/20 split over records.  Reported only for comparability with the
    repository's existing benchmark -- 5 837 records sit at 529 coordinates and
    every predictor is a coordinate-level climatology, so this split puts
    replicates of the same feature vector on both sides and measures how well
    the site mean is memorised.
``site``
    Grouped K-fold with one group per coordinate.  No feature vector appears in
    both training and test.
``spatial``
    Grouped K-fold on 5-degree spatial blocks, so whole regions are held out and
    short-range spatial autocorrelation cannot leak either.
``site_mean``
    As ``spatial``, but on one record per coordinate (529 rows, each a distinct
    feature vector, target averaged).  This removes the weighting artefact by
    which a few intensively sampled coordinates dominate record-level metrics.

Alongside R^2/RMSE/MAE each scheme reports the empirical coverage of TabPFN's
80 % predictive interval, which says whether the uncertainty drawn on the global
map can be believed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split

from . import backend, config, data


@dataclass
class Scores:
    scheme: str
    n_test: int
    r2: float
    rmse: float
    mae: float
    bias: float
    interval_coverage: float
    interval_width: float


def _score(scheme: str, y_true: np.ndarray, prediction: dict[str, np.ndarray]) -> Scores:
    point = prediction.get("median", prediction.get("mean"))
    low, high = prediction.get("q0.1"), prediction.get("q0.9")
    if low is not None and high is not None:
        coverage = float(np.mean((y_true >= low) & (y_true <= high)))
        width = float(np.mean(high - low))
    else:
        coverage = width = float("nan")
    return Scores(
        scheme=scheme,
        n_test=int(len(y_true)),
        r2=float(r2_score(y_true, point)),
        rmse=float(np.sqrt(mean_squared_error(y_true, point))),
        mae=float(mean_absolute_error(y_true, point)),
        bias=float(np.mean(point - y_true)),
        interval_coverage=coverage,
        interval_width=width,
    )


def random_split(X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> tuple[Scores, pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    if verbose:
        print(f"\nrandom 80/20 split: {len(X_train)} train / {len(X_test)} test")
    model = backend.fit(X_train, y_train, fit_mode="fit_preprocessors", verbose=verbose)
    prediction = backend.with_retry(lambda: backend.predict_distribution(model, X_test), 4, verbose)
    scores = _score("random", y_test.to_numpy(), prediction)
    frame = pd.DataFrame({
        "scheme": "random",
        "observed": y_test.to_numpy(),
        "predicted": prediction.get("median", prediction.get("mean")),
        "q10": prediction.get("q0.1"),
        "q90": prediction.get("q0.9"),
    })
    if verbose:
        _print_scores(scores)
    return scores, frame


def grouped_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    scheme: str,
    n_folds: int | None = None,
    verbose: bool = True,
) -> tuple[Scores, pd.DataFrame]:
    """Out-of-fold predictions with whole groups held out."""
    n_folds = n_folds or config.N_FOLDS
    n_groups = len(np.unique(groups))
    n_folds = min(n_folds, n_groups)
    splitter = GroupKFold(n_splits=n_folds)

    if verbose:
        print(f"\n{scheme} grouped {n_folds}-fold CV over {n_groups} groups")

    pieces = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        if verbose:
            print(f"  fold {fold}/{n_folds}: {len(train_idx)} train / {len(test_idx)} test")
        model = backend.fit(
            X.iloc[train_idx], y.iloc[train_idx], fit_mode="fit_preprocessors", verbose=verbose
        )
        prediction = backend.with_retry(
            lambda: backend.predict_distribution(model, X.iloc[test_idx]), 4, verbose
        )
        pieces.append(pd.DataFrame({
            "scheme": scheme,
            "fold": fold,
            "observed": y.iloc[test_idx].to_numpy(),
            "predicted": prediction.get("median", prediction.get("mean")),
            "q10": prediction.get("q0.1"),
            "q90": prediction.get("q0.9"),
        }))

    frame = pd.concat(pieces, ignore_index=True)
    prediction = {
        "median": frame["predicted"].to_numpy(),
        "q0.1": frame["q10"].to_numpy(),
        "q0.9": frame["q90"].to_numpy(),
    }
    scores = _score(scheme, frame["observed"].to_numpy(), prediction)
    if verbose:
        _print_scores(scores)
        per_fold = frame.groupby("fold").apply(
            lambda g: r2_score(g["observed"], g["predicted"]), include_groups=False
        )
        print(f"    per-fold R2: {', '.join(f'{v:.3f}' for v in per_fold)}")
    return scores, frame


def _print_scores(s: Scores) -> None:
    print(f"    R2 {s.r2:+.3f} | RMSE {s.rmse:.4f} | MAE {s.mae:.4f} | bias {s.bias:+.4f} "
          f"| 80% interval coverage {s.interval_coverage:.3f} (width {s.interval_width:.3f})")


def run(
    schemes: tuple[str, ...] = ("random", "site", "spatial", "site_mean"),
    n_folds: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the requested validation schemes and write results to the output directory."""
    df = data.load_training_frame()
    X, y = data.features_and_target(df)
    replication = data.describe_replication(df)
    if verbose:
        print(f"Validating TabPFN-3.5 on {replication['n_records']:,} records at "
              f"{replication['n_sites']} coordinates "
              f"({replication['n_spatial_blocks']} 5-degree blocks)")
        print(f"missing predictor values: {X.isna().to_numpy().mean()*100:.1f}% "
              "(passed to TabPFN as NaN, not imputed)")

    backend.ensure_client(verbose=verbose)
    if verbose:
        print(f"API usage before validation: {backend.api_usage()}")

    all_scores, all_frames = [], []
    for scheme in schemes:
        if scheme == "random":
            scores, frame = random_split(X, y, verbose=verbose)
        elif scheme == "site":
            scores, frame = grouped_cv(X, y, data.site_groups(df), "site", n_folds, verbose)
        elif scheme == "spatial":
            scores, frame = grouped_cv(
                X, y, data.spatial_block_groups(df), "spatial", n_folds, verbose
            )
        elif scheme == "site_mean":
            site_df = data.site_mean_frame(df)
            Xs, ys = data.features_and_target(site_df)
            scores, frame = grouped_cv(
                Xs, ys, data.spatial_block_groups(site_df), "site_mean", n_folds, verbose
            )
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        all_scores.append(scores)
        all_frames.append(frame)

    scores_df = pd.DataFrame([asdict(s) for s in all_scores])
    predictions_df = pd.concat(all_frames, ignore_index=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(config.OUTPUT_DIR / "validation_scores.csv", index=False)
    predictions_df.to_csv(config.OUTPUT_DIR / "validation_predictions.csv", index=False)
    with open(config.OUTPUT_DIR / "validation_summary.json", "w") as handle:
        json.dump(
            {
                "model": f"TabPFN-3.5 ({config.MODEL_VERSION}) via the Prior Labs API",
                "target": config.TARGET,
                "features": config.FEATURES,
                "replication": replication,
                "scores": [asdict(s) for s in all_scores],
            },
            handle,
            indent=2,
        )
    if verbose:
        print("\n" + scores_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\nwritten to {config.OUTPUT_DIR}")
        print(f"API usage after validation: {backend.api_usage()}")
    return scores_df, predictions_df
