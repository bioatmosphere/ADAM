"""Apply TabPFN-3.5 to the global 0.5 degree predictor stack.

The model is fitted once on every usable record and then predicts the 10th, 50th
and 90th percentiles of the BNPP-fraction posterior for each grid cell, so the
product is a map *with* an uncertainty field rather than a bare point estimate.

By default only cells inside the applicability domain are sent to the API: they
are the cells where a prediction means something, and skipping the rest roughly
halves the metered work.  ``--all-land`` predicts everywhere and keeps the domain
flag as a layer instead.

The model is fitted on one record per coordinate (the site mean) rather than on
all 5 837 records.  Every predictor is constant within a coordinate, so the
records add no information about the inputs -- what they do add is weight: one
intensively sampled site contributes 588 records, and fitting on the raw records
pulls the fitted relationship towards it.  Validation makes the consequence
visible: spatially blocked CV scores R2 = 0.38 per coordinate but about 0 per
record.  ``--use-records`` fits on the raw records instead.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr

from . import backend, config, data, stack


def _empty_grid(like: xr.Dataset) -> np.ndarray:
    return np.full((like.sizes["lat"], like.sizes["lon"]), np.nan, dtype="float32")


def run(
    all_land: bool = False,
    use_records: bool = False,
    batch_size: int | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> xr.Dataset | None:
    predictor_stack = stack.load_stack()
    cells = stack.stack_to_frame(predictor_stack)
    features = cells[config.FEATURES]

    df = data.load_training_frame()
    n_coordinates = int(df[["lat", "lon"]].drop_duplicates().shape[0])
    fit_frame = df if use_records else data.site_mean_frame(df)
    X, y = data.features_and_target(fit_frame)
    domain_X, _ = data.features_and_target(df)

    if verbose:
        basis = "records" if use_records else "site means"
        print(f"Training on {len(X):,} rows ({basis}) from {n_coordinates} coordinates")
        print(f"Land cells in stack: {len(cells):,}")

    domain = data.ApplicabilityDomain.fit(domain_X)
    distance = domain.distance(features)
    inside = domain.inside(features)
    if verbose:
        print(f"Applicability domain (threshold {domain.threshold:.2f}): "
              f"{int(inside.sum()):,} cells in domain ({100*inside.mean():.1f}% of land)")

    selection = np.ones(len(cells), dtype=bool) if all_land else inside
    if limit:
        chosen = np.flatnonzero(selection)[:limit]
        selection = np.zeros(len(cells), dtype=bool)
        selection[chosen] = True
    target_features = features[selection]
    if verbose:
        print(f"Cells to predict: {int(selection.sum()):,}"
              f"{' (all land)' if all_land else ' (in domain)'}")

    if dry_run:
        print("dry run: stack and domain are ready, no API calls made")
        return None

    backend.ensure_client(verbose=verbose)
    quote = backend.estimate_cost(X, target_features)
    if verbose:
        print(f"Cost estimate for the global pass: {quote}")
        print(f"API usage before: {backend.api_usage()}")

    model = backend.fit(X, y, fit_mode="fit_with_cache", verbose=verbose)
    prediction = backend.predict_batched(
        model, target_features, batch_size=batch_size, verbose=verbose
    )
    if verbose:
        print(f"API usage after: {backend.api_usage()}")

    median = prediction.get("median", prediction.get("q0.5"))
    low, high = prediction.get("q0.1"), prediction.get("q0.9")
    if median is None:
        raise RuntimeError("the API returned no median prediction")

    # BNPP/TNPP is a fraction; the posterior can run slightly outside [0, 1].
    median = np.clip(median, 0.0, 1.0)
    if low is not None:
        low = np.clip(low, 0.0, 1.0)
    if high is not None:
        high = np.clip(high, 0.0, 1.0)

    out = xr.Dataset(coords={"lat": predictor_stack.lat, "lon": predictor_stack.lon})
    land = predictor_stack["land_mask"].values.astype(bool)
    row, col = np.where(land)
    row, col = row[selection], col[selection]

    def scatter(values, name, **attrs):
        grid = _empty_grid(predictor_stack)
        if values is not None:
            grid[row, col] = values
        out[name] = xr.DataArray(grid, dims=("lat", "lon"), attrs=attrs)

    scatter(median, "bnpp_fraction",
            long_name="Belowground NPP fraction (BNPP/TNPP), posterior median",
            units="1")
    scatter(low, "bnpp_fraction_q10",
            long_name="BNPP fraction, 10th posterior percentile", units="1")
    scatter(high, "bnpp_fraction_q90",
            long_name="BNPP fraction, 90th posterior percentile", units="1")
    if low is not None and high is not None:
        scatter(high - low, "bnpp_fraction_interval_width",
                long_name="Width of the 80% predictive interval", units="1")

    # Domain layers cover all land cells, not just the predicted ones.
    full_distance = _empty_grid(predictor_stack)
    full_distance[np.where(land)] = distance
    out["domain_distance"] = xr.DataArray(
        full_distance, dims=("lat", "lon"),
        attrs={"long_name": "Distance to the nearest training point in standardised "
                            "feature space",
               "threshold": domain.threshold})
    in_domain = _empty_grid(predictor_stack)
    in_domain[np.where(land)] = inside.astype("float32")
    out["in_domain"] = xr.DataArray(
        in_domain, dims=("lat", "lon"),
        attrs={"long_name": "1 where the cell is inside the applicability domain"})

    out.attrs.update(
        title="Global belowground NPP fraction predicted by TabPFN-3.5",
        model=f"TabPFN-3.5 ({config.MODEL_VERSION}) via the Prior Labs API",
        target=config.TARGET,
        predictors=", ".join(config.FEATURES),
        n_training_rows=len(X),
        training_basis="records" if use_records else "site means (one row per coordinate)",
        n_training_coordinates=n_coordinates,
        applicability_domain_threshold=domain.threshold,
        cells_predicted=int(selection.sum()),
        climate_source=predictor_stack.attrs.get("climate_source", ""),
        soil_source=predictor_stack.attrs.get("soil_source", ""),
    )

    config.PREDICTION_NC.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(config.PREDICTION_NC, encoding=encoding)
    if verbose:
        print(f"\nPredictions written to {config.PREDICTION_NC}")

    summary = {
        "cells_predicted": int(selection.sum()),
        "median_of_medians": float(np.nanmedian(median)),
        "mean_of_medians": float(np.nanmean(median)),
        "p5": float(np.nanpercentile(median, 5)),
        "p95": float(np.nanpercentile(median, 95)),
        "mean_interval_width": float(np.nanmean(high - low)) if low is not None else None,
        "in_domain_cells": int(inside.sum()),
        "land_cells": int(len(cells)),
    }
    with open(config.OUTPUT_DIR / "global_prediction_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    if verbose:
        print(json.dumps(summary, indent=2))
    return out
