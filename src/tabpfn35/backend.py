"""TabPFN-3.5 inference through the Prior Labs API.

This is the only place in the pipeline that talks to a model.  There is no
local-weights path and no fallback estimator: every number the pipeline reports
comes from TabPFN-3.5 served by ``platform.priorlabs.ai``.

Three things make the global pass (tens of thousands of grid cells) survivable
on a metered API:

* ``fit_mode="fit_with_cache"`` -- the training set is uploaded and fitted once,
  and every prediction batch reuses that fit.
* batch sizes derived from the server's own limits (``test_set_max_rows``, and
  the train-rows x test-rows budget), not guessed.
* every batch is written to disk as soon as it returns, so an interrupted or
  quota-limited run resumes instead of re-spending on completed work.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# A fit held in the server-side cache accepts at most this many test rows per
# call, independently of the model's own test_set_max_rows.
CACHED_FIT_MAX_TEST_ROWS = 10_000

TOKEN_HELP = (
    "No TabPFN API token found. Create one at "
    "https://platform.priorlabs.ai/account/api-keys and export it:\n"
    '    export TABPFN_TOKEN="<your-token>"'
)


def ensure_client(verbose: bool = True) -> None:
    """Authenticate against the Prior Labs API, failing fast without a token."""
    import tabpfn_client

    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError(TOKEN_HELP)
    tabpfn_client.init()
    if verbose:
        print(f"TabPFN API ready (model {config.MODEL_VERSION})")


def api_usage() -> str | None:
    import tabpfn_client

    try:
        return tabpfn_client.get_api_usage()
    except Exception as exc:  # usage reporting must never break a run
        return f"usage unavailable ({exc})"


def estimate_cost(X_train, X_test=None, operation: str = "predict") -> dict:
    """Quote an operation without uploading data or spending quota."""
    import tabpfn_client

    try:
        result = tabpfn_client.estimate_cost(
            X_train, X_test, model_version=config.MODEL_VERSION, operation=operation
        )
        return {
            "estimated_cost": getattr(result, "estimated_cost", None),
            "pricing_version": getattr(result, "pricing_version", None),
            "inputs": getattr(result, "inputs", None),
        }
    except Exception as exc:
        return {"error": str(exc)}


def make_regressor(fit_mode: str = "fit_with_cache", **overrides):
    """A TabPFN-3.5 regressor on the API."""
    from tabpfn_client import TabPFNRegressor

    options = {"random_state": config.RANDOM_STATE, "fit_mode": fit_mode}
    options.update(overrides)
    return TabPFNRegressor.create_default_for_version(config.MODEL_VERSION, **options)


def fit(X: pd.DataFrame, y: pd.Series, fit_mode: str = "fit_with_cache", verbose: bool = True):
    """Fit TabPFN-3.5 on the API and return the fitted estimator."""
    limits = model_limits()
    if limits is not None and len(X) > limits.train_set_max_rows:
        raise ValueError(
            f"{len(X)} training rows exceed the API limit of {limits.train_set_max_rows} "
            f"for {config.MODEL_VERSION}"
        )
    model = make_regressor(fit_mode=fit_mode)
    start = time.time()
    model.fit(X, y)
    if verbose:
        print(f"  fitted on {len(X)} rows x {X.shape[1]} features in {time.time() - start:.1f}s")
    return model


def model_limits():
    """The server's dataset limits for TabPFN-3.5, or None if unavailable."""
    try:
        from tabpfn_client.api_models import ModelVersion
        from tabpfn_client.client import ServiceClient

        settings = ServiceClient.get_settings()
        if settings is None:
            return None
        return settings.model_limits[ModelVersion(config.MODEL_VERSION)]
    except Exception:
        return None


def max_test_rows(
    train_rows: int, requested: int | None = None, cached_fit: bool = True
) -> int:
    """Largest batch the API accepts for a fit of ``train_rows`` rows."""
    ceiling = CACHED_FIT_MAX_TEST_ROWS if cached_fit else None
    limits = model_limits()
    if limits is None:
        fallback = requested or 5_000
        return min(fallback, ceiling) if ceiling else fallback
    allowed = min(limits.test_set_max_rows, limits.predict_row_pairs_budget // max(train_rows, 1))
    allowed = min(allowed, limits.test_set_max_cells // max(len(config.FEATURES), 1))
    if ceiling:
        allowed = min(allowed, ceiling)
    if requested:
        allowed = min(allowed, requested)
    return max(int(allowed), 1)


# --- prediction -------------------------------------------------------------
def _batch_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha1(np.ascontiguousarray(frame.to_numpy(dtype="float64")).tobytes()).hexdigest()


def predict_distribution(
    model, X: pd.DataFrame, quantiles=config.PREDICTION_QUANTILES
) -> dict[str, np.ndarray]:
    """Central estimate and predictive quantiles for ``X``, in as few calls as possible.

    The API can return several summaries of the predictive distribution in one
    response (``output_type="main"``).  When the server does not support that,
    this falls back to asking for the quantiles alone, which still yields the
    median used as the point estimate.
    """
    quantiles = list(quantiles)
    out: dict[str, np.ndarray] = {}

    try:
        main = model.predict(X, output_type="main")
    except Exception:
        main = None

    if isinstance(main, dict):
        for key in ("mean", "median"):
            if key in main:
                out[key] = np.asarray(main[key], dtype=float).ravel()
        if "quantiles" in main:
            block = np.asarray(main["quantiles"], dtype=float)
            # Server-side quantile grids are not guaranteed to match the request,
            # so only trust them when the shape lines up.
            if block.ndim == 2 and block.shape[0] == len(quantiles):
                for q, row in zip(quantiles, block):
                    out[f"q{q}"] = row.ravel()

    missing = [q for q in quantiles if f"q{q}" not in out]
    if missing:
        block = model.predict(X, output_type="quantiles", quantiles=missing)
        for q, row in zip(missing, block):
            out[f"q{q}"] = np.asarray(row, dtype=float).ravel()

    if "median" not in out and "q0.5" in out:
        out["median"] = out["q0.5"]
    return out


def predict_batched(
    model,
    X: pd.DataFrame,
    quantiles=config.PREDICTION_QUANTILES,
    batch_size: int | None = None,
    cache_dir: Path | None = None,
    verbose: bool = True,
    max_retries: int = 4,
) -> dict[str, np.ndarray]:
    """Predict for every row of ``X``, batch by batch, caching as it goes.

    Completed batches are written to ``cache_dir`` keyed by their position and a
    digest of their feature values, so re-running after an interruption or a
    quota stop only pays for what is still missing.
    """
    quantiles = list(quantiles)
    train_rows = int(getattr(model, "n_train_rows_", 0) or 0)
    cached_fit = getattr(model, "fit_mode", None) == "fit_with_cache"
    batch_size = max_test_rows(
        train_rows or len(X), requested=batch_size or 10_000, cached_fit=cached_fit
    )
    cache_dir = cache_dir or config.BATCH_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_batches = int(np.ceil(len(X) / batch_size))
    if verbose:
        print(f"  predicting {len(X):,} rows in {n_batches} batch(es) of up to {batch_size:,}")

    collected: list[dict[str, np.ndarray]] = []
    reused = 0
    started = time.time()
    for index in range(n_batches):
        start, stop = index * batch_size, min((index + 1) * batch_size, len(X))
        batch = X.iloc[start:stop]
        cache_path = cache_dir / f"batch_{index:05d}_{_batch_digest(batch)[:12]}.npz"

        if cache_path.exists():
            cached = np.load(cache_path)
            if all(f"q{q}" in cached for q in quantiles):
                collected.append({k: cached[k] for k in cached.files})
                reused += 1
                continue

        values = with_retry(
            lambda: predict_distribution(model, batch, quantiles), max_retries, verbose
        )
        np.savez_compressed(cache_path, **values)
        collected.append(values)
        if verbose:
            elapsed = time.time() - started
            print(f"    batch {index + 1}/{n_batches} ({stop:,} rows, {elapsed:.0f}s elapsed)")

    if verbose and reused:
        print(f"  reused {reused} cached batch(es)")

    keys = set.intersection(*(set(d) for d in collected)) if collected else set()
    return {key: np.concatenate([d[key] for d in collected]) for key in sorted(keys)}


def _is_transient(exc: Exception) -> bool:
    """Whether retrying could plausibly help.

    A 4xx other than 429 means the request itself is wrong -- a different batch
    size, say -- so repeating it only burns time.
    """
    message = str(exc)
    if "HTTP 429" in message:
        return True
    return not any(f"HTTP {code}" in message for code in range(400, 500))


def with_retry(call, max_retries: int, verbose: bool):
    delay = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == max_retries or not _is_transient(exc):
                raise
            if verbose:
                print(f"    attempt {attempt} failed ({type(exc).__name__}: {exc}); "
                      f"retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")
