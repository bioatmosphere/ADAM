"""Build the 0.5 deg global predictor stack that TabPFN-3.5 is applied to.

The one rule this module exists to enforce: every layer is produced the same way
the matching column of the training table was produced (see ``config`` for the
provenance notes).  In particular

* climate layers are means of *monthly* TerraClimate values over 2001-2010 --
  not annual sums, which would put ppt/aet/pet roughly 12x outside the range the
  model was fitted on;
* soil layers come from SoilGrids 2.0 (the 5 km aggregation of the same product
  the site values were sampled from) and carry the same unit divisors;
* soil moisture comes from ``ec_ors.nc``, the file the training values came from.

``build_stack`` writes the stack to NetCDF and ``check_provenance`` samples it at
the training coordinates and reports, per predictor, how well it reproduces the
stored training values.  That table is the pipeline's main sanity artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .data import load_training_frame


def target_grid(resolution: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    res = resolution or config.GRID_RESOLUTION
    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(-180 + res / 2, 180, res)
    return lat, lon


def fetch_soilgrids(verbose: bool = True) -> None:
    """Download the SoilGrids 2.0 5 km rasters the stack is built from."""
    import urllib.request

    config.SOILGRIDS_5KM_DIR.mkdir(parents=True, exist_ok=True)
    for name, (relpath, _) in config.SOIL_SOURCES.items():
        destination = config.SOILGRIDS_5KM_DIR / relpath.split("/")[-1]
        if destination.exists() and destination.stat().st_size > 0:
            if verbose:
                print(f"  {destination.name} already present")
            continue
        url = f"{config.SOILGRIDS_BASE_URL}/{relpath}"
        if verbose:
            print(f"  downloading {name} from {url}")
        urllib.request.urlretrieve(url, destination)
    if verbose:
        print(f"SoilGrids rasters in {config.SOILGRIDS_5KM_DIR}")


# --- climate ----------------------------------------------------------------
def climate_climatology(resolution: float | None = None, verbose: bool = True) -> xr.Dataset:
    """Mean monthly TerraClimate value per variable over ``config.CLIMATE_YEARS``."""
    res = resolution or config.GRID_RESOLUTION
    lat, lon = target_grid(res)
    native = 1.0 / 24.0
    factor = int(round(res / native))
    if not np.isclose(factor * native, res):
        raise ValueError(f"resolution {res} is not a multiple of TerraClimate's 1/24 deg grid")

    layers = {}
    for var in config.CLIMATE_FEATURES:
        yearly = []
        for year in config.CLIMATE_YEARS:
            path = config.TERRACLIMATE_DIR / f"TerraClimate_{var}_{year}.nc"
            if not path.exists():
                if verbose:
                    print(f"    missing {path.name}, skipping")
                continue
            with xr.open_dataset(path) as ds:
                # Mean over the 12 months: the same statistic the site extraction
                # stored for every variable, including ppt/aet/pet.
                monthly_mean = ds[var].mean(dim="time", skipna=True)
                coarse = monthly_mean.coarsen(lat=factor, lon=factor, boundary="exact").mean()
                yearly.append(coarse.load())
        if not yearly:
            raise FileNotFoundError(f"no TerraClimate files found for {var}")
        stacked = xr.concat(yearly, dim="year").mean(dim="year", skipna=True)
        stacked = stacked.sortby("lat").sortby("lon")
        stacked = stacked.assign_coords(lat=np.round(stacked.lat.values, 6),
                                        lon=np.round(stacked.lon.values, 6))
        layers[var] = stacked.interp(lat=lat, lon=lon, method="nearest")
        if verbose:
            v = layers[var].values
            finite = np.isfinite(v)
            print(f"    {var}: {finite.mean()*100:.1f}% land, "
                  f"{np.nanmin(v):.2f} to {np.nanmax(v):.2f}")
    return xr.Dataset(layers)


# --- soil -------------------------------------------------------------------
def soil_layers(resolution: float | None = None, verbose: bool = True) -> xr.Dataset:
    """SoilGrids 5 km rasters warped to the target grid, in training-table units."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    res = resolution or config.GRID_RESOLUTION
    lat, lon = target_grid(res)
    # North-up destination, flipped to the south-up grid after warping.
    dst_transform = from_origin(-180.0, 90.0, res, res)
    dst_shape = (len(lat), len(lon))

    layers = {}
    for name, (relpath, divisor) in config.SOIL_SOURCES.items():
        path = config.SOILGRIDS_5KM_DIR / relpath.split("/")[-1]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run `python -m src.tabpfn35.run fetch-soil` first"
            )
        destination = np.full(dst_shape, np.nan, dtype="float32")
        with rasterio.open(path) as src:
            source = src.read(1).astype("float32")
            nodata = src.nodata if src.nodata is not None else -32768.0
            source[source == nodata] = np.nan
            reproject(
                source=source,
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
        values = destination[::-1, :] / divisor
        da = xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name=name)
        layers[name] = da
        if verbose:
            v = da.values
            print(f"    {name}: {np.isfinite(v).mean()*100:.1f}% coverage, "
                  f"{np.nanmin(v):.2f} to {np.nanmax(v):.2f}")
    return xr.Dataset(layers)


# --- soil moisture and elevation -------------------------------------------
def soil_moisture_layer(resolution: float | None = None) -> xr.DataArray:
    lat, lon = target_grid(resolution)
    with xr.open_dataset(config.SOIL_MOISTURE_NC) as ds:
        sm = ds["sm"].isel(depth=0).mean(dim="time", skipna=True).load()
    sm = sm.sortby("lat").sortby("lon")
    return sm.interp(lat=lat, lon=lon, method="nearest").rename("soil_moisture")


def elevation_layer(resolution: float | None = None) -> xr.DataArray:
    lat, lon = target_grid(resolution)
    with xr.open_dataset(config.ELEVATION_NC) as ds:
        elev = ds["elevation"].load()
    elev = elev.sortby("lat").sortby("lon")
    return elev.interp(lat=lat, lon=lon, method="nearest").rename("elevation")


# --- assembly ---------------------------------------------------------------
def build_stack(resolution: float | None = None, verbose: bool = True) -> xr.Dataset:
    res = resolution or config.GRID_RESOLUTION
    if verbose:
        print(f"Building global predictor stack at {res} deg")
        print("  TerraClimate 2001-2010 monthly-mean climatology")
    stack = climate_climatology(res, verbose=verbose)

    if verbose:
        print("  SoilGrids 2.0 (5 km aggregation, training-table units)")
    for name, da in soil_layers(res, verbose=verbose).data_vars.items():
        stack[name] = da

    if verbose:
        print("  soil moisture (ec_ors.nc, shallowest depth, time mean)")
    stack["soil_moisture"] = soil_moisture_layer(res)
    if verbose:
        print("  elevation (0.5 deg mean)")
    stack["elevation"] = elevation_layer(res)

    # Land mask: TerraClimate is defined over land only, which makes a finite
    # climate climatology the most reliable land indicator in the stack.
    land = np.isfinite(stack["aet"].values)
    for var in config.CLIMATE_FEATURES[1:]:
        land &= np.isfinite(stack[var].values)
    stack["land_mask"] = xr.DataArray(land, dims=("lat", "lon"),
                                      coords={"lat": stack.lat, "lon": stack.lon})
    stack.attrs.update(
        title="Global predictor stack for TabPFN-3.5 BNPP fraction prediction",
        climate_source=f"TerraClimate {config.CLIMATE_YEARS[0]}-{config.CLIMATE_YEARS[-1]} "
                       "mean of monthly values",
        soil_source="SoilGrids 2.0 aggregated 5 km, warped with area averaging",
        soil_moisture_source="ancillary/soilmoisture/ec_ors.nc, depth 0, time mean",
        elevation_source="ancillary/elevation/global_elevation_0.5deg.nc",
        resolution_degrees=res,
    )
    if verbose:
        print(f"  land cells: {int(land.sum()):,} of {land.size:,}")
    return stack


def save_stack(stack: xr.Dataset, path=None) -> None:
    path = path or config.STACK_NC
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in stack.data_vars}
    stack.to_netcdf(path, encoding=encoding)
    print(f"Stack written to {path}")


def load_stack(path=None) -> xr.Dataset:
    path = path or config.STACK_NC
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `run.py stack` first")
    return xr.open_dataset(path)


def stack_to_frame(stack: xr.Dataset) -> pd.DataFrame:
    """Land cells of the stack as a tidy frame of lat, lon and the 17 predictors."""
    land = stack["land_mask"].values.astype(bool)
    lat2d, lon2d = np.meshgrid(stack.lat.values, stack.lon.values, indexing="ij")
    frame = {"lat": lat2d[land], "lon": lon2d[land]}
    for feature in config.FEATURES:
        frame[feature] = stack[feature].values[land]
    return pd.DataFrame(frame)


# --- quality control --------------------------------------------------------
def check_provenance(stack: xr.Dataset, verbose: bool = True) -> pd.DataFrame:
    """Compare stack values at the training coordinates with the stored columns.

    A predictor whose global layer disagrees with the training column means the
    model would be applied outside the feature distribution it was fitted on, so
    this table is checked before any global prediction is made.
    """
    df = load_training_frame()
    sites = df.drop_duplicates(subset=["lat", "lon"])[["lat", "lon"] + config.FEATURES]

    lat_sel = xr.DataArray(sites["lat"].to_numpy(), dims="site")
    lon_sel = xr.DataArray(sites["lon"].to_numpy(), dims="site")

    rows = []
    for feature in config.FEATURES:
        sampled = stack[feature].sel(lat=lat_sel, lon=lon_sel, method="nearest").values
        stored = sites[feature].to_numpy(dtype=float)
        ok = np.isfinite(sampled) & np.isfinite(stored)
        if ok.sum() < 3:
            rows.append({"feature": feature, "n": int(ok.sum())})
            continue
        a, b = sampled[ok], stored[ok]
        rows.append({
            "feature": feature,
            "n": int(ok.sum()),
            "stack_mean": float(a.mean()),
            "training_mean": float(b.mean()),
            "bias": float((a - b).mean()),
            "rmse": float(np.sqrt(((a - b) ** 2).mean())),
            "correlation": float(np.corrcoef(a, b)[0, 1]),
            "mean_ratio": float(a.mean() / b.mean()) if b.mean() != 0 else np.nan,
        })
    table = pd.DataFrame(rows)
    if verbose:
        print("\nProvenance check (global stack sampled at training coordinates):")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return table
