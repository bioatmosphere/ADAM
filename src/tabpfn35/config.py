"""Paths, feature definitions and provenance constants for the TabPFN-3.5 BNPP pipeline.

Everything the training table and the global predictor stack must agree on lives
here, so that a feature can never be built one way for fitting and another way
for global application.

Provenance of the 17 predictors in ``productivity/earth/aggregated_data.csv``
(established by reverse-engineering the extraction scripts and checking values
against the stored columns):

* climate (aet/pet/ppt/tmax/tmin/vpd) -- TerraClimate, mean of the *monthly*
  values over 2001-2010.  NOTE: these are monthly means, not annual totals;
  ``aet`` averages 55 mm month-1, not ~670 mm year-1.
* soil (9 variables) -- SoilGrids 2.0.  ``soil_carbon_stock`` is OCS 0-30 cm in
  t C ha-1; the other eight are the 0-5 cm mean layers in SoilGrids' native
  integer units divided by the factors in ``SOIL_SOURCES`` (verified against
  ``ancillary/soilgrids/extracted_soil_data.csv``: ratio raw/stored is exactly
  10 for clay/silt/sand/pH, 100 for nitrogen/bulk density, 1 for CEC/cfvo).
* soil_moisture -- ``ancillary/soilmoisture/ec_ors.nc``, shallowest depth,
  averaged over time (99.8 % of stored values reproduce exactly).
* elevation -- point elevation at the site.  The global stack necessarily uses
  the 0.5 deg mean instead (r = 0.92, RMSE 429 m against the site values); this
  is the one predictor whose spatial support differs between fit and application.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- inputs -----------------------------------------------------------------
TRAINING_CSV = REPO_ROOT / "productivity" / "earth" / "aggregated_data.csv"
TERRACLIMATE_DIR = REPO_ROOT / "ancillary" / "terraclimate"
SOILGRIDS_DIR = REPO_ROOT / "ancillary" / "soilgrids"
SOILGRIDS_5KM_DIR = SOILGRIDS_DIR / "aggregated_5000m"
SOIL_MOISTURE_NC = REPO_ROOT / "ancillary" / "soilmoisture" / "ec_ors.nc"
ELEVATION_NC = REPO_ROOT / "ancillary" / "elevation" / "global_elevation_0.5deg.nc"

# --- outputs ----------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "output" / "tabpfn35"
STACK_NC = OUTPUT_DIR / "global_predictor_stack_0.5deg.nc"
STACK_QC_CSV = OUTPUT_DIR / "stack_provenance_check.csv"
PREDICTION_NC = OUTPUT_DIR / "global_bnpp_fraction_tabpfn35.nc"
BATCH_CACHE_DIR = OUTPUT_DIR / "prediction_batches"
FIGURE_DIR = OUTPUT_DIR / "figures"

# --- model / target ---------------------------------------------------------
TARGET = "BNPP_fraction"
MODEL_VERSION = "v3.5"          # TabPFN-3.5 on the Prior Labs API
RANDOM_STATE = 42

CLIMATE_FEATURES = ["aet", "pet", "ppt", "tmax", "tmin", "vpd"]
SOIL_FEATURES = [
    "soil_carbon_stock",
    "clay_content",
    "silt_content",
    "sand_content",
    "nitrogen_content",
    "cation_exchange_capacity",
    "ph_in_water",
    "bulk_density",
    "coarse_fragments",
]
OTHER_FEATURES = ["soil_moisture", "elevation"]
FEATURES = CLIMATE_FEATURES + SOIL_FEATURES + OTHER_FEATURES
assert len(FEATURES) == 17

# --- global grid ------------------------------------------------------------
GRID_RESOLUTION = 0.5           # degrees
CLIMATE_YEARS = tuple(range(2001, 2011))

# --- SoilGrids 2.0 aggregated 5 km rasters ---------------------------------
# (remote filename, divisor to reach the units stored in the training table)
SOILGRIDS_BASE_URL = "https://files.isric.org/soilgrids/latest/data_aggregated/5000m"
SOIL_SOURCES: dict[str, tuple[str, float]] = {
    "soil_carbon_stock": ("ocs/ocs_0-30cm_mean_5000.tif", 1.0),
    "clay_content": ("clay/clay_0-5cm_mean_5000.tif", 10.0),
    "silt_content": ("silt/silt_0-5cm_mean_5000.tif", 10.0),
    "sand_content": ("sand/sand_0-5cm_mean_5000.tif", 10.0),
    "nitrogen_content": ("nitrogen/nitrogen_0-5cm_mean_5000.tif", 100.0),
    "cation_exchange_capacity": ("cec/cec_0-5cm_mean_5000.tif", 1.0),
    "ph_in_water": ("phh2o/phh2o_0-5cm_mean_5000.tif", 10.0),
    "bulk_density": ("bdod/bdod_0-5cm_mean_5000.tif", 100.0),
    "coarse_fragments": ("cfvo/cfvo_0-5cm_mean_5000.tif", 1.0),
}

# --- cross-validation -------------------------------------------------------
SPATIAL_BLOCK_DEGREES = 5.0     # size of the blocks held out in spatial CV
N_FOLDS = 5
PREDICTION_QUANTILES = (0.1, 0.5, 0.9)

# Applicability domain: a grid cell counts as out-of-domain when its distance to
# the nearest training point in standardised feature space exceeds this
# percentile of the training set's own nearest-neighbour distances.
AOA_PERCENTILE = 95.0
