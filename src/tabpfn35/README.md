# Global belowground productivity with TabPFN-3.5

**TabPFN-3.5 Hackathon submission — Prior Labs, September–October 2026**

A single foundation model, called through the Prior Labs API, predicts the
**belowground share of net primary productivity** (BNPP / TNPP) for every
half-degree land cell on Earth, with a calibrated predictive interval and an
explicit statement of where the prediction may be trusted.

No other model is used anywhere in this pipeline: TabPFN-3.5 does the fitting,
the validation and the global extrapolation.

## Why this problem

How plants split their production between leaves and roots sets how much carbon
enters the soil, and it is one of the least constrained parameters in land
surface models. Field measurements of belowground productivity are destructive,
scarce and heavily clustered: the ADAM benchmark table holds 5 837 records, but
they sit at only **529 distinct coordinates**. That is precisely the regime
TabPFN is built for — a few hundred effective samples, seventeen heterogeneous
predictors, a third of the soil values missing.

## What the pipeline does

| Step | Command | What happens |
|---|---|---|
| 1 | `run.py fetch-soil` | Downloads the nine SoilGrids 2.0 5 km rasters |
| 2 | `run.py stack` | Builds the 0.5° predictor stack and checks it against the training table |
| 3 | `run.py validate` | Random / site-grouped / spatially-blocked validation of TabPFN-3.5 |
| 4 | `run.py global` | Fits once, predicts q10/q50/q90 for every in-domain land cell |
| 5 | `run.py figures` | Maps, uncertainty map, latitudinal profile, validation panels |

```bash
export TABPFN_TOKEN="<token from platform.priorlabs.ai/account/api-keys>"
uv run python -m src.tabpfn35.run check     # what is on disk, what can run
uv run python -m src.tabpfn35.run validate  # ~6 min, ~0.9 M credits
uv run python -m src.tabpfn35.run global    # ~1 min, ~0.1 M credits
```

## Results

### Validation — the split decides the score

| Split | Scored on | R² | RMSE | 80 % interval coverage |
|---|---|---|---|---|
| Random 80/20 | 1 168 records | **0.54** | 0.141 | 0.81 |
| Grouped by coordinate | 5 837 records | 0.05 | 0.200 | 0.59 |
| Blocked 5° spatial CV | 5 837 records | −0.15 | 0.221 | 0.71 |
| **Blocked 5° spatial CV, one record per coordinate** | **530 sites** | **0.38** | 0.163 | **0.82** |

Read top to bottom, this is the whole story of the dataset. The random split's
R² = 0.54 is close to the 0.588 the repository's benchmark reported for TabPFN —
and it is not a measure of prediction, because replicates of the same feature
vector sit on both sides of the split.

The two record-level grouped rows then look like total failure, but they are not:
they are dominated by replication. One coordinate contributes 588 of the 5 837
records, so its single prediction is scored 588 times. Collapse each coordinate
to one record — which loses no predictor information, since the predictors are
constant within a coordinate — and TabPFN-3.5 recovers **R² = 0.38 on entirely
unseen 5° regions**, with an 80 % predictive interval that covers 82 % of the
held-out observations.

That last row is the number this map stands on: a third of the between-site
variance in belowground allocation is predictable from climate, soil and
elevation alone, and the model's own uncertainty estimate is honest about the
rest.

### The global product

44 788 half-degree cells predicted (46.9 % of land; the rest is outside the
applicability domain), median across predicted cells BNPP/TNPP **0.42**,
5th–95th percentile 0.22–0.75. The pattern is the expected one: a high belowground share
across boreal, tundra and cold-steppe systems (Siberia, northern Canada, the
Tibetan Plateau, Patagonia) and a low share in the humid tropics — recovered from
point measurements alone, with no biome map among the predictors.

The whole job — four validation schemes, sixteen fits and the global pass — cost
**1.08 M of the 20 M credit allowance**, and the 44 788-cell prediction itself ran
in 25 seconds.

## Three things this submission does carefully

**1. Missing values stay missing.** Roughly 13 % of the training predictors and
31 % of the global soil layers are absent. TabPFN handles NaN natively, so
nothing is imputed — the earlier pipeline filled gaps with column medians, which
invents structure the data does not have.

**2. The validation split is chosen to match the task.** Because every predictor
is a coordinate-level climatology, records sharing a coordinate share an
identical feature vector. A random split therefore puts copies of the same input
on both sides and scores memorisation of the site mean. The honest number comes
from holding out whole 5° blocks and scoring one record per coordinate. All four
splits are reported side by side rather than the flattering one alone.

For the same reason the global model is fitted on the 530 site means, not on the
5 837 raw records: the records add no information about the inputs, only weight,
and fitting on them pulls the relationship towards a handful of intensively
sampled sites. `--use-records` fits the other way for comparison.

**3. The map says where it should not be believed.** Two layers travel with the
estimate: the width of TabPFN's 80 % predictive interval, and a nearest-neighbour
applicability domain (after Meyer & Pebesma 2021) that flags cells whose
predictors fall outside the sampled range. Deserts, ice sheets and other
unsampled regimes are left unpredicted rather than coloured in.

## Reproducing this from a fresh clone

Everything in the Results section reproduces from what is committed here, plus a
TabPFN API token. The raw source datasets are *not* needed: the data the model
actually consumes -- the training table it is fitted on, and the predictor stack
it is applied to -- travel with the repository. `run.py check` prints the state
of each input and which commands it unblocks.

Inference is deterministic: re-running the random split against the API with
`random_state=42` reproduces the stored R² of 0.540008 and RMSE of 0.141131
exactly, so a reviewer's numbers should match these digit for digit rather than
merely closely.

**Committed with the submission (9 MB)**

| File | What it is |
|---|---|
| `productivity/earth/aggregated_data.csv` | the 5 837 BNPP records with their 17 predictors |
| `output/tabpfn35/global_predictor_stack_0.5deg.nc` | the 17 predictors on the global 0.5° grid |

With those two, `validate`, `global` and `figures` all run. That covers every
number and every figure in this submission.

**Fetched on demand (72 MB)**

`run.py fetch-soil` pulls the nine SoilGrids 2.0 5 km rasters from
`files.isric.org`. Needed only to rebuild the stack.

**Not shipped (~18 GB), needed only to rebuild the stack from raw sources**

| Source | Size | How to get it |
|---|---|---|
| TerraClimate 2001-2010, six variables | ~9 GB | `python src/ancillary/terraclimate.py --mode download --variables aet pet ppt tmax tmin vpd --start-year 2001 --end-year 2010` |
| Soil moisture, `ancillary/soilmoisture/ec_ors.nc` | ~9 GB | **not a public download** — an EC ORS field provided by its authors (contact address is in the file's attributes) |
| Elevation, `ancillary/elevation/global_elevation_0.5deg.nc` | 2 MB | `python src/ancillary/download_global_elevation.py` |

The soil-moisture field is the reason the stack travels with the submission
rather than being treated as a build artefact: it cannot be re-downloaded at
all, so shipping the 0.5° stack is what makes the global prediction
independently reproducible. `stack_provenance_check.csv` is the audit trail for that stack —
it shows the committed layers reproducing the training table's own values at all
529 coordinates.

## Data provenance, and three bugs it caught

Every global layer is built the way the matching training column was built.
Re-deriving that correspondence from the extraction scripts exposed three
mismatches in the repository's earlier global application:

* **Climate units.** The site values are means of *monthly* TerraClimate values;
  the old global script summed the twelve months, putting precipitation, AET and
  PET roughly 12× outside the fitted range.
* **Soil source.** Site soil came from SoilGrids 2.0; the old global stack used
  OpenLandMap exports, whose organic-carbon values differ from the training
  column by a factor of ~38.
* **Soil moisture file.** Training used `ec_ors.nc`, the old global script used
  `olc_ors.nc` (+0.026 m³/m³ bias).

`output/tabpfn35/stack_provenance_check.csv` samples the finished stack at the
529 training coordinates and reports bias, RMSE and correlation for all
seventeen predictors. Correlations run 0.86–1.00 with near-zero bias; the single
expected exception is elevation, where a 0.5° mean necessarily differs from a
point measurement (r = 0.91, RMSE 429 m).

## Outputs

```
output/tabpfn35/
├── global_predictor_stack_0.5deg.nc     # the 17 predictors on the global grid
├── stack_provenance_check.csv           # stack vs training table, per predictor
├── validation_scores.csv                # R², RMSE, MAE, bias, interval coverage
├── validation_predictions.csv           # every out-of-fold prediction
├── global_bnpp_fraction_tabpfn35.nc     # q10 / median / q90 + domain layers
├── global_prediction_summary.json
└── figures/
```

## Module layout

```
src/tabpfn35/
├── config.py        paths, the 17 predictors, provenance constants
├── data.py          training table, CV groupings, applicability domain
├── stack.py         global predictor stack + provenance check
├── backend.py       the only code that calls TabPFN-3.5 (API, batching, resume)
├── validation.py    random / site / spatial-block validation
├── apply_global.py  global prediction with uncertainty
├── figures.py       maps and validation panels
└── run.py           command line entry point
```
