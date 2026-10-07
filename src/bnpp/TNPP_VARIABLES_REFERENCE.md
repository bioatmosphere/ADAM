# Environmental & Ecological Variables in ForC TNPP Dataset

## Overview

For each of the **368 sites with TNPP_C** (Total Net Primary Production) measurements, the dataset includes **52 variables** organized into 8 categories. This document provides a complete reference of what environmental, ecological, and methodological information is available.

---

## Variable Categories Summary

| Category | # Variables | Data Completeness |
|----------|-------------|-------------------|
| **Geographic** | 5 | 80-100% complete |
| **Climate** | 5 | 8-100% complete |
| **Soil** | 2 | 99-100% complete |
| **Vegetation** | 4 | 8-100% complete |
| **Productivity** | 4 | 100% complete |
| **Measurement** | 6 | 61-100% complete |
| **Methodology** | 3 | 0-65% complete |
| **Quality Control** | 8 | 0-100% complete |

---

## 1. GEOGRAPHIC VARIABLES (100% complete for lat/lon)

### Available Variables:

| Variable | Completeness | Range/Values | Description |
|----------|--------------|--------------|-------------|
| **lat** | 100% | -35.4° to 66.4° | Latitude (decimal degrees) |
| **lon** | 100% | -156.3° to 167.2° | Longitude (decimal degrees) |
| **country** | 100% | 30 countries | Country name |
| **continent** | 100% | 7 continents | Continent name |
| **masl** | 79.9% | 16 to 3,537 m | Elevation (meters above sea level) |

### Geographic Distribution:
- **North America**: 175 sites (47.6%)
- **Europe**: 107 sites (29.1%)
- **South America**: 43 sites (11.7%)
- **Asia**: 16 sites (4.3%)
- **Oceania**: 14 sites (3.8%)
- **Africa**: 8 sites (2.2%)
- **Australia**: 5 sites (1.4%)

### Top Countries:
1. United States: 113 sites
2. Canada: 68 sites
3. Russia: 23 sites
4. Germany: 22 sites
5. China: 21 sites

---

## 2. CLIMATE VARIABLES (82-100% complete)

### Available Variables:

| Variable | Completeness | Range/Values | Description |
|----------|--------------|--------------|-------------|
| **mat** | 82.6% | -8.9°C to 31.5°C | Mean Annual Temperature (°C) |
| **map** | 84.0% | 220 to 5,302 mm | Mean Annual Precipitation (mm) |
| **Koeppen** | 100% | 13 climate types | Köppen climate classification |
| **FAO.ecozone** | 100% | 14 ecozones | FAO ecological zone |
| **climate.notes** | 7.9% | Text descriptions | Additional climate notes |

### Climate Statistics:
- **Mean Temperature**: 9.2°C (±9.4°C)
- **Mean Precipitation**: 1,299 mm (±927 mm)
- **Latitudinal Range**: Tropical to boreal (35°S to 66°N)

### Köppen Climate Classifications:

| Climate Code | Climate Type | N Sites | % |
|--------------|-------------|---------|---|
| **Dfc** | Cold, no dry season, cold summer | 93 | 25.3% |
| **Cfb** | Temperate oceanic | 83 | 22.6% |
| **Csb** | Mediterranean warm summer | 76 | 20.7% |
| **Dfb** | Continental humid | 31 | 8.4% |
| **Cfa** | Humid subtropical | 30 | 8.2% |
| **Af** | Tropical rainforest | 28 | 7.6% |
| **Aw** | Tropical savanna | 10 | 2.7% |
| Others | Various | 17 | 4.6% |

### FAO Ecological Zones:

| Ecozone | N Sites | % |
|---------|---------|---|
| **Temperate mountain system** | 108 | 29.3% |
| **Boreal coniferous forest** | 88 | 23.9% |
| **Temperate oceanic forest** | 43 | 11.7% |
| **Temperate continental forest** | 30 | 8.2% |
| **Tropical mountain system** | 28 | 7.6% |
| **Tropical rainforest** | 26 | 7.1% |
| **Tropical moist forest** | 9 | 2.4% |
| **Tropical dry forest** | 8 | 2.2% |
| Others | 28 | 7.6% |

---

## 3. SOIL VARIABLES (98-100% complete)

### Available Variables:

| Variable | Completeness | # Values | Description |
|----------|--------------|----------|-------------|
| **soil.texture** | 98.6% | 13 textures | Soil texture class |
| **soil.classification** | 100% | 49 types | Soil taxonomy classification |

### Soil Texture Distribution:

| Texture | N Sites | % | Description |
|---------|---------|---|-------------|
| **NAC** | 113 | 30.7% | Not Available/Confirmed |
| **Clay loam** | 80 | 21.7% | Moderate clay + silt + sand |
| **Loam** | 78 | 21.2% | Balanced mixture (ideal) |
| **Clay** | 44 | 12.0% | Fine particles, high water retention |
| **Sand** | 14 | 3.8% | Coarse particles, low water retention |
| **Sandy loam** | 10 | 2.7% | Sand-dominated loam |
| **Loamy sand** | 8 | 2.2% | Slightly loamy sand |
| Others | 21 | 5.7% | Various textures |

### Soil Classification:
- 49 unique soil taxonomy classifications (USDA/FAO systems)
- Includes Alfisols, Andisols, Entisols, Histosols, Inceptisols, Mollisols, Oxisols, Spodosols, Ultisols, etc.

---

## 4. VEGETATION VARIABLES (8-100% complete)

### Available Variables:

| Variable | Completeness | # Values | Description |
|----------|--------------|----------|-------------|
| **dominant.life.form** | 100% | 2 types | Primary vegetation life form |
| **dominant.veg** | 100% | 9 types | Dominant vegetation community |
| **stand.age** | 95.7% | 0-999 years | Forest stand age (years) |
| **scientific.name** | 8.2% | 11 species | Scientific name of dominant species |

### Life Forms:
- **Woody**: 367 sites (99.7%)
- **Woody + Grass**: 1 site (0.3%)

### Forest Type Distribution:

| Code | Forest Type | N Sites | % | Examples |
|------|-------------|---------|---|----------|
| **2TEN** | Temperate Evergreen Needleleaf | 193 | 52.4% | Pine, spruce, fir forests |
| **2TDB** | Temperate Deciduous Broadleaf | 96 | 26.1% | Oak, beech, maple forests |
| **2TEB** | Temperate Evergreen Broadleaf | 58 | 15.8% | Evergreen oak, laurel forests |
| **2TM** | Tropical Mixed | 9 | 2.4% | Mixed tropical species |
| **2TREE** | Tree (unspecified) | 4 | 1.1% | General tree cover |
| **2TDN** | Temperate Deciduous Needleleaf | 2 | 0.5% | Larch forests |
| Others | Various | 6 | 1.6% | Mixed types |

### Stand Age Statistics:
- **Mean**: 203 years
- **Median**: 87 years
- **Range**: 0-999 years
- **Distribution**: Wide range from young plantations to old-growth forests

---

## 5. PRODUCTIVITY VARIABLES (100% complete)

### Available Variables (all 100% complete):

| Variable | Mean ± SD | Median | Range | Units |
|----------|-----------|--------|-------|-------|
| **BNPP_root_C** | 2.13 ± 1.54 | 1.83 | 0.02 - 9.00 | Mg C ha⁻¹ yr⁻¹ |
| **ANPP_2_C** | 4.24 ± 3.18 | 3.58 | 0.00 - 14.46 | Mg C ha⁻¹ yr⁻¹ |
| **TNPP_C** | 6.37 ± 4.17 | 5.67 | 0.06 - 19.70 | Mg C ha⁻¹ yr⁻¹ |
| **BNPP_fraction** | 0.36 ± 0.17 | 0.33 | 0.05 - 1.00 | dimensionless |

### Variable Definitions:

**BNPP_root_C** (Belowground Net Primary Production):
- Annual production of fine and coarse roots
- Measured via root coring, ingrowth cores, or minirhizotrons
- Includes root biomass increment + root turnover

**ANPP_2_C** (Aboveground Net Primary Production - Complete):
- Formula: ANPP_2_C = foliage + woody_stem + woody_branch
- Most complete ANPP measure (includes branch turnover)
- Measured via litterfall collection + diameter increment

**TNPP_C** (Total Net Primary Production):
- Formula: TNPP_C = BNPP_root_C + ANPP_2_C
- Total annual carbon fixation allocated to growth

**BNPP_fraction** (Belowground Allocation Fraction):
- Formula: BNPP_fraction = BNPP_root_C / TNPP_C
- Proportion of NPP allocated belowground
- Indicates carbon allocation strategy

---

## 6. MEASUREMENT VARIABLES (61-100% complete)

### Temporal Information:

| Variable | Completeness | Description |
|----------|--------------|-------------|
| **date** | 73.9% | Measurement year or date |
| **start.date** | 60.6% | Measurement period start |
| **end.date** | 60.6% | Measurement period end |

**Temporal Coverage**: 1960-2013 (53 years)

### Sampling Design:

| Variable | Completeness | Description |
|----------|--------------|-------------|
| **area.sampled** | 100% | Plot area sampled (m²) |
| **depth** | 65.2% | Root sampling depth (cm) |
| **min.dbh** | 65.5% | Minimum tree diameter measured (cm) |

---

## 7. METHODOLOGY VARIABLES (0-65% complete)

### Available Variables:

| Variable | Completeness | Description |
|----------|--------------|-------------|
| **method.ID** | 65.2% | Methodology identifier code |
| **method.category** | 0% | Methodology category (not populated) |
| **method.notes** | 0% | Detailed methodology notes (not populated) |

### Top Measurement Methods:
- 34 unique methodology codes
- Most common: NAC (not available/confirmed), 120, 221, 171
- Methods include sequential coring, ingrowth cores, minirhizotrons, allometric equations

---

## 8. QUALITY CONTROL VARIABLES (0-100% complete)

### Uncertainty Estimates:

| Variable | Completeness | Description |
|----------|--------------|-------------|
| **sd** | 0% | Standard deviation (not populated) |
| **se** | 1.1% | Standard error (rarely provided) |
| **n** | 6.8% | Sample size (rarely provided) |
| **lower95CI** | 100% | Lower 95% confidence interval |
| **upper95CI** | 100% | Upper 95% confidence interval |

### Data Quality Flags:

| Variable | Completeness | Description |
|----------|--------------|-------------|
| **conflicts** | 100% | Data conflicts identified |
| **flag.suspicious** | 100% | Flagged as suspicious (6% of sites) |
| **checked.ori.pub** | 100% | Checked against original publication (8% of sites) |

---

## Example Data Record

Here's a complete example showing all variables for a single site:

```
Site: Andrews 1 (H.J. Andrews Experimental Forest, Oregon, USA)

GEOGRAPHIC:
  Latitude: 44.26°N
  Longitude: -122.20°W
  Elevation: Not available
  Country: United States of America
  Continent: North America

CLIMATE:
  Mean Annual Temperature: Not available
  Mean Annual Precipitation: Not available
  Köppen: Csb (Mediterranean warm summer)
  FAO Ecozone: Temperate mountain system

SOIL:
  Texture: Loam
  Classification: Andisol

VEGETATION:
  Forest Type: 2TEN (Temperate Evergreen Needleleaf)
  Life Form: Woody
  Stand Age: 19 years
  Scientific Name: Not available

PRODUCTIVITY:
  BNPP: 2.27 Mg C ha⁻¹ yr⁻¹
  ANPP: 8.74 Mg C ha⁻¹ yr⁻¹
  TNPP: 11.01 Mg C ha⁻¹ yr⁻¹
  BNPP Fraction: 0.206 (20.6% belowground)

MEASUREMENT:
  Year: 2001
  Method: NAC (Sequential coring)
```

---

## Data Quality Assessment

### Completeness by Category:

**Excellent (>95% complete):**
- Geographic coordinates (lat/lon): 100%
- Climate classification (Köppen, FAO): 100%
- Soil information: 99-100%
- Vegetation type: 96-100%
- Productivity values: 100%

**Good (80-95% complete):**
- Climate variables (MAT, MAP): 83-84%
- Elevation: 80%

**Moderate (60-80% complete):**
- Temporal information: 61-74%
- Sampling design: 65%

**Limited (<60% complete):**
- Species names: 8%
- Uncertainty estimates: 0-7%
- Detailed methodology: 0%

---

## Usage Recommendations

### For Machine Learning Models:

**Strong Predictors (high completeness, high relevance):**
- Geographic: `lat`, `lon`, `masl`
- Climate: `mat`, `map`, `Koeppen`, `FAO.ecozone`
- Soil: `soil.texture`, `soil.classification`
- Vegetation: `dominant.veg`, `stand.age`
- **Productivity**: `ANPP_2_C` (for predicting BNPP)

**Moderate Predictors (moderate completeness):**
- Sampling design: `depth` (root sampling depth)
- Temporal: `date` (for temporal trends)

**Weak Predictors (low completeness):**
- Species-level: `scientific.name`
- Uncertainty: `sd`, `se`, `n`

### Handling Missing Data:

**Climate variables (MAT, MAP):**
- 16-17% missing
- Can be filled using TerraClimate or WorldClim grids based on lat/lon

**Elevation (MASL):**
- 20% missing
- Can be filled using SRTM DEM or similar elevation datasets

**Stand Age:**
- 4% missing
- Consider binning (young/mature/old-growth) or imputation

---

## References

**ForC Database:**
- Anderson-Teixeira, K. J., et al. (2018). ForC: a global database of forest carbon and flux data. *Ecology*, 99(6), 1507.
- GitHub: https://github.com/forc-db/ForC
- Variable definitions: https://github.com/forc-db/ForC/tree/master/data

**Climate Classifications:**
- Köppen: Kottek et al. (2006). World Map of Köppen-Geiger Climate Classification
- FAO Ecozones: FAO (2012). Global Ecological Zones for FAO Forest Reporting

**Soil Classifications:**
- USDA Soil Taxonomy
- FAO-UNESCO Soil Classification System

---

## File Information

**Dataset**: `ForC_BNPP_ANPP_TNPP_processed.csv`

**Location**: `/Users/6lw/Desktop/2_models/ADAM/productivity/forc/`

**Size**: 368 sites × 52 variables

**Generated**: November 4, 2025

**Script**: `ForC_data.py` (modified for BNPP+ANPP+TNPP integration)
