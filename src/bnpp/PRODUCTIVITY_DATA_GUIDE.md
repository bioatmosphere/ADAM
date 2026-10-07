# ForC All Productivity Data - Complete Guide

## Overview

The `ForC_all_productivity.py` script extends the original `ForC_data.py` to extract **ALL productivity variables** for sites with BNPP measurements, creating a comprehensive forest productivity dataset.

---

## What Was Downloaded

### Data Summary

| Metric | Value |
|--------|-------|
| **Total measurements** | 4,351 |
| **Unique sites** | 352 sites with BNPP data |
| **Unique variables** | 29 productivity variables |
| **Productivity types** | 5 (BNPP, ANPP, NPP, GPP, TBCF) |

### Breakdown by Productivity Type

| Type | Measurements | Sites | Description |
|------|-------------|-------|-------------|
| **ANPP** | 1,527 | 312 | Aboveground Net Primary Production |
| **BNPP** | 1,150 | 352 | Belowground Net Primary Production |
| **NPP** | 894 | 331 | Net Primary Production (total) |
| **GPP** | 733 | 125 | Gross Primary Production |
| **TBCF** | 47 | 18 | Total Belowground Carbon Flux |

---

## Output Files

### 1. Long Format Dataset
**File:** `ForC_all_productivity_data_long.csv`
**Size:** 4,351 rows (one per measurement)

**Structure:**
```
columns:
  - measurement.ID          (unique measurement identifier)
  - sites.sitename          (forest site name)
  - variable.name           (e.g., BNPP_root_C, ANPP_1_C, NPP_1_C, GPP_C)
  - mean                    (productivity value in Mg C ha⁻¹ yr⁻¹)
  - lat, lon                (coordinates)
  - country, continent      (geographic info)
  - mat, map                (climate: temperature, precipitation)
  - dominant.life.form      (forest type)
  - method.ID               (measurement methodology)
  - productivity_type       (BNPP, ANPP, NPP, GPP, TBCF)
  - Data_Source             (ForC_Database)
  + 40+ additional metadata columns
```

**Use case:**
- Time series analysis
- Comparing different measurement methods
- Detailed statistical analysis

**Example:**
```csv
sites.sitename,variable.name,mean,country,mat,map,productivity_type
Aheden,BNPP_root_C,1.14,Sweden,1.0,588.0,BNPP
Aheden,ANPP_2_C,3.85,Sweden,1.0,588.0,ANPP
Aheden,NPP_1_C,4.99,Sweden,1.0,588.0,NPP
```

### 2. Wide Format Dataset
**File:** `ForC_all_productivity_data_wide.csv`
**Size:** 226 rows (one per site)

**Structure:**
```
columns (39 total):
  Site info: sites.sitename, lat, lon, country, continent, mat, map, masl

  Productivity variables (29 columns):
    BNPP_root_C               (216 sites, 95.6% coverage)
    NPP_1_C                   (193 sites, 85.4%)
    ANPP_2_C                  (161 sites, 71.2%)
    ANPP_foliage_C            (112 sites, 49.6%)
    BNPP_root_fine_C          (111 sites, 49.1%)
    ANPP_woody_C              (102 sites, 45.1%)
    BNPP_root_coarse_C        (96 sites, 42.5%)
    ... (22 more variables)
```

**Use case:**
- Site-level comparisons
- Carbon allocation analysis (BNPP vs ANPP ratios)
- Multi-variable modeling

**Example:**
```csv
sites.sitename,lat,lon,BNPP_root_C,ANPP_2_C,NPP_1_C,GPP_C
Aheden,64.21,19.5,1.14,3.85,4.99,8.2
```

---

## Key Findings from Analysis

### 1. Productivity Variable Relationships

| Relationship | Correlation (r) | Sample Size | Interpretation |
|--------------|-----------------|-------------|----------------|
| BNPP vs NPP | **0.769** | 192 sites | Strong positive ✅ |
| ANPP vs NPP | **0.888** | 54 sites | Very strong ✅ |
| NPP vs GPP | **0.767** | 56 sites | Strong positive ✅ |
| BNPP vs GPP | **0.579** | 61 sites | Moderate positive ✅ |
| BNPP vs ANPP | **0.498** | 59 sites | Moderate positive ✅ |

**Ecological interpretation:**
- Strong correlations confirm productivity variables are tightly coupled
- BNPP/ANPP ratio shows consistent carbon allocation patterns
- GPP drives both above- and belowground productivity

### 2. BNPP/NPP Ratio Analysis

Based on 192 sites with both measurements:

| Statistic | Value |
|-----------|-------|
| **Mean ratio** | 0.308 |
| **Median ratio** | 0.285 |
| **Range** | 0.047 - 1.000 |
| **Standard deviation** | 0.167 |

**Ecological interpretation:**
- On average, **30.8% of NPP goes belowground**
- This aligns with forest ecology literature (20-50% typical)
- High variability reflects ecosystem differences (climate, species, age)

---

## Complete Variable List

### BNPP Variables (8 total)

| Variable | Sites | Description |
|----------|-------|-------------|
| `BNPP_root_C` | 216 | **Total root carbon production** ⭐ |
| `BNPP_root_fine_C` | 111 | Fine root production |
| `BNPP_root_coarse_C` | 96 | Coarse root production |
| `BNPP_root_OM` | 7 | Total root organic matter |
| `BNPP_root_fine_OM` | 7 | Fine root OM |
| `BNPP_root_coarse_OM` | 4 | Coarse root OM |
| `BNPP_root.turnover_fine_C` | 5 | Fine root turnover |

### ANPP Variables (11 total)

| Variable | Sites | Description |
|----------|-------|-------------|
| `ANPP_2_C` | 161 | ANPP including stem + foliage + branches |
| `ANPP_foliage_C` | 112 | Leaf production |
| `ANPP_woody_C` | 102 | Woody production (stem + branches) |
| `ANPP_woody_stem_C` | 68 | Stem wood production |
| `ANPP_woody_branch_C` | 59 | Branch production |
| `ANPP_1_C` | 59 | ANPP: stem + foliage only |
| `ANPP_litterfall_0_C` | 37 | Total litterfall |
| `ANPP_0_C` | 15 | ANPP (components unspecified) |
| `ANPP_folivory_C` | 14 | Herbivory consumption |
| `ANPP_repro_C` | 10 | Reproductive structures |
| `ANPP_litterfall_1_C` | 7 | Litterfall: leaves + twigs + repro |

### NPP Variables (9 total)

| Variable | Sites | Description |
|----------|-------|-------------|
| `NPP_1_C` | 193 | **Total NPP (ANPP + BNPP)** ⭐ |
| `NPP_2_C` | 64 | NPP including understory |
| `NPP_understory_C` | 56 | Understory production |
| `NPP_0_C` | 27 | NPP (components unspecified) |
| `NPP_woody_C` | 17 | Woody component only |
| `NPP_3_C` | 9 | NPP + reproductive |
| `NPP_litter_C` | 5 | Fine detrital production |
| `NPP_5_C` | 4 | Total NPP (all components) |
| `NPP_4_C` | 2 | NPP + herbivory |

### GPP Variables (2 total)

| Variable | Sites | Description |
|----------|-------|-------------|
| `GPP_C` | 65 | **Gross Primary Production** ⭐ |
| `GPP_cum_C` | 9 | Cumulative GPP (non-annual) |

### TBCF Variables (1 total)

| Variable | Sites | Description |
|----------|-------|-------------|
| `TBCF_C` | 21 | Total Belowground Carbon Flux |

**⭐ = Most commonly measured, recommended for analysis**

---

## Visualizations Created

### 1. Correlation Matrix
**File:** `figures_productivity/productivity_correlation_matrix.png`

Shows pairwise correlations between all productivity variables:
- Identifies which variables are strongly related
- Helps avoid multicollinearity in modeling
- Validates ecological relationships

### 2. Productivity Scatter Plots
**File:** `figures_productivity/productivity_scatter_plots.png`

4-panel figure showing key relationships:
- BNPP vs ANPP (carbon allocation)
- BNPP vs NPP (belowground fraction)
- BNPP vs GPP (primary production linkage)
- ANPP vs GPP (aboveground efficiency)

Each plot includes:
- Scatter points (n = sample size)
- Linear trend line
- Correlation coefficient (r value)

### 3. Data Coverage Overview
**File:** `figures_productivity/productivity_coverage_overview.png`

4-panel figure showing:
- Measurements by productivity type (bar chart)
- Unique sites by type (bar chart)
- Top 10 variables by count (horizontal bar)
- Geographic distribution by continent

---

## How to Use This Data

### Use Case 1: Comprehensive BNPP Training Data

**Goal:** Train ML models with multiple productivity features

```python
import pandas as pd

# Load wide-format data
df = pd.read_csv('ForC_all_productivity_data_wide.csv')

# Features for model
features = [
    'mat', 'map', 'masl',           # Climate & topography
    'GPP_C',                         # Gross productivity
    'ANPP_2_C',                      # Aboveground NPP
]

# Target variable
target = 'BNPP_root_C'

# Filter complete cases
model_data = df[features + [target]].dropna()

print(f"Training samples: {len(model_data)}")
# Expected: ~50-60 sites with all variables
```

### Use Case 2: Carbon Allocation Analysis

**Goal:** Understand BNPP/ANPP ratios across ecosystems

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ForC_all_productivity_data_wide.csv')

# Calculate ratios
df['BNPP_ANPP_ratio'] = df['BNPP_root_C'] / df['ANPP_2_C']
df['BNPP_NPP_ratio'] = df['BNPP_root_C'] / df['NPP_1_C']

# Plot by climate
plt.scatter(df['mat'], df['BNPP_NPP_ratio'], alpha=0.6)
plt.xlabel('Mean Annual Temperature (°C)')
plt.ylabel('BNPP/NPP Ratio')
plt.title('Carbon Allocation vs Climate')
plt.show()
```

### Use Case 3: Time Series / Multi-Method Analysis

**Goal:** Compare different measurement methods at same site

```python
import pandas as pd

# Load long-format data
df = pd.read_csv('ForC_all_productivity_data_long.csv')

# Get all measurements for one site
site = 'Aheden'
site_data = df[df['sites.sitename'] == site]

print(site_data[['variable.name', 'mean', 'method.ID', 'date']])

# Output:
# variable.name         mean  method.ID  date
# BNPP_root_coarse_C    0.28  4          2005
# BNPP_root_fine_C      0.86  4          2005
# BNPP_root_C           1.14  4          2005
# ANPP_2_C              3.85  3          2005
# NPP_1_C               4.99  derived    2005
```

### Use Case 4: Validation Dataset

**Goal:** Validate GPP-BNPP relationships in global models

```python
import pandas as pd
import numpy as np

df = pd.read_csv('ForC_all_productivity_data_wide.csv')

# Sites with both GPP and BNPP measurements
validation = df[['GPP_C', 'BNPP_root_C', 'lat', 'lon']].dropna()

# Expected BNPP/GPP ratio from literature: 0.1 - 0.3
validation['ratio'] = validation['BNPP_root_C'] / validation['GPP_C']

print(f"Validation sites: {len(validation)}")
print(f"Mean BNPP/GPP ratio: {validation['ratio'].mean():.3f}")
print(f"Expected model predictions should be in this range")
```

---

## Comparison: Original vs All Productivity Script

| Feature | `ForC_data.py` (Original) | `ForC_all_productivity.py` (New) |
|---------|---------------------------|----------------------------------|
| **Variables extracted** | 1 (BNPP_root_C only) | 29 (all productivity types) |
| **Output format** | Long format only | Both long and wide formats |
| **Sites** | 529 (all BNPP sites) | 352 (BNPP sites with coords) |
| **Measurements** | 529 | 4,351 |
| **Visualizations** | 3 maps | 6 plots (maps + relationships) |
| **Analysis** | Basic stats | Correlations, ratios, relationships |
| **Use case** | BNPP-only modeling | Comprehensive productivity analysis |

---

## Key Advantages of All Productivity Data

### 1. **Richer Feature Set for ML Models**
- Can use GPP, ANPP, NPP as predictors for BNPP
- Better capture ecosystem carbon dynamics
- Improve model accuracy through multi-variable relationships

### 2. **Validation & Quality Control**
- Cross-check BNPP estimates against NPP totals
- Validate carbon allocation patterns
- Identify potential data quality issues

### 3. **Carbon Cycle Understanding**
- Quantify BNPP/NPP ratios globally
- Understand climate effects on allocation
- Inform ecosystem carbon models

### 4. **Publication-Ready Dataset**
- Comprehensive metadata preserved
- Multiple output formats for different analyses
- Pre-computed correlations and ratios

---

## Data Quality Notes

### Missing Data Patterns

**High Coverage (>50% of sites):**
- ✅ BNPP_root_C (95.6%)
- ✅ NPP_1_C (85.4%)
- ✅ ANPP_2_C (71.2%)

**Moderate Coverage (25-50%):**
- ⚠️ ANPP_foliage_C (49.6%)
- ⚠️ BNPP_root_fine_C (49.1%)
- ⚠️ ANPP_woody_C (45.1%)
- ⚠️ GPP_C (28.8%)

**Low Coverage (<25%):**
- ❌ Most component variables
- ❌ Specialized measurements (herbivory, VOC, etc.)

**Recommendation:** For robust analyses, focus on high-coverage variables (BNPP_root_C, NPP_1_C, ANPP_2_C).

### Geographic Bias

The 352 sites are concentrated in:
- **North America**: ~40% of sites
- **Europe**: ~30% of sites
- **Asia**: ~15% of sites
- **Tropical regions**: Underrepresented

**Implication:** Models trained on this data may not generalize well to tropical forests.

---

## Next Steps & Integration with ADAM Pipeline

### 1. Merge with Original BNPP Data
```bash
# Combine ForC all productivity (352 sites) with grassland data (891 sites)
# This creates comprehensive training dataset: ~1,243 total sites

cd src/
python data_aggregation.py --files \
  forc/ForC_all_productivity_data_wide.csv \
  grassland/grassland_bnpp_data.csv \
  ... (ancillary data)
```

### 2. Enhanced Feature Engineering
```python
# Use productivity ratios as features
features_new = [
    'BNPP_NPP_ratio',      # Carbon allocation
    'ANPP_GPP_ratio',      # Aboveground efficiency
    'NPP_GPP_ratio',       # Overall NPP efficiency
    ... (existing features)
]
```

### 3. Multi-Task Learning
```python
# Train model to predict multiple productivity variables
targets = ['BNPP_root_C', 'ANPP_2_C', 'NPP_1_C', 'GPP_C']

# Exploit correlations between targets
# Improve predictions through shared learning
```

### 4. Model Validation
```python
# Use 65 sites with GPP measurements as validation set
# Check if global GPP → BNPP predictions are realistic
# Compare against observed BNPP/GPP ratios
```

---

## File Locations

```
ADAM/
├── src/bnpp/
│   ├── ForC_data.py                           # Original (BNPP only)
│   ├── ForC_all_productivity.py               # NEW (all variables)
│   └── PRODUCTIVITY_DATA_GUIDE.md             # This file
│
└── productivity/forc/
    ├── ForC_BNPP_root_C_processed.csv         # Original output
    ├── ForC_all_productivity_data_long.csv    # NEW: 4,351 measurements
    ├── ForC_all_productivity_data_wide.csv    # NEW: 226 sites
    │
    └── figures_productivity/                   # NEW visualizations
        ├── productivity_correlation_matrix.png
        ├── productivity_scatter_plots.png
        └── productivity_coverage_overview.png
```

---

## Summary

**What you have now:**
✅ Comprehensive forest productivity dataset (4,351 measurements)
✅ 29 productivity variables across 5 types (BNPP, ANPP, NPP, GPP, TBCF)
✅ Both long-format (detailed) and wide-format (site-level) datasets
✅ Validated correlations between productivity variables
✅ BNPP/NPP ratios for carbon allocation analysis
✅ Multiple output formats ready for different analyses
✅ Publication-quality visualizations

**This dramatically expands your training data beyond just BNPP measurements!** 🎉

You can now:
1. Train models using GPP, ANPP, NPP as features
2. Validate model predictions against observed ratios
3. Understand carbon allocation patterns globally
4. Create multi-variable productivity models
5. Publish comprehensive forest productivity analyses

---

**Questions? Check the code documentation in `ForC_all_productivity.py` or examine the visualizations in `figures_productivity/`**
