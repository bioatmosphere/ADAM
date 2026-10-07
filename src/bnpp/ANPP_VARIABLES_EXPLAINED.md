# ANPP Variables Explained - Complete Guide

## Overview

ForC database has **11 different ANPP (Aboveground Net Primary Production) variables** that measure different **components** or **levels of completeness** in forest productivity. Understanding these differences is crucial for data analysis and modeling.

---

## The ANPP Hierarchy

ANPP variables are organized in a **building-block structure**, from individual components to increasingly complete totals:

```
Level 1: Individual Components
├── ANPP_foliage_C        (leaves/needles)
├── ANPP_woody_stem_C     (stem wood growth)
├── ANPP_woody_branch_C   (branch turnover)
├── ANPP_repro_C          (flowers, fruits, seeds)
└── ANPP_folivory_C       (leaf consumption by herbivores)

Level 2: Composite Components
├── ANPP_woody_C = woody_stem + woody_branch
└── ANPP_litterfall_0_C = foliage + twigs + reproductive parts

Level 3: Aggregated Totals (Most Commonly Used)
├── ANPP_1_C = foliage + woody_stem
├── ANPP_2_C = foliage + woody_stem + woody_branch
└── ANPP_0_C = unspecified components
```

---

## Main ANPP Variables (Most Important)

### **ANPP_1_C** - Basic Aboveground Production
**Formula:** `ANPP_1_C = ANPP_foliage_C + ANPP_woody_stem_C`

**What it includes:**
- ✅ Leaf/needle production (litterfall)
- ✅ Stem wood growth (from tree ring measurements)
- ❌ Branch turnover (NOT included)
- ❌ Reproductive structures
- ❌ Understory

**Measurement method:**
- Foliage: Collect fallen leaves in litter traps
- Stem: Measure diameter growth, use allometry to estimate biomass increment

**Coverage in dataset:** 59 sites (26.1%)

**When to use:**
- When you want the most basic, commonly-measured ANPP
- When branch data is unavailable
- For comparing with older studies that only measured leaves + stems

---

### **ANPP_2_C** ⭐ - Complete Aboveground Production (RECOMMENDED)
**Formula:** `ANPP_2_C = ANPP_foliage_C + ANPP_woody_stem_C + ANPP_woody_branch_C`

**What it includes:**
- ✅ Leaf/needle production
- ✅ Stem wood growth
- ✅ Branch turnover (fallen branches)
- ❌ Reproductive structures
- ❌ Understory

**Why it's more complete than ANPP_1:**
- Accounts for energy spent producing branches that later fall
- More accurate representation of carbon allocation
- Branch production can be 10-20% of woody production

**Coverage in dataset:** 161 sites (71.2%) - **MOST COMMON**

**When to use:**
- **Recommended for most analyses** (highest coverage + comprehensive)
- Standard definition in forest ecology
- Best for carbon cycle studies

---

### **ANPP_0_C** - Unspecified Components
**Formula:** Not standardized (components vary by study)

**What it includes:**
- ❓ Variable - depends on original study
- May or may not include branches, reproductive parts, etc.
- Use with caution!

**Coverage in dataset:** 15 sites (6.6%)

**When to use:**
- Only when ANPP_1_C or ANPP_2_C are unavailable
- Note the uncertainty in your analysis
- Check original publication for what was measured

---

## Component Variables (Building Blocks)

### **ANPP_foliage_C** - Leaf Production
**What it measures:** Annual production of leaves or needles

**Measurement method:**
1. Place litter traps under forest canopy
2. Collect and sort leaf material monthly
3. Dry, weigh, measure carbon content
4. Sum annual total

**Ecological notes:**
- Evergreen forests: Continuous leaf fall over year
- Deciduous forests: Concentrated in autumn
- Does NOT include herbivory (leaves eaten before falling)

**Coverage:** 112 sites (49.6%)

**Typical values:** 0.5-3.0 Mg C ha⁻¹ yr⁻¹

---

### **ANPP_woody_stem_C** - Stem Growth
**What it measures:** Annual increment of stem wood (including branches attached to stem)

**Measurement method:**
1. Measure diameter of all trees
2. Re-measure after 1 year
3. Calculate biomass increment using allometric equations
4. Account for recruitment (new trees) and mortality

**Ecological notes:**
- Includes bark growth
- Includes small branches still attached to stem
- Does NOT include branches that fall off

**Coverage:** 68 sites (30.1%)

**Typical values:** 1.0-5.0 Mg C ha⁻¹ yr⁻¹

---

### **ANPP_woody_branch_C** - Branch Turnover
**What it measures:** Annual production of branches that subsequently fall

**Measurement method:**
1. Collect fallen branches from litter traps or ground surveys
2. Measure diameter, length, weight
3. Calculate carbon content
4. Sum annual total

**Ecological notes:**
- Represents branches produced in current or previous years that fall
- Can be 10-30% of total woody production
- Often omitted in older studies (labor-intensive to measure)

**Coverage:** 59 sites (26.1%)

**Typical values:** 0.2-1.5 Mg C ha⁻¹ yr⁻¹

---

### **ANPP_woody_C** - Total Woody Production
**Formula:** `ANPP_woody_C = ANPP_woody_stem_C + ANPP_woody_branch_C`

**What it measures:** All woody aboveground production (stems + branches)

**Coverage:** 102 sites (45.1%)

**When to use:**
- When you want total woody allocation
- To compare wood vs leaf allocation
- For woody biomass carbon models

---

### **ANPP_repro_C** - Reproductive Production
**What it measures:** Annual production of flowers, fruits, seeds, cones

**Measurement method:**
- Collect reproductive structures from litter traps
- Measure mass and carbon content

**Ecological notes:**
- Highly variable year-to-year (mast years)
- Small component (usually <5% of ANPP)
- Often omitted from measurements

**Coverage:** 10 sites (4.4%)

**Typical values:** 0.01-0.5 Mg C ha⁻¹ yr⁻¹

---

### **ANPP_folivory_C** - Herbivory
**What it measures:** Leaves consumed by herbivores (insects, mammals)

**Measurement method:**
- Estimate percentage of leaf area consumed
- Calculate mass of consumed leaves
- Add to ANPP_foliage to get total leaf production

**Why it's separate:**
- Standard leaf litter collection misses eaten leaves
- True ANPP includes production that's immediately consumed
- Important in some ecosystems (tropical forests: 5-15%)

**Coverage:** 14 sites (6.2%)

**Typical values:** 0.05-1.0 Mg C ha⁻¹ yr⁻¹

---

### **ANPP_litterfall_0_C, _1_C, _2_C** - Litterfall Totals
**What they measure:** All material falling from canopy

**Differences:**
- **ANPP_litterfall_0_C**: Unspecified components (37 sites)
- **ANPP_litterfall_1_C**: Leaves + twigs + reproductive (7 sites)
- **ANPP_litterfall_2_C**: Leaves + twigs + reproductive + small branches (0 sites)

**Note:** These are NOT full ANPP estimates (missing stem growth increment)

---

## Visual Comparison: What Each ANPP Variable Includes

```
Component:          ANPP_1_C  ANPP_2_C  ANPP_woody_C  ANPP_foliage_C  ANPP_repro_C
─────────────────────────────────────────────────────────────────────────────────
Foliage (leaves)       ✓         ✓           ✗             ✓              ✗
Stem wood growth       ✓         ✓           ✓             ✗              ✗
Branch turnover        ✗         ✓           ✓             ✗              ✗
Reproductive parts     ✗         ✗           ✗             ✗              ✓
Herbivory              ✗         ✗           ✗             ✗              ✗
Understory             ✗         ✗           ✗             ✗              ✗
─────────────────────────────────────────────────────────────────────────────────
Completeness:         Basic    Complete    Woody-only   Leaves-only   Repro-only
─────────────────────────────────────────────────────────────────────────────────
Coverage (sites):      59       161          102           112            10
Recommended use:       ★★       ★★★★★        ★★★           ★★★            ★
```

---

## Typical Magnitudes & Relationships

Based on ForC database analysis:

| Variable | Typical Range | % of ANPP_2_C |
|----------|---------------|---------------|
| **ANPP_2_C** | **2-10 Mg C ha⁻¹ yr⁻¹** | 100% |
| ANPP_foliage_C | 0.5-3.0 | 20-40% |
| ANPP_woody_stem_C | 1.0-5.0 | 40-60% |
| ANPP_woody_branch_C | 0.2-1.5 | 10-20% |
| ANPP_repro_C | 0.01-0.5 | 1-5% |
| ANPP_folivory_C | 0.05-1.0 | 2-10% |

**Key relationships:**
- Woody production (stem + branch) usually > foliage production
- ANPP_2_C ≈ ANPP_1_C × 1.1-1.3 (branches add 10-30%)
- Tropical forests: higher foliage fraction
- Boreal forests: higher woody fraction

---

## Which ANPP Variable Should I Use?

### For Machine Learning Models (Predicting BNPP):

**Best choice: ANPP_2_C** ✅
- **Reason:** Highest coverage (161 sites) + most complete measurement
- Captures full aboveground carbon allocation
- Standard in forest ecology

**Alternative: ANPP_1_C**
- Use if ANPP_2_C unavailable at some sites
- Note: Underestimates ANPP by ~15% (missing branches)
- Adjust for branch component if possible

**Avoid: ANPP_0_C**
- Unknown components = uncertain interpretation
- Only use as last resort

### For Carbon Allocation Analysis (BNPP/ANPP ratio):

**Best practice:**
1. Use ANPP_2_C where available
2. For sites with only ANPP_1_C, estimate: `ANPP_2_C ≈ ANPP_1_C × 1.20`
3. Document which sites were adjusted

### For Global Comparisons:

**Recommendation:**
- Always specify which ANPP definition you used
- ANPP_2_C is becoming the standard in recent literature
- Convert older studies (ANPP_1_C) to ANPP_2_C equivalent where possible

---

## Data Quality Checks

When working with ANPP data:

### 1. **Check component consistency:**
```python
# ANPP_2_C should approximately equal sum of components
df['calculated_ANPP_2'] = (
    df['ANPP_foliage_C'] +
    df['ANPP_woody_stem_C'] +
    df['ANPP_woody_branch_C']
)

# Flag inconsistencies
df['ANPP_mismatch'] = abs(df['ANPP_2_C'] - df['calculated_ANPP_2']) > 0.5
```

### 2. **Check ecological plausibility:**
```python
# ANPP_2_C should be greater than ANPP_1_C
assert (df['ANPP_2_C'] >= df['ANPP_1_C']).all()

# Branch component should be 10-30% of woody
branch_fraction = df['ANPP_woody_branch_C'] / df['ANPP_woody_C']
# Flag if outside 0.05-0.50 range
```

### 3. **Check BNPP/ANPP ratios:**
```python
df['BNPP_ANPP_ratio'] = df['BNPP_root_C'] / df['ANPP_2_C']

# Typical range: 0.2-2.0 (most forests: 0.3-0.8)
outliers = df[
    (df['BNPP_ANPP_ratio'] < 0.1) |
    (df['BNPP_ANPP_ratio'] > 3.0)
]
# Investigate outliers for measurement errors
```

---

## Example Analysis: Comparing ANPP Variables

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ForC_all_productivity_data_wide.csv')

# Sites with multiple ANPP measurements
multi_anpp = df[['sites.sitename', 'ANPP_1_C', 'ANPP_2_C']].dropna()

# Calculate difference
multi_anpp['branch_contribution'] = multi_anpp['ANPP_2_C'] - multi_anpp['ANPP_1_C']
multi_anpp['branch_fraction'] = multi_anpp['branch_contribution'] / multi_anpp['ANPP_2_C']

print(f"Sites with both ANPP_1_C and ANPP_2_C: {len(multi_anpp)}")
print(f"\nBranch contribution statistics:")
print(f"  Mean: {multi_anpp['branch_fraction'].mean():.1%}")
print(f"  Median: {multi_anpp['branch_fraction'].median():.1%}")
print(f"  Range: {multi_anpp['branch_fraction'].min():.1%} - {multi_anpp['branch_fraction'].max():.1%}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(multi_anpp['ANPP_1_C'], multi_anpp['ANPP_2_C'], alpha=0.6)
plt.plot([0, 20], [0, 20], 'r--', label='1:1 line')
plt.plot([0, 20], [0, 24], 'b--', label='+20% line')
plt.xlabel('ANPP_1_C (Mg C ha⁻¹ yr⁻¹)')
plt.ylabel('ANPP_2_C (Mg C ha⁻¹ yr⁻¹)')
plt.title('Relationship: ANPP_1_C vs ANPP_2_C')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Common Questions

### Q: Why are there so many different ANPP variables?
**A:** Different research groups measure different components based on:
- Available resources (time, equipment, labor)
- Research questions (some studies focus on woody production)
- Historical context (older studies didn't always measure branches)
- Ecosystem type (branch turnover varies by forest type)

### Q: Can I combine ANPP_1_C and ANPP_2_C in the same analysis?
**A:** Yes, but with caveats:
1. Adjust ANPP_1_C upward by ~20% to approximate ANPP_2_C
2. Include a variable indicating which measurement type
3. Document the adjustment in your methods
4. Sensitivity analysis: check if results change

### Q: Which is more comparable to NPP measurements?
**A:** ANPP_2_C is more comparable because:
- NPP includes all components (above + below)
- ANPP_2_C captures full aboveground allocation
- ANPP_1_C underestimates aboveground production

### Q: Why don't more sites have ANPP_2_C?
**A:** Branch collection is labor-intensive:
- Requires year-round field work
- Large branches are heavy and difficult to process
- Not all litter traps capture large branches
- Some studies prioritize other measurements

### Q: Can I estimate ANPP_woody_branch_C from other variables?
**A:** Rough approximation:
```python
# Typical branch contribution: 15-25% of woody production
ANPP_woody_branch_C ≈ ANPP_woody_stem_C × 0.20

# Or if you have ANPP_1_C and ANPP_2_C:
ANPP_woody_branch_C ≈ ANPP_2_C - ANPP_1_C
```

---

## Recommendations Summary

| Use Case | Recommended Variable | Alternative | Notes |
|----------|---------------------|-------------|-------|
| **ML models (BNPP prediction)** | ANPP_2_C | ANPP_1_C × 1.2 | Highest coverage + complete |
| **Carbon allocation analysis** | ANPP_2_C | NPP_1_C - BNPP_root_C | Use NPP if ANPP unavailable |
| **Woody vs leaf allocation** | ANPP_foliage_C + ANPP_woody_C | Component analysis | Need both components |
| **Global comparisons** | ANPP_2_C | State which definition used | Document in methods |
| **Temporal trends** | Consistent definition | Convert older ANPP_1 → ANPP_2 | Note conversions |

---

## References

Key papers defining ANPP measurement standards:
- Clark et al. (2001) *Ecology* - NPP methodology
- Malhi et al. (2011) *Phil Trans R Soc B* - Tropical forest NPP
- Gough et al. (2008) *Oecologia* - Woody production methods
- Litton et al. (2007) *Ecol Appl* - Carbon allocation patterns

---

## Summary

**Key takeaways:**

1. **ANPP_2_C is the gold standard** (most complete, highest coverage)
2. **ANPP_1_C is acceptable but underestimates** by ~15-20%
3. **ANPP_0_C is variable** - use only when necessary
4. **Component variables** (foliage, woody) are building blocks
5. **Branches matter** - they add 10-30% to ANPP
6. **Always document** which ANPP definition you used
7. **Check consistency** between totals and components

**For ADAM project:** Use **ANPP_2_C** (161 sites) as primary ANPP variable for models and analysis. 🎯
