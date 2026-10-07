# GLASS GPP Global Processing - Installation & Usage Guide

This guide explains how to process GLASS GPP_YEARLY HDF tiles (500m resolution) into a global 0.5° grid for ADAM model applications.

## 📋 Overview

**Problem**: The original global application uses synthetic GPP patterns instead of real satellite data.

**Solution**: Process actual GLASS HDF tiles to create accurate global GPP grids.

**Benefits**:
- Real satellite-derived GPP data
- Better spatial patterns
- More accurate BNPP predictions
- Improved model performance

---

## 🔧 Installation

### **Step 1: Install Required Dependencies**

The GLASS processor requires several Python packages for HDF4 file handling:

```bash
# Using pip
pip install pyhdf numpy xarray rasterio pyproj tqdm pandas matplotlib cartopy

# OR using conda (recommended for pyhdf)
conda install -c conda-forge pyhdf numpy xarray rasterio pyproj tqdm pandas matplotlib cartopy
```

### **Step 2: Verify Installation**

```bash
python3 << 'EOF'
try:
    from pyhdf.SD import SD, SDC
    print("✓ pyhdf installed")
except ImportError:
    print("✗ pyhdf NOT installed")

try:
    import rasterio
    print("✓ rasterio installed")
except ImportError:
    print("✗ rasterio NOT installed")

try:
    from pyproj import Proj
    print("✓ pyproj installed")
except ImportError:
    print("✗ pyproj NOT installed")

print("\nAll dependencies installed successfully!")
EOF
```

### **Common Installation Issues**

#### **Issue 1: pyhdf installation fails**
```bash
# Solution: Install HDF4 library first
# On macOS:
brew install hdf4

# On Ubuntu/Debian:
sudo apt-get install libhdf4-dev

# Then install pyhdf:
pip install pyhdf
```

#### **Issue 2: GDAL/rasterio conflicts**
```bash
# Use conda for better dependency management:
conda create -n glass python=3.9
conda activate glass
conda install -c conda-forge pyhdf rasterio pyproj xarray tqdm
```

---

## 🚀 Usage

### **Method 1: Process GLASS Tiles Standalone**

Process GLASS GPP tiles and create global grid:

```bash
cd src/ancillary/

python glass_global_processor.py \
  --glass-dir ../../ancillary/glass/GPP_YEARLY \
  --year 2010 \
  --output ../../ancillary/glass/global_gpp_yearly_2010_0.5deg.nc
```

**Output**:
- `global_gpp_yearly_2010_0.5deg.nc` - NetCDF file (360×720 grid)
- `global_gpp_yearly_2010_0.5deg.png` - Global map visualization

### **Method 2: Integrate with Model Application**

Use processed GPP in global BNPP predictions:

```bash
cd src/models/application/

# Option A: Process GPP and apply model in one step
python apply_RF_globally_v2.py

# Option B: Pre-process GPP, then apply model
python ../../ancillary/glass_global_processor.py --year 2010
python apply_RF_globally_v2.py
```

The v2 script automatically:
1. Checks for existing processed GPP grid
2. If not found, processes HDF tiles automatically
3. Applies model using real GLASS GPP data
4. Creates global BNPP predictions

---

## 📊 Expected Processing Time

| Step | Time | Memory |
|------|------|--------|
| Read 288 HDF tiles | ~5-10 min | ~2 GB |
| Reproject to geographic | ~10-15 min | ~4 GB |
| Aggregate to 0.5° grid | ~2-5 min | ~1 GB |
| **Total** | **~20-30 min** | **~4-6 GB** |

*Times vary based on system performance*

---

## 🔍 Understanding the Processing Workflow

### **Input: GLASS HDF Tiles**
```
ancillary/glass/GPP_YEARLY/2010_001/
├── GLASS12E11.V60.A2010001.h00v08.2022100.hdf
├── GLASS12E11.V60.A2010001.h01v09.2022100.hdf
├── ... (288 tiles total)
└── GLASS12E11.V60.A2010001.h35v17.2022100.hdf
```

**Tile specifications**:
- Product: GLASS12E11 (Yearly GPP)
- Resolution: 500m × 500m
- Grid: MODIS Sinusoidal projection
- Tile size: 2400 × 2400 pixels
- Coverage: h00-35 (horizontal), v00-17 (vertical)

### **Processing Steps**:

1. **Read HDF tiles** → Extract GPP dataset, apply scale factors
2. **Reproject** → MODIS Sinusoidal → Geographic (WGS84)
3. **Mosaic** → Combine 288 tiles into global grid
4. **Aggregate** → Resample 500m → 0.5° (~55km)
5. **Export** → NetCDF format for model application

### **Output: Global GPP Grid**
```
global_gpp_yearly_2010_0.5deg.nc
├── Dimensions: lat(360), lon(720)
├── Variable: gpp_yearly (360×720)
├── Units: gC m⁻² year⁻¹
├── Resolution: 0.5° (~55 km)
└── Coverage: Global land areas
```

---

## 🧪 Testing & Validation

### **Test 1: Quick Validation**

```python
import xarray as xr
import numpy as np

# Load processed GPP
gpp = xr.open_dataarray('../../ancillary/glass/global_gpp_yearly_2010_0.5deg.nc')

# Check statistics
print(f"Shape: {gpp.shape}")
print(f"Valid pixels: {np.sum(~np.isnan(gpp.values)):,}")
print(f"GPP range: {gpp.min().values:.1f} to {gpp.max().values:.1f}")
print(f"Mean GPP: {gpp.mean().values:.1f} gC m⁻² yr⁻¹")

# Expected values (approximate):
# - Valid pixels: ~50,000-80,000 (land areas)
# - GPP range: 0-3000 gC m⁻² yr⁻¹
# - Mean GPP: 800-1200 gC m⁻² yr⁻¹
```

### **Test 2: Compare with Training Point Data**

```python
import pandas as pd

# Load training data
train = pd.read_csv('../../productivity/earth/aggregated_data_cleaned.csv')

# Load global GPP grid
gpp_global = xr.open_dataarray('../../ancillary/glass/global_gpp_yearly_2010_0.5deg.nc')

# Extract GPP at training locations
for idx, row in train.iterrows():
    gpp_value = gpp_global.sel(lat=row['lat'], lon=row['lon'], method='nearest').values
    train_gpp = row['gpp_yearly']
    print(f"Point {idx}: Training GPP={train_gpp:.1f}, Global GPP={gpp_value:.1f}")
```

### **Test 3: Visual Inspection**

```bash
# Create visualization
python << 'EOF'
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

gpp = xr.open_dataarray('../../ancillary/glass/global_gpp_yearly_2010_0.5deg.nc')

fig, ax = plt.subplots(figsize=(15, 8), subplot_kw={'projection': ccrs.PlateCarree()})
gpp.plot(ax=ax, cmap='YlGn', vmin=0, vmax=2000, transform=ccrs.PlateCarree())
ax.coastlines()
ax.set_title('Global GLASS GPP - 2010')
plt.savefig('test_gpp_map.png', dpi=150)
print("✓ Map saved to test_gpp_map.png")
EOF
```

---

## 📈 Performance Comparison

### **Before (Synthetic GPP)**:
```
Global BNPP Predictions:
  Mean: 351.5 g C m⁻² yr⁻¹
  Range: 260-449 g C m⁻² yr⁻¹
  Std Dev: 28.8 g C m⁻² yr⁻¹
  Test R²: 0.5382
```

### **After (Real GLASS GPP)** - Expected:
```
Global BNPP Predictions:
  Mean: [will vary based on real data]
  Range: [wider range expected]
  Std Dev: [higher variability expected]
  Test R²: [should improve or stay similar]
```

---

## 🐛 Troubleshooting

### **Error: "No module named 'pyhdf'"**
```bash
# Install pyhdf using conda
conda install -c conda-forge pyhdf
```

### **Error: "HDF file not recognized"**
- Check HDF file integrity: `h4dump -H file.hdf`
- Ensure HDF4 (not HDF5) support is installed

### **Error: "Memory Error" during processing**
```bash
# Process tiles in batches or use a machine with more RAM
# Alternatively, increase system swap space
```

### **Error: "GPP values all NaN"**
- Check dataset name in HDF file (may vary between versions)
- Verify scale factor and fill value attributes

---

## 💡 Tips & Best Practices

1. **Pre-process once, use many times**: Process GPP grid once and reuse for multiple model runs
2. **Cache processed files**: Keep NetCDF outputs to avoid reprocessing
3. **Quality control**: Always visually inspect GPP maps before model application
4. **Disk space**: Ensure ~10-20 GB free for intermediate files
5. **Parallel processing**: For multiple years, process in parallel on HPC

---

## 📝 File Structure

```
ADAM/
├── ancillary/
│   ├── glass/
│   │   ├── GPP_YEARLY/
│   │   │   └── 2010_001/          # HDF tiles
│   │   └── global_gpp_yearly_2010_0.5deg.nc  # Processed output
│   └── glass_global_processor.py   # Processing script
├── src/
│   └── models/
│       └── application/
│           ├── apply_RF_globally.py     # Original (synthetic GPP)
│           └── apply_RF_globally_v2.py  # Updated (real GPP)
└── productivity/
    └── earth/
        ├── global_bnpp_predictions_rf.nc     # Original predictions
        └── global_bnpp_predictions_rf_v2.nc  # Updated predictions
```

---

## 🎯 Next Steps

After processing GLASS GPP:

1. **Validate** - Check output statistics and visualizations
2. **Compare** - Run model with synthetic vs. real GPP
3. **Analyze** - Assess impact on BNPP predictions
4. **Document** - Record differences in model performance
5. **Scale** - Process additional years (2000-2022)

---

## 📚 References

- GLASS Product: https://www.glass.hku.hk/
- MODIS Tiles: https://modis-land.gsfc.nasa.gov/MODLAND_grid.html
- Paper: Liang et al. (2021) - GLASS GPP Product

---

**Questions?** Check the ADAM documentation or open an issue on GitHub.
