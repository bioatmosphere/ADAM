# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ADAM is a TAM (Transport and Absorptive roots with Mycorrhizal fungi) benchmark pipeline that synthesizes biomass and productivity data from multiple sources using machine learning. The project focuses on Below-ground Net Primary Productivity (BNPP) prediction using environmental predictors.

## Common Commands

### Running the Complete Pipeline
```bash
# Run the full 4-stage pipeline
python src/main.py --verbose

# Run specific pipeline stages
python src/main.py --stage 1  # Data retrieval and processing
python src/main.py --stage 2  # Data integration  
python src/main.py --stage 3  # Machine learning
python src/main.py --stage 4  # Global application
```

### Data Processing
```bash
# Process individual data sources (run from src/ directory)
python bnpp/ForC_data.py           # Forest carbon database
python bnpp/grassland_data.py      # Grassland productivity data
python ancillary/terraclimate.py   # Climate data extraction
python ancillary/soilgrids.py      # Soil data extraction
```

### Model Training and Application
```bash
# Train models individually (run from src/models/ directory)
python RF_model.py    # Random Forest model
python MLP_model.py   # Multi-Layer Perceptron model
```

## Architecture Overview

### Pipeline Structure
The project follows a 4-stage data science pipeline:

1. **Stage 1 - Data Retrieval**: Processes BNPP data from ForC, Gherardi-Sala, and grassland databases, plus ancillary climate/soil data
2. **Stage 2 - Data Integration**: Aggregates all data sources into a unified dataset with spatial/temporal alignment
3. **Stage 3 - Machine Learning**: Trains Random Forest and MLP models for BNPP prediction
4. **Stage 4 - Global Application**: Applies trained models to global land cover data for worldwide predictions

### Core Components

**Main Orchestrator** (`src/main.py`):
- `TAMPipeline` class manages the complete workflow
- Handles logging, directory setup, and stage coordination
- Supports running individual stages or the full pipeline

**Data Processing Modules**:
- `src/bnpp/`: BNPP data sources (ForC forest data, grassland productivity)
- `src/ancillary/`: Environmental data (TerraClimate, SoilGrids, soil moisture)
- `src/landcover/`: Land cover classification (SYNMAP, biomes, bioregions)
- `src/data_aggregation.py`: Spatial data integration and merging

**Machine Learning**:
- `src/models/RF_model.py`: Random Forest implementation with hyperparameter tuning
- `src/models/MLP_model.py`: Multi-Layer Perceptron with neural network architecture

### Data Sources Integrated
- **ForC**: Global forest carbon database
- **Gherardi-Sala**: Grassland belowground productivity data
- **TerraClimate**: Climate variables (precipitation, temperature, VPD)
- **SoilGrids**: Soil carbon and physical properties
- **GLASS**: Gross Primary Productivity satellite data
- **SYNMAP**: Global vegetation maps
- **Biome Classifications**: Olson biomes and bioregions

### Key Dependencies
- **Geospatial**: `xarray`, `cartopy`, `gdal`, `rasterio` for netCDF/raster processing
- **ML/Data**: `scikit-learn`, `pandas`, `numpy` for machine learning and data manipulation
- **Visualization**: `matplotlib`, `seaborn` for publication-quality figures
- **Climate Data**: `requests`, `tqdm` for downloading TerraClimate data

### Output Structure
```
output/
├── processed_data/     # Stage 1 outputs
├── integrated_data/    # Stage 2 unified dataset
├── models/            # Stage 3 trained models
├── global_predictions/ # Stage 4 global maps
└── figures/           # Visualizations
```

## Important Notes

- The pipeline expects specific data directory structures under `ancillary/`, `productivity/`, `landcover/`, etc.
- All data processing includes comprehensive logging and error handling
- Models support both training and global application phases
- Point data extraction uses spatial coordinates for climate/soil variable alignment
- Global predictions require processed land cover datasets for spatial context