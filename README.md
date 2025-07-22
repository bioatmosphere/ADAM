# ADAM
**A**gentic **D**ata **A**ggregation and **M**odelling

## Project Overview

ADAM is a comprehensive data synthesis and machine learning pipeline for predicting Below-ground Net Primary Productivity (BNPP) across global ecosystems. The project integrates multiple global datasets and applies eight state-of-the-art machine learning models to create benchmark predictions for ecosystem productivity research.

## Key Features

- **Multi-source Data Integration**: Combines forest, grassland, climate, soil, and elevation datasets from global sources
- **4-Stage Pipeline**: Automated workflow from data retrieval to global application
- **8 Machine Learning Models**: Comprehensive benchmark suite including Random Forest, XGBoost, MLP, LightGBM, TabNet, TabPFN, Deep Ensemble, and Elastic Net
- **Advanced Data Cleaning**: Automated outlier detection with multiple statistical methods
- **Elevation Integration**: SRTM digital elevation model as topographic predictor
- **Global Predictions**: Applies trained models to worldwide land cover data
- **Scientific Reproducibility**: Comprehensive logging and modular architecture

## Quick Start

### Prerequisites
- Python 3.8+
- Required packages: `xarray`, `cartopy`, `scikit-learn`, `pandas`, `matplotlib`, `xgboost`, `pytorch`
- GDAL for geospatial data processing

### Running the Pipeline

```bash
# Run the complete 4-stage pipeline
python src/main.py --verbose

# Run individual stages
python src/main.py --stage 1  # Data retrieval
python src/main.py --stage 2  # Data integration
python src/main.py --stage 3  # Machine learning
python src/main.py --stage 4  # Global application
```

### Training Individual Models

```bash
# Train specific models (run from src/models/ directory)
python RF_model.py             # Random Forest
python XGBoost_model.py        # XGBoost
python MLP_model.py            # Multi-Layer Perceptron
python LightGBM_model.py       # LightGBM
python ElasticNet_model.py     # Elastic Net
python TabNet_model.py         # TabNet
python TabPFN_model.py         # TabPFN (Prior-Fitted Networks)
python DeepEnsemble_model.py   # Deep Ensemble

# Apply models globally (run from src/models/application/ directory)
python apply_RF_globally.py       # Random Forest global predictions
python apply_MLP_globally.py      # MLP global predictions
python apply_XGBoost_globally.py  # XGBoost global predictions
```

## Directory Structure

```
ADAM/
├── src/                     # Source code
│   ├── main.py             # Pipeline orchestrator
│   ├── data_aggregation.py # Data integration
│   ├── bnpp/               # BNPP data processing
│   ├── ancillary/          # Climate & soil data
│   ├── landcover/          # Land cover processing
│   └── models/             # ML implementations
│       ├── RF_model.py     # Random Forest model
│       ├── MLP_model.py    # Multi-Layer Perceptron
│       ├── XGBoost_model.py # XGBoost model
│       └── application/    # Global model application
│           ├── apply_RF_globally.py
│           ├── apply_MLP_globally.py
│           └── apply_XGBoost_globally.py
├── ancillary/              # Environmental data
│   ├── terraclimate/       # Climate variables
│   ├── soilgrids/          # Soil properties
│   ├── glass/              # GPP satellite data
│   └── elevation_points.csv # SRTM elevation data
├── productivity/           # Productivity datasets
│   ├── forc/               # ForC forest data
│   ├── grassland/          # Grassland BNPP
│   └── globe/              # Global datasets
├── landcover/              # Land cover data
│   ├── synmap/             # SYNMAP vegetation
│   └── data/               # Biome classifications
└── output/                 # Pipeline outputs
    ├── processed_data/
    ├── integrated_data/
    ├── models/
    └── global_predictions/
```

## Data Sources

- **[ForC](https://github.com/forc-db/ForC)**: Global forest carbon database
- **[TerraClimate](https://climate.northwestknowledge.net/TERRACLIMATE-DATA)**: Climate variables (1958-2019)
- **[SoilGrids](https://soilgrids.org/)**: Global soil information
- **[GLASS](http://www.glass.umd.edu/)**: Global Land Surface Satellite products
- **[SYNMAP](https://www.earthenv.org/)**: Global vegetation mapping
- **[SRTM](https://www2.jpl.nasa.gov/srtm/)**: Shuttle Radar Topography Mission elevation data
- **Gherardi-Sala**: Grassland belowground productivity database

## Pipeline Stages

1. **Data Retrieval**: Download and process source datasets
2. **Data Integration**: Spatially align and merge all data sources with outlier detection
3. **Machine Learning**: Train 8 machine learning models with comprehensive benchmarking
4. **Global Application**: Apply trained models to create worldwide BNPP maps

## Machine Learning Models

The pipeline implements eight state-of-the-art machine learning models for comprehensive benchmarking:

### Tree-Based Models
- **Random Forest (`src/models/RF_model.py`)**: Ensemble method with hyperparameter tuning
- **XGBoost (`src/models/XGBoost_model.py`)**: Gradient boosting with advanced regularization  
- **LightGBM (`src/models/LightGBM_model.py`)**: Fast gradient boosting framework

### Neural Network Models
- **Multi-Layer Perceptron (`src/models/MLP_model.py`)**: Deep learning with PyTorch
- **TabNet (`src/models/TabNet_model.py`)**: Attention-based neural network for tabular data
- **TabPFN (`src/models/TabPFN_model.py`)**: Prior-Fitted Networks with zero-shot learning
- **Deep Ensemble (`src/models/DeepEnsemble_model.py`)**: Ensemble of neural networks

### Linear Model
- **Elastic Net (`src/models/ElasticNet_model.py`)**: Regularized linear regression with L1/L2 penalties

### Global Application (`src/models/application/`)
- Dedicated scripts for worldwide model application
- Climate and satellite data integration
- 0.5-degree resolution global predictions

## Data Processing and Quality Control

### Outlier Detection
The pipeline includes comprehensive outlier detection with multiple methods:
- **IQR Method**: Interquartile Range-based statistical outliers
- **Z-Score Method**: Standard deviation-based outliers  
- **Modified Z-Score**: Median Absolute Deviation-based detection
- **Domain-Based**: Ecological constraints on BNPP values
- **Geographic**: Spatial clustering of outliers

### Environmental Predictors (18 features)
- **Climate (6)**: Actual evapotranspiration, potential evapotranspiration, precipitation, max/min temperature, vapor pressure deficit
- **Satellite (1)**: Yearly Gross Primary Productivity from GLASS
- **Soil Properties (4)**: Carbon stock, clay/silt/sand content
- **Soil Chemistry (3)**: Nitrogen content, cation exchange capacity, pH
- **Soil Physics (3)**: Bulk density, coarse fragments, soil moisture
- **Topography (1)**: Elevation from SRTM

## Output

The pipeline generates:
- Cleaned datasets with outlier removal (1,367 samples from 1,482 original)
- 8 trained machine learning models with comprehensive summaries
- Global BNPP prediction maps at 0.5-degree resolution
- Model evaluation metrics and cross-validation results
- Feature importance analysis and visualization plots
- Comprehensive execution logs and model benchmarking

## Development

For development guidance and detailed architecture information, see [CLAUDE.md](./CLAUDE.md).

## Citation

If you use this code or data in your research, please cite:
```
[Citation information to be added]
```

## License

[License to be specified]

## Contact

For questions or collaboration opportunities, please open an issue on this repository.