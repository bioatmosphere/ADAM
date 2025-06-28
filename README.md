# ADAM
**A**gentic **D**ata **A**ggregation and **M**odelling

## Project Overview

ADAM is a comprehensive data synthesis and machine learning pipeline for predicting Below-ground Net Primary Productivity (BNPP) using the TAM (Transport and Absorptive roots with Mycorrhizal fungi) conceptual framework. The project integrates multiple global datasets and applies machine learning models to create benchmark predictions for ecosystem productivity research.

## Key Features

- **Multi-source Data Integration**: Combines forest, grassland, climate, and soil datasets from global sources
- **4-Stage Pipeline**: Automated workflow from data retrieval to global application
- **Machine Learning Models**: Random Forest and Multi-Layer Perceptron implementations
- **Global Predictions**: Applies trained models to worldwide land cover data
- **Scientific Reproducibility**: Comprehensive logging and modular architecture

## Quick Start

### Prerequisites
- Python 3.8+
- Required packages: `xarray`, `cartopy`, `scikit-learn`, `pandas`, `matplotlib`
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
├── ancillary/              # Environmental data
│   ├── terraclimate/       # Climate variables
│   ├── soilgrids/          # Soil properties
│   └── glass/              # GPP satellite data
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
- **Gherardi-Sala**: Grassland belowground productivity database

## Pipeline Stages

1. **Data Retrieval**: Download and process source datasets
2. **Data Integration**: Spatially align and merge all data sources
3. **Machine Learning**: Train Random Forest and MLP models
4. **Global Application**: Apply models to create worldwide BNPP maps

## Output

The pipeline generates:
- Processed datasets in standardized formats
- Trained machine learning models (`.pkl` files)
- Global BNPP prediction maps
- Model evaluation metrics and visualizations
- Comprehensive execution logs

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