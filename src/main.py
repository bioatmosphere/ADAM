"""
Master script for the end-to-end machine learning pipeline for TAM benchmark.

This pipeline implements the TAM (Transport and Absorptive roots with Mycorrhizal fungi)
conceptual framework through comprehensive data synthesis and machine learning.

Pipeline stages:
1. Retrieves and processes the source data: BNPP and ancillary data
2. Integrates the data to form a whole new dataset
3. Performs machine learning on the integrated data, including algorithms like Random Forest and MLP
4. Applies the machine learning model to the entire globe, involving global land cover map

Usage:
    python main.py [--stage STAGE] [--config CONFIG_FILE] [--verbose]

Author: TAM Development Team
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import pipeline components
try:
    from data_aggregation import aggregate_all_data, create_integrated_dataset
    from bnpp.ForC_data import process_forc_data
    from bnpp.GherardiSala_data_2020 import process_gherardi_sala_data
    from bnpp.grassland_data import process_grassland_data
    from ancillary.terraclimate import extract_point_data as extract_climate_data
    from ancillary.soilgrids import extract_point_data as extract_soil_data
    from landcover.SYNMAP import process_synmap_data
    from landcover.biomes import process_biomes_data
    from models.RF_model import train_random_forest, apply_global_rf
    from models.MLP_model import train_mlp, apply_global_mlp
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some pipeline stages may not be available.")


class TAMPipeline:
    """
    Main TAM benchmark pipeline orchestrator.
    """
    
    def __init__(self, config=None, verbose=False):
        """
        Initialize the TAM pipeline.
        
        Args:
            config (dict): Configuration parameters
            verbose (bool): Enable verbose logging
        """
        self.config = config or {}
        self.verbose = verbose
        self.setup_logging()
        self.setup_directories()
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'tam_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """Create necessary output directories."""
        base_dir = Path(__file__).parent.parent
        self.data_dir = base_dir
        self.output_dir = base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "processed_data").mkdir(exist_ok=True)
        (self.output_dir / "integrated_data").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "global_predictions").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
    def stage_1_data_retrieval(self):
        """
        Stage 1: Retrieve and process source data (BNPP and ancillary data).
        """
        self.logger.info("=== Stage 1: Data Retrieval and Processing ===")
        
        # Process BNPP data sources
        self.logger.info("Processing BNPP data sources...")
        
        try:
            # ForC global forest carbon database
            self.logger.info("Processing ForC data...")
            forc_data = process_forc_data(
                data_dir=self.data_dir / "productivity" / "forc",
                output_dir=self.output_dir / "processed_data"
            )
            
            # Gherardi-Sala grassland productivity data
            self.logger.info("Processing Gherardi-Sala data...")
            gherardi_data = process_gherardi_sala_data(
                data_dir=self.data_dir / "productivity" / "globe",
                output_dir=self.output_dir / "processed_data"
            )
            
            # Additional grassland data
            self.logger.info("Processing grassland data...")
            grassland_data = process_grassland_data(
                data_dir=self.data_dir / "productivity" / "grassland",
                output_dir=self.output_dir / "processed_data"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing BNPP data: {e}")
            
        # Process ancillary data
        self.logger.info("Processing ancillary data...")
        
        try:
            # Extract climate data for all sites
            self.logger.info("Extracting TerraClimate data...")
            climate_data = extract_climate_data(
                data_dir=self.data_dir / "ancillary" / "terraclimate",
                coordinates_file=self.data_dir / "productivity" / "earth" / "lat_lon.csv",
                variables=['ppt', 'tmax', 'tmin'],
                output_format='csv'
            )
            
            # Extract soil data
            self.logger.info("Extracting SoilGrids data...")
            soil_data = extract_soil_data(
                data_dir=self.data_dir / "ancillary" / "soilgrids",
                coordinates_file=self.data_dir / "productivity" / "earth" / "lat_lon.csv"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing ancillary data: {e}")
            
        self.logger.info("Stage 1 completed successfully")
        
    def stage_2_data_integration(self):
        """
        Stage 2: Integrate data to form a unified dataset.
        """
        self.logger.info("=== Stage 2: Data Integration ===")
        
        try:
            # Aggregate all processed data
            self.logger.info("Aggregating all data sources...")
            integrated_dataset = aggregate_all_data(
                processed_data_dir=self.output_dir / "processed_data",
                output_dir=self.output_dir / "integrated_data"
            )
            
            # Create the final integrated dataset with all features
            self.logger.info("Creating integrated dataset...")
            final_dataset = create_integrated_dataset(
                bnpp_data=integrated_dataset,
                climate_data_dir=self.data_dir / "ancillary" / "terraclimate" / "point_extractions",
                soil_data_dir=self.data_dir / "ancillary" / "soilgrids",
                output_dir=self.output_dir / "integrated_data"
            )
            
            self.logger.info(f"Integrated dataset created with {len(final_dataset)} records")
            
        except Exception as e:
            self.logger.error(f"Error in data integration: {e}")
            raise
            
        self.logger.info("Stage 2 completed successfully")
        
    def stage_3_machine_learning(self):
        """
        Stage 3: Perform machine learning on the integrated data.
        """
        self.logger.info("=== Stage 3: Machine Learning ===")
        
        try:
            # Load integrated dataset
            dataset_file = self.output_dir / "integrated_data" / "final_integrated_dataset.csv"
            if not dataset_file.exists():
                self.logger.error(f"Integrated dataset not found: {dataset_file}")
                return
                
            # Train Random Forest model
            self.logger.info("Training Random Forest model...")
            rf_model, rf_metrics = train_random_forest(
                data_file=dataset_file,
                output_dir=self.output_dir / "models",
                target_variable='bnpp',  # Below-ground net primary productivity
                test_size=0.2,
                random_state=42
            )
            
            self.logger.info(f"Random Forest trained - R²: {rf_metrics.get('r2', 'N/A'):.3f}")
            
            # Train MLP model
            self.logger.info("Training Multi-Layer Perceptron model...")
            mlp_model, mlp_metrics = train_mlp(
                data_file=dataset_file,
                output_dir=self.output_dir / "models",
                target_variable='bnpp',
                test_size=0.2,
                random_state=42
            )
            
            self.logger.info(f"MLP trained - R²: {mlp_metrics.get('r2', 'N/A'):.3f}")
            
            # Compare models and select the best
            best_model = 'RF' if rf_metrics.get('r2', 0) > mlp_metrics.get('r2', 0) else 'MLP'
            self.logger.info(f"Best performing model: {best_model}")
            
        except Exception as e:
            self.logger.error(f"Error in machine learning stage: {e}")
            raise
            
        self.logger.info("Stage 3 completed successfully")
        
    def stage_4_global_application(self):
        """
        Stage 4: Apply the machine learning model to the entire globe.
        """
        self.logger.info("=== Stage 4: Global Application ===")
        
        try:
            # Process global land cover data
            self.logger.info("Processing global land cover data...")
            synmap_data = process_synmap_data(
                data_dir=self.data_dir / "landcover",
                output_dir=self.output_dir / "global_predictions"
            )
            
            biomes_data = process_biomes_data(
                data_dir=self.data_dir / "landcover", 
                output_dir=self.output_dir / "global_predictions"
            )
            
            # Apply trained models globally
            model_dir = self.output_dir / "models"
            
            # Apply Random Forest globally
            if (model_dir / "random_forest_model.pkl").exists():
                self.logger.info("Applying Random Forest model globally...")
                rf_predictions = apply_global_rf(
                    model_file=model_dir / "random_forest_model.pkl",
                    global_data_dir=self.output_dir / "global_predictions",
                    output_dir=self.output_dir / "global_predictions"
                )
            
            # Apply MLP globally
            if (model_dir / "mlp_model.pkl").exists():
                self.logger.info("Applying MLP model globally...")
                mlp_predictions = apply_global_mlp(
                    model_file=model_dir / "mlp_model.pkl",
                    global_data_dir=self.output_dir / "global_predictions",
                    output_dir=self.output_dir / "global_predictions"
                )
            
        except Exception as e:
            self.logger.error(f"Error in global application stage: {e}")
            raise
            
        self.logger.info("Stage 4 completed successfully")
        
    def run_full_pipeline(self):
        """
        Run the complete TAM benchmark pipeline.
        """
        start_time = time.time()
        self.logger.info("=== Starting TAM Benchmark Pipeline ===")
        
        try:
            # Run all stages sequentially
            self.stage_1_data_retrieval()
            self.stage_2_data_integration()
            self.stage_3_machine_learning()
            self.stage_4_global_application()
            
            # Generate final summary
            self.generate_pipeline_summary()
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            self.logger.info(f"Pipeline completed in {elapsed_time/60:.1f} minutes")
            
    def run_stage(self, stage_number):
        """
        Run a specific stage of the pipeline.
        
        Args:
            stage_number (int): Stage number to run (1-4)
        """
        stage_methods = {
            1: self.stage_1_data_retrieval,
            2: self.stage_2_data_integration,
            3: self.stage_3_machine_learning,
            4: self.stage_4_global_application
        }
        
        if stage_number not in stage_methods:
            raise ValueError(f"Invalid stage number: {stage_number}. Must be 1-4.")
            
        self.logger.info(f"Running Stage {stage_number} only...")
        stage_methods[stage_number]()
        
    def generate_pipeline_summary(self):
        """
        Generate a summary report of the pipeline execution.
        """
        self.logger.info("=== Generating Pipeline Summary ===")
        
        summary_file = self.output_dir / "pipeline_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("TAM Benchmark Pipeline Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Check output files
            f.write("Generated Files:\n")
            for subdir in ['processed_data', 'integrated_data', 'models', 'global_predictions']:
                subdir_path = self.output_dir / subdir
                if subdir_path.exists():
                    files = list(subdir_path.glob('*'))
                    f.write(f"  {subdir}/: {len(files)} files\n")
                    
            f.write(f"\nAll outputs saved to: {self.output_dir}\n")
            
        self.logger.info(f"Pipeline summary saved to: {summary_file}")


def main():
    """
    Main function with command line interface.
    """
    parser = argparse.ArgumentParser(description='TAM Benchmark Pipeline')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4],
                       help='Run specific stage only (1-4)')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TAMPipeline(verbose=args.verbose)
    
    try:
        if args.stage:
            # Run specific stage
            pipeline.run_stage(args.stage)
        else:
            # Run full pipeline
            pipeline.run_full_pipeline()
            
        print("\n✓ TAM Benchmark Pipeline completed successfully!")
        print(f"✓ Results saved to: {pipeline.output_dir}")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()