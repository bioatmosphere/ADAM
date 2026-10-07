import re

# Read the current file
with open('apply_MLP_globally.py', 'r') as f:
    content = f.read()

# Find and replace the main function
old_main_start = 'def main():'
new_main = '''def main():
    """
    Main function to apply MLP model globally for BNPP fraction prediction with GPP-based ecological constraints.
    """
    try:
        print("="*70)
        print("GLOBAL MLP BNPP_FRACTION PREDICTION")
        print("WITH GPP-BASED ECOLOGICAL CONSTRAINTS")
        print("="*70)

        # Configuration
        GPP_THRESHOLD = 0.0  # Minimum GPP for valid prediction (gC m⁻² yr⁻¹) - exclude only non-vegetated areas
        YEAR = 2010  # Year for climate and GPP data

        # 1. Load trained MLP model
        print("\\n[1/9] Loading trained model...")
        model, model_data = load_trained_mlp_model()
        # Get feature names from the scaler (the actual features the model was trained with)
        if hasattr(model_data['scaler'], 'feature_names_in_'):
            required_features = list(model_data['scaler'].feature_names_in_')
        else:
            # Fallback: BNPP_fraction model uses these 17 features (not gpp_yearly!)
            required_features = ['aet', 'pet', 'ppt', 'tmax', 'tmin', 'vpd',
                                'soil_carbon_stock', 'clay_content', 'silt_content', 'sand_content',
                                'nitrogen_content', 'cation_exchange_capacity', 'ph_in_water',
                                'bulk_density', 'coarse_fragments', 'soil_moisture', 'elevation']
        scaler = model_data['scaler']
        print(f"  Note: GPP is used for filtering only, NOT as a model feature")

        # 2. Load global climate data
        print("\\n[2/9] Loading global climate data...")
        climate_ds = load_global_terraclimate_data(year=YEAR)

        # 3. Load GLASS GPP data for ecological filtering
        print("\\n[3/9] Loading GLASS GPP data...")
        gpp_da = load_glass_gpp_data()

        # 4. Interpolate to common grid
        print("\\n[4/9] Interpolating data to common grid...")
        combined_ds = interpolate_to_common_grid(climate_ds, gpp_da)

        # 5. Prepare features for model (GPP kept separate for filtering)
        print("\\n[5/9] Preparing global features...")
        features_df, coords_df, gpp_df = prepare_global_features(combined_ds, required_features)

        # 6. Apply MLP model globally
        print("\\n[6/9] Applying MLP model...")
        predictions = apply_mlp_model_globally(model, features_df, scaler)

        # 7. Apply GPP-based ecological constraints
        print("\\n[7/9] Applying GPP-based ecological constraints...")
        constrained_predictions, stats = apply_ecological_constraints(
            predictions,
            gpp_df,
            gpp_threshold=GPP_THRESHOLD
        )

        # 8. Create global prediction map
        print("\\n[8/9] Creating outputs...")
        bnpp_da = create_global_prediction_map(constrained_predictions, coords_df)

        # 9. Plot global map
        plot_global_bnpp_map(bnpp_da)

        # Print summary statistics
        print("\\n" + "="*70)
        print("GLOBAL BNPP_FRACTION PREDICTION SUMMARY (GPP-CONSTRAINED)")
        print("="*70)
        print(f"Total grid points processed: {stats['n_total']:,}")
        print(f"Valid predictions (GPP > {GPP_THRESHOLD}): {stats['n_kept']:,} ({stats['n_kept']/stats['n_total']*100:.1f}%)")
        print(f"Filtered (GPP ≤ {GPP_THRESHOLD}): {stats['n_filtered']:,} ({stats['filter_percentage']:.1f}%)")

        print(f"\\nGlobal BNPP_fraction statistics (productive areas only):")
        print(f"  Minimum: {stats['min']:.4f}")
        print(f"  Maximum: {stats['max']:.4f}")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Standard deviation: {stats['std']:.4f}")

        # Percentiles on valid data
        valid_predictions = constrained_predictions[~np.isnan(constrained_predictions)]
        print(f"\\nPercentiles (productive areas only):")
        print(f"  25th: {np.percentile(valid_predictions, 25):.4f}")
        print(f"  50th (median): {np.percentile(valid_predictions, 50):.4f}")
        print(f"  75th: {np.percentile(valid_predictions, 75):.4f}")
        print(f"  95th: {np.percentile(valid_predictions, 95):.4f}")

        print(f"\\nGPP range in productive areas:")
        print(f"  Minimum: {stats['gpp_min']:.1f} gC m⁻² yr⁻¹")
        print(f"  Maximum: {stats['gpp_max']:.1f} gC m⁻² yr⁻¹")
        print(f"  Mean: {stats['gpp_mean']:.1f} gC m⁻² yr⁻¹")

        print("\\n" + "="*70)
        print("✓ Global MLP BNPP_fraction application completed successfully!")
        print("="*70)

        print("\\nOutput files created:")
        print("  - productivity/earth/global_bnpp_fraction_predictions_mlp_gpp_constrained.nc")
        print("  - productivity/earth/global_bnpp_fraction_map_mlp_gpp_constrained.png")
        print("\\nEcological filter applied:")
        print(f"  - Excluded areas with GPP ≤ {GPP_THRESHOLD} gC m⁻² yr⁻¹ (non-vegetated areas only)")
        print(f"  - This removes: Ice sheets, permanent snow, barren rock, water bodies")
        print(f"  - Includes: All vegetated areas including low-productivity ecosystems")
        print(f"  - GPP used for filtering only, NOT as a model predictor")

    except Exception as e:
        print(f"\\n✗ Error in global MLP application: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
'''

# Find the start of the main function
main_idx = content.find('def main():')
if main_idx == -1:
    print("Could not find main function!")
    exit(1)

# Replace from main function to end of file
content = content[:main_idx] + new_main

# Write back
with open('apply_MLP_globally.py', 'w') as f:
    f.write(content)

print("Successfully updated main function!")
