# ADAM Model Benchmark Suite - Comprehensive Results

## Overview
This document presents the comprehensive benchmarking results for 12 different machine learning algorithms applied to Below-ground Net Primary Productivity (BNPP) prediction using the ADAM (Transport and Absorptive roots with Mycorrhizal fungi) framework.

## Dataset Information
- **Total samples**: 1,420 measurements from global ecosystems
- **Features**: 18 environmental predictors including:
  - Climate variables (aet, pet, ppt, tmax, tmin, vpd)
  - Productivity metrics (gpp_yearly)
  - Soil properties (carbon, texture, chemistry, moisture)
  - Topography (elevation)
- **Target variable**: BNPP in g C m⁻² yr⁻¹
- **Data source**: Integrated ForC forest database and global grassland productivity data
- **Outlier handling**: Conservative IQR method (multiplier = 2.5)

## Model Performance Ranking (by Test R²)

### 🏆 **Top Tier (R² > 0.50)**

#### 1. Random Forest - **R² = 0.5382** 🥇
- **Test RMSE**: 185.13 g C m⁻² yr⁻¹
- **Test MAE**: 125.19 g C m⁻² yr⁻¹
- **Training time**: ~2 seconds
- **Key strengths**: Excellent interpretability, robust to outliers, handles non-linear relationships well

#### 2. AutoGluon AutoML - **R² = 0.5340** 🥈
- **Test RMSE**: 185.98 g C m⁻² yr⁻¹
- **Test MAE**: 122.65 g C m⁻² yr⁻¹
- **Training time**: 282 seconds
- **Key strengths**: Automated ensemble, minimal tuning required, competitive performance

#### 3. CatBoost - **R² = 0.5145** 🥉
- **Test RMSE**: 189.83 g C m⁻² yr⁻¹
- **Test MAE**: 130.31 g C m⁻² yr⁻¹
- **Training time**: ~15 seconds
- **Key strengths**: Handles categorical features well, gradient boosting efficiency

#### 4. XGBoost - **R² = 0.5047**
- **Test RMSE**: 191.73 g C m⁻² yr⁻¹
- **Test MAE**: 127.39 g C m⁻² yr⁻¹
- **Training time**: ~8 seconds
- **Key strengths**: Industry standard, excellent performance-speed balance

#### 5. LightGBM - **R² = 0.5005**
- **Test RMSE**: 192.54 g C m⁻² yr⁻¹
- **Test MAE**: 127.54 g C m⁻² yr⁻¹
- **Training time**: ~3 seconds
- **Key strengths**: Very fast training, memory efficient

### 🔬 **Mid Tier (R² 0.30-0.50)**

#### 6. TabPFN - **R² = 0.4840**
- **Test RMSE**: 204.07 g C m⁻² yr⁻¹
- **Test MAE**: 143.90 g C m⁻² yr⁻¹
- **Training time**: ~600 seconds (500 samples)
- **Key strengths**: Foundation model, no hyperparameter tuning needed

#### 7. Multi-Layer Perceptron (MLP) - **R² = 0.4478**
- **Test RMSE**: 202.45 g C m⁻² yr⁻¹
- **Test MAE**: 135.24 g C m⁻² yr⁻¹
- **Training time**: ~45 seconds
- **Key strengths**: Universal approximator capability

#### 8. Deep Ensemble - **R² = 0.3384**
- **Test RMSE**: 221.59 g C m⁻² yr⁻¹
- **Test MAE**: 153.35 g C m⁻² yr⁻¹
- **Training time**: ~180 seconds
- **Key strengths**: Uncertainty quantification, ensemble robustness

#### 9. TabNet - **R² = 0.3362**
- **Test RMSE**: 221.97 g C m⁻² yr⁻¹
- **Test MAE**: 148.47 g C m⁻² yr⁻¹
- **Training time**: ~90 seconds
- **Key strengths**: Attention mechanism, interpretable deep learning

### 📉 **Lower Tier (R² < 0.30)**

#### 10. Elastic Net - **R² = 0.1767**
- **Test RMSE**: 247.19 g C m⁻² yr⁻¹
- **Test MAE**: 174.83 g C m⁻² yr⁻¹
- **Training time**: <1 second
- **Key strengths**: Regularization, linear interpretability

#### 11. Ridge Regression - **R² = 0.1520**
- **Test RMSE**: 250.88 g C m⁻² yr⁻¹
- **Test MAE**: 178.63 g C m⁻² yr⁻¹
- **Training time**: <1 second
- **Key strengths**: Simple, fast, regularized linear model

#### 12. Support Vector Machine - **R² = 0.1074**
- **Test RMSE**: 257.40 g C m⁻² yr⁻¹
- **Test MAE**: 161.71 g C m⁻² yr⁻¹
- **Training time**: ~2 seconds
- **Key strengths**: Kernel methods, robust to outliers

## Key Insights

### 🌳 **Tree-Based Model Dominance**
- **Top 5 models** are all tree-based ensemble methods
- Clear performance advantage over other paradigms
- Excellent handling of non-linear ecological relationships
- Natural feature interaction modeling

### 🤖 **AutoML Effectiveness**
- **AutoGluon** achieved 2nd place with minimal manual tuning
- Demonstrates value of automated model selection and ensembling
- Nearly matches best manual approach (Random Forest)

### 📊 **Performance Clustering**
- **Tier 1**: Tree-based models (R² > 0.50)
- **Tier 2**: Neural networks and foundation models (R² 0.30-0.50)
- **Tier 3**: Linear models (R² < 0.30)

### 🧮 **Linear Model Limitations**
- Traditional linear approaches struggle with BNPP prediction
- Suggests strong non-linear relationships in ecological data
- Complex environmental interactions beyond linear assumptions

### 🚀 **Speed vs Performance Trade-offs**
- **Fastest**: Ridge, Elastic Net (<1s)
- **Best balance**: Random Forest, LightGBM (2-3s, R² > 0.50)
- **Slowest**: TabPFN (600s), AutoGluon (282s)

## Recommendations

### 🎯 **For Production Use**
1. **Random Forest**: Best overall performance + interpretability
2. **LightGBM**: Excellent performance + speed for large-scale applications
3. **AutoGluon**: When minimizing manual tuning effort

### 🔬 **For Research Applications**
1. **Random Forest**: Feature importance analysis
2. **Deep Ensemble**: Uncertainty quantification
3. **TabPFN**: Foundation model capabilities

### ⚡ **For Real-Time Applications**
1. **LightGBM**: Best performance-speed balance
2. **Random Forest**: Good performance, fast inference
3. **Ridge Regression**: Ultra-fast but limited accuracy

## Technical Implementation Notes

- All models use 80/20 train-test split with random_state=42
- Cross-validation applied where appropriate
- Missing values handled per model requirements
- Feature scaling applied for neural networks and linear models
- Hyperparameter tuning performed for applicable models

## Dataset Challenges Identified

1. **Missing values**: Up to 17% missing in some soil variables
2. **Non-linear relationships**: Linear models consistently underperform
3. **Complex interactions**: Tree models excel at capturing feature interactions
4. **Scale variations**: BNPP ranges from 2 to 1,333 g C m⁻² yr⁻¹

## Conclusion

The benchmark demonstrates that **tree-based ensemble methods** are the optimal choice for BNPP prediction using environmental predictors. **Random Forest** emerges as the best performer, achieving R² = 0.5382, while **AutoGluon** provides an excellent automated alternative with minimal tuning effort.

The strong performance of tree-based models suggests that BNPP prediction involves complex non-linear relationships and feature interactions that are naturally captured by decision tree ensembles, making them well-suited for ecological modeling applications.

---
*Generated by ADAM Benchmark Pipeline - 13 Model Comprehensive Evaluation*
*Dataset: 1,420 global BNPP measurements with 18 environmental predictors*
*Date: 2025-08-04*