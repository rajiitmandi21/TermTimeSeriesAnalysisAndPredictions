# Hotel Price Forecasting Report

**Generated on:** 2025-07-05 03:50:05

---

# Model Training and Evaluation

- **Training Date:** 2025-07-05 03:50:05

- **Hyperparameter Search Iterations:** 10

- **Cross-Validation Folds:** 5

## Outlier Detection and Clipping

- **Method:** Deviation from Smooth Trend

- **Window Size:** 21 days

- **Threshold:** ±3.0 standard deviations

- **Total Data Points:** 1,476

- **Outliers Detected:** 2

- **Outlier Rate:** 0.14%


![outlier_clipping_analysis](images/outlier_clipping_analysis.png)

## Filling Missing Values

- **Date column:** Date found

- **Missing dates:** 1

 Filling missing values with interpolation

## Feature Engineering

- **Total Features Created:** 19

- **Feature Categories:**

  - Basic Time: year, month, dayofyear, days_since_start

  - Cyclical: month_sin/cos, dayofyear_sin/cos, year_sin/cos

- **Training Data Date Range:** 2012-01-01 to 2016-01-16


![feature_correlation_heatmap_before_selection](images/feature_correlation_heatmap_before_selection.png)

- **Feature `year_sin`** removed due to high correlation with: `year` (corr=0.93)

- **Feature `year_cos`** removed due to high correlation with: `year` (corr=0.97)

- **Feature `days_since_start`** removed due to high correlation with: `year` (corr=0.97)

- **Feature `week_sin`** removed due to high correlation with: `month_sin` (corr=0.97)

- **Feature `dayofyear_sin`** removed due to high correlation with: `month_sin` (corr=0.95)

- **Feature `dayofyear`** removed due to high correlation with: `month_sin` (corr=0.76)

- **Feature `month`** removed due to high correlation with: `month_sin` (corr=0.76)

- **Feature `week`** removed due to high correlation with: `month_sin` (corr=0.75)

- **Feature `week_cos`** removed due to high correlation with: `dayofyear_cos` (corr=1.00)

- **Feature `month_cos`** removed due to high correlation with: `dayofyear_cos` (corr=0.95)

- **Feature `day_of_month_sin`** removed due to high correlation with: `day_of_month` (corr=0.80)

- **Feature `day_of_week`** removed due to high correlation with: `day_of_week_sin` (corr=0.73)

- **Total features removed:** 12


![feature_correlation_heatmap_after_selection](images/feature_correlation_heatmap_after_selection.png)

## Dataset Statistics

- **Total Samples:** 1,477

- **Features:** 7

- **Target Variable:** Price

- **Target Range:** 81.00 - 193.00

- **Target Mean:** 114.01

- **Target Std:** 20.08

- **Training Data Stored:** Available for analysis

## Linear Regression Training

### Cross-Validation Performance (Primary Metrics)

- **CV RMSE:** 11.4582 (±0.9192)

- **CV R² Score:** 0.6699 (±0.0300)

- **CV MAE:** 8.6199 (±0.4720)

### Training Performance (Full Dataset)

- **Train R² Score:** 0.6792

- **Train RMSE:** 11.3673

- **Train MAE:** 8.5320

### Top 5 Feature Coefficients

  - year: 12.5060

  - month_sin: 10.9226

  - dayofyear_cos: -1.3133

  - day_of_week_cos: -0.5895

  - day_of_month_cos: -0.3160

## Random Forest Training

### Cross-Validation Performance (Primary Metrics)

- **CV RMSE:** 5.2389 (±0.3064)

- **CV R² Score:** 0.9306 (±0.0079)

- **CV MAE:** 3.9891 (±0.1397)

### Training Performance (Full Dataset)

- **Train RMSE:** 2.6378

- **Train MAE:** 2.0554

- **Train R² Score:** 0.9827

### Hyperparameter Tuning Results

- **Best Parameters:** {'n_estimators': 200, 'max_depth': 10}

### Top 5 Feature Importances

  - month_sin: 0.4530

  - year: 0.4419

  - dayofyear_cos: 0.0606

  - day_of_month: 0.0173

  - day_of_month_cos: 0.0122

## XGBoost Training

### Cross-Validation Performance (Primary Metrics)

- **CV RMSE:** 5.3305 (±0.4265)

- **CV R² Score:** 0.9281 (±0.0098)

- **CV MAE:** 4.0267 (±0.1777)

### Training Performance (Full Dataset)

- **Train RMSE:** 3.5028

- **Train MAE:** 2.6544

- **Train R² Score:** 0.9695

### Hyperparameter Tuning Results

- **Best Parameters:** {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}

### Top 5 Feature Importances

  - month_sin: 0.5730

  - year: 0.3586

  - dayofyear_cos: 0.0261

  - day_of_month: 0.0130

  - day_of_month_cos: 0.0111


![feature_importance_comparison](images/feature_importance_comparison.png)


![cv_performance_comparison](images/cv_performance_comparison.png)

## Training Summary

- **Models Successfully Trained:** 3

- **Model Performance Comparison (Cross-Validation):**

  - Linear Regression CV RMSE: 11.4582 (±0.9192)

  - Linear Regression CV R²: 0.6699 (±0.0300)

  - Random Forest CV RMSE: 5.2389 (±0.3064)

  - Random Forest CV R²: 0.9306 (±0.0079)

  - XGBoost CV RMSE: 5.3305 (±0.4265)

  - XGBoost CV R²: 0.9281 (±0.0098)

## Saving Trained Models

- **LINEAR Model:** pricelabs/forecast_outputs/models/linear_model.joblib

- **RF Model:** pricelabs/forecast_outputs/models/rf_model.joblib

- **XGB Model:** pricelabs/forecast_outputs/models/xgb_model.joblib

- **Scaler:** pricelabs/forecast_outputs/models/scaler.joblib

- **Metadata:** pricelabs/forecast_outputs/models/metadata.pkl

- **Total Models Saved:** 3

- **Models Directory:** pricelabs/forecast_outputs/models

# Training Data Analysis

## Feature Correlation with Target

- **Top 10 Most Correlated Features:**

  - year: 0.6297

  - month_sin: 0.5367

  - dayofyear_cos: 0.0985

  - day_of_week_cos: 0.0330

  - day_of_month_cos: 0.0232

  - day_of_month: 0.0155

  - day_of_week_sin: 0.0005

## Feature Statistics

|       |       year |      month_sin |   dayofyear_cos |   day_of_week_cos |   day_of_month_cos |   day_of_month |   day_of_week_sin |
|:------|-----------:|---------------:|----------------:|------------------:|-------------------:|---------------:|------------------:|
| count | 1477       | 1477           |    1477         |    1477           |       1477         |     1477       |    1477           |
| mean  | 2013.53    |    0.000683561 |       0.0106835 |      -3.12697e-17 |         -0.0190724 |       15.6513  |       6.76507e-18 |
| std   |    1.14239 |    0.704106    |       0.710879  |       0.707346    |          0.701011  |        8.80031 |       0.707346    |
| min   | 2012       |   -1           |      -0.999979  |      -0.900969    |         -0.994869  |        1       |      -0.974928    |
| 25%   | 2013       |   -0.5         |      -0.69831   |      -0.900969    |         -0.758758  |        8       |      -0.781831    |
| 50%   | 2014       |    1.22465e-16 |       0.0182766 |      -0.222521    |         -0.0506492 |       16       |       0           |
| 75%   | 2015       |    0.5         |       0.724004  |       0.62349     |          0.688967  |       23       |       0.781831    |
| max   | 2016       |    1           |       0.999991  |       1           |          1         |       31       |       0.974928    |

## Data Quality: No missing values in training data


![feature_distributions](images/feature_distributions.png)

# Prediction Generation

- **Prediction Period:** 2012-01-01 to 2020-12-31

- **Prediction Points:** 3,288

- **Features Used:** 7

- **Ensemble Weights:**

  - linear: 1.00

  - rf: 0.00

  - xgb: 0.00

## Prediction Statistics

- **Forecast Range:** 80.95 - 201.20

- **Forecast Mean:** 141.04

- **Forecast Std:** 30.25

## Sample Predictions

|    | Date                |   Forecast |
|---:|:--------------------|-----------:|
|  0 | 2012-01-01 00:00:00 |    102.002 |
|  1 | 2012-01-02 00:00:00 |    101.782 |
|  2 | 2012-01-03 00:00:00 |    102.208 |
|  3 | 2012-01-04 00:00:00 |    103.001 |
|  4 | 2012-01-05 00:00:00 |    103.618 |
|  5 | 2012-01-06 00:00:00 |    103.657 |
|  6 | 2012-01-07 00:00:00 |    103.16  |
|  7 | 2012-01-08 00:00:00 |    102.576 |
|  8 | 2012-01-09 00:00:00 |    102.423 |
|  9 | 2012-01-10 00:00:00 |    102.894 |


![forecast_linear_regression_only](images/forecast_linear_regression_only.png)

# Prediction Generation

- **Prediction Period:** 2012-01-01 to 2020-12-31

- **Prediction Points:** 3,288

- **Features Used:** 7

- **Ensemble Weights:**

  - linear: 0.00

  - rf: 1.00

  - xgb: 0.00

## Prediction Statistics

- **Forecast Range:** 84.36 - 190.53

- **Forecast Mean:** 127.75

- **Forecast Std:** 21.39

## Sample Predictions

|    | Date                |   Forecast |
|---:|:--------------------|-----------:|
|  0 | 2012-01-01 00:00:00 |    97.9962 |
|  1 | 2012-01-02 00:00:00 |    95.984  |
|  2 | 2012-01-03 00:00:00 |    95.6751 |
|  3 | 2012-01-04 00:00:00 |    94.8017 |
|  4 | 2012-01-05 00:00:00 |    93.3615 |
|  5 | 2012-01-06 00:00:00 |    93.9745 |
|  6 | 2012-01-07 00:00:00 |    95.4253 |
|  7 | 2012-01-08 00:00:00 |    96.1103 |
|  8 | 2012-01-09 00:00:00 |    97.1568 |
|  9 | 2012-01-10 00:00:00 |    96.9577 |


![forecast_random_forest_only](images/forecast_random_forest_only.png)

# Prediction Generation

- **Prediction Period:** 2012-01-01 to 2020-12-31

- **Prediction Points:** 3,288

- **Features Used:** 7

- **Ensemble Weights:**

  - linear: 0.00

  - rf: 0.00

  - xgb: 1.00

## Prediction Statistics

- **Forecast Range:** 84.63 - 188.74

- **Forecast Mean:** 128.66

- **Forecast Std:** 21.97

## Sample Predictions

|    | Date                |   Forecast |
|---:|:--------------------|-----------:|
|  0 | 2012-01-01 00:00:00 |    99.6058 |
|  1 | 2012-01-02 00:00:00 |    95.2199 |
|  2 | 2012-01-03 00:00:00 |    93.9756 |
|  3 | 2012-01-04 00:00:00 |    92.9587 |
|  4 | 2012-01-05 00:00:00 |    92.9637 |
|  5 | 2012-01-06 00:00:00 |    93.2465 |
|  6 | 2012-01-07 00:00:00 |    94.2062 |
|  7 | 2012-01-08 00:00:00 |    94.709  |
|  8 | 2012-01-09 00:00:00 |    94.1858 |
|  9 | 2012-01-10 00:00:00 |    93.601  |


![forecast_xgboost_only](images/forecast_xgboost_only.png)

# Prediction Generation

- **Prediction Period:** 2012-01-01 to 2020-12-31

- **Prediction Points:** 3,288

- **Features Used:** 7

- **Ensemble Weights:**

  - linear: 0.60

  - rf: 0.00

  - xgb: 0.40

## Prediction Statistics

- **Forecast Range:** 83.91 - 196.06

- **Forecast Mean:** 136.09

- **Forecast Std:** 25.58

## Sample Predictions

|    | Date                |   Forecast |
|---:|:--------------------|-----------:|
|  0 | 2012-01-01 00:00:00 |   101.044  |
|  1 | 2012-01-02 00:00:00 |    99.1574 |
|  2 | 2012-01-03 00:00:00 |    98.9151 |
|  3 | 2012-01-04 00:00:00 |    98.984  |
|  4 | 2012-01-05 00:00:00 |    99.3562 |
|  5 | 2012-01-06 00:00:00 |    99.493  |
|  6 | 2012-01-07 00:00:00 |    99.5786 |
|  7 | 2012-01-08 00:00:00 |    99.4295 |
|  8 | 2012-01-09 00:00:00 |    99.1284 |
|  9 | 2012-01-10 00:00:00 |    99.1769 |


![forecast_balanced_ensemble](images/forecast_balanced_ensemble.png)

# Prediction Generation

- **Prediction Period:** 2012-01-01 to 2020-12-31

- **Prediction Points:** 3,288

- **Features Used:** 7

- **Ensemble Weights:**

  - linear: 0.40

  - rf: 0.00

  - xgb: 0.60

## Prediction Statistics

- **Forecast Range:** 84.85 - 193.49

- **Forecast Mean:** 133.61

- **Forecast Std:** 23.83

## Sample Predictions

|    | Date                |   Forecast |
|---:|:--------------------|-----------:|
|  0 | 2012-01-01 00:00:00 |   100.564  |
|  1 | 2012-01-02 00:00:00 |    97.8449 |
|  2 | 2012-01-03 00:00:00 |    97.2686 |
|  3 | 2012-01-04 00:00:00 |    96.9756 |
|  4 | 2012-01-05 00:00:00 |    97.2253 |
|  5 | 2012-01-06 00:00:00 |    97.4108 |
|  6 | 2012-01-07 00:00:00 |    97.7878 |
|  7 | 2012-01-08 00:00:00 |    97.856  |
|  8 | 2012-01-09 00:00:00 |    97.4809 |
|  9 | 2012-01-10 00:00:00 |    97.3183 |


![forecast_xgboost-linear_hybrid](images/forecast_xgboost-linear_hybrid.png)

# Prediction Generation

- **Prediction Period:** 2020-02-01 to 2020-02-29

- **Prediction Points:** 29

- **Features Used:** 7

- **Ensemble Weights:**

  - linear: 0.40

  - rf: 0.30

  - xgb: 0.30

## Prediction Statistics

- **Forecast Range:** 163.10 - 178.61

- **Forecast Mean:** 172.09

- **Forecast Std:** 4.47

## Sample Predictions

|    | Date                |   Forecast |
|---:|:--------------------|-----------:|
|  0 | 2020-02-01 00:00:00 |    163.856 |
|  1 | 2020-02-02 00:00:00 |    163.104 |
|  2 | 2020-02-03 00:00:00 |    164.473 |
|  3 | 2020-02-04 00:00:00 |    167.793 |
|  4 | 2020-02-05 00:00:00 |    171.577 |
|  5 | 2020-02-06 00:00:00 |    168.694 |
|  6 | 2020-02-07 00:00:00 |    168.491 |
|  7 | 2020-02-08 00:00:00 |    168.239 |
|  8 | 2020-02-09 00:00:00 |    166.844 |
|  9 | 2020-02-10 00:00:00 |    166.456 |


![final_forecast_feb_2020](images/final_forecast_feb_2020.png)

## Training Data Export

- **Feature-engineered data saved to:** pricelabs/forecast_outputs/training_data_features.csv

- **Raw features with target saved to:** pricelabs/forecast_outputs/training_data_raw_features.csv

# Comprehensive Analysis Summary

---

## Dataset Overview

- **Data Source:** Price time series data

- **Date Range:** 0 to 1475

- **Total Records:** 1,476

- **Data Quality:** 100.0% complete

## Data Processing Summary

- **Outlier Clipping:** Enabled

- **Feature Engineering:** 7 features created

- **Data Scaling:** StandardScaler applied

- **Training Data Stored:** 1477 samples with 7 features

- **Available Components:** Raw features, scaled features, target variable, feature-engineered dataframe

## Model Performance Summary

- **Linear Regression Training R² Score:** 0.6699

- **Linear Regression Training RMSE:** 11.4582

- **Linear Regression CV RMSE:** 11.4582 (±0.9192)

- **Linear Regression CV R²:** 0.6699 (±0.0300)

- **Random Forest Training R² Score:** 0.9306

- **Random Forest Training RMSE:** 5.2389

- **Random Forest CV RMSE:** 5.2389 (±0.3064)

- **Random Forest CV R²:** 0.9306 (±0.0079)

- **XGBoost Training R² Score:** 0.9281

- **XGBoost Training RMSE:** 5.3305

- **XGBoost CV RMSE:** 5.3305 (±0.4265)

- **XGBoost CV R²:** 0.9281 (±0.0098)

## Forecast Summary

### Linear Regression Only

- **Forecast Period:** 0 to 3287

- **Forecast Points:** 3,288

- **Value Range:** 80.95 - 201.20

- **Average Value:** 141.04

### Random Forest Only

- **Forecast Period:** 0 to 3287

- **Forecast Points:** 3,288

- **Value Range:** 84.36 - 190.53

- **Average Value:** 127.75

### XGBoost Only

- **Forecast Period:** 0 to 3287

- **Forecast Points:** 3,288

- **Value Range:** 84.63 - 188.74

- **Average Value:** 128.66

### Balanced Ensemble

- **Forecast Period:** 0 to 3287

- **Forecast Points:** 3,288

- **Value Range:** 83.91 - 196.06

- **Average Value:** 136.09

### XGBoost-Linear Hybrid

- **Forecast Period:** 0 to 3287

- **Forecast Points:** 3,288

- **Value Range:** 84.85 - 193.49

- **Average Value:** 133.61

### Final Forecast (Feb 2020)

- **Forecast Period:** 0 to 28

- **Forecast Points:** 29

- **Value Range:** 163.10 - 178.61

- **Average Value:** 172.09

## Technical Details

- **Model Types:** Linear Regression, Random Forest, XGBoost

- **Ensemble Method:** Weighted average of individual predictions

- **Hyperparameter Tuning:** RandomizedSearchCV with cross-validation

- **Feature Types:** Time-based features with cyclical encoding

## Key Insights and Recommendations

- **Best Performing Model (CV RMSE):** Random Forest (5.2389)

- **Model Performance Rankings (CV RMSE):**

  1. Random Forest: 5.2389

  2. XGBoost: 5.3305

  3. Linear Regression: 11.4582

- **Data Volatility:** Moderate volatility - normal price fluctuations

## Notes

- All models were trained using time-based features to capture seasonal patterns

- Ensemble predictions combine the strengths of different modeling approaches

- Cross-validation ensures robust performance estimates

- Feature importance analysis identifies key predictive factors

---

**Report Generated:** 2025-07-05 03:50:39

**End of Analysis**

## Saving Trained Models

- **LINEAR Model:** pricelabs/forecast_outputs/models/linear_model.joblib

- **RF Model:** pricelabs/forecast_outputs/models/rf_model.joblib

- **XGB Model:** pricelabs/forecast_outputs/models/xgb_model.joblib

- **Scaler:** pricelabs/forecast_outputs/models/scaler.joblib

- **Metadata:** pricelabs/forecast_outputs/models/metadata.pkl

- **Total Models Saved:** 3

- **Models Directory:** pricelabs/forecast_outputs/models

