# Hotel Price Forecasting Report

**Generated on:** 2025-07-10 16:24:10

---

# Model Training and Evaluation

- **Training Date:** 2025-07-10 16:24:13

- **Hyperparameter Search Iterations:** 10

- **Cross-Validation Folds:** 5

## Outlier Detection and Clipping

- **Method:** Deviation from Smooth Trend

- **Window Size:** 21 days

- **Threshold:** ±3.0 standard deviations

- **Total Data Points:** 1,476

- **Outliers Detected:** 0

- **Outlier Rate:** 0.00%


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

