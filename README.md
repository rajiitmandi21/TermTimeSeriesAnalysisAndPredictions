# PriceLabs Assignment
# Hotel Price Forecasting

## Project Overview

This project provides a comprehensive pipeline for hotel price forecasting using time series analysis and machine learning ensemble models. It includes modules for exploratory data analysis (EDA), model training, forecasting with Prophet and ensemble methods, and report generation. The project is designed for reproducibility and extensibility, supporting both research and production use cases.

## Installation Instructions

1. **Clone the repository** and navigate to the project directory.
2. **Set up the environment** (recommended: Conda):
   ```bash
   bash setup.sh
   ```
   This script will:
   - Create a new Conda environment (`assignment_env`) with Python 3.11
   - Install all required dependencies from `requirements.txt`
   - Check for the presence of `price_data.csv` (place your data file in the root directory)

3. **Manual setup (optional):**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

- **Run the full pipeline (EDA, ensemble training, Prophet forecasting):**
  ```bash
  python main.py --step all --data_path price_data.csv --output_dir pricelabs/forecast_outputs
  ```

- **Run only EDA:**
  ```bash
  python main.py --step eda --data_path price_data.csv --output_dir pricelabs/forecast_outputs
  ```

- **Run only ensemble model training:**
  ```bash
  python main.py --step ensemble --data_path price_data.csv --output_dir pricelabs/forecast_outputs
  ```

- **Run only Prophet forecasting:**
  ```bash
  python main.py --step prophet --data_path price_data.csv --output_dir pricelabs/forecast_outputs
  ```

- **Train and save a complete pipeline:**
  ```bash
  python pipeline.py
  ```

- **Perform EDA directly:**
  ```bash
  python eda.py
  ```

## Directory Structure

```
.
├── assignment.py                # Main forecasting and model training logic
├── eda.py                       # Exploratory Data Analysis module
├── main.py                      # Pipeline entry point (EDA, training, forecasting)
├── pipeline.py                  # Build and train a complete ML pipeline
├── short_term_model.py          # Time series analysis and Prophet forecasting
├── setup.sh                     # Environment setup script
├── requirements.txt             # Python dependencies
├── price_data.csv               # Input data (not included in repo)
├── pricelabs/
│   ├── forecast_outputs/
│   │   ├── images/              # Output plots and visualizations
│   │   ├── forecasting_report.md/pdf # Generated reports
│   │   ├── training_data_*.csv  # Processed training data
│   ├── models/                  # Saved model files and metadata
│   ├── images/                  # Additional generated images
│   ├── detailed report.pdf      # Detailed analysis report
│   ├── forecasting_report.pdf   # Forecasting report
├── forecast_outputs/            # (if used) Additional outputs
│   ├── models/                  # (if used) Model outputs
│   ├── images/                  # (if used) Output images
```

## Dependencies

All dependencies are listed in `requirements.txt`:

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- tabulate
- prophet
- plotly
- xgboost
- joblib
- scikit-learn

To install them manually:
```bash
pip install -r requirements.txt
```
