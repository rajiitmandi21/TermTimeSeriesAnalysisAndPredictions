
import argparse
import subprocess
import pandas as pd
from eda import HPEdA
from assignment import PricePredictor
from short_term_model import analyze_timeseries

def run_eda(data_path, output_dir):
    """Runs the exploratory data analysis."""
    print("--- Running Exploratory Data Analysis ---")
    df = pd.read_csv(data_path)
    config = {
        "output_dir": output_dir,
        "output_file": "eda_output.md",
        "date_col": "Date",
        "price_col": "Price",
        "write_mode": "w",
    }
    eda = HPEdA(df, verbose=True, config=config)
    eda.run_complete_eda()
    print("--- EDA complete ---")

def run_ensemble_training(data_path, output_dir):
    """Runs the ensemble model training."""
    print("--- Running Ensemble Model Training ---")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    predictor = PricePredictor(df, "Date", "Price", output_dir=output_dir)
    predictor.tune_and_evaluate(n_iter=10, cv=3)
    print("--- Ensemble model training complete ---")

def run_prophet_forecasting(data_path, output_dir):
    """Runs the Prophet forecasting."""
    print("--- Running Prophet Forecasting ---")
    md_path = f"{output_dir}/prophet_model_output.md"
    with open(md_path, "w") as f:
        f.write("")  # Clear the file

    df = pd.read_csv(data_path)
    analyze_timeseries(
        df,
        periods=44,
        test_periods=60,
        freq="D",
        fill_dates=True,
        country_holidays="US",
        forecast_path=f"{output_dir}/feb_2016_forecast.csv",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        md_path=md_path,
        plot_prefix="prophet",
    )
    print("--- Prophet forecasting complete ---")

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="PriceLabs Assignment Pipeline")
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "eda", "ensemble", "prophet"],
        default="all",
        help="Which step of the pipeline to run.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="price_data.csv",
        help="Path to the price data CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the outputs.",
    )
    args = parser.parse_args()

    if args.step == "all" or args.step == "eda":
        run_eda(args.data_path, args.output_dir)
    if args.step == "all" or args.step == "ensemble":
        run_ensemble_training(args.data_path, args.output_dir)
    if args.step == "all" or args.step == "prophet":
        run_prophet_forecasting(args.data_path, args.output_dir)

if __name__ == "__main__":
    main()
