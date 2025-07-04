import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")

def save_plot_and_log(fig, plot_name, md_path):
    images_dir = os.path.join(os.path.dirname(md_path), "images")
    os.makedirs(images_dir, exist_ok=True)
    image_path = os.path.join(images_dir, f"{plot_name}.png")
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    rel_path = os.path.relpath(image_path, os.path.dirname(md_path))
    with open(md_path, "a") as f:
        f.write(f"\n![{plot_name}]({rel_path})\n\n")



def prepare_data_for_prophet(df):
    """
    Prepare data for Prophet model by handling missing dates and formatting

    Args:
        df: DataFrame with columns ['Date', 'Price']

    Returns:
        DataFrame formatted for Prophet (columns: ds, y)
    """
    df_clean = df.copy()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    df_clean = df_clean.drop_duplicates(subset=["Date"], keep="first")

    # make sure the date range is complete
    date_range = pd.date_range(df_clean["Date"].min(), df_clean["Date"].max(), freq="D")
    date_range_df = pd.DataFrame(date_range, columns=["Date"])
    df_clean = date_range_df.merge(df_clean, on="Date", how="left")

    # fill missing values with linear interpolation
    df_clean.set_index("Date", inplace=True)
    df_clean["Price"] = df_clean["Price"].interpolate(method="time")
    df_clean = df_clean.sort_values("Date").reset_index()

    # reset index
    df_clean.reset_index(inplace=True)
    df_prophet = df_clean.rename(columns={"Date": "ds", "Price": "y"})

    return df_prophet


def fill_missing_dates(df_prophet, freq="D"):
    """
    Fill missing dates in the time series

    Args:
        df_prophet: DataFrame with columns ['ds', 'y']
        freq: Frequency string ('D' for daily, 'W' for weekly, 'M' for monthly)

    Returns:
        DataFrame with filled missing dates
    """
    # Create complete date range
    date_range = pd.date_range(
        start=df_prophet["ds"].min(), end=df_prophet["ds"].max(), freq=freq
    )

    # Create complete dataframe
    complete_df = pd.DataFrame({"ds": date_range})

    # Merge with original data
    df_filled = complete_df.merge(df_prophet, on="ds", how="left")

    # Fill missing values using interpolation
    df_filled["y"] = df_filled["y"].interpolate(method="linear")

    # If still NaN at beginning or end, use forward/backward fill
    df_filled["y"] = df_filled["y"].fillna(method="bfill").fillna(method="ffill")

    return df_filled


def train_prophet_model(df_prophet, country_holidays=None, **kwargs):
    """
    Train Prophet model

    Args:
        df_prophet: DataFrame with columns ['ds', 'y']
        country_holidays: Optional country name for holidays
        **kwargs: Additional Prophet parameters

    Returns:
        Trained Prophet model
    """
    # Default Prophet parameters
    prophet_params = {
        "daily_seasonality": True,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "changepoint_prior_scale": 0.05,  # Flexibility of trend changes
        "seasonality_prior_scale": 10.0,  # Flexibility of seasonality
        "interval_width": 0.95,  # Uncertainty interval width
    }

    # Update with user-provided parameters
    prophet_params.update(kwargs)

    # Initialize and fit model
    model = Prophet(**prophet_params)

    if country_holidays:
        model.add_country_holidays(country_name=country_holidays)

    model.fit(df_prophet)

    return model


def make_predictions(model, periods=30, freq="D"):
    """
    Make future predictions

    Args:
        model: Trained Prophet model
        periods: Number of periods to predict
        freq: Frequency of predictions

    Returns:
        DataFrame with predictions
    """
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq=freq)

    # Make predictions
    forecast = model.predict(future)

    return forecast


def plot_results(model, forecast, md_path=None, plot_prefix="prophet"):
    """
    Plot the results and save to markdown if md_path is provided
    """
    # Plot forecast
    fig1 = model.plot(forecast, figsize=(12, 6))
    plt.title("Prophet Time Series Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    if md_path:
        save_plot_and_log(fig1, f"{plot_prefix}_forecast", md_path)
    else:
        plt.show()

    # Plot components
    fig2 = model.plot_components(forecast, figsize=(12, 8))
    if md_path:
        save_plot_and_log(fig2, f"{plot_prefix}_components", md_path)
    else:
        plt.show()


def analyze_timeseries(
    df,
    periods=30,
    freq="D",
    fill_dates=True,
    test_periods=30,
    country_holidays=None,
    forecast_path=None,
    md_path=None,
    plot_prefix="prophet",
    **prophet_kwargs,
):
    """
    Complete time series analysis pipeline

    Args:
        df: DataFrame with columns ['Date', 'Price']
        periods: Number of periods to forecast into the future
        freq: Frequency ('D', 'W', 'M', etc.)
        fill_dates: Whether to fill missing dates
        test_periods: Number of periods to use for testing
        country_holidays: Optional country name for holidays
        forecast_path: Optional path to save the forecast CSV
        md_path: Optional path to save markdown output (for print and images)
        plot_prefix: Prefix for plot image names
        **prophet_kwargs: Additional Prophet parameters

    Returns:
        Dictionary with model, forecast, and metrics
    """
    log_and_write("Starting time series analysis...", md_path)

    df_prophet = prepare_data_for_prophet(df)
    log_and_write(f"   Data shape: {df_prophet.shape}", md_path)
    log_and_write(f"   Date range: {df_prophet['ds'].min()} to {df_prophet['ds'].max()}", md_path)

    if fill_dates:
        df_prophet = fill_missing_dates(df_prophet, freq=freq)
        log_and_write(f"   Filled data shape: {df_prophet.shape}", md_path)

    # Train/test split
    train_df = df_prophet.iloc[:-test_periods]
    test_df = df_prophet.iloc[-test_periods:]
    log_and_write(f"\n   Training data shape: {train_df.shape}", md_path)
    log_and_write(f"   Test data shape:     {test_df.shape}", md_path)

    model = train_prophet_model(
        train_df, country_holidays=country_holidays, **prophet_kwargs
    )

    # Predict for the test period + the future period
    future_dataframe = model.make_future_dataframe(periods=test_periods + periods, freq=freq)
    forecast = model.predict(future_dataframe)

    plot_results(model, forecast, md_path=md_path, plot_prefix=plot_prefix)

    if forecast_path:
        # Select only the future forecast dates
        future_forecast_only = forecast.tail(periods)
        # Select and rename columns
        forecast_to_save = future_forecast_only[["ds", "yhat"]].rename(
            columns={"ds": "Date", "yhat": "Price"}
        )
        forecast_to_save.to_csv(forecast_path, index=False)
        log_and_write(f"\n   Forecast for future {periods} periods saved to {forecast_path}", md_path)

    # --- Evaluation on Test Set ---
    # Isolate the forecast for the test period
    test_forecast = forecast.iloc[-(test_periods + periods) : -periods]
    eval_df = test_df.merge(test_forecast[['ds', 'yhat']], on='ds', how='left')
    mae = np.mean(np.abs(eval_df["y"] - eval_df["yhat"]))
    mape = np.mean(np.abs((eval_df["y"] - eval_df["yhat"]) / eval_df["y"])) * 100
    rmse = np.sqrt(np.mean((eval_df["y"] - eval_df["yhat"]) ** 2))
    log_and_write("\nEvaluation on Test Set:", md_path)
    log_and_write(f"-   MAE: {mae:.2f}", md_path)
    log_and_write(f"-   MAPE: {mape:.2f}%", md_path)
    log_and_write(f"-   RMSE: {rmse:.2f}", md_path)

    # --- Cross-validation ---
    log_and_write("\nRunning cross-validation...", md_path)
    try:
        # Use parallel processing for speed
        df_cv = cross_validation(
            model, initial=f'{3 * test_periods} days', period=f'{test_periods // 2} days', horizon=f'{test_periods} days', parallel="processes"
        )
        df_p = performance_metrics(df_cv)
        log_and_write("\nCross-validation performance metrics:", md_path)
        log_and_write(df_to_markdown_table(df_p.head()), md_path)
    except Exception as e:
        log_and_write(f"   Could not run cross-validation: {e}", md_path)
        df_p = None

    results = {
        "model": model,
        "forecast": forecast,
        "test_metrics": {"mae": mae, "mape": mape, "rmse": rmse},
        "cross_validation_metrics": df_p,
    }
    return results


def log_and_write(text, md_path):
    print(text)
    with open(md_path, "a") as f:
        f.write(str(text) + "\n\n")

def df_to_markdown_table(df):
    try:
        import tabulate
        return tabulate.tabulate(df, headers="keys", tablefmt="pipe")
    except ImportError:
        return df.to_string()


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    pwd = Path(__file__).parent
    md_path = pwd / "outputs" / "prophet_model_output.md"
    # Clean the markdown file at the start
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    open(md_path, "w").close()

    df = pd.read_csv(pwd / "price_data.csv")
    log_and_write("Sample data:", md_path)
    log_and_write(df_to_markdown_table(df.sample(5)), md_path)
    log_and_write(f"Data shape: {df.shape}", md_path)

    results = analyze_timeseries(
        df,
        periods=44,  # Forecast 44 days for all of Feb 2016
        test_periods=60,
        freq="D",
        fill_dates=True,
        country_holidays="US",
        forecast_path=pwd / "outputs" / "feb_2016_forecast.csv",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        md_path=md_path,
        plot_prefix="prophet",
    )

    log_and_write("\n--- Analysis Complete ---", md_path)
    log_and_write("\nTest Set Metrics:", md_path)
    log_and_write(results["test_metrics"], md_path)
    if results["cross_validation_metrics"] is not None:
        log_and_write("\nCross-validation metrics (summary):", md_path)
        log_and_write(df_to_markdown_table(results["cross_validation_metrics"].head()), md_path)

    log_and_write("\nFinal Forecast summary (last 10 points of the full forecast):", md_path)
    log_and_write(
        df_to_markdown_table(results["forecast"][["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10)), md_path
    )
