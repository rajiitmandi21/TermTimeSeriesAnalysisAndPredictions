import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Custom transformer for creating time-based features
class TimeFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Takes a DataFrame with a 'Date' column and generates time-based features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # The input X is expected to be a DataFrame with a 'Date' column.
        df = X.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        features = pd.DataFrame(index=df.index)
        features["year"] = df["Date"].dt.year
        features["month"] = df["Date"].dt.month
        features["day"] = df["Date"].dt.day
        features["dayofweek"] = df["Date"].dt.dayofweek
        features["quarter"] = df["Date"].dt.quarter
        features["is_weekend"] = features["dayofweek"].isin([5, 6]).astype(int)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
        features["dayofweek_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7)
        features["dayofweek_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7)

        # Calculate days_since_start based on the min date in the current dataset
        start_date = df["Date"].min()
        features["days_since_start"] = (df["Date"] - start_date).dt.days

        return features.values  # Return as numpy array for the pipeline


def build_and_train_pipeline(df, date_column="Date", target_column="Price"):
    """
    Builds, trains, and saves a complete scikit-learn pipeline.
    """
    print("--- Building and Training Model Pipeline ---")

    # 1. Pre-process the data (handle interpolation for the target variable)
    df_processed = df.copy()
    df_processed[date_column] = pd.to_datetime(df_processed[date_column])
    df_processed.set_index(date_column, inplace=True)
    df_processed[target_column] = df_processed[target_column].interpolate(method="time")
    df_processed.reset_index(inplace=True)

    X = df_processed[[date_column]]
    y = df_processed[target_column]

    # 2. Define the models for the ensemble
    # Using default parameters inspired by the original `assignment.py` exploration
    xgb_reg = xgb.XGBRegressor(
        random_state=42, n_estimators=300, max_depth=5, learning_rate=0.1
    )
    rf_reg = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)
    lr_reg = LinearRegression()

    # 3. Create the ensemble model with weights
    ensemble = VotingRegressor(
        estimators=[("xgb", xgb_reg), ("rf", rf_reg), ("linear", lr_reg)],
        weights=[0.5, 0.3, 0.2],
    )

    # 4. Create the full pipeline
    model_pipeline = Pipeline(
        [
            ("feature_generator", TimeFeatureGenerator()),
            ("scaler", StandardScaler()),
            ("ensemble", ensemble),
        ]
    )

    # 5. Train the pipeline
    print("Training the pipeline...")
    model_pipeline.fit(X, y)
    print("Training complete.")

    return model_pipeline


def main():
    """
    Main function to run the pipeline creation process.
    """
    try:
        df = pd.read_csv("price_data.csv")
    except FileNotFoundError:
        print("Error: price_data.csv not found. Make sure it's in the same directory.")
        return

    # Build and train the pipeline
    trained_pipeline = build_and_train_pipeline(df)

    # Save the trained pipeline to a file
    pipeline_path = "outputs/complete_model_pipeline.joblib"
    joblib.dump(trained_pipeline, pipeline_path)
    print(f"--- Complete model pipeline saved to {pipeline_path} ---")

    # Example of how to load and use the pipeline
    print("\n--- Example: Loading and using the saved pipeline ---")
    loaded_pipeline = joblib.load(pipeline_path)
    print("Pipeline loaded successfully.")

    # Create some future dates to predict
    future_dates = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])}
    )

    # Make predictions
    predictions = loaded_pipeline.predict(future_dates)
    print("Example predictions:")
    for date, pred in zip(future_dates["Date"], predictions):
        print(f"  - Date: {date.date()}, Predicted Price: {pred:.2f}")


if __name__ == "__main__":
    main()
