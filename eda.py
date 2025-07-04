"""
Hotel Price EDA Script

Performs exploratory data analysis on hotel price time series data,
including feature engineering, anomaly detection, and visualization.
Outputs results and plots to a Markdown file for easy review.

Dependencies:
    pandas, numpy, matplotlib, seaborn, statsmodels, tabulate

Usage:
    python assignment.py
"""

import os
from calendar import month_name
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

cs = "₹"  # currency symbol

week_day_name = [
    "",  # to match with the index and day_of_week
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thurshday",
    "Friday",
    "Saturday",
    "Sunday",
]


### EDA Class
class HPEdA:
    """
    Exploratory Data Analysis (EDA) class for hotel price time series.

    This class provides methods to clean, prepare, and analyze hotel price data,
    including feature engineering, statistical summaries, anomaly detection,
    correlation analysis, seasonality, time series decomposition, and visualization.
    All analysis is performed on the raw data.

    Attributes:
        raw_df (pd.DataFrame): Original input DataFrame.
        df (pd.DataFrame): Processed DataFrame with features.
        date_col (str): Name of the date column.
        price_col (str): Name of the price column.
        verbose (bool): If True, writes detailed logs and outputs.
        file_path (str): Path to the Markdown output file.
    """

    DEFAULT_CONFIG = {
        "output_dir": "pricelabs/outputs",
        "output_file": "output.md",
        "image_dir": "images",
        "date_col": "Date",
        "price_col": "Price",
        "window_sizes": [7, 14, 30, 90],
        "lag_sizes": [1, 7, 14, 30, 365],
        "anomaly_zscore_threshold": 3,
        "write_mode": "w",  # 'w' for overwrite, 'a' for append
    }

    def __init__(self, df, date_col=None, price_col=None, verbose=False, config=None):
        """
        Initialize the EDA class for hotel price analysis.

        Args:
            df (pd.DataFrame): Input DataFrame.
            date_col (str): Name of the date column.
            price_col (str): Name of the price column.
            verbose (bool): If True, enables detailed logging and output.
            config (dict, optional): Configuration dictionary.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        self.verbose = verbose
        self.raw_df = df.copy()
        self.date_col = date_col or self.config["date_col"]
        self.price_col = price_col or self.config["price_col"]
        self.output_dir = self.config["output_dir"]
        self.image_dir = os.path.join(self.output_dir, self.config["image_dir"])
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        self.file_path = os.path.join(self.output_dir, self.config["output_file"])
        # Overwrite or append mode
        open(self.file_path, self.config["write_mode"]).close()
        self._logged_sections = set()

    def _write_to_file(self, text):
        """
        Write text to the Markdown output file.

        Args:
            text (str): Text to write.
        """
        with open(self.file_path, "a") as f:
            f.write(str(text))

    def _save_plot_and_log(self, fig, plot_name):
        """
        Save a matplotlib figure as an image and log its Markdown path.

        Args:
            fig (matplotlib.figure.Figure): Figure to save.
            plot_name (str): Name for the saved plot image.
        Side Effects:
            - Saves the plot image to the images directory.
            - Writes a Markdown image link to the output file.
        """
        image_path = os.path.join(self.image_dir, f"{plot_name}.png")
        fig.savefig(image_path, bbox_inches="tight")
        plt.close(fig)
        rel_path = os.path.relpath(image_path, os.path.dirname(self.file_path))
        self._write_to_file(f"\n![{plot_name}]({rel_path})\n")

    def _log_and_write(self, text, section=None):
        """
        Write text to the Markdown output file with double line breaks.

        Args:
            text (str): Text to write.
            section (str, optional): Section for logging.
        """
        if section:
            if section in self._logged_sections:
                return
            self._logged_sections.add(section)
        self._write_to_file(text + "\n\n")

    def _df_to_markdown_table(self, df):
        """
        Convert a DataFrame to a Markdown-formatted table.

        Args:
            df (pd.DataFrame): DataFrame to convert.
        Returns:
            str: Markdown table as a string.
        """
        try:
            import tabulate

            return tabulate.tabulate(df, headers="keys", tablefmt="pipe")
        except ImportError:
            return df.to_string()

    def raw_df_summary(self, verbose=None):
        """
        Generate and log a summary of the raw input DataFrame.

        Args:
            verbose (bool or None): If True, writes summary to file. If None, uses self.verbose.
        Side Effects:
            - Writes summary statistics and sample data to the Markdown output file.
            - Saves and logs basic plots of the raw data.
        """
        if verbose is None:
            verbose = self.verbose

        if verbose:
            self._log_and_write("\n# Raw DataFrame Summary\n")
            self._log_and_write(f"**Columns:** {self.raw_df.columns.tolist()}")
            self._log_and_write(f"**Rows, Columns:** {self.raw_df.shape}")
            self._log_and_write(f"**Data Types:**\n{self.raw_df.dtypes}")
            self._log_and_write(
                f"**Descriptive Statistics:**\n{self._df_to_markdown_table(self.raw_df.describe())}"
            )
            self._log_and_write(f"**Null Values:**\n{self.raw_df.isnull().sum()}")
            # plot scatter plot at top complete row
            # plot histogram at bottom first column
            # plot boxplot at bottom second column
            fig1 = plt.figure(figsize=(10, 10))
            plt.plot(self.raw_df["Date"], self.raw_df[self.price_col])
            plt.title("Price vs Date")
            plt.xlabel("Date")
            plt.ylabel("Price")
            self._save_plot_and_log(fig1, "price_vs_date")

            fig2 = plt.figure(figsize=(10, 10))
            plt.subplot(2, 2, 1)
            plt.hist(self.raw_df[self.price_col], bins=20, color="skyblue", alpha=0.7)
            plt.title("Price Distribution")
            plt.xlabel("Price")
            plt.ylabel("Count")

            plt.subplot(2, 2, 2)
            plt.boxplot(self.raw_df[self.price_col])
            plt.title("Price Boxplot")
            plt.xlabel("Price")
            plt.ylabel("Count")

            self._save_plot_and_log(fig2, "price_distribution_boxplot")

    def prepare_data(self, verbose=None):
        """
        Prepare and clean the data for EDA.

        - Converts price column to float.
        - Converts date column to datetime.
        - Fills missing dates and interpolates missing prices.

        Args:
            verbose (bool or None): If True, logs actions. If None, uses self.verbose.
        Side Effects:
            - Updates self.df with cleaned and indexed data.
            - Logs missing data information if verbose.
        """
        if verbose is None:
            verbose = self.verbose

        self.orginal_df_col = self.raw_df.columns.tolist()
        self.raw_df[self.price_col] = self.raw_df[self.price_col].astype(float)
        self.raw_df["dt"] = pd.to_datetime(
            self.raw_df[self.date_col]
        )  # date time columns
        date_range = self.raw_df["dt"].min(), self.raw_df["dt"].max()

        if verbose:
            self._log_and_write(f"Date Range: {date_range[0]} to {date_range[1]}")

        # date series with all date from min to max
        date_df = pd.date_range(start=date_range[0], end=date_range[1])

        # create a date range dataframe
        date_df = pd.DataFrame(
            pd.date_range(start=date_range[0], end=date_range[1]), columns=["dt"]
        )

        # merge the date range dataframe with the raw dataframe
        self.df = date_df.merge(self.raw_df, on="dt", how="left")

        # sort the dataframe by date
        self.df = self.df.sort_values(by="dt").reset_index(drop=True)
        self.df.set_index("dt", inplace=True)

        self._log_and_write(
            f"Price data is missing for {self.df[self.price_col].isnull().sum()} days"
        )
        self._log_and_write("Missing dates:")
        self._log_and_write(
            self._df_to_markdown_table(
                self.df[self.df[self.price_col].isnull()].index.astype(str).tolist()
            )
        )

        # keep a flag for na filled values and count of na filled values
        self.df["real_value"] = self.df[self.price_col].notna()
        # fillna with linear interpolation
        self.df[self.price_col] = self.df[self.price_col].fillna(
            self.df[self.price_col].interpolate(method="time")
        )

    def create_features(self, verbose=None):
        """
        Create time series and date-based features for EDA.

        Args:
            verbose (bool or None): If True, logs actions. If None, uses self.verbose.
        Side Effects:
            - Adds new feature columns to self.df.
            - Logs the number of features created if verbose.
        """
        # now extract the detailed date features for EDA
        self.df["year"] = self.df.index.year
        self.df["quarter_number"] = self.df.index.quarter  # 1-4
        self.df["month_number"] = self.df.index.month  # 1-12
        self.df["week_number"] = self.df.index.isocalendar().week  # 1-52

        self.df["day_of_week"] = self.df.index.dayofweek  # 0-6
        self.df["day_of_month"] = self.df.index.day  # 1-31
        self.df["day_of_year"] = self.df.index.dayofyear  # 1-366

        self.df["month_name"] = self.df["month_number"].map(
            {k: v[:3] for k, v in enumerate(month_name)}
        )

        self.df["week_number_name"] = self.df["day_of_week"].map(
            {k: v[:3] for k, v in enumerate(week_day_name)}
        )

        # weekend, month start, month end
        self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(int)
        self.df["is_month_start"] = (self.df["day_of_month"] <= 5).astype(int)
        self.df["is_month_end"] = (self.df["day_of_month"] >= 25).astype(int)

        # lag features
        for lag in self.config["lag_sizes"]:
            self.df[f"price_lag_{lag}"] = self.df[self.price_col].shift(lag)

        # rolling statistics
        for window in self.config["window_sizes"]:
            self.df[f"rolling_mean_{window}"] = (
                self.df[self.price_col].rolling(window=window).mean()
            )
            self.df[f"rolling_std_{window}"] = (
                self.df[self.price_col].rolling(window=window).std()
            )
            self.df[f"rolling_min_{window}"] = (
                self.df[self.price_col].rolling(window=window).min()
            )
            self.df[f"rolling_max_{window}"] = (
                self.df[self.price_col].rolling(window=window).max()
            )

        # price changes and volatility
        self.df["price_change"] = self.df[self.price_col].diff()
        self.df["price_pct_change"] = self.df[self.price_col].pct_change()
        self.df["price_volatility_7d"] = self.df["price_pct_change"].rolling(7).std()

        # z-scores for anomaly detection
        self.df["price_zscore_7d"] = (
            self.df[self.price_col] - self.df["rolling_mean_7"]
        ) / self.df["rolling_std_7"]
        self.df["price_zscore_30d"] = (
            self.df[self.price_col] - self.df["rolling_mean_30"]
        ) / self.df["rolling_std_30"]

        # cyclical features (sine/cosine encoding)
        self.df["day_of_year_sin"] = np.sin(2 * np.pi * self.df["day_of_year"] / 365.25)
        self.df["day_of_year_cos"] = np.cos(2 * np.pi * self.df["day_of_year"] / 365.25)
        self.df["day_of_week_sin"] = np.sin(2 * np.pi * self.df["day_of_week"] / 7)
        self.df["day_of_week_cos"] = np.cos(2 * np.pi * self.df["day_of_week"] / 7)

        self._log_and_write(f"Created {len(self.df.columns) - 1} features")

    def basic_statistics(self, verbose=None):
        """
        Compute and log basic statistics for the price data.

        Args:
            verbose (bool or None): If True, logs statistics. If None, uses self.verbose.
        Side Effects:
            - Writes summary tables and plots to the Markdown output file.
        """
        if verbose is None:
            verbose = self.verbose

        if verbose:
            self._log_and_write("\n## Basic Statistics\n")
            self._log_and_write(
                self._df_to_markdown_table(
                    self.df[self.price_col].describe().to_frame().T
                )
            )
            self._log_and_write("\n### Yearly Statistics\n")
            year_df = self.df.groupby("year")[self.price_col].agg(
                ["count", "mean", "std", "min", "max"]
            )
            self._log_and_write(self._df_to_markdown_table(year_df))
            self.plot_x_y(
                year_df, True, True, "year", "Price", plot_name="year_vs_price"
            )

            self._log_and_write("\n### Day of Week Statistics\n")
            week_df = self.df.groupby("day_of_week")[self.price_col].agg(
                ["mean", "std"]
            )
            self._log_and_write(self._df_to_markdown_table(week_df))
            self.plot_x_y(
                week_df,
                True,
                True,
                "day_of_week",
                "Price",
                plot_name="dow_vs_price",
            )

            self._log_and_write("\n### Monthly Statistics\n")
            df_agg = self.df.groupby("month_name")[self.price_col].agg(["mean", "std"])
            self._log_and_write(self._df_to_markdown_table(df_agg))
            self.plot_x_y(
                df_agg, True, True, "month_name", "Price", plot_name="month_vs_price"
            )

            self._log_and_write("\n### Week Number Statistics\n")
            df_agg = self.df.groupby("week_number")[self.price_col].agg(["mean", "std"])
            self._log_and_write(self._df_to_markdown_table(df_agg))
            self.plot_x_y(
                df_agg, True, True, "week_number", "Price", plot_name="week_vs_price"
            )

            self._log_and_write("\n### Day of Month Statistics\n")
            df_agg = self.df.groupby("day_of_month")[self.price_col].agg(
                ["mean", "std"]
            )
            self._log_and_write(self._df_to_markdown_table(df_agg))
            self.plot_x_y(
                df_agg, True, True, "day_of_month", "Price", plot_name="dom_vs_price"
            )

    def stationarity_tests(self, verbose=True):
        """
        Perform stationarity tests (ADF) on the price series.

        Args:
            verbose (bool or None): If True, logs results. If None, uses self.verbose.
        Side Effects:
            - Writes test results to the Markdown output file.
        """
        if verbose is None:
            verbose = self.verbose

        # ADF Test
        adf_result = adfuller(self.df[self.price_col])

        if verbose:
            self._log_and_write("\n## Stationarity Tests\n")
            self._log_and_write("### Augmented Dickey-Fuller Test:")
            self._log_and_write(f"- ADF Statistic: {adf_result[0]:.6f}")
            self._log_and_write(f"- p-value: {adf_result[1]:.6f}")
            self._log_and_write(f"- Critical Values: {adf_result[4]}")
            self._log_and_write(
                f"- Stationary: {'Yes' if adf_result[1] < 0.05 else 'No'}"
            )

    def detect_anomalies(self, verbose=True):
        """
        Detect anomalies in the price series using z-score and IQR methods.

        Args:
            verbose (bool or None): If True, logs results. If None, uses self.verbose.
        Side Effects:
            - Adds anomaly columns to self.df.
            - Writes anomaly statistics and plots to the Markdown output file.
        """
        if verbose is None:
            verbose = self.verbose

        if verbose:
            self._log_and_write("\n## Anomaly Detection\n")
            # z-score
            z_anomalies_30d = (
                np.abs(self.df["price_zscore_30d"])
                > self.config["anomaly_zscore_threshold"]
            )
            self._log_and_write(
                f"Z-score anomalies for 30 days: {z_anomalies_30d.sum()}"
            )
            self._log_and_write(
                f"Dates: {self.df[z_anomalies_30d].index.astype(str).tolist()}"
            )
            self.df["is_anomaly_zscore"] = z_anomalies_30d
            # iqr
            Q1 = self.df[self.price_col].quantile(0.25)
            Q3 = self.df[self.price_col].quantile(0.75)
            iqr = Q3 - Q1
            self._log_and_write(f"IQR: {iqr}")
            iqr_anomalies = (self.df[self.price_col] > Q3 + 1.5 * iqr) | (
                self.df[self.price_col] < Q1 - 1.5 * iqr
            )
            self._log_and_write(f"IQR anomalies: {iqr_anomalies.sum()}")
            self._log_and_write(
                f"Dates: {self.df[iqr_anomalies].index.astype(str).tolist()}"
            )
            self.df["is_anomaly_iqr"] = iqr_anomalies
            # plot both anomalies in a single plot with different colors
            plt.figure(figsize=(10, 6))
            plt.plot(self.df.index, self.df[self.price_col], label="Price")
            plt.plot(
                self.df[z_anomalies_30d].index,
                self.df[z_anomalies_30d][self.price_col],
                "x",
                label="Z-score anomalies",
            )
            fig = plt.gcf()
            self._save_plot_and_log(fig, "anomaly_detection")

    def correlation_analysis(self, verbose=True):
        """
        Analyze correlations between numeric features and price.

        Args:
            verbose (bool or None): If True, logs results. If None, uses self.verbose.
        Side Effects:
            - Writes top correlations to the Markdown output file.
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            self._log_and_write("\n## Correlation Analysis\n")

            # Select numeric columns for correlation
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            correlation_matrix = self.df[numeric_cols].corr()

            # Top correlations with Price
            price_corr = (
                correlation_matrix[self.price_col].abs().sort_values(ascending=False)
            )
            self._log_and_write("### Top correlations with Price:")
            for col, corr in price_corr.head(10).items():
                if col != self.price_col:
                    self._log_and_write(f"- {col}: {corr:.3f}")

    def seasonal_patterns(self, verbose=True):
        """
        Analyze and log seasonal patterns in the price data.

        Args:
            verbose (bool or None): If True, logs results. If None, uses self.verbose.
        Side Effects:
            - Writes seasonal statistics and plots to the Markdown output file.
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            self._log_and_write("\n## Seasonal Analysis\n")

            # Monthly patterns
            monthly_stats = self.df.groupby("month_name")[self.price_col].agg(
                ["mean", "std", "count"]
            )
            if verbose:
                self._log_and_write("### Monthly Statistics:")
                self._log_and_write(self._df_to_markdown_table(monthly_stats))

                # Day-of-week patterns
                # dow_stats = self.df.groupby("day_of_week")["Price"].mean()
                # dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                self._log_and_write("\n### Weekend vs Weekday Average:")
                self._log_and_write(
                    f"- Weekday Average: {cs}{self.df[self.df['is_weekend'] == 0][self.price_col].mean():.2f}"
                )
                self._log_and_write(
                    f"- Weekend Average: {cs}{self.df[self.df['is_weekend'] == 1][self.price_col].mean():.2f}"
                )
            self.plot_x_y(
                monthly_stats,
                True,
                True,
                "month_name",
                "Price",
                plot_name="seasonal_monthly_stats",
            )

    def time_series_decomposition(self, verbose=True):
        """
        Perform and log time series decomposition (trend, seasonality, residual).

        Args:
            verbose (bool or None): If True, logs results. If None, uses self.verbose.
        Side Effects:
            - Adds decomposition columns to self.df.
            - Writes decomposition results to the Markdown output file.
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            self._log_and_write("\n## Time Series Decomposition\n")

            # Classical decomposition
            try:
                decomposition = seasonal_decompose(
                    self.df[self.price_col], model="additive", period=365
                )
                trend_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(
                    decomposition.trend.dropna() + decomposition.resid.dropna()
                )
                seasonal_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(
                    decomposition.seasonal.dropna() + decomposition.resid.dropna()
                )

                if verbose:
                    self._log_and_write(f"- Trend Strength: {trend_strength:.3f}")
                    self._log_and_write(f"- Seasonal Strength: {seasonal_strength:.3f}")

                # Store decomposition components
                self.df["trend"] = decomposition.trend
                self.df["seasonal"] = decomposition.seasonal
                self.df["residual"] = decomposition.resid

            except Exception as e:
                self._log_and_write(f"Decomposition failed: {str(e)}")

    def create_visualizations(self):
        """
        Create and log comprehensive visualizations for the EDA report.

        Side Effects:
            - Saves multiple plots and logs their paths to the Markdown output file.
        """
        self._log_and_write("\n## Detailed Visualizations\n")

        # Set up the plotting area
        fig = plt.figure(figsize=(20, 24))

        # - Time series plot
        fig.add_subplot(6, 3, 1)
        plt.plot(self.df.index, self.df[self.price_col], linewidth=0.8, alpha=0.8)
        plt.title("Price Time Series", fontsize=12, fontweight="bold")
        plt.ylabel(f"Price ({cs})")
        plt.xticks(rotation=45)

        # - Price distribution
        plt.subplot(6, 3, 2)
        plt.hist(self.df[self.price_col], bins=50, alpha=0.7, edgecolor="black")
        plt.title("Price Distribution", fontsize=12, fontweight="bold")
        plt.xlabel(f"Price ({cs})")
        plt.ylabel("Frequency")

        # - Box plot by year
        plt.subplot(6, 3, 3)
        self.df.boxplot(column="Price", by="year", ax=plt.gca())
        plt.title("Price Distribution by Year", fontsize=12, fontweight="bold")
        plt.suptitle("")

        # -. Monthly seasonality
        plt.subplot(6, 3, 4)
        monthly_avg = self.df.groupby("month_name")["Price"].mean()
        plt.plot(
            monthly_avg.index, monthly_avg.values, marker="o", linewidth=2, markersize=6
        )
        plt.title("Monthly Average Price", fontsize=12, fontweight="bold")
        plt.xlabel("Month")
        plt.ylabel(f"Average Price ({cs})")
        plt.xticks(range(1, 13))

        # - Day-of-week patterns
        plt.subplot(6, 3, 5)
        dow_avg = self.df.groupby("day_of_week")["Price"].mean()
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        plt.bar(dow_names, dow_avg.values, alpha=0.7)
        plt.title("Average Price by Day of Week", fontsize=12, fontweight="bold")
        plt.ylabel(f"Average Price ({cs})")
        plt.xticks(rotation=45)

        # - Rolling statistics
        plt.subplot(6, 3, 6)
        plt.plot(self.df.index, self.df[self.price_col], alpha=0.3, label="Daily Price")
        plt.plot(
            self.df.index, self.df["rolling_mean_30"], linewidth=2, label="30-day MA"
        )
        plt.plot(
            self.df.index, self.df["rolling_mean_90"], linewidth=2, label="90-day MA"
        )
        plt.title("Price with Moving Averages", fontsize=12, fontweight="bold")
        plt.ylabel(f"Price ({cs})")
        plt.legend()
        plt.xticks(rotation=45)

        # - Volatility analysis
        plt.subplot(6, 3, 7)
        plt.plot(self.df.index, self.df["price_volatility_7d"], linewidth=1)
        plt.title("7-Day Price Volatility", fontsize=12, fontweight="bold")
        plt.ylabel("Volatility")
        plt.xticks(rotation=45)

        # - Autocorrelation plot
        plt.subplot(6, 3, 8)
        price_clean = self.df[self.price_col].dropna()
        lags = range(1, min(50, len(price_clean) // 4))
        autocorr = [price_clean.autocorr(lag=lag) for lag in lags]
        plt.plot(lags, autocorr, marker="o", markersize=3)
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.axhline(y=0.05, color="red", linestyle="--", alpha=0.5)
        plt.axhline(y=-0.05, color="red", linestyle="--", alpha=0.5)
        plt.title("Autocorrelation Function", fontsize=12, fontweight="bold")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")

        # - Price changes distribution
        plt.subplot(6, 3, 9)
        price_changes = self.df["price_change"].dropna()
        plt.hist(price_changes, bins=50, alpha=0.7, edgecolor="black")
        plt.title("Price Changes Distribution", fontsize=12, fontweight="bold")
        plt.xlabel(f"Price Change ({cs})")
        plt.ylabel("Frequency")

        # - Anomalies visualization
        plt.subplot(6, 3, 10)
        plt.plot(self.df.index, self.df[self.price_col], alpha=0.5, label="Price")
        anomaly_dates = self.df[self.df["is_anomaly_zscore"]].index
        anomaly_prices = self.df[self.df["is_anomaly_zscore"]][self.price_col]
        plt.scatter(
            anomaly_dates,
            anomaly_prices,
            color="red",
            s=30,
            label="Z-score Anomalies",
            zorder=5,
        )
        plt.title("Anomaly Detection (Z-score)", fontsize=12, fontweight="bold")
        plt.ylabel(f"Price ({cs})")
        plt.legend()
        plt.xticks(rotation=45)

        # -  Quarterly trends
        plt.subplot(6, 3, 11)
        quarterly_avg = (
            self.df.groupby(["year", "quarter_number"])["Price"].mean().reset_index()
        )
        quarterly_avg["period"] = (
            quarterly_avg["year"].astype(str)
            + "-Q"
            + quarterly_avg["quarter_number"].astype(str)
        )
        plt.plot(
            range(len(quarterly_avg)),
            quarterly_avg["Price"],
            marker="o",
            linewidth=2,
            markersize=4,
        )
        plt.title("Quarterly Price Trends", fontsize=12, fontweight="bold")
        plt.ylabel(f"Average Price ({cs})")
        plt.xticks(
            range(0, len(quarterly_avg), 4),
            [quarterly_avg.iloc[i]["period"] for i in range(0, len(quarterly_avg), 4)],
            rotation=45,
        )

        # - Weekend vs Weekday comparison
        plt.subplot(6, 3, 12)
        weekend_data = [
            self.df[self.df["is_weekend"] == 0]["Price"].values,
            self.df[self.df["is_weekend"] == 1]["Price"].values,
        ]
        plt.boxplot(weekend_data, tick_labels=["Weekday", "Weekend"])
        plt.title("Weekday vs Weekend Prices", fontsize=12, fontweight="bold")
        plt.ylabel(f"Price ({cs})")

        # - Year-over-year growth
        plt.subplot(6, 3, 13)
        yearly_avg = self.df.groupby("year")["Price"].mean()
        yoy_growth = yearly_avg.pct_change() * 100
        plt.bar(yoy_growth.index[1:], yoy_growth.values[1:], alpha=0.7)
        plt.title("Year-over-Year Price Growth (%)", fontsize=12, fontweight="bold")
        plt.ylabel("Growth Rate (%)")
        plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # - Seasonal decomposition (if available)
        if "trend" in self.df.columns:
            plt.subplot(6, 3, 14)
            plt.plot(self.df.index, self.df["trend"], linewidth=1.5, label="Trend")
            plt.title("Trend Component", fontsize=12, fontweight="bold")
            plt.ylabel(f"Price ({cs})")
            plt.xticks(rotation=45)

            plt.subplot(6, 3, 15)
            plt.plot(self.df.index, self.df["seasonal"], linewidth=1, alpha=0.8)
            plt.title("Seasonal Component", fontsize=12, fontweight="bold")
            plt.ylabel("Seasonal Effect")
            plt.xticks(rotation=45)

        # - Correlation heatmap (top features)
        plt.subplot(6, 3, 16)
        numeric_cols = [
            "Price",
            "rolling_mean_7",
            "rolling_mean_30",
            "price_volatility_7d",
            "day_of_week",
            "month_number",
            "is_weekend",
        ]
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        corr_matrix = self.df[available_cols].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature Correlation Matrix", fontsize=12, fontweight="bold")

        # - Price momentum
        plt.subplot(6, 3, 17)
        momentum_3d = self.df[self.price_col].rolling(3).mean().pct_change()
        plt.plot(self.df.index, momentum_3d, alpha=0.7, linewidth=0.8)
        plt.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        plt.title("3-Day Price Momentum", fontsize=12, fontweight="bold")
        plt.ylabel("Momentum")
        plt.xticks(rotation=45)

        # - Price range analysis
        plt.subplot(6, 3, 18)
        monthly_range = self.df.groupby("month_name")[self.price_col].agg(
            lambda x: x.max() - x.min()
        )
        plt.bar(monthly_range.index, monthly_range.values, alpha=0.7)
        plt.title("Monthly Price Range", fontsize=12, fontweight="bold")
        plt.xlabel("Month")
        plt.ylabel(f"Price Range ({cs})")
        plt.xticks(range(1, 13))

        plt.tight_layout()
        self._save_plot_and_log(fig, "comprehensive_visualizations")

    def generate_summary_report(self):
        """
        Generate a comprehensive summary report of the EDA.

        Side Effects:
            - Writes summary statistics, trends, and recommendations to the Markdown output file.
        """
        self._log_and_write("\n# COMPREHENSIVE EDA SUMMARY REPORT\n")
        # Dataset overview
        self._log_and_write("## DATASET OVERVIEW:")
        self._log_and_write(
            f"- Time Period: {self.df.index.min().strftime('%Y-%m-%d')} to {self.df.index.max().strftime('%Y-%m-%d')}"
        )
        self._log_and_write(f"- Total Days: {len(self.df):,}")
        self._log_and_write(f"- Years Covered: {self.df.index.year.nunique()}")
        self._log_and_write(
            f"- Data Completeness: {(1 - self.df['Price'].isnull().mean()) * 100:.1f}%"
        )
        # Price statistics
        self._log_and_write("\n## PRICE STATISTICS:")
        self._log_and_write(f"- Average Price: {cs}{self.df['Price'].mean():.2f}")
        self._log_and_write(f"- Median Price: {cs}{self.df['Price'].median():.2f}")
        self._log_and_write(
            f"- Price Range: {cs}{self.df['Price'].min():.2f} - {cs}{self.df['Price'].max():.2f}"
        )
        self._log_and_write(f"- Standard Deviation: {cs}{self.df['Price'].std():.2f}")
        self._log_and_write(
            f"- Coefficient of Variation: {(self.df['Price'].std() / self.df['Price'].mean()) * 100:.1f}%"
        )
        # Trend analysis
        yearly_avg = self.df.groupby("year")["Price"].mean()
        if len(yearly_avg) > 1:
            total_growth = ((yearly_avg.iloc[-1] / yearly_avg.iloc[0]) - 1) * 100
            years_span = yearly_avg.index[-1] - yearly_avg.index[0]
            cagr = (
                pow(yearly_avg.iloc[-1] / yearly_avg.iloc[0], 1 / years_span) - 1
            ) * 100
            self._log_and_write("\n## TREND ANALYSIS:")
            self._log_and_write(f"- Total Growth: {total_growth:+.1f}%")
            self._log_and_write(f"- CAGR: {cagr:+.1f}%")
        # Seasonality insights
        monthly_avg = self.df.groupby("month_number")["Price"].mean()
        peak_month = monthly_avg.idxmax()
        trough_month = monthly_avg.idxmin()
        self._log_and_write("\n## SEASONALITY INSIGHTS:")
        self._log_and_write(
            f"- Peak Month: {month_name[peak_month]} ({cs}{monthly_avg.max():.2f})"
        )
        self._log_and_write(
            f"- Lowest Month: {month_name[trough_month]} ({cs}{monthly_avg.min():.2f})"
        )
        self._log_and_write(
            f"- Seasonal Variation: {((monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean()) * 100:.1f}%"
        )
        weekend_avg = self.df[self.df["is_weekend"] == 1]["Price"].mean()
        weekday_avg = self.df[self.df["is_weekend"] == 0]["Price"].mean()
        weekend_premium = ((weekend_avg / weekday_avg) - 1) * 100
        self._log_and_write(f"- Weekend Premium: {weekend_premium:+.1f}%")
        # Volatility analysis
        self._log_and_write("\n## VOLATILITY ANALYSIS:")
        daily_volatility = self.df["price_pct_change"].std() * 100
        self._log_and_write(f"- Daily Volatility: {daily_volatility:.2f}%")
        if "price_volatility_7d" in self.df.columns:
            avg_weekly_vol = self.df["price_volatility_7d"].mean() * 100
            self._log_and_write(f"- Average Weekly Volatility: {avg_weekly_vol:.2f}%")
        # Anomaly summary
        if "is_anomaly_zscore" in self.df.columns:
            anomaly_count = self.df["is_anomaly_zscore"].sum()
            anomaly_rate = (anomaly_count / len(self.df)) * 100
            self._log_and_write("\n## ANOMALY DETECTION:")
            self._log_and_write(f"- Total Anomalies: {anomaly_count}")
            self._log_and_write(f"- Anomaly Rate: {anomaly_rate:.2f}%")
        # Key recommendations
        self._log_and_write("\n## KEY INSIGHTS & RECOMMENDATIONS:")
        # Price stability
        cv = (self.df[self.price_col].std() / self.df[self.price_col].mean()) * 100
        if cv < 10:
            self._log_and_write(f"- Price is relatively stable (CV: {cv:.1f}%)")
        elif cv < 20:
            self._log_and_write(f"- Price shows moderate volatility (CV: {cv:.1f}%)")
        else:
            self._log_and_write(f"- Price is highly volatile (CV: {cv:.1f}%)")
        # Seasonal patterns
        if abs(weekend_premium) > 5:
            self._log_and_write(
                f"- Strong weekend effect detected ({weekend_premium:+.1f}%)"
            )
        # Trend direction
        if len(yearly_avg) > 1:
            recent_trend = yearly_avg.iloc[-3:].pct_change().mean() * 100
            if recent_trend > 2:
                self._log_and_write("- Recent upward trend in pricing")
            elif recent_trend < -2:
                self._log_and_write("- Recent downward trend in pricing")
            else:
                self._log_and_write("- Pricing trend is relatively stable")
        self._log_and_write("\n" + "=" * 60)
        self._log_and_write("                    END OF REPORT")
        self._log_and_write("=" * 60)

    def run_complete_eda(self):
        """
        Run the complete EDA pipeline: summary, cleaning, feature engineering, statistics, tests, anomaly detection, correlation, seasonality, decomposition, visualization, and report.

        Side Effects:
            - Writes all results and plots to the Markdown output file.
        """
        self.raw_df_summary(verbose=self.verbose)
        self.prepare_data(verbose=self.verbose)
        self.create_features(verbose=self.verbose)
        self.basic_statistics(verbose=self.verbose)
        self.stationarity_tests(verbose=self.verbose)
        self.detect_anomalies(verbose=self.verbose)
        self.correlation_analysis(verbose=self.verbose)
        self.seasonal_patterns(verbose=self.verbose)
        self.time_series_decomposition(verbose=self.verbose)
        self.create_visualizations()
        self.generate_summary_report()

    def plot_x_y(
        self, df, index_as_x=False, y_all=False, col_x=None, col_y=None, plot_name=None
    ):
        """
        Plot one or more columns against an x-axis and save the plot.

        Args:
            df (pd.DataFrame): DataFrame to plot.
            index_as_x (bool): If True, use DataFrame index as x-axis.
            y_all (bool): If True, plot all columns as y.
            col_x (str): Column to use as x-axis if index_as_x is False.
            col_y (str): Column to use as y-axis if y_all is False.
            plot_name (str or None): Name for saving the plot image.
        Returns:
            list: List of columns skipped during plotting.
        Side Effects:
            - Saves the plot image and logs its path to the Markdown file.
        """
        skipped_cols = []
        plt.figure(figsize=(10, 6))
        if y_all:
            for col in df.columns:
                try:
                    if col != self.date_col:
                        plt.plot(
                            df.index if index_as_x else df[col_x],
                            df[col],
                            label=col + " " + col_y if col_y else col,
                        )
                    else:
                        skipped_cols.append(col)
                except Exception:
                    skipped_cols.append(col)
        else:
            plt.plot(df.index if index_as_x else df[col_x], df[col_y], label=col_y)
        plt.title(f"{col_y.title()} over {col_x.title()}")
        plt.xlabel(col_x.title())
        plt.ylabel(col_y.title())
        plt.legend()
        fig = plt.gcf()
        if plot_name:
            self._save_plot_and_log(fig, plot_name)
        else:
            self._save_plot_and_log(fig, f"plot_{col_x}_vs_{col_y}")
        return skipped_cols


if __name__ == "__main__":
    # from datetime import datetime
    import os
    from pathlib import Path

    import pandas as pd

    pwd = Path(__file__).parent
    raw_df = pd.read_csv(pwd / "price_data.csv")
    config = {
        "output_dir": pwd / "outputs",
        "output_file": "eda_output.md",  # fixed file name
        "date_col": "Date",
        "price_col": "Price",
        "write_mode": "a",  # always append after cleaning
    }
    # Clean the file at the start of each run
    output_path = os.path.join(config["output_dir"], config["output_file"])
    open(output_path, "w").close()
    eda = HPEdA(raw_df, verbose=True, config=config)
    eda.run_complete_eda()
