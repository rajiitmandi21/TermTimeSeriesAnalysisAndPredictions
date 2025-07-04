import os
import pickle
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class SimpleFutureForecaster:
    def __init__(
        self,
        df,
        date_col,
        target_col,
        output_dir="forecast_outputs",
        clip_outliers=True,
        verbose=True,
    ):
        self.raw_df = df.copy()
        self.df = None
        self.date_col = date_col
        self.target_col = target_col
        self.output_dir = output_dir
        self.image_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        self.models = {}
        self.scaler = StandardScaler()
        self.start_date = pd.to_datetime(self.raw_df[self.date_col]).min()
        self.feature_names = []
        self.clip_outliers = clip_outliers
        self.verbose = verbose
        self.report_file = os.path.join(self.output_dir, "forecasting_report.md")
        self.model_performance = {}
        self.final_selected_features = []
        self.models_dir = os.path.join(self.output_dir, "models")
        self.X_train = None
        self.y_train = None
        self.training_mode = True
        os.makedirs(self.models_dir, exist_ok=True)

        # Initialize report file
        if self.verbose:
            self._init_report_file()

    def _init_report_file(self):
        """Initialize the markdown report file with header."""
        with open(self.report_file, "w") as f:
            f.write("# Hotel Price Forecasting Report\n\n")
            f.write(
                f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            f.write("---\n\n")

    def _write_to_file(self, text):
        """Write text to the markdown report file."""
        with open(self.report_file, "a") as f:
            f.write(str(text))

    def _log_and_write(self, text):
        """Log text to console and write to markdown file."""
        if self.verbose:
            print(text)
            self._write_to_file(text + "\n\n")

    def _save_plot_and_log(self, fig, plot_name):
        """Save a matplotlib figure and log its markdown path."""
        image_path = os.path.join(self.image_dir, f"{plot_name}.png")
        fig.savefig(image_path, bbox_inches="tight", dpi=300)
        plt.close(fig)

        # Create relative path for markdown
        rel_path = os.path.relpath(image_path, os.path.dirname(self.report_file))
        self._write_to_file(f"\n![{plot_name}]({rel_path})\n\n")

    def _df_to_markdown_table(self, df):
        """Convert DataFrame to markdown table format."""
        try:
            import tabulate

            return tabulate.tabulate(df, headers="keys", tablefmt="pipe")
        except ImportError:
            return df.to_string()

    def _clip_outliers_from_trend(self, df, window=21, threshold=3.0):
        """Clip outliers based on deviation from smooth trend."""
        if self.verbose:
            self._log_and_write("## Outlier Detection and Clipping")
            self._log_and_write("- **Method:** Deviation from Smooth Trend")
            self._log_and_write(f"- **Window Size:** {window} days")
            self._log_and_write(f"- **Threshold:** ±{threshold} standard deviations")

        df = df.copy()
        series = df[self.target_col]

        smooth_baseline = series.rolling(window=window, center=True).mean()
        diff = series - smooth_baseline
        rolling_std = diff.rolling(window=window, center=True).std() + 1e-6
        z_score = diff / rolling_std

        outlier_mask = z_score.abs() > threshold
        outlier_count = outlier_mask.sum()

        if self.verbose:
            self._log_and_write(f"- **Total Data Points:** {len(series):,}")
            self._log_and_write(f"- **Outliers Detected:** {outlier_count}")
            self._log_and_write(
                f"- **Outlier Rate:** {(outlier_count / len(series) * 100):.2f}%"
            )

        df_clipped = df.copy()
        df_clipped.loc[outlier_mask, self.target_col] = smooth_baseline[outlier_mask]

        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            pd.to_datetime(df[self.date_col]),
            df[self.target_col],
            label="Original Data",
            color="blue",
            alpha=0.6,
            linewidth=1,
        )
        ax.plot(
            pd.to_datetime(df_clipped[self.date_col]),
            df_clipped[self.target_col],
            label="Clipped Data",
            color="orange",
            linewidth=1.5,
        )
        ax.scatter(
            pd.to_datetime(df[outlier_mask][self.date_col]),
            df[outlier_mask][self.target_col],
            color="red",
            s=20,
            label=f"Outliers ({outlier_count})",
            zorder=5,
        )

        ax.set_title("Outlier Detection and Clipping", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(self.target_col, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = (
            f"Window: {window} days\n"
            f"Threshold: ±{threshold}σ\n"
            f"Outliers: {outlier_count}/{len(series)}\n"
            f"Rate: {(outlier_count / len(series) * 100):.1f}%"
        )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round", facecolor="lightyellow", alpha=0.9, edgecolor="black"
            ),
        )

        plt.tight_layout()

        if self.verbose:
            self._save_plot_and_log(fig, "outlier_clipping_analysis")
        else:
            plt.savefig(os.path.join(self.output_dir, "outlier_clipping_plot.png"))
            plt.show()

        return df_clipped

    def create_time_features(self, df):
        """Create time-based features from the date column."""
        df = df.copy()
        if df.index.dtype != "datetime64[ns]":
            df[self.date_col] = pd.to_datetime(df[self.date_col])

        # Basic time features
        df["year"] = df.index.year
        df["dayofyear"] = df.index.dayofyear

        df["month"] = df.index.month
        df["day_of_month"] = df.index.day

        df["week"] = df.index.isocalendar().week
        df["day_of_week"] = df.index.dayofweek

        df["days_since_start"] = (df.index - self.start_date).days

        # Cyclical features
        df["year_sin"] = np.sin(2 * np.pi * df["year"] / 10)
        df["year_cos"] = np.cos(2 * np.pi * df["year"] / 10)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
        df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
        df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

        if (
            self.verbose and len(self.feature_names) == 0
        ):  # Only log once during training
            self._log_and_write("## Feature Engineering")
            feature_list = [
                col for col in df.columns if col not in [self.date_col, self.target_col]
            ]
            self._log_and_write(f"- **Total Features Created:** {len(feature_list)}")
            self._log_and_write("- **Feature Categories:**")
            self._log_and_write(
                "  - Basic Time: year, month, dayofyear, days_since_start"
            )
            self._log_and_write(
                "  - Cyclical: month_sin/cos, dayofyear_sin/cos, year_sin/cos"
            )
            self._log_and_write(
                f"- **Training Data Date Range:** {df.index.min().date()} to {df.index.max().date()}"
            )

        return df

    def _fill_missing_values(self, df):
        """Fill missing values in the dataframe."""
        if self.verbose:
            self._log_and_write("## Filling Missing Values")
        df = df.copy()

        if self.date_col in df.columns and df.index.dtype != "datetime64[ns]":
            if self.verbose:
                self._log_and_write(f"- **Date column:** {self.date_col} found")
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df.set_index(self.date_col, inplace=True)
        else:
            raise ValueError(f"Date column {self.date_col} not found in dataframe")

        # # index is already datetime then do not interpolate only fill missing values
        # elif df.index.dtype == "datetime64[ns]":
        #     pass

        # create a date range for the entire period
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        # to make sure if there are any missing dates then fill them with NaN
        df = pd.merge(
            pd.DataFrame(date_range, columns=[self.date_col]),
            df,
            on=self.date_col,
            how="left",
        )
        if self.verbose:
            self._log_and_write(
                f"- **Missing dates:** {df[self.target_col].isna().sum()}\n\n Filling missing values with interpolation"
            )
        if df.index.dtype != "datetime64[ns]":
            df.set_index(self.date_col, inplace=True)
        df[self.target_col] = df[self.target_col].interpolate(method="time")

        return df

    def train_models(self, n_iter=20, cv=5):
        """Train ensemble models with hyperparameter tuning and KFold cross validation."""
        if self.verbose:
            self._log_and_write("# Model Training and Evaluation")
            self._log_and_write(
                f"- **Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self._log_and_write(f"- **Hyperparameter Search Iterations:** {n_iter}")
            self._log_and_write(f"- **Cross-Validation Folds:** {cv}")

        # Data preparation
        if self.clip_outliers:
            self.df = self._clip_outliers_from_trend(self.raw_df)
        else:
            self.df = self.raw_df.copy()
            if self.verbose:
                self._log_and_write("## Data Preparation")
                self._log_and_write("- **Outlier Clipping:** Disabled")
        self.df = self._fill_missing_values(self.df)
        df_feat = self.create_time_features(self.df)
        df_feat = self._feature_selection(df_feat)
        X = df_feat.drop(columns=[self.date_col, self.target_col], errors="ignore")
        y = df_feat[self.target_col]

        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)

        # Initialize KFold cross validator
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

        # Store training data for analysis
        self.training_data = {
            "df_features": df_feat.copy(),  # Full feature-engineered dataframe
            "X_raw": X.copy(),  # Raw features before scaling
            "X_scaled": X_scaled.copy(),  # Scaled features
            "y": y.copy(),  # Target variable
            "feature_names": self.feature_names.copy(),
        }

        if self.verbose:
            self._log_and_write("## Dataset Statistics")
            self._log_and_write(f"- **Total Samples:** {len(X):,}")
            self._log_and_write(f"- **Features:** {len(self.feature_names)}")
            self._log_and_write(f"- **Target Variable:** {self.target_col}")
            self._log_and_write(f"- **Target Range:** {y.min():.2f} - {y.max():.2f}")
            self._log_and_write(f"- **Target Mean:** {y.mean():.2f}")
            self._log_and_write(f"- **Target Std:** {y.std():.2f}")
            self._log_and_write("- **Training Data Stored:** Available for analysis")

            # Linear Regression
        if self.verbose:
            self._log_and_write("## Linear Regression Training")
        lr = LinearRegression()

        # Calculate KFold Cross Validation scores BEFORE final training
        lr_cv_scores_rmse = cross_val_score(
            lr, X_scaled, y, cv=kfold, scoring="neg_mean_squared_error"
        )
        lr_cv_rmse_scores = np.sqrt(-lr_cv_scores_rmse)
        lr_cv_r2_scores = cross_val_score(lr, X_scaled, y, cv=kfold, scoring="r2")
        lr_cv_mae_scores = -cross_val_score(
            lr, X_scaled, y, cv=kfold, scoring="neg_mean_absolute_error"
        )

        # Now train the final model on all data
        lr.fit(X_scaled, y)
        self.models["linear"] = lr
        lr_coefs = pd.Series(lr.coef_, index=self.feature_names)

        # Calculate training performance (for comparison only)
        lr_score = lr.score(X_scaled, y)
        lr_y_pred = lr.predict(X_scaled)
        lr_rmse_train = np.sqrt(mean_squared_error(y, lr_y_pred))
        lr_mae_train = mean_absolute_error(y, lr_y_pred)

        # Store CV metrics as primary metrics
        self.model_performance["linear"] = {
            "r2_score": lr_cv_r2_scores.mean(),
            "rmse": lr_cv_rmse_scores.mean(),
            "mae": lr_cv_mae_scores.mean(),
            "cv_rmse_mean": lr_cv_rmse_scores.mean(),
            "cv_rmse_std": lr_cv_rmse_scores.std(),
            "cv_r2_mean": lr_cv_r2_scores.mean(),
            "cv_r2_std": lr_cv_r2_scores.std(),
            "cv_mae_mean": lr_cv_mae_scores.mean(),
            "cv_mae_std": lr_cv_mae_scores.std(),
            "train_r2_score": lr_score,
            "train_rmse": lr_rmse_train,
            "train_mae": lr_mae_train,
        }

        if self.verbose:
            self._log_and_write("### Cross-Validation Performance (Primary Metrics)")
            self._log_and_write(
                f"- **CV RMSE:** {lr_cv_rmse_scores.mean():.4f} (±{lr_cv_rmse_scores.std():.4f})"
            )
            self._log_and_write(
                f"- **CV R² Score:** {lr_cv_r2_scores.mean():.4f} (±{lr_cv_r2_scores.std():.4f})"
            )
            self._log_and_write(
                f"- **CV MAE:** {lr_cv_mae_scores.mean():.4f} (±{lr_cv_mae_scores.std():.4f})"
            )
            self._log_and_write("### Training Performance (Full Dataset)")
            self._log_and_write(f"- **Train R² Score:** {lr_score:.4f}")
            self._log_and_write(f"- **Train RMSE:** {lr_rmse_train:.4f}")
            self._log_and_write(f"- **Train MAE:** {lr_mae_train:.4f}")
            self._log_and_write("### Top 5 Feature Coefficients")
            top_coefs = lr_coefs.abs().sort_values(ascending=False).head(5)
            for feature, coef in top_coefs.items():
                self._log_and_write(f"  - {feature}: {lr_coefs[feature]:.4f}")

        # Random Forest
        if self.verbose:
            self._log_and_write("## Random Forest Training")
        rf_params = {
            "n_estimators": [100, 200],
            "max_depth": [5, 7, 10, None],
        }
        rf_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            n_iter=n_iter,
            cv=kfold,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rf_search.fit(X_scaled, y)
        self.models["rf"] = rf_search.best_estimator_
        rf_importances = pd.Series(
            self.models["rf"].feature_importances_, index=self.feature_names
        )

        # Calculate KFold Cross Validation scores AFTER training
        rf_cv_scores = cross_val_score(
            self.models["rf"], X_scaled, y, cv=kfold, scoring="neg_mean_squared_error"
        )
        rf_cv_rmse_scores = np.sqrt(-rf_cv_scores)
        rf_cv_r2_scores = cross_val_score(
            self.models["rf"], X_scaled, y, cv=kfold, scoring="r2"
        )
        rf_cv_mae_scores = -cross_val_score(
            self.models["rf"], X_scaled, y, cv=kfold, scoring="neg_mean_absolute_error"
        )

        # Calculate training performance (for comparison only)
        rf_y_pred = self.models["rf"].predict(X_scaled)
        rf_r2_train = r2_score(y, rf_y_pred)
        rf_rmse_train = np.sqrt(mean_squared_error(y, rf_y_pred))
        rf_mae_train = mean_absolute_error(y, rf_y_pred)

        # Store CV metrics as primary metrics
        self.model_performance["rf"] = {
            "best_params": rf_search.best_params_,
            "r2_score": rf_cv_r2_scores.mean(),
            "rmse": rf_cv_rmse_scores.mean(),
            "mae": rf_cv_mae_scores.mean(),
            "cv_rmse_mean": rf_cv_rmse_scores.mean(),
            "cv_rmse_std": rf_cv_rmse_scores.std(),
            "cv_r2_mean": rf_cv_r2_scores.mean(),
            "cv_r2_std": rf_cv_r2_scores.std(),
            "cv_mae_mean": rf_cv_mae_scores.mean(),
            "cv_mae_std": rf_cv_mae_scores.std(),
            "train_r2_score": rf_r2_train,
            "train_rmse": rf_rmse_train,
            "train_mae": rf_mae_train,
        }

        if self.verbose:
            self._log_and_write("### Cross-Validation Performance (Primary Metrics)")
            self._log_and_write(
                f"- **CV RMSE:** {rf_cv_rmse_scores.mean():.4f} (±{rf_cv_rmse_scores.std():.4f})"
            )
            self._log_and_write(
                f"- **CV R² Score:** {rf_cv_r2_scores.mean():.4f} (±{rf_cv_r2_scores.std():.4f})"
            )
            self._log_and_write(
                f"- **CV MAE:** {rf_cv_mae_scores.mean():.4f} (±{rf_cv_mae_scores.std():.4f})"
            )
            self._log_and_write("### Training Performance (Full Dataset)")
            self._log_and_write(f"- **Train RMSE:** {rf_rmse_train:.4f}")
            self._log_and_write(f"- **Train MAE:** {rf_mae_train:.4f}")
            self._log_and_write(f"- **Train R² Score:** {rf_r2_train:.4f}")
            self._log_and_write("### Hyperparameter Tuning Results")
            self._log_and_write(f"- **Best Parameters:** {rf_search.best_params_}")
            self._log_and_write("### Top 5 Feature Importances")
            top_importances = rf_importances.sort_values(ascending=False).head(5)
            for feature, importance in top_importances.items():
                self._log_and_write(f"  - {feature}: {importance:.4f}")

                # XGBoost
        if self.verbose:
            self._log_and_write("## XGBoost Training")
        xgb_params = {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        }
        xgb_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42),
            xgb_params,
            n_iter=n_iter,
            cv=kfold,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        xgb_search.fit(X_scaled, y)
        self.models["xgb"] = xgb_search.best_estimator_
        xgb_importances = pd.Series(
            self.models["xgb"].feature_importances_, index=self.feature_names
        )

        # Calculate KFold Cross Validation scores AFTER training
        xgb_cv_scores = cross_val_score(
            self.models["xgb"], X_scaled, y, cv=kfold, scoring="neg_mean_squared_error"
        )
        xgb_cv_rmse_scores = np.sqrt(-xgb_cv_scores)
        xgb_cv_r2_scores = cross_val_score(
            self.models["xgb"], X_scaled, y, cv=kfold, scoring="r2"
        )
        xgb_cv_mae_scores = -cross_val_score(
            self.models["xgb"], X_scaled, y, cv=kfold, scoring="neg_mean_absolute_error"
        )

        # Calculate training performance (for comparison only)
        xgb_y_pred = self.models["xgb"].predict(X_scaled)
        xgb_r2_train = r2_score(y, xgb_y_pred)
        xgb_rmse_train = np.sqrt(mean_squared_error(y, xgb_y_pred))
        xgb_mae_train = mean_absolute_error(y, xgb_y_pred)

        # Store CV metrics as primary metrics
        self.model_performance["xgb"] = {
            "best_params": xgb_search.best_params_,
            "r2_score": xgb_cv_r2_scores.mean(),
            "rmse": xgb_cv_rmse_scores.mean(),
            "mae": xgb_cv_mae_scores.mean(),
            "cv_rmse_mean": xgb_cv_rmse_scores.mean(),
            "cv_rmse_std": xgb_cv_rmse_scores.std(),
            "cv_r2_mean": xgb_cv_r2_scores.mean(),
            "cv_r2_std": xgb_cv_r2_scores.std(),
            "cv_mae_mean": xgb_cv_mae_scores.mean(),
            "cv_mae_std": xgb_cv_mae_scores.std(),
            "train_r2_score": xgb_r2_train,
            "train_rmse": xgb_rmse_train,
            "train_mae": xgb_mae_train,
        }

        if self.verbose:
            self._log_and_write("### Cross-Validation Performance (Primary Metrics)")
            self._log_and_write(
                f"- **CV RMSE:** {xgb_cv_rmse_scores.mean():.4f} (±{xgb_cv_rmse_scores.std():.4f})"
            )
            self._log_and_write(
                f"- **CV R² Score:** {xgb_cv_r2_scores.mean():.4f} (±{xgb_cv_r2_scores.std():.4f})"
            )
            self._log_and_write(
                f"- **CV MAE:** {xgb_cv_mae_scores.mean():.4f} (±{xgb_cv_mae_scores.std():.4f})"
            )
            self._log_and_write("### Training Performance (Full Dataset)")
            self._log_and_write(f"- **Train RMSE:** {xgb_rmse_train:.4f}")
            self._log_and_write(f"- **Train MAE:** {xgb_mae_train:.4f}")
            self._log_and_write(f"- **Train R² Score:** {xgb_r2_train:.4f}")
            self._log_and_write("### Hyperparameter Tuning Results")
            self._log_and_write(f"- **Best Parameters:** {xgb_search.best_params_}")
            self._log_and_write("### Top 5 Feature Importances")
            top_importances = xgb_importances.sort_values(ascending=False).head(5)
            for feature, importance in top_importances.items():
                self._log_and_write(f"  - {feature}: {importance:.4f}")

        # Create feature importance comparison plot
        if self.verbose:
            self._create_feature_importance_plot(
                lr_coefs, rf_importances, xgb_importances
            )
            self._create_cv_performance_comparison_plot()
            self._log_and_write("## Training Summary")
            self._log_and_write(
                f"- **Models Successfully Trained:** {len(self.models)}"
            )
            self._log_and_write(
                "- **Model Performance Comparison (Cross-Validation):**"
            )
            if "linear" in self.model_performance:
                self._log_and_write(
                    f"  - Linear Regression CV RMSE: {self.model_performance['linear']['cv_rmse_mean']:.4f} (±{self.model_performance['linear']['cv_rmse_std']:.4f})"
                )
                self._log_and_write(
                    f"  - Linear Regression CV R²: {self.model_performance['linear']['cv_r2_mean']:.4f} (±{self.model_performance['linear']['cv_r2_std']:.4f})"
                )
            if "rf" in self.model_performance:
                self._log_and_write(
                    f"  - Random Forest CV RMSE: {self.model_performance['rf']['cv_rmse_mean']:.4f} (±{self.model_performance['rf']['cv_rmse_std']:.4f})"
                )
                self._log_and_write(
                    f"  - Random Forest CV R²: {self.model_performance['rf']['cv_r2_mean']:.4f} (±{self.model_performance['rf']['cv_r2_std']:.4f})"
                )
            if "xgb" in self.model_performance:
                self._log_and_write(
                    f"  - XGBoost CV RMSE: {self.model_performance['xgb']['cv_rmse_mean']:.4f} (±{self.model_performance['xgb']['cv_rmse_std']:.4f})"
                )
                self._log_and_write(
                    f"  - XGBoost CV R²: {self.model_performance['xgb']['cv_r2_mean']:.4f} (±{self.model_performance['xgb']['cv_r2_std']:.4f})"
                )
        self.X_train = X_scaled
        self.y_train = y

    def _create_feature_importance_plot(
        self, lr_coefs, rf_importances, xgb_importances
    ):
        """Create a comparison plot of feature importances across models."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Linear Regression Coefficients
        top_lr = lr_coefs.abs().sort_values(ascending=False).head(10)
        axes[0, 0].barh(range(len(top_lr)), top_lr.values)
        axes[0, 0].set_yticks(range(len(top_lr)))
        axes[0, 0].set_yticklabels(top_lr.index)
        axes[0, 0].set_title(
            "Linear Regression - Top 10 Feature Coefficients (Absolute)",
            fontweight="bold",
        )
        axes[0, 0].set_xlabel("Absolute Coefficient Value")

        # Random Forest Importances
        top_rf = rf_importances.sort_values(ascending=False).head(10)
        axes[0, 1].barh(range(len(top_rf)), top_rf.values)
        axes[0, 1].set_yticks(range(len(top_rf)))
        axes[0, 1].set_yticklabels(top_rf.index)
        axes[0, 1].set_title(
            "Random Forest - Top 10 Feature Importances", fontweight="bold"
        )
        axes[0, 1].set_xlabel("Importance Score")

        # XGBoost Importances
        top_xgb = xgb_importances.sort_values(ascending=False).head(10)
        axes[1, 0].barh(range(len(top_xgb)), top_xgb.values)
        axes[1, 0].set_yticks(range(len(top_xgb)))
        axes[1, 0].set_yticklabels(top_xgb.index)
        axes[1, 0].set_title("XGBoost - Top 10 Feature Importances", fontweight="bold")
        axes[1, 0].set_xlabel("Importance Score")

        # Combined comparison for top 5 features
        top_features = set(
            list(top_lr.head(5).index)
            + list(top_rf.head(5).index)
            + list(top_xgb.head(5).index)
        )

        comparison_data = pd.DataFrame(
            {
                "Linear_Reg": [lr_coefs.abs().get(feat, 0) for feat in top_features],
                "Random_Forest": [rf_importances.get(feat, 0) for feat in top_features],
                "XGBoost": [xgb_importances.get(feat, 0) for feat in top_features],
            },
            index=list(top_features),
        )

        comparison_data.plot(kind="bar", ax=axes[1, 1])
        axes[1, 1].set_title(
            "Feature Importance Comparison (Top Features)", fontweight="bold"
        )
        axes[1, 1].set_xlabel("Features")
        axes[1, 1].set_ylabel("Importance Score")
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        self._save_plot_and_log(fig, "feature_importance_comparison")

    def _create_cv_performance_comparison_plot(self):
        """Create a comprehensive cross-validation performance comparison plot."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        models = ["Linear Regression", "Random Forest", "XGBoost"]
        model_keys = ["linear", "rf", "xgb"]
        colors = ["#2E86AB", "#A23B72", "#F18F01"]

        # Extract CV performance data
        cv_rmse_means = [
            self.model_performance[key]["cv_rmse_mean"] for key in model_keys
        ]
        cv_rmse_stds = [
            self.model_performance[key]["cv_rmse_std"] for key in model_keys
        ]
        cv_r2_means = [self.model_performance[key]["cv_r2_mean"] for key in model_keys]
        cv_r2_stds = [self.model_performance[key]["cv_r2_std"] for key in model_keys]
        cv_mae_means = [
            self.model_performance[key]["cv_mae_mean"] for key in model_keys
        ]
        cv_mae_stds = [self.model_performance[key]["cv_mae_std"] for key in model_keys]

        # RMSE comparison
        axes[0].bar(
            models, cv_rmse_means, yerr=cv_rmse_stds, capsize=5, color=colors, alpha=0.7
        )
        axes[0].set_title(
            "Cross-Validation RMSE Comparison", fontweight="bold", fontsize=14
        )
        axes[0].set_ylabel("RMSE")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(cv_rmse_means, cv_rmse_stds)):
            axes[0].text(
                i,
                mean + std + 0.5,
                f"{mean:.3f}\n±{std:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # R² Score comparison
        axes[1].bar(
            models, cv_r2_means, yerr=cv_r2_stds, capsize=5, color=colors, alpha=0.7
        )
        axes[1].set_title(
            "Cross-Validation R² Score Comparison", fontweight="bold", fontsize=14
        )
        axes[1].set_ylabel("R² Score")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(cv_r2_means, cv_r2_stds)):
            axes[1].text(
                i,
                mean + std + 0.01,
                f"{mean:.3f}\n±{std:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # MAE comparison
        axes[2].bar(
            models, cv_mae_means, yerr=cv_mae_stds, capsize=5, color=colors, alpha=0.7
        )
        axes[2].set_title(
            "Cross-Validation MAE Comparison", fontweight="bold", fontsize=14
        )
        axes[2].set_ylabel("MAE")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(cv_mae_means, cv_mae_stds)):
            axes[2].text(
                i,
                mean + std + 0.3,
                f"{mean:.3f}\n±{std:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        self._save_plot_and_log(fig, "cv_performance_comparison")

    def save_models(self, save_performance=True):
        """Save trained models, scaler, and metadata to files."""
        if not self.models:
            raise ValueError("No models to save. Please train models first.")

        if self.verbose:
            self._log_and_write("## Saving Trained Models")

        # Save individual models
        for model_name, model in self.models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}_model.joblib")
            joblib.dump(model, model_path)
            if self.verbose:
                self._log_and_write(f"- **{model_name.upper()} Model:** {model_path}")

        # Save scaler
        model_dump_path = os.path.join(self.models_dir, "model_dump.joblib")
        joblib.dump(self, model_dump_path)
        print("model_dump saved successfully at ", model_dump_path)
        if self.verbose:
            self._log_and_write(f"- **Model Dump:** {model_dump_path}")

        # Save feature names and metadata
        metadata = {
            "feature_names": self.feature_names,
            "final_selected_features": self.final_selected_features,
            "date_col": self.date_col,
            "target_col": self.target_col,
            "start_date": self.start_date,
            "clip_outliers": self.clip_outliers,
        }

        if save_performance:
            metadata["model_performance"] = self.model_performance

        metadata_path = os.path.join(self.models_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        if self.verbose:
            self._log_and_write(f"- **Metadata:** {metadata_path}")

        if self.verbose:
            self._log_and_write(f"- **Total Models Saved:** {len(self.models)}")
            self._log_and_write(f"- **Models Directory:** {self.models_dir}")

        print(f"All models saved successfully to: {self.models_dir}")
        return self.models_dir

    @classmethod
    def load_models(cls, models_dir, verbose=True):
        """Class method to load trained models and create a new forecaster instance.

        Args:
            models_dir (str): Directory containing saved models
            verbose (bool): Whether to enable verbose logging

        Returns:
            SimpleFutureForecaster: New instance with loaded models
        """
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

        model_dump_path = os.path.join(models_dir, "model_dump.joblib")
        if not os.path.exists(model_dump_path):
            raise FileNotFoundError(f"Model dump file not found: {model_dump_path}")

        forecaster = joblib.load(model_dump_path)
        if verbose:
            print(f"Models loaded successfully from: {models_dir}")
            forecaster.verbose = verbose
            forecaster.training_mode = False

        return forecaster

    def _load_models_from_directory(self, models_dir, metadata):
        """Internal method to load models into existing instance."""
        if self.verbose:
            self._log_and_write("## Loading Trained Models")

        # Restore metadata
        self.feature_names = metadata["feature_names"]
        self.final_selected_features = metadata["final_selected_features"]
        self.date_col = metadata["date_col"]
        self.target_col = metadata["target_col"]
        self.start_date = metadata["start_date"]
        self.clip_outliers = metadata["clip_outliers"]

        if "model_performance" in metadata:
            self.model_performance = metadata["model_performance"]

        # Load scaler
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        if self.verbose:
            self._log_and_write(f"- **Scaler loaded:** {scaler_path}")

        # Load models
        self.models = {}
        model_files = ["linear_model.joblib", "rf_model.joblib", "xgb_model.joblib"]

        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                model_name = model_file.replace("_model.joblib", "")
                self.models[model_name] = joblib.load(model_path)
                if self.verbose:
                    self._log_and_write(
                        f"- **{model_name.upper()} Model loaded:** {model_path}"
                    )

        if self.verbose:
            self._log_and_write(f"- **Total Models Loaded:** {len(self.models)}")
            self._log_and_write(f"- **Feature Names:** {len(self.feature_names)}")
            self._log_and_write(
                f"- **Selected Features:** {len(self.final_selected_features)}"
            )

    @classmethod
    def load_from_saved(cls, models_dir, verbose=True):
        """Create a new SimpleFutureForecaster instance from saved models.

        This is an alias for load_models() for backward compatibility.
        """
        return cls.load_models(models_dir, verbose)

    def predict_from_saved(self, start_date, end_date, models_weights=None):
        """Generate predictions using loaded models."""
        if not self.models:
            raise ValueError(
                "No models loaded. Please load models first using load_models()."
            )

        if self.verbose:
            self._log_and_write("# Prediction from Saved Models")
            self._log_and_write(f"- **Prediction Period:** {start_date} to {end_date}")
            self._log_and_write(f"- **Models Available:** {list(self.models.keys())}")

        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        future_df = pd.DataFrame(index=date_range)

        # Create features
        future_feat = self.create_time_features(future_df)

        # Select features (use final_selected_features if available, otherwise use feature_names)
        if self.final_selected_features:
            selected_features = self.final_selected_features
        else:
            selected_features = self.feature_names

        X_future = future_feat[selected_features]
        X_future_scaled = self.scaler.transform(X_future)

        if self.verbose:
            self._log_and_write(f"- **Prediction Points:** {len(date_range):,}")
            self._log_and_write(f"- **Features Used:** {len(selected_features)}")

        # Generate predictions
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X_future_scaled)

        # Configure ensemble weights
        if models_weights is None:
            num_models = len(self.models)
            models_weights = {name: 1 / num_models for name in self.models.keys()}

        if self.verbose:
            self._log_and_write("- **Ensemble Weights:**")
            for model_name, weight in models_weights.items():
                self._log_and_write(f"  - {model_name}: {weight:.2f}")

        # Create ensemble prediction
        ensemble = np.sum(
            [preds[name] * models_weights.get(name, 0) for name in preds.keys()], axis=0
        )
        future_df["Forecast"] = ensemble

        # Individual model predictions for analysis
        for name, pred in preds.items():
            future_df[f"{name}_prediction"] = pred

        if self.verbose:
            self._log_and_write("## Prediction Statistics")
            self._log_and_write(
                f"- **Forecast Range:** {ensemble.min():.2f} - {ensemble.max():.2f}"
            )
            self._log_and_write(f"- **Forecast Mean:** {ensemble.mean():.2f}")
            self._log_and_write(f"- **Forecast Std:** {ensemble.std():.2f}")

            # Show sample predictions
            future_df_copy = future_df.copy()
            future_df_copy.reset_index(inplace=True)
            future_df_copy.rename(columns={"index": self.date_col}, inplace=True)
            sample_preds = future_df_copy.head(10)[[self.date_col, "Forecast"]].copy()
            self._log_and_write("## Sample Predictions")
            self._log_and_write(self._df_to_markdown_table(sample_preds))

        return future_df

    def predict(self, start_date, end_date, models_weights=None):
        """Generate predictions for the specified date range."""
        if self.verbose:
            self._log_and_write("# Prediction Generation")
            self._log_and_write(f"- **Prediction Period:** {start_date} to {end_date}")

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        future_df = pd.DataFrame(index=date_range)
        future_feat = self.create_time_features(future_df)
        X_future = future_feat[self.final_selected_features]
        X_future_scaled = self.scaler.transform(X_future)

        if self.verbose:
            self._log_and_write(f"- **Prediction Points:** {len(date_range):,}")
            self._log_and_write(f"- **Features Used:** {len(self.feature_names)}")

        preds = {
            name: model.predict(X_future_scaled) for name, model in self.models.items()
        }

        # Configure ensemble weights
        if models_weights is None:
            models_weights = {"linear": 1 / 3, "rf": 1 / 3, "xgb": 1 / 3}

        if self.verbose:
            self._log_and_write("- **Ensemble Weights:**")
            for model_name, weight in models_weights.items():
                self._log_and_write(f"  - {model_name}: {weight:.2f}")

        ensemble = np.sum(
            [preds[name] * models_weights[name] for name in preds.keys()], axis=0
        )
        future_df["Forecast"] = ensemble

        # Individual model predictions for analysis
        for name, pred in preds.items():
            future_df[f"{name}_prediction"] = pred

        if self.verbose:
            self._log_and_write("## Prediction Statistics")
            self._log_and_write(
                f"- **Forecast Range:** {ensemble.min():.2f} - {ensemble.max():.2f}"
            )
            self._log_and_write(f"- **Forecast Mean:** {ensemble.mean():.2f}")
            self._log_and_write(f"- **Forecast Std:** {ensemble.std():.2f}")

            # Show sample predictions
            future_df.reset_index(inplace=True)
            future_df.rename(columns={"index": self.date_col}, inplace=True)
            sample_preds = future_df.head(10)[[self.date_col, "Forecast"]].copy()
            self._log_and_write("## Sample Predictions")
            self._log_and_write(self._df_to_markdown_table(sample_preds))

        return future_df

    def plot_forecast(self, forecast_df, text_note=None, plot_name=None):
        """Create and save forecast visualization."""
        fig, ax = plt.subplots(figsize=(15, 8))
        forecast_df = forecast_df.copy()

        if self.training_mode:
            # Plot historical data
            hist_dates = pd.to_datetime(self.df.index)
            hist_values = self.df[self.target_col]
            ax.plot(
                hist_dates,
                hist_values,
                label="Historical Data",
                color="blue",
                alpha=0.7,
                linewidth=1,
            )

        # Plot forecast
        if forecast_df.index.dtype != "datetime64[ns]":
            forecast_df.set_index(self.date_col, inplace=True)
        forecast_dates = pd.to_datetime(forecast_df.index)
        forecast_values = forecast_df["Forecast"]
        ax.plot(
            forecast_dates,
            forecast_values,
            label="Ensemble Forecast",
            color="red",
            linewidth=2,
        )

        # Plot individual model predictions if available
        if self.training_mode:
            colors = {"linear": "green", "rf": "orange", "xgb": "purple"}
            for model_name in ["linear", "rf", "xgb"]:
                pred_col = f"{model_name}_prediction"
                if pred_col in forecast_df.columns:
                    ax.plot(
                        forecast_dates,
                        forecast_df[pred_col],
                        label=f"{model_name.upper()} Forecast",
                        color=colors.get(model_name, "gray"),
                        alpha=0.6,
                        linestyle="--",
                    )

            # Formatting
            ax.set_title(
                "Historical Data and Forecast Comparison",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel(f"{self.target_col} Value", fontsize=12)
            ax.legend(loc="upper left", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add vertical line to separate historical and forecast
            if len(hist_dates) > 0 and len(forecast_dates) > 0:
                separation_date = max(hist_dates)
                ax.axvline(
                    x=separation_date,
                    color="gray",
                    linestyle=":",
                    alpha=0.7,
                    label="Forecast Start",
                )

        # Add statistics text box
        if text_note:
            props = dict(
                boxstyle="round", facecolor="lightcyan", alpha=0.9, edgecolor="navy"
            )
            ax.text(
                0.02,
                0.98,
                text_note,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

        plt.tight_layout()

        # Save plot
        if self.verbose:
            if plot_name is None:
                plot_name = f"forecast_plot_{forecast_dates.min().strftime('%Y%m%d')}_{forecast_dates.max().strftime('%Y%m%d')}"
            self._save_plot_and_log(fig, plot_name)
        else:
            plt.savefig(os.path.join(self.output_dir, "forecast_plot.png"))
            plt.show()

        return fig

    def get_training_data(self):
        """Get the stored training data after feature engineering."""
        if not hasattr(self, "training_data"):
            raise ValueError(
                "Training data not available. Please run train_models() first."
            )
        return self.training_data

    def get_training_features(self, scaled=True):
        """Get the training features (X).

        Args:
            scaled (bool): If True, returns scaled features. If False, returns raw features.
        """
        if not hasattr(self, "training_data"):
            raise ValueError(
                "Training data not available. Please run train_models() first."
            )
        return self.training_data["X_scaled"] if scaled else self.training_data["X_raw"]

    def get_training_target(self):
        """Get the training target variable (y)."""
        if not hasattr(self, "training_data"):
            raise ValueError(
                "Training data not available. Please run train_models() first."
            )
        return self.training_data["y"]

    def get_feature_engineered_dataframe(self):
        """Get the complete feature-engineered dataframe."""
        if not hasattr(self, "training_data"):
            raise ValueError(
                "Training data not available. Please run train_models() first."
            )
        return self.training_data["df_features"]

    def export_training_data(self, prefix="training_data"):
        """Export training data to CSV files.

        Args:
            prefix (str): Prefix for the exported CSV filenames.
        """
        if not hasattr(self, "training_data"):
            raise ValueError(
                "Training data not available. Please run train_models() first."
            )

        # Export feature-engineered dataframe
        features_path = os.path.join(self.output_dir, f"{prefix}_features.csv")
        self.training_data["df_features"].to_csv(features_path, index=False)

        # Export raw features with target
        raw_features_with_target = self.training_data["X_raw"].copy()
        raw_features_with_target[self.target_col] = self.training_data["y"]
        raw_path = os.path.join(self.output_dir, f"{prefix}_raw_features.csv")
        raw_features_with_target.to_csv(raw_path, index=False)

        if self.verbose:
            self._log_and_write("## Training Data Export")
            self._log_and_write(
                f"- **Feature-engineered data saved to:** {features_path}"
            )
            self._log_and_write(f"- **Raw features with target saved to:** {raw_path}")

        print("Training data exported:")
        print(f"- Feature-engineered data: {features_path}")
        print(f"- Raw features with target: {raw_path}")

    def analyze_training_data(self):
        """Analyze and log training data statistics."""
        if not hasattr(self, "training_data"):
            raise ValueError(
                "Training data not available. Please run train_models() first."
            )

        if self.verbose:
            self._log_and_write("# Training Data Analysis")
            # self._feature_selection(self.training_data["df_features"])
            # Feature correlation analysis
            feature_corr = self.training_data["df_features"].corr()

            target_corr = (
                feature_corr[self.target_col].abs().sort_values(ascending=False)
            )

            self._log_and_write("## Feature Correlation with Target")
            self._log_and_write("- **Top 10 Most Correlated Features:**")
            for feature, corr in target_corr.head(11).items():
                if (
                    feature != self.target_col
                    and feature in self.training_data["X_raw"].columns
                ):
                    self._log_and_write(f"  - {feature}: {corr:.4f}")

            # Feature statistics
            self._log_and_write("## Feature Statistics")
            feature_stats = self.training_data["X_raw"].describe()
            self._log_and_write(self._df_to_markdown_table(feature_stats))

            # Missing values check
            missing_values = self.training_data["df_features"].isnull().sum()
            if missing_values.sum() > 0:
                self._log_and_write("## Missing Values in Training Data")
                self._log_and_write(
                    self._df_to_markdown_table(missing_values[missing_values > 0])
                )
            else:
                self._log_and_write(
                    "## Data Quality: No missing values in training data"
                )

            # Feature distribution plots
            self._create_feature_distribution_plots()

    def _feature_selection(self, df_feat, top_k=10, threshold=0.6):
        """
        Select top K features based on correlation with target.
        If two features are highly correlated, remove the one with lower correlation to the target.
        Also logs reasons for removal and their correlated counterparts.
        """
        df_feat = df_feat.copy()
        corr_matrix = df_feat.corr()
        target_corr = (
            corr_matrix[self.target_col]
            .drop(self.target_col)
            .abs()
            .sort_values(ascending=False)
        )
        selected = list(target_corr.index)

        self._plot_heatmap(
            corr_matrix,
            "feature_correlation_heatmap_before_selection",
        )

        to_remove = {}
        # kept = set()

        for i in range(len(selected)):
            f1 = selected[i]
            if f1 in to_remove:
                continue
            for j in range(i + 1, len(selected)):
                f2 = selected[j]
                # if f2 in to_remove:
                #     continue
                corr_val = abs(corr_matrix.loc[f1, f2])
                if corr_val > threshold:
                    weaker = f2 if target_corr[f1] >= target_corr[f2] else f1
                    stronger = f1 if weaker == f2 else f2
                    # if any of the two is already there then add other one as reason val as well
                    if weaker in to_remove:
                        to_remove[weaker].append((stronger, corr_val))
                    elif stronger in to_remove:
                        to_remove[stronger].append((weaker, corr_val))
                    else:
                        to_remove[weaker] = [(stronger, corr_val)]

        # Logging which feature is removed and why
        for feat, related_feats in to_remove.items():
            reasons = ", ".join(
                [f"`{f2}` (corr={val:.2f})" for f2, val in related_feats]
            )
            self._log_and_write(
                f"- **Feature `{feat}`** removed due to high correlation with: {reasons}"
            )

        self._log_and_write(
            f"- **Total features removed:** {len(to_remove)}"
            if to_remove
            else "- **No features removed**"
        )

        final_candidates = [f for f in selected if f not in to_remove]
        final_features = final_candidates[:top_k]

        self._plot_heatmap(
            df_feat[final_features].corr(),
            "feature_correlation_heatmap_after_selection",
        )
        self.final_selected_features = final_features
        return df_feat[final_features + [self.target_col]]

    def _plot_heatmap(self, feature_corr, plot_name="feature_correlation_heatmap"):
        """Plot the correlation heatmap."""
        plt.figure(figsize=(10, 8))
        # make the labels smaller and slant at 45 degrees
        plt.rcParams["font.size"] = 8
        plt.rcParams["font.style"] = "italic"
        plt.rcParams["xtick.labelsize"] = 8
        plt.rcParams["ytick.labelsize"] = 8
        # plt.rcParams["xtick.rotation"] = 45
        # plt.rcParams["ytick.rotation"] = 45
        sns.heatmap(
            feature_corr,
            annot=True,
            cmap="coolwarm",
            # center=0,
            # mask=np.triu(np.ones_like(feature_corr)),
        )
        plt.title(f"{plot_name.replace('_', ' ').title()}")
        plt.savefig(os.path.join(self.image_dir, f"{plot_name}.png"))

        plt.tight_layout()
        if self.verbose:
            self._save_plot_and_log(plt.gcf(), plot_name)
            plt.show()
        plt.close()

    def _create_feature_distribution_plots(self):
        """Create distribution plots for key features."""
        if not hasattr(self, "training_data"):
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Select top 6 features for visualization
        feature_corr = self.training_data["df_features"].corr()
        target_corr = feature_corr[self.target_col].abs().sort_values(ascending=False)

        # Only include features that are in X_raw (exclude Date and target)
        available_features = [
            f
            for f in target_corr.index
            if f != self.target_col and f in self.training_data["X_raw"].columns
        ]
        top_features = available_features[:6]

        for i, feature in enumerate(top_features):
            if i < 6:
                axes[i].hist(
                    self.training_data["X_raw"][feature],
                    bins=30,
                    alpha=0.7,
                    edgecolor="black",
                )
                axes[i].set_title(f"{feature} Distribution", fontweight="bold")
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(top_features), 6):
            axes[i].set_visible(False)

        plt.tight_layout()
        if self.verbose:
            self._save_plot_and_log(fig, "feature_distributions")

    def generate_comprehensive_report(self, forecast_results):
        """Generate a comprehensive summary report of the entire analysis."""
        if not self.verbose:
            return

        self._log_and_write("# Comprehensive Analysis Summary")
        self._log_and_write("---")

        # Dataset Overview
        self._log_and_write("## Dataset Overview")
        self._log_and_write(f"- **Data Source:** {self.target_col} time series data")
        self._log_and_write(
            f"- **Date Range:** {self.raw_df.index.min()} to {self.raw_df.index.max()}"
        )
        self._log_and_write(f"- **Total Records:** {len(self.raw_df):,}")
        self._log_and_write(
            f"- **Data Quality:** {(1 - self.raw_df[self.target_col].isnull().sum() / len(self.raw_df)) * 100:.1f}% complete"
        )

        # Data Processing Summary
        self._log_and_write("## Data Processing Summary")
        self._log_and_write(
            f"- **Outlier Clipping:** {'Enabled' if self.clip_outliers else 'Disabled'}"
        )
        self._log_and_write(
            f"- **Feature Engineering:** {len(self.feature_names)} features created"
        )
        self._log_and_write("- **Data Scaling:** StandardScaler applied")

        # Training data storage info
        if hasattr(self, "training_data"):
            self._log_and_write(
                f"- **Training Data Stored:** {len(self.training_data['X_raw'])} samples with {len(self.training_data['feature_names'])} features"
            )
            self._log_and_write(
                "- **Available Components:** Raw features, scaled features, target variable, feature-engineered dataframe"
            )

        # Model Performance Summary
        self._log_and_write("## Model Performance Summary")
        if self.model_performance:
            if "linear" in self.model_performance:
                self._log_and_write(
                    f"- **Linear Regression Training R² Score:** {self.model_performance['linear']['r2_score']:.4f}"
                )
                self._log_and_write(
                    f"- **Linear Regression Training RMSE:** {self.model_performance['linear']['rmse']:.4f}"
                )
                self._log_and_write(
                    f"- **Linear Regression CV RMSE:** {self.model_performance['linear']['cv_rmse_mean']:.4f} (±{self.model_performance['linear']['cv_rmse_std']:.4f})"
                )
                self._log_and_write(
                    f"- **Linear Regression CV R²:** {self.model_performance['linear']['cv_r2_mean']:.4f} (±{self.model_performance['linear']['cv_r2_std']:.4f})"
                )
            if "rf" in self.model_performance:
                self._log_and_write(
                    f"- **Random Forest Training R² Score:** {self.model_performance['rf']['r2_score']:.4f}"
                )
                self._log_and_write(
                    f"- **Random Forest Training RMSE:** {self.model_performance['rf']['rmse']:.4f}"
                )
                self._log_and_write(
                    f"- **Random Forest CV RMSE:** {self.model_performance['rf']['cv_rmse_mean']:.4f} (±{self.model_performance['rf']['cv_rmse_std']:.4f})"
                )
                self._log_and_write(
                    f"- **Random Forest CV R²:** {self.model_performance['rf']['cv_r2_mean']:.4f} (±{self.model_performance['rf']['cv_r2_std']:.4f})"
                )
            if "xgb" in self.model_performance:
                self._log_and_write(
                    f"- **XGBoost Training R² Score:** {self.model_performance['xgb']['r2_score']:.4f}"
                )
                self._log_and_write(
                    f"- **XGBoost Training RMSE:** {self.model_performance['xgb']['rmse']:.4f}"
                )
                self._log_and_write(
                    f"- **XGBoost CV RMSE:** {self.model_performance['xgb']['cv_rmse_mean']:.4f} (±{self.model_performance['xgb']['cv_rmse_std']:.4f})"
                )
                self._log_and_write(
                    f"- **XGBoost CV R²:** {self.model_performance['xgb']['cv_r2_mean']:.4f} (±{self.model_performance['xgb']['cv_r2_std']:.4f})"
                )

        # Forecast Summary
        if forecast_results:
            self._log_and_write("## Forecast Summary")
            for i, (description, forecast_df) in enumerate(forecast_results):
                self._log_and_write(f"### {description}")
                self._log_and_write(
                    f"- **Forecast Period:** {forecast_df.index.min()} to {forecast_df.index.max()}"
                )
                self._log_and_write(f"- **Forecast Points:** {len(forecast_df):,}")
                self._log_and_write(
                    f"- **Value Range:** {forecast_df['Forecast'].min():.2f} - {forecast_df['Forecast'].max():.2f}"
                )
                self._log_and_write(
                    f"- **Average Value:** {forecast_df['Forecast'].mean():.2f}"
                )

        # Technical Details
        self._log_and_write("## Technical Details")
        self._log_and_write(
            "- **Model Types:** Linear Regression, Random Forest, XGBoost"
        )
        self._log_and_write(
            "- **Ensemble Method:** Weighted average of individual predictions"
        )
        self._log_and_write(
            "- **Hyperparameter Tuning:** RandomizedSearchCV with cross-validation"
        )
        self._log_and_write(
            "- **Feature Types:** Time-based features with cyclical encoding"
        )

        # Recommendations
        self._log_and_write("## Key Insights and Recommendations")

        # Performance insights
        if self.model_performance:
            if (
                "rf" in self.model_performance
                and "xgb" in self.model_performance
                and "linear" in self.model_performance
            ):
                rf_cv_rmse = self.model_performance["rf"]["cv_rmse_mean"]
                xgb_cv_rmse = self.model_performance["xgb"]["cv_rmse_mean"]
                linear_cv_rmse = self.model_performance["linear"]["cv_rmse_mean"]

                # Find best performing model based on CV RMSE
                best_rmse = min(rf_cv_rmse, xgb_cv_rmse, linear_cv_rmse)

                if best_rmse == rf_cv_rmse:
                    self._log_and_write(
                        f"- **Best Performing Model (CV RMSE):** Random Forest ({rf_cv_rmse:.4f})"
                    )
                elif best_rmse == xgb_cv_rmse:
                    self._log_and_write(
                        f"- **Best Performing Model (CV RMSE):** XGBoost ({xgb_cv_rmse:.4f})"
                    )
                else:
                    self._log_and_write(
                        f"- **Best Performing Model (CV RMSE):** Linear Regression ({linear_cv_rmse:.4f})"
                    )

                # Performance comparison
                self._log_and_write("- **Model Performance Rankings (CV RMSE):**")
                performances = [
                    ("Random Forest", rf_cv_rmse),
                    ("XGBoost", xgb_cv_rmse),
                    ("Linear Regression", linear_cv_rmse),
                ]
                performances.sort(key=lambda x: x[1])
                for i, (model_name, rmse) in enumerate(performances, 1):
                    self._log_and_write(f"  {i}. {model_name}: {rmse:.4f}")

            elif "rf" in self.model_performance and "xgb" in self.model_performance:
                rf_cv_rmse = self.model_performance["rf"]["cv_rmse_mean"]
                xgb_cv_rmse = self.model_performance["xgb"]["cv_rmse_mean"]
                if rf_cv_rmse < xgb_cv_rmse:
                    self._log_and_write(
                        "- **Best Performing Model:** Random Forest shows superior performance"
                    )
                elif xgb_cv_rmse < rf_cv_rmse:
                    self._log_and_write(
                        "- **Best Performing Model:** XGBoost shows superior performance"
                    )
                else:
                    self._log_and_write(
                        "- **Model Performance:** Random Forest and XGBoost show similar performance"
                    )

        # Data insights
        if hasattr(self, "df") and self.df is not None:
            price_cv = self.df[self.target_col].std() / self.df[self.target_col].mean()
            if price_cv < 0.1:
                self._log_and_write(
                    "- **Data Volatility:** Low volatility - prices are relatively stable"
                )
            elif price_cv < 0.3:
                self._log_and_write(
                    "- **Data Volatility:** Moderate volatility - normal price fluctuations"
                )
            else:
                self._log_and_write(
                    "- **Data Volatility:** High volatility - significant price variations"
                )

        # Final notes
        self._log_and_write("## Notes")
        self._log_and_write(
            "- All models were trained using time-based features to capture seasonal patterns"
        )
        self._log_and_write(
            "- Ensemble predictions combine the strengths of different modeling approaches"
        )
        self._log_and_write("- Cross-validation ensures robust performance estimates")
        self._log_and_write(
            "- Feature importance analysis identifies key predictive factors"
        )

        self._log_and_write("---")
        self._log_and_write(
            f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._log_and_write("**End of Analysis**")


def train_main():
    print("Starting Hotel Price Forecasting Analysis")
    print("=" * 50)

    # Load data
    df = pd.read_csv("price_data.csv")

    # Initialize forecaster with verbose logging
    forecaster = SimpleFutureForecaster(
        df,
        "Date",
        "Price",
        output_dir="pricelabs",
        clip_outliers=True,
        verbose=True,
    )

    # Train models
    print("\nTraining Models...")
    forecaster.train_models(n_iter=10, cv=5)

    # Save trained models
    print("\nSaving Trained Models...")
    forecaster.save_models()

    # Analyze training data
    print("\nAnalyzing Training Data...")
    forecaster.analyze_training_data()

    # Store forecast results for final report
    forecast_results = []

    # Generate forecasts with different model weights
    print("\nGenerating Forecasts...")

    weight_configurations = [
        {"linear": 1, "rf": 0, "xgb": 0, "name": "Linear Regression Only"},
        {"linear": 0, "rf": 1, "xgb": 0, "name": "Random Forest Only"},
        {"linear": 0, "rf": 0, "xgb": 1, "name": "XGBoost Only"},
        {"linear": 0.6, "rf": 0.0, "xgb": 0.4, "name": "Balanced Ensemble"},
        {"linear": 0.4, "rf": 0.0, "xgb": 0.6, "name": "XGBoost-Linear Hybrid"},
    ]

    for config in weight_configurations:
        weight_dict = {k: v for k, v in config.items() if k != "name"}
        forecast = forecaster.predict(
            "2012-01-01",
            "2020-12-31",
            models_weights=weight_dict,
        )

        # Create descriptive text for plot
        weight_text = f"{' ' * 50}Model weights: Linear={config['linear']}, RF={config['rf']}, XGB={config['xgb']}"

        # Plot forecast
        forecaster.plot_forecast(
            forecast,
            text_note=weight_text,
            plot_name=f"forecast_{config['name'].lower().replace(' ', '_')}",
        )

        # Store for final report
        forecast_results.append((config["name"], forecast))

    # Generate final prediction for specific period
    print("\nGenerating Final Prediction...")
    final_forecast = forecaster.predict(
        "2020-02-01",
        "2020-02-29",
        models_weights={"linear": 0.4, "rf": 0.3, "xgb": 0.3},
    )

    # Plot final forecast
    forecaster.plot_forecast(
        final_forecast,
        text_note="Final ensemble model weights: Linear=0.4, RF=0.3, XGB=0.3",
        plot_name="final_forecast_feb_2020",
    )

    # Save final forecast to CSV
    final_forecast_export = final_forecast[[forecaster.date_col, "Forecast"]].copy()
    final_forecast_export.rename(columns={"Forecast": "Price"}, inplace=True)
    final_forecast_export.to_csv("forecast.csv", index=False)

    # Add final forecast to results
    forecast_results.append(("Final Forecast (Feb 2020)", final_forecast))

    # Export training data
    print("\nExporting Training Data...")
    forecaster.export_training_data()

    # Generate comprehensive report
    print("\nGenerating Comprehensive Report...")
    forecaster.generate_comprehensive_report(forecast_results)

    # Final summary
    print("\nAnalysis Complete!")
    print("=" * 50)
    print(f"Results saved to: {forecaster.output_dir}")
    print(f"Report file: {forecaster.report_file}")
    print(f"Images saved to: {forecaster.image_dir}")
    print("Forecast CSV: forecast.csv")

    # Training data summary
    print("\nTraining Data Access:")
    print(f"- Stored training samples: {len(forecaster.training_data['X_raw']):,}")
    print(f"- Features available: {len(forecaster.training_data['feature_names'])}")
    print(
        "- Access methods: get_training_data(), get_training_features(), get_training_target()"
    )

    print("\nFinal Forecast Preview:")
    print(final_forecast_export.head(10).to_string(index=False))

    # Save models
    forecaster.save_models(save_performance=True)


def load_and_predict():
    forecaster = SimpleFutureForecaster.load_from_saved(
        models_dir="pricelabs/models", verbose=True
    )

    # Generate predictions for any date range
    predictions = forecaster.predict_from_saved(
        start_date="2020-02-01",
        end_date="2020-02-29",
        models_weights={"linear": 0.4, "rf": 0.3, "xgb": 0.3},
    )

    # Save predictions
    predictions.to_csv("feb_2020_forecast.csv")
    # plot the predictions
    forecaster.plot_forecast(
        predictions,
        text_note="Final ensemble model weights: Linear=0.4, RF=0.3, XGB=0.3",
        plot_name="final_forecast_feb_2020",
    )


if __name__ == "__main__":
    train_main()
    load_and_predict()
