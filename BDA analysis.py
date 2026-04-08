import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILE = "modified_air_quality.csv"
MODEL_FILE = "aqi_model.pkl"
TARGET = "AQI"
FEATURES = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]
EPS = 1e-8

CITY_SELECTED = "All Cities"
YEAR_SELECTED = "All Years"
BACKTEST_DAYS = 30

# ==========================================
# FUNCTIONS
# ==========================================

def load_model(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Model file '{filepath}' not found.")
        return None
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_data(filepath):
    print(f"Loading and Preprocessing AQI dataset: {filepath}...")
    df = pd.read_csv(filepath)

    # Convert Datetime and drop invalid dates/missing targets
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime", TARGET])
    
    # Preprocessing: Fill missing feature values with median (Condition 2)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    
    df = df.sort_values("Datetime")
    return df

def get_recommendation(aqi):
    """Actionable decisions based on AQI values (Condition 5)"""
    if aqi <= 50: return "Good: Air quality is satisfactory."
    elif aqi <= 100: return "Satisfactory: Minor breathing discomfort to sensitive people."
    elif aqi <= 200: return "Moderate: Breathing discomfort to people with lungs/heart disease."
    elif aqi <= 300: return "Poor: Breathing discomfort to most people on prolonged exposure."
    else: return "Severe: Affects healthy people and seriously impacts those with existing diseases."

def backtest_accuracy(model, df, days):
    test_df = df.iloc[-days:].copy()
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    smape = np.mean(2 * np.abs(preds - y_test) / (np.abs(y_test) + np.abs(preds) + EPS)) * 100
    accuracy = (1 - mae / (np.mean(y_test) + EPS)) * 100

    result_df = test_df[["Datetime", TARGET]].copy()
    result_df["Predicted_AQI"] = preds.round(2)
    return result_df, mae, r2, smape, accuracy

def plot_aqi_dashboard(df, target_col):
    df_plot = df.copy()
    df_plot["Year"] = df_plot["Datetime"].dt.year
    df_plot["Month"] = df_plot["Datetime"].dt.month

    monthly_trend = df_plot.groupby(["Year", "Month"])[target_col].mean().unstack(fill_value=0)
    yearly_avg = df_plot.groupby("Year")[target_col].mean()
    years = sorted(df_plot["Year"].unique())

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Air Quality Index Big Data Analysis", fontsize=16, fontweight='bold')

    # 1. Monthly Trend
    ax1 = axs[0, 0]
    for month in monthly_trend.columns:
        ax1.plot(monthly_trend.index, monthly_trend[month], marker='o', label=f"M{month}")
    ax1.set_title("Monthly AQI Trend")
    ax1.legend(ncol=3, fontsize='small')
    ax1.grid(True)

    # 2. Year Comparison
    ax2 = axs[0, 1]
    x = np.arange(len(years))
    width = 0.15
    for i, month in enumerate(list(monthly_trend.columns)[:4]): # Plot first 4 months
        ax2.bar(x + width*i, monthly_trend[month], width, label=f"M{month}")
    ax2.set_xticks(x + width, years)
    ax2.set_title("Year-wise Seasonal AQI")

    # 3. Yearly Average
    ax3 = axs[1, 0]
    bars = ax3.bar(yearly_avg.index.astype(str), yearly_avg.values, color='skyblue')
    ax3.set_title("Average AQI Per Year")
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{bar.get_height():.1f}", ha='center')

    # 4. Correlation
    ax4 = axs[1, 1]
    corr = df[FEATURES + [target_col]].corr()[target_col].drop(target_col)
    ax4.barh(corr.index, corr.values, color='salmon')
    ax4.set_title("Pollutant Correlation with AQI")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" AIR QUALITY BIG DATA ANALYTICS ENGINE ")
    print("="*50)

    if not os.path.exists(DATA_FILE):
        print(f"Dataset '{DATA_FILE}' not found. Please check the filename.")
        exit()

    # 1. Load Data
    df = load_data(DATA_FILE)

    # 2. Filter Data
    if CITY_SELECTED != "All Cities" and "City" in df.columns:
        df = df[df["City"] == CITY_SELECTED]
    if YEAR_SELECTED != "All Years":
        df = df[df["Datetime"].dt.year == int(YEAR_SELECTED)]

    print(f"Dataset Processed. Total Records: {len(df):,}")

    # 3. Load Model
    model = load_model(MODEL_FILE)
    if model is None:
        print("Execution halted: Model not found.")
        exit()

    # 4. Evaluate Performance (Condition 4)
    split_idx = int(len(df) * 0.8)
    X_test = df[FEATURES].iloc[split_idx:]
    y_test = df[TARGET].iloc[split_idx:]

    preds = model.predict(X_test)

    # NaN Masking for Safety
    mask = ~np.isnan(y_test) & ~np.isnan(preds)
    y_test_clean = y_test[mask]
    preds_clean = preds[mask]

    print("\n--- Model Performance Metrics ---")
    if len(y_test_clean) > 0:
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_clean, preds_clean):.4f}")
        print(f"R2 Score: {r2_score(y_test_clean, preds_clean):.4f}")
    else:
        print("No valid data points for metric calculation.")

    # 5. Backtesting & Prediction CSV
    bt_df, bt_mae, bt_r2, bt_smape, bt_acc = backtest_accuracy(model, df, BACKTEST_DAYS)
    print(f"Recent {BACKTEST_DAYS} Days Accuracy: {bt_acc:.2f}%")
    
    # 6. Recommendation (Condition 5)
    latest_aqi = bt_df["Predicted_AQI"].iloc[-1]
    print("\n" + "-"*30)
    print(f"Latest Predicted AQI: {latest_aqi}")
    print(f"Recommendation: {get_recommendation(latest_aqi)}")
    print("-"*30)

    # 7. Dashboards
    print("\nGenerating Visualizations...")
    fig1 = plot_aqi_dashboard(df, TARGET)

    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt_df["Datetime"], bt_df[TARGET], label="Actual AQI", marker="o", color="blue")
    ax.plot(bt_df["Datetime"], bt_df["Predicted_AQI"], label="Predicted AQI", linestyle="--", color="red")
    ax.set_title("Recent 30-Day AQI Prediction Accuracy")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

    # 8. Save Results
    output_csv = "aqi_predicted_vs_actual.csv"
    bt_df.to_csv(output_csv, index=False)
    print(f"\nPredictions saved to: {output_csv}")