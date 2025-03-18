import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def forecast_material_consumption(df, mat_number, lag_time=2):
    """
    Forecasts weekly consumption for a given material number using XGBoost.

    Args:
        df (pd.DataFrame): Input DataFrame containing material consumption data.
        mat_number (str): Material number to forecast.
        lag_time (int): Number of lag weeks to use as features (default 2).

    Returns:
        tuple: A tuple containing the plot and the forecasted DataFrame.
    """

    # Filter data for the specified Material Number
    df_material = df[df['Material Number'] == mat_number].reset_index(drop=True)

    # Reshape the data into a time series format (weekly consumption data)
    weeks = ['WW' + str(i) + '_Consumption' for i in range(1, 53)]
    df_material = df_material[weeks]

    # Create a DataFrame with weeks as the index
    weekly_data = df_material.transpose().reset_index()
    weekly_data.columns = ['week', 'consumption']
    weekly_data['week'] = weekly_data['week'].str.extract('(\d+)').astype(int)

    # Sort data by week
    weekly_data = weekly_data.sort_values('week')

    # Feature Engineering
    weekly_data['year'] = (weekly_data['week'] - 1) // 52 + 2024
    weekly_data['month'] = (weekly_data['week'] - 1) % 12 + 1

    for i in range(1, lag_time + 1):
        weekly_data[f'lag_{i}'] = weekly_data['consumption'].shift(i)

    weekly_data['rolling_mean_6'] = weekly_data['consumption'].shift(1).rolling(window=6).mean()
    weekly_data['rolling_std_6'] = weekly_data['consumption'].shift(1).rolling(window=6).std()

    # Drop NaN values due to lag and rolling features, but do NOT drop zeros
    weekly_data = weekly_data.dropna(subset=[f'lag_{i}' for i in range(1, lag_time + 1)] + ['rolling_mean_6', 'rolling_std_6'])

    # Define features and target
    features = ['year', 'month'] + [f'lag_{i}' for i in range(1, lag_time + 1)] + ['rolling_mean_6', 'rolling_std_6']
    X = weekly_data[features]
    y = weekly_data['consumption']

    # Train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1, max_depth=6)
    model.fit(X, y)

    # Generate additional 10 weeks data for 2025 (forecasting weeks 1 to 10)
    forecast_weeks = pd.DataFrame({'week': np.arange(53, 63)})

    # Add feature columns for the forecast data
    forecast_weeks['year'] = 2025
    forecast_weeks['month'] = (forecast_weeks['week'] - 1) % 12 + 1

    # Use the last available consumption data for lag features
    last_data = weekly_data.iloc[-1]
    for i in range(1, lag_time + 1):
        forecast_weeks[f'lag_{i}'] = last_data['consumption']

    # For rolling mean and rolling std, we assume similar trends for simplicity
    forecast_weeks['rolling_mean_6'] = weekly_data['rolling_mean_6'].iloc[-1]
    forecast_weeks['rolling_std_6'] = weekly_data['rolling_std_6'].iloc[-1]

    # Predict future consumption for weeks 53 to 62 using the trained model
    X_forecast = forecast_weeks[features]
    forecast_weeks['predicted_consumption'] = model.predict(X_forecast)

    # Plot actual vs predicted consumption
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_data['week'], y, label='Actual Consumption (2024)', color='blue')
    plt.plot(forecast_weeks['week'], forecast_weeks['predicted_consumption'], label='Forecasted Consumption (2025)', linestyle='dashed', color='red')
    plt.xlabel('Week')
    plt.ylabel('Consumption')
    plt.title(f'Consumption Forecasting for Material {mat_number} (Weeks 1-10, 2025)')
    plt.legend()
    plt.show()

    return plt, forecast_weeks[['week', 'predicted_consumption']]

def plot_acf_pacf_material_consumption(df, mat_number):
    """
    Plots the ACF and PACF of weekly consumption for a given material number.

    Args:
        df (pd.DataFrame): Input DataFrame containing material consumption data.
        mat_number (str): Material number to analyze.

    Returns:
        None (displays the plots).
    """

    # Filter data for the specified Material Number
    df_material = df[df['Material Number'] == mat_number].reset_index(drop=True)

    # Reshape the data into a time series format (weekly consumption data)
    weeks = ['WW' + str(i) + '_Consumption' for i in range(1, 53)]
    df_material = df_material[weeks]

    # Create a DataFrame with weeks as the index
    weekly_data = df_material.transpose().reset_index()
    weekly_data.columns = ['week', 'consumption']
    weekly_data['week'] = weekly_data['week'].str.extract('(\d+)').astype(int)

    # Sort data by week
    weekly_data = weekly_data.sort_values('week')

    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(weekly_data['consumption'], ax=ax1, lags=20)  # Adjust lags as needed
    plot_pacf(weekly_data['consumption'], ax=ax2, lags=20)  # Adjust lags as needed
    plt.suptitle(f'ACF and PACF for Material {mat_number} Weekly Consumption')
    plt.show()