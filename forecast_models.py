import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm



def forecast_weekly_consumption_xgboost(df, forecast_weeks_ahead=6, seasonality='No'):
    """
    Forecasts weekly consumption for a given material using XGBoost and recursive forecasting.

    Args:
        file_path (str): Path to the Excel file containing consumption data.
        material_number (str): Material number to forecast.
        forecast_weeks_ahead (int): Number of weeks to forecast into the future.
        seasonality (str): 'Y' to include year and week as features, 'N' otherwise.

    Returns:
        pandas.DataFrame: DataFrame containing the forecasted consumption.
    """
    material_number = df['Material Number'][0]
    df_material = df
    weeks = ['WW' + str(i) + '_Consumption' for i in range(1, 53)]
    df_material = df_material[weeks]
    weekly_data = df_material.transpose().reset_index()
    weekly_data.columns = ['week', 'consumption']
    weekly_data['week'] = weekly_data['week'].str.extract('(\d+)').astype(int)
    weekly_data = weekly_data.sort_values('week')

    weekly_data['year'] = 2024
    weekly_data['lag_1'] = weekly_data['consumption'].shift(1)
    weekly_data['lag_2'] = weekly_data['consumption'].shift(2)
    weekly_data['rolling_mean_6'] = weekly_data['consumption'].shift(1).rolling(window=6, min_periods=1).mean()
    weekly_data['rolling_std_6'] = weekly_data['consumption'].shift(1).rolling(window=6, min_periods=1).std()

    #Fill NA with 0
    weekly_data[['lag_1', 'lag_2', 'rolling_mean_6', 'rolling_std_6']] = weekly_data[['lag_1', 'lag_2', 'rolling_mean_6', 'rolling_std_6']].fillna(0)

    if seasonality == 'Yes':
        features = ['year', 'week', 'lag_1', 'lag_2', 'rolling_mean_6', 'rolling_std_6']
    else:
        features = ['lag_1', 'lag_2', 'rolling_mean_6', 'rolling_std_6']

    X = weekly_data[features]
    y = weekly_data['consumption']

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1, max_depth=6)
    model.fit(X, y)

    forecast_weeks = pd.DataFrame({'week': np.arange(1, forecast_weeks_ahead+1)})
    forecast_weeks['year'] = 2025

    forecast_results = []
    last_data = weekly_data.iloc[-1].copy()
    for index, row in forecast_weeks.iterrows():
        if seasonality == 'Yes':
            row['year'] = 2025
            row['week'] = row['week']
        row['lag_1'] = last_data['consumption']
        row['lag_2'] = last_data['lag_1']
        rolling_data = pd.Series([last_data['lag_2'], last_data['lag_1'], last_data['consumption']])
        rolling_data = pd.concat([weekly_data['consumption'].tail(3), rolling_data])
        row['rolling_mean_6'] = rolling_data.tail(6).mean()
        row['rolling_std_6'] = rolling_data.tail(6).std()
        X_forecast = pd.DataFrame([row[features]])
        predicted_consumption = model.predict(X_forecast)[0]
        forecast_results.append({'week': row['week'], 'predicted_consumption': predicted_consumption})
        last_data['lag_2'] = last_data['lag_1']
        last_data['lag_1'] = last_data['consumption']
        last_data['consumption'] = predicted_consumption
        #print(last_data)

    forecast_results_df = pd.DataFrame(forecast_results)
    forecast_results_df['year'] = 2025

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_data['week'], y, label='Actual Consumption (2024)', color='blue')
    plt.plot(forecast_results_df['week'] + 52, forecast_results_df['predicted_consumption'], label='Forecasted Consumption (2025)', linestyle='dashed', color='red')
    plt.xlabel('Week')
    plt.ylabel('Consumption')
    plt.title(f'Recursive Consumption Forecasting for Material {material_number} (Weeks 1-{forecast_weeks_ahead}, 2025)')
    plt.legend()
    plt.show()

    return forecast_results_df, plt

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


def find_arima_order(series, d):
    """
    Finds the d, p, and q parameters for a SARIMA model.

    Args:
        series (pd.Series): The time series data.

    Returns:
        tuple: (d, p, q) - the determined SARIMA order.   
    """

    acf_values, ci_acf = sm.tsa.acf(series, alpha=0.05)
    pacf_values, ci_pacf = sm.tsa.pacf(series, alpha=0.05)

    
    acf_data = acf_values[1:] #exclude lag 0
    pacf_data = pacf_values[1:] #exclude lag 0

    conf_int_acf = 1.96 / np.sqrt(len(series))
    conf_int_pacf = 1.96 / np.sqrt(len(series))

    # Determine 'q' based on ACF
    q = 0
    for i, val in enumerate(acf_data):
        if abs(val) > conf_int_acf:
            q = i + 1
        else:
            break

    # Determine 'p' based on PACF
    p = 0
    for i, val in enumerate(pacf_data):
        if abs(val) > conf_int_pacf:
            p = i + 1
        else:
            break

    #print(f"AR order (p): {p}")
    #print(f"MA order (q): {q}")

    return (p, q)

def check_stationarity(series, title):
    """
    Checks stationarity of a time series using visual inspection, ACF/PACF plots, and ADF test.

    Args:
        series (pd.Series): The time series data.
        title (str): The title for the plots.
    """
    print(f"Stationarity Check: {title}")

    # Visual Inspection
    plt.figure(figsize=(12, 4))
    plt.plot(series)
    plt.title(f"Time Series Plot: {title}")
    plt.show()

    # ACF and PACF Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, ax=ax1, lags=20)
    plot_pacf(series, ax=ax2, lags=20)
    plt.show()

    # Augmented Dickey-Fuller Test to see if series is stationary
    adf_result = adfuller(series)
    print("ADF Test:")
    print(f"  ADF Statistic: {adf_result[0]}")
    print(f"  p-value: {adf_result[1]}")
    print(f"  Critical Values: {adf_result[4]}")
    if adf_result[1] <= 0.05:
        print("  Result: Series is likely stationary.")
    else:
        print("  Result: Series is likely non-stationary.")
    print("-" * 40)

def difference_series(series, order=1):
    """
    Differences a time series.

    Args:
        series (pd.Series): The time series data.
        order (int): The order of differencing.

    Returns:
        pd.Series: The differenced time series.
    """
    differenced_series = series.diff(order).dropna()
    return differenced_series

def forecast_weekly_consumption_arima(df, forecast_weeks_ahead=6, seasonality = "No"):
    """
    Forecasts weekly consumption for a given material using SARIMA.

    Args:
        file_path (str): Path to the Excel file containing consumption data.
        material_number (str): Material number to forecast.
        forecast_weeks_ahead (int): Number of weeks to forecast into the future.
        seasonal_order (tuple): Seasonal order for SARIMA (p, d, q, s).

    Returns:
        pandas.DataFrame: DataFrame containing the forecasted consumption.
    """

    material_number = df['Material Number'][0]
    df_material = df
    weeks = ['WW' + str(i) + '_Consumption' for i in range(1, 53)]
    df_material = df_material[weeks]
    weekly_data = df_material.transpose().reset_index()
    weekly_data.columns = ['week', 'consumption']
    weekly_data['week'] = weekly_data['week'].str.extract('(\d+)').astype(int)
    date_data = pd.DataFrame()
    date_data['date'] = pd.to_datetime('2024-01-01') + pd.to_timedelta((weekly_data['week'] - 1) * 7, unit='days')
    date_data['consumption'] = weekly_data['consumption'] 
    weekly_data = weekly_data.sort_values('week')
    weekly_data = weekly_data.set_index('week')
    date_data = date_data.set_index('date')
    date_data.index.freq = 'W-MON'
    

    # Check stationarity and difference if needed
    #check_stationarity(weekly_data['consumption'], "Original Series")
    d = 0
    temp_series = weekly_data['consumption'].copy()
    while adfuller(temp_series)[1] > 0.05:
        temp_series = difference_series(temp_series)
        d += 1
        #check_stationarity(temp_series, f"Differenced Series (d={d})")

    # Find SARIMA order
    if d > 0:
        differenced_consumption = difference_series(weekly_data['consumption'], d)
        p, q = find_arima_order(differenced_consumption, d)
        #plot_acf(differenced_consumption, lags=20)
        #plot_pacf(differenced_consumption, lags=20)
        #plt.show()
    else:
        p, q = find_arima_order(weekly_data['consumption'], d)
    print(f"ARIMA Order: (p, d, q) = ({p}, {d}, {q})")
    
    order = (p, d, q)
    seasonal_order = (p, d, q, 52)

    if seasonality == "Yes":
        model = SARIMAX(date_data['consumption'], order=order, seasonal_order = seasonal_order)
        model_fit = model.fit(disp=False)

    else:
        model = ARIMA(date_data['consumption'], order=order)
        model_fit = model.fit()

    # Forecast
    forecast = model_fit.get_forecast(steps=forecast_weeks_ahead+1)
    forecast_values = forecast.predicted_mean

    # Corrected forecast index calculation starting from 2025
    freq = date_data.index.freq or pd.Timedelta(days=7) # Get the freq or default to weekly.
    
    #Start from 2025:
    start_forecast_date = pd.to_datetime('2025-01-01') #First Day of 2025.
    
    forecast_index = pd.date_range(start=start_forecast_date, periods=forecast_weeks_ahead, freq=freq)

    forecast_series = pd.Series(forecast_values, index=forecast_index)

    # Create DataFrame for results
    forecast_results_df = pd.DataFrame({'date': forecast_series.index, 'predicted_consumption': forecast_series.values})


    # Plotting
    if seasonality == "Yes":
        model_name = "SARIMAX"
    else:
        model_name = "ARIMA"
    plt.figure(figsize=(12, 6))
    plt.plot(date_data.index, date_data['consumption'], label='Actual Consumption (2024)', color='blue')
    plt.plot(forecast_results_df['date'], forecast_results_df['predicted_consumption'], label='Forecasted Consumption (2025)', linestyle='dashed', color='red')
    plt.xlabel('Date')
    plt.ylabel('Consumption')
    plt.title(f'{model_name} Consumption Forecasting for Material {material_number} (Weeks 1-{forecast_weeks_ahead}, 2025)')
    plt.legend()

    return forecast_results_df, plt
