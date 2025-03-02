import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px


def plot_average_consumption(df):
    df_grouped = df.groupby(['Pstng Date', 'Material Number', 'Site'])['Quantity'].mean().reset_index()
    
    fig = px.line(df_grouped, x='Pstng Date', y='Quantity', color='Material Number', line_group='Site',facet_col='Site',
                  title="Average Consumption Pattern by Material and Site",
                  labels={'Quantity': 'Average Quantity', 'Pstng Date': 'Posting Date'})
    
    return fig

def test_stationarity(series):
    """
    Test the stationarity of a time series using the Augmented Dickey-Fuller test.
    """
    # Ensure series has enough data points
    if len(series) < 15:  # ADF needs a minimum of 15-20 points for reliable results
        print("Not enough data points for stationarity test.")
        return 1.0  # Return a high p-value (non-stationary)

    # Perform ADF Test
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    
    return p_value  # Return the p-value from ADF test


# Make series stationary by differencing
def make_stationary(series):
    diff_series = series.diff().dropna()  # Differencing the series
    return diff_series

# Plot the trend using Plotly
def plot_trend(df, material_group):
    df_grouped = df.groupby('Pstng Date')['Quantity'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_grouped['Pstng Date'], y=df_grouped['Quantity'], mode='lines', name=f'Trend - Group {material_group}'))
    fig.update_layout(title=f'Consumption Trend for Material Group {material_group}', xaxis_title='Date', yaxis_title='Quantity')
    return fig

def forecast_demand(df, periods=12):
    if df.empty or 'Pstng Date' not in df.columns or 'Quantity' not in df.columns:
        st.error("Error: DataFrame is empty or missing required columns.")
        return None

    # Convert 'Pstng Date' to datetime and sort
    df['Pstng Date'] = pd.to_datetime(df['Pstng Date'], errors='coerce')
    df = df.sort_values('Pstng Date')

    # Ensure Quantity is numeric
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

    # Drop rows where Quantity is NaN
    df = df.dropna(subset=['Quantity'])
    
    if df['Quantity'].isnull().all() or df.empty:
        st.error("Error: No valid quantity data for forecasting.")
        return None

    # Aggregate data by week
    df_grouped = df.groupby('Pstng Date')['Quantity'].sum()

    if df_grouped.nunique() <= 1 or len(df_grouped) < 10:
        st.warning("Not enough variation in demand data to fit ARIMA.")
        return None

    # Set frequency to weekly, forward fill missing values
    df_grouped = df_grouped.asfreq('W', method='ffill')

    # Check if the data is stationary
    p_value = test_stationarity(df_grouped)
    if p_value > 0.05:
        st.warning("Data is not stationary. Applying differencing.")
        df_grouped = make_stationary(df_grouped)

    # Fit ARIMA model (try-except to catch errors)
    try:
        model = ARIMA(df_grouped, order=(5, 1, 0))  # Adjust order as needed
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)
    except Exception as e:
        st.error(f"ARIMA model failed: {e}")
        return None

    # Create forecast DataFrame
    forecast_index = pd.date_range(start=df_grouped.index[-1], periods=periods + 1, freq='W')[1:]
    forecast_df = pd.DataFrame({'Pstng Date': forecast_index, 'Forecast': forecast.values})

    return forecast_df

    
# Monte Carlo simulation for demand
def monte_carlo_simulation(df, simulations=1000):
    demand_data = df['Quantity'].values
    simulated_demands = [np.random.choice(demand_data, len(demand_data), replace=True).sum() for _ in range(simulations)]
    return simulated_demands