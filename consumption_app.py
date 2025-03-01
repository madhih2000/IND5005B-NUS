import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.subplots as sp

# Load data
def load_data(file):
    df = pd.read_excel(file)
    df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])
    df['SLED/BBD'] = df['SLED/BBD'].fillna('No Expiry')  # Handle empty expiry dates
    return df

# Stationarity test
def test_stationarity(series):
    if series.nunique() <= 1:  # Check if the series has only one unique value
        return 1.0  # Return a high p-value indicating non-stationarity
    result = adfuller(series)
    return result[1]  # p-value

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
    df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])
    df = df.sort_values('Pstng Date')

    # Ensure Quantity is numeric
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    
    if df['Quantity'].isnull().all():
        st.error("Error: No valid quantity data for forecasting.")
        return None

    # Aggregate data by week
    df_grouped = df.groupby('Pstng Date')['Quantity'].sum()

    if df_grouped.nunique() <= 1:
        st.warning("Not enough variation in demand data to fit ARIMA.")
        return None

    # Set frequency to weekly if there are gaps
    df_grouped = df_grouped.asfreq('W', method='ffill')

    # Stationarity check
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
    forecast_df = pd.DataFrame({'Pstng Date': forecast_index, 'Forecast': forecast})

    return forecast_df

# Monte Carlo simulation for demand
def monte_carlo_simulation(df, simulations=1000):
    demand_data = df['Quantity'].values
    simulated_demands = [np.random.choice(demand_data, len(demand_data), replace=True).sum() for _ in range(simulations)]
    return simulated_demands

# Set the page config with the title centered
st.set_page_config(page_title="Micron SupplySense", layout="wide")

# Center title at the top
st.markdown(
    """
    <h1 style="text-align: center; color: #4B9CD3;">Micron SupplySense</h1>
    """, 
    unsafe_allow_html=True
)

# Create a sidebar for navigation (for a dashboard-style layout)
tabs = st.sidebar.radio("Select an Analysis Type:", ["Material Consumption Analysis", "Other Analysis", "Settings"])

if tabs == "Material Consumption Analysis":
    st.title("Material Consumption Analysis")
    # Add selection for Material Group(s)
    group_selection = st.radio(
        "Select Material Group(s) to Analyze:",
        ("Material Group 260", "Material Group 453", "Both")
    )

    # Upload files based on the selection
    file_260 = None
    file_453 = None
    if group_selection == "Material Group 260" or group_selection == "Both":
        file_260 = st.file_uploader("Upload Excel for Material Group 260", type=["xlsx"])

    if group_selection == "Material Group 453" or group_selection == "Both":
        file_453 = st.file_uploader("Upload Excel for Material Group 453", type=["xlsx"])

    # Create subplots for side-by-side layout when both groups are selected
    if group_selection == "Material Group 260" and file_260:
        st.subheader("Material Group 260 Analysis")
        df_260 = load_data(file_260)

        # Trend Plot for Material Group 260
        trend_260 = plot_trend(df_260, 260)
        st.plotly_chart(trend_260, use_container_width=True)

        # Forecast Plot for Material Group 260
        forecast_periods = st.slider("Select Forecast Periods for 260", 1, 24, 12)  # Select number of forecast periods
        forecast_260 = forecast_demand(df_260, periods=forecast_periods)
        fig_forecast_260 = go.Figure(go.Scatter(x=forecast_260['Pstng Date'], y=forecast_260['Forecast'], mode='lines', name="Forecast - Group 260"))
        fig_forecast_260.update_layout(title="Demand Forecast - Material Group 260", xaxis_title="Date", yaxis_title="Forecast")
        st.plotly_chart(fig_forecast_260, use_container_width=True)

        # Monte Carlo Simulation for Material Group 260
        simulation_260 = monte_carlo_simulation(df_260)
        fig_simulation_260 = go.Figure(go.Histogram(x=simulation_260, nbinsx=50, name="Simulation - Group 260", marker_color='blue'))
        fig_simulation_260.update_layout(title="Monte Carlo Simulation - Material Group 260", xaxis_title="Simulated Demand", yaxis_title="Frequency")
        st.plotly_chart(fig_simulation_260, use_container_width=True)

        # Material selection dropdown
        unique_materials = df_260['Material Number'].unique()
        material_number = st.selectbox("Select Material Number", unique_materials)
        df_material = df_260[df_260['Material Number'] == material_number]
        if not df_material.empty:
            st.subheader(f"Analysis for Material {material_number}")
            st.plotly_chart(plot_trend(df_material, f"Trend - {material_number}"), use_container_width=True)
            forecast_df = forecast_demand(df_material, periods=12)
            fig_forecast = go.Figure(go.Scatter(x=forecast_df['Pstng Date'], y=forecast_df['Forecast'], mode='lines', name=f"Forecast - {material_number}"))
            fig_forecast.update_layout(title=f"Forecast - Material {material_number}", xaxis_title="Date", yaxis_title="Forecast")
            st.plotly_chart(fig_forecast, use_container_width=True)
            simulation_data = monte_carlo_simulation(df_material)
            fig_simulation = go.Figure(go.Histogram(x=simulation_data, nbinsx=50, marker_color='green'))
            fig_simulation.update_layout(title=f"Monte Carlo Simulation - Material {material_number}", xaxis_title="Simulated Demand", yaxis_title="Frequency")
            st.plotly_chart(fig_simulation, use_container_width=True)


    elif group_selection == "Material Group 453" and file_453:
        st.subheader("Material Group 453 Analysis")
        df_453 = load_data(file_453)

        # Trend Plot for Material Group 453
        trend_453 = plot_trend(df_453, 453)
        st.plotly_chart(trend_453, use_container_width=True)

        # Forecast Plot for Material Group 453
        forecast_periods = st.slider("Select Forecast Periods for 453", 1, 24, 12)  # Select number of forecast periods
        forecast_453 = forecast_demand(df_453, periods=forecast_periods)
        fig_forecast_453 = go.Figure(go.Scatter(x=forecast_453['Pstng Date'], y=forecast_453['Forecast'], mode='lines', name="Forecast - Group 453"))
        fig_forecast_453.update_layout(title="Demand Forecast - Material Group 453", xaxis_title="Date", yaxis_title="Forecast")
        st.plotly_chart(fig_forecast_453, use_container_width=True)

        # Monte Carlo Simulation for Material Group 453
        simulation_453 = monte_carlo_simulation(df_453)
        fig_simulation_453 = go.Figure(go.Histogram(x=simulation_453, nbinsx=50, name="Simulation - Group 453", marker_color='green'))
        fig_simulation_453.update_layout(title="Monte Carlo Simulation - Material Group 453", xaxis_title="Simulated Demand", yaxis_title="Frequency")
        st.plotly_chart(fig_simulation_453, use_container_width=True)


        # Material selection dropdown
        unique_materials = df_453['Material Number'].unique()
        material_number = st.selectbox("Select Material Number", unique_materials)
        df_material = df_453[df_453['Material Number'] == material_number]
        if not df_material.empty:
            st.subheader(f"Analysis for Material {material_number}")
            st.plotly_chart(plot_trend(df_material, f"Trend - {material_number}"), use_container_width=True)
            forecast_df = forecast_demand(df_material, periods=12)
            fig_forecast = go.Figure(go.Scatter(x=forecast_df['Pstng Date'], y=forecast_df['Forecast'], mode='lines', name=f"Forecast - {material_number}"))
            fig_forecast.update_layout(title=f"Forecast - Material {material_number}", xaxis_title="Date", yaxis_title="Forecast")
            st.plotly_chart(fig_forecast, use_container_width=True)
            simulation_data = monte_carlo_simulation(df_material)
            fig_simulation = go.Figure(go.Histogram(x=simulation_data, nbinsx=50, marker_color='green'))
            fig_simulation.update_layout(title=f"Monte Carlo Simulation - Material {material_number}", xaxis_title="Simulated Demand", yaxis_title="Frequency")
            st.plotly_chart(fig_simulation, use_container_width=True)


    elif group_selection == "Both" and file_260 and file_453:
        st.subheader("Material Group 260 and 453 Comparison")

        # Load data for both material groups
        df_260 = load_data(file_260)
        df_453 = load_data(file_453)

        # Plot the trend
        fig_trend = sp.make_subplots(rows=1, cols=2, subplot_titles=("Material Group 260 Trend", "Material Group 453 Trend"))

        # Plot for Material Group 260
        trend_260 = plot_trend(df_260, 260)
        for trace in trend_260.data:
            fig_trend.add_trace(trace, row=1, col=1)

        # Plot for Material Group 453
        trend_453 = plot_trend(df_453, 453)
        for trace in trend_453.data:
            fig_trend.add_trace(trace, row=1, col=2)

        fig_trend.update_layout(title="Consumption Trend Comparison", showlegend=False)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Forecast demand for both material groups
        forecast_periods = st.slider("Select Forecast Periods", 1, 24, 12)  # Select number of forecast periods
        
        forecast_260 = forecast_demand(df_260, periods=forecast_periods)
        forecast_453 = forecast_demand(df_453, periods=forecast_periods)

        # Create subplots for Forecast
        fig_forecast = sp.make_subplots(rows=1, cols=2, subplot_titles=("Material Group 260 Forecast", "Material Group 453 Forecast"))

        # Forecast for Material Group 260
        fig_forecast.add_trace(go.Scatter(x=forecast_260['Pstng Date'], y=forecast_260['Forecast'], mode='lines', name="Forecast - Group 260"), row=1, col=1)
        
        # Forecast for Material Group 453
        fig_forecast.add_trace(go.Scatter(x=forecast_453['Pstng Date'], y=forecast_453['Forecast'], mode='lines', name="Forecast - Group 453"), row=1, col=2)

        fig_forecast.update_layout(title="Demand Forecast Comparison", showlegend=False)
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Monte Carlo Simulation for both material groups
        simulation_260 = monte_carlo_simulation(df_260)
        simulation_453 = monte_carlo_simulation(df_453)

        # Create subplots for Monte Carlo Simulation
        fig_simulation = sp.make_subplots(rows=1, cols=2, subplot_titles=("Material Group 260 Simulation", "Material Group 453 Simulation"))

        # Simulation for Material Group 260
        fig_simulation.add_trace(go.Histogram(x=simulation_260, nbinsx=50, name="Simulation - Group 260", marker_color='blue'), row=1, col=1)

        # Simulation for Material Group 453
        fig_simulation.add_trace(go.Histogram(x=simulation_453, nbinsx=50, name="Simulation - Group 453", marker_color='green'), row=1, col=2)

        fig_simulation.update_layout(title="Monte Carlo Simulation Comparison", showlegend=False)
        st.plotly_chart(fig_simulation, use_container_width=True)

        # **Material Selection for Both Groups**
        st.subheader("Material-Specific Analysis")

        # Create separate dropdowns for Material Groups 260 and 453
        col1, col2 = st.columns(2)

        with col1:
            unique_materials_260 = df_260['Material Number'].unique()
            material_260 = st.selectbox("Select Material from Group 260", unique_materials_260)
            df_material_260 = df_260[df_260['Material Number'] == material_260]
            if not df_material_260.empty:
                st.subheader(f"Analysis for Material {material_260} (Group 260)")
                st.plotly_chart(plot_trend(df_material_260, f"Trend - {material_260}"), use_container_width=True)
                forecast_260 = forecast_demand(df_material_260, periods=12)
                fig_forecast_260 = go.Figure(go.Scatter(x=forecast_260['Pstng Date'], y=forecast_260['Forecast'], mode='lines', name=f"Forecast - {material_260}"))
                fig_forecast_260.update_layout(title=f"Forecast - Material {material_260}", xaxis_title="Date", yaxis_title="Forecast")
                st.plotly_chart(fig_forecast_260, use_container_width=True)
                simulation_260 = monte_carlo_simulation(df_material_260)
                fig_sim_260 = go.Figure(go.Histogram(x=simulation_260, nbinsx=50, marker_color='blue'))
                fig_sim_260.update_layout(title=f"Monte Carlo Simulation - Material {material_260}", xaxis_title="Simulated Demand", yaxis_title="Frequency")
                st.plotly_chart(fig_sim_260, use_container_width=True)

        with col2:
            unique_materials_453 = df_453['Material Number'].unique()
            material_453 = st.selectbox("Select Material from Group 453", unique_materials_453)
            df_material_453 = df_453[df_453['Material Number'] == material_453]
            if df_material_453.empty:
                st.warning(f"No data available for Material {material_453} in Group 453.")
            else:
                st.subheader(f"Analysis for Material {material_453} (Group 453)")
                st.plotly_chart(plot_trend(df_material_453, f"Trend - {material_453}"), use_container_width=True)

                # Ensure df_material_453 has at least a few data points before forecasting
                if len(df_material_453) < 5:
                    st.warning(f"Not enough data points to forecast Material {material_453}.")
                else:
                    forecast_453 = forecast_demand(df_material_453, periods=12)
                    fig_forecast_453 = go.Figure(go.Scatter(x=forecast_453['Pstng Date'], y=forecast_453['Forecast'], mode='lines', name=f"Forecast - {material_453}"))
                    fig_forecast_453.update_layout(title=f"Forecast - Material {material_453}", xaxis_title="Date", yaxis_title="Forecast")
                    st.plotly_chart(fig_forecast_453, use_container_width=True)

                # Monte Carlo Simulation
                simulation_453 = monte_carlo_simulation(df_material_453)
                fig_sim_453 = go.Figure(go.Histogram(x=simulation_453, nbinsx=50, marker_color='green'))
                fig_sim_453.update_layout(title=f"Monte Carlo Simulation - Material {material_453}", xaxis_title="Simulated Demand", yaxis_title="Frequency")
                st.plotly_chart(fig_sim_453, use_container_width=True)

# Other Analysis or Settings Page (for navigation)
elif tabs == "Other Analysis":
    st.title("Other Analysis")
    st.write("You can add more analysis sections here.")

elif tabs == "Settings":
    st.title("Settings")
    st.write("Adjust your settings here.")





