import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import plotly.figure_factory as ff
from utils import *
from consumption_utils import *
from order_placement_utils import *

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
tabs = st.sidebar.radio("Select an Analysis Type:", ["Material Consumption Analysis", "Order Placement Analysis", "Goods Receipt Analysis"])

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

        avg_consumption_260 = plot_average_consumption(df_260)
        st.plotly_chart(avg_consumption_260, use_container_width=True)

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

            if forecast_df is not None:
                for (site, plant), group in forecast_df.groupby(['Site', 'Plant']):
                    fig = go.Figure(go.Scatter(
                        x=group['Pstng Date'], 
                        y=group['Forecast'], 
                        mode='lines', 
                        name=f"Forecast - {site}, {plant}"
                    ))

                    fig.update_layout(
                        title=f"Demand Forecast - Site {site}, Plant {plant}",
                        xaxis_title="Date",
                        yaxis_title="Forecast",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No forecast data available.")


            # if forecast_df is not None:
            #     fig_forecast = go.Figure(go.Scatter(x=forecast_df['Pstng Date'], y=forecast_df['Forecast'], mode='lines', name=f"Forecast - {material_number}"))
            #     fig_forecast.update_layout(title=f"Forecast - Material {material_number}", xaxis_title="Date", yaxis_title="Forecast")
            #     st.plotly_chart(fig_forecast, use_container_width=True)
            # else:
            #     st.warning("Forecasting failed due to insufficient or invalid data.")

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

        avg_consumption_453 = plot_average_consumption(df_453)
        st.plotly_chart(avg_consumption_453, use_container_width=True)

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

            if forecast_df is not None:
                for (site, plant), group in forecast_df.groupby(['Site', 'Plant']):
                    fig = go.Figure(go.Scatter(
                        x=group['Pstng Date'], 
                        y=group['Forecast'], 
                        mode='lines', 
                        name=f"Forecast - {site}, {plant}"
                    ))

                    fig.update_layout(
                        title=f"Demand Forecast - Site {site}, Plant {plant}",
                        xaxis_title="Date",
                        yaxis_title="Forecast",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No forecast data available.")

            # if forecast_df is not None:
            #     fig_forecast = go.Figure(go.Scatter(x=forecast_df['Pstng Date'], y=forecast_df['Forecast'], mode='lines', name=f"Forecast - {material_number}"))
            #     fig_forecast.update_layout(title=f"Forecast - Material {material_number}", xaxis_title="Date", yaxis_title="Forecast")
            #     st.plotly_chart(fig_forecast, use_container_width=True)
            # else:
            #     st.warning("Forecasting failed due to insufficient or invalid data.")

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

        # Plot the trend
        fig_consumption = sp.make_subplots(rows=1, cols=2, subplot_titles=("Material Group 260 Average Consumption", "Material Group 453 Average Consumption"))

        # Plot for Material Group 260
        avg_consumption_260  = plot_average_consumption(df_260)
        for trace in avg_consumption_260.data:
            fig_consumption.add_trace(trace, row=1, col=1)

        # Plot for Material Group 453
        avg_consumption_453 = plot_average_consumption(df_453)
        for trace in avg_consumption_453.data:
            fig_consumption.add_trace(trace, row=1, col=2)

        fig_consumption.update_layout(title="Average Consumption Comparison", showlegend=True)
        st.plotly_chart(fig_consumption, use_container_width=True)


        # Forecast demand for both material groups
        forecast_periods = st.slider("Select Forecast Periods", 1, 24, 12)  # Select number of forecast periods
        
        forecast_260 = forecast_demand(df_260, periods=forecast_periods)
        forecast_453 = forecast_demand(df_453, periods=forecast_periods)

        # Create subplots (1 row, 2 columns)
        fig_forecast = sp.make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Material Group 260 Forecast", "Material Group 453 Forecast")
        )

        # Add forecasts to subplots
        add_forecast_traces(fig_forecast, forecast_260, col=1)
        add_forecast_traces(fig_forecast, forecast_453, col=2)

        # Update layout and display the chart if there is at least one valid forecast
        if (forecast_260 is not None and not forecast_260.empty) or (forecast_453 is not None and not forecast_453.empty):
            fig_forecast.update_layout(
                title="Demand Forecast Comparison",
                xaxis_title="Date",
                yaxis_title="Forecast",
                template="plotly_white",
                showlegend=True
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        # # Create subplots for Forecast
        # fig_forecast = sp.make_subplots(rows=1, cols=2, subplot_titles=("Material Group 260 Forecast", "Material Group 453 Forecast"))

        # # Forecast for Material Group 260
        # if forecast_260 is not None and not forecast_260.empty:
        #     fig_forecast.add_trace(
        #         go.Scatter(x=forecast_260['Pstng Date'], y=forecast_260['Forecast'], mode='lines', name="Forecast - Group 260"),
        #         row=1, col=1
        #     )
        # else:
        #     st.warning("Forecasting failed for Material Group 260 due to insufficient or invalid data.")
            

        # # Forecast for Material Group 453
        # if forecast_453 is not None and not forecast_453.empty:
        #     fig_forecast.add_trace(
        #         go.Scatter(x=forecast_453['Pstng Date'], y=forecast_453['Forecast'], mode='lines', name="Forecast - Group 453"),
        #         row=1, col=2
        #     )
        # else:
        #     st.warning("Forecasting failed for Material Group 453 due to insufficient or invalid data.")

        # # Update layout and display the chart if there is at least one valid forecast
        # if (forecast_260 is not None and not forecast_260.empty) or (forecast_453 is not None and not forecast_453.empty):
        #     fig_forecast.update_layout(title="Demand Forecast Comparison", showlegend=False)
        #     st.plotly_chart(fig_forecast, use_container_width=True)

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
                
                if forecast_260 is not None:
                    fig_forecast_260 = go.Figure(go.Scatter(x=forecast_260['Pstng Date'], y=forecast_260['Forecast'], mode='lines', name=f"Forecast - {material_260}"))
                    fig_forecast_260.update_layout(title=f"Forecast - Material {material_260}", xaxis_title="Date", yaxis_title="Forecast")
                    st.plotly_chart(fig_forecast_260, use_container_width=True)
                else:
                    st.warning("Forecasting failed due to insufficient or invalid data.")

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
                    if forecast_453 is not None:
                        fig_forecast_453 = go.Figure(go.Scatter(x=forecast_453['Pstng Date'], y=forecast_453['Forecast'], mode='lines', name=f"Forecast - {material_453}"))
                        fig_forecast_453.update_layout(title=f"Forecast - Material {material_453}", xaxis_title="Date", yaxis_title="Forecast")
                        st.plotly_chart(fig_forecast_453, use_container_width=True)
                    else:
                        st.warning("Forecasting failed due to insufficient or invalid data.")

                # Monte Carlo Simulation
                simulation_453 = monte_carlo_simulation(df_material_453)
                fig_sim_453 = go.Figure(go.Histogram(x=simulation_453, nbinsx=50, marker_color='green'))
                fig_sim_453.update_layout(title=f"Monte Carlo Simulation - Material {material_453}", xaxis_title="Simulated Demand", yaxis_title="Frequency")
                st.plotly_chart(fig_sim_453, use_container_width=True)


elif tabs == "Order Placement Analysis":
    st.title("Order Placement Analysis")

     # Add selection for Material Group(s)
    group_selection = st.radio(
        "Select Material Group(s) to Analyze:",
        ("Material Group 260", "Material Group 453", "Both")
    )

    # Upload files based on the selection
    file_260 = None
    file_453 = None
    if group_selection == "Material Group 260" or group_selection == "Both":
        file_260 = st.file_uploader("Upload Order Placement Dataset for Material Group 260", type=["xlsx"])

    if group_selection == "Material Group 453" or group_selection == "Both":
        file_453 = st.file_uploader("Upload Order Placement Dataset for Material Group 453", type=["xlsx"])

    if group_selection == "Material Group 260" and file_260:
        st.subheader("Material Group 260 Analysis")

        df_260 = preprocess_order_data(file_260)
        # 1. **Order Quantity Trends by Plant**
        st.subheader("Order Quantity Trends by Plant")
        data_grouped = df_260.groupby('Plant')['Order Quantity'].sum().reset_index()
        fig = px.bar(data_grouped, x='Plant', y='Order Quantity', title="Total Order Quantity by Plant")
        st.plotly_chart(fig)

        # 2. **Supplier Performance**
        st.subheader("Supplier Performance")
        data_grouped = df_260.groupby('Supplier')['Order Quantity'].sum().reset_index()
        fig = px.bar(data_grouped, x='Supplier', y='Order Quantity', title="Total Order Quantity by Supplier")
        st.plotly_chart(fig)

        # 3. **Material Number Consumption**
        st.subheader("Material Number Order")
        data_grouped = df_260.groupby('Material Number')['Order Quantity'].sum().reset_index()
        fig = px.bar(data_grouped, x='Material Number', y='Order Quantity', title="Order Quantity by Material Number")
        st.plotly_chart(fig)

        # 4. **Order Fulfillment Status by Plant (Ordered vs Delivered)**
        st.subheader("Order Fulfillment Status by Plant")
        # Grouping by Plant to get Ordered and Delivered quantities
        data_grouped = df_260.groupby('Plant').agg({
            'Order Quantity': 'sum',
            'Delivery Status': 'sum'
        }).reset_index()

        # Create a side-by-side bar chart comparing Order Quantity and Delivered Quantity (Delivery Status)
        fig = go.Figure(data=[
            go.Bar(
                x=data_grouped['Plant'],
                y=data_grouped['Order Quantity'],
                name='Ordered Quantity',
                marker_color='lightblue'
            ),
            go.Bar(
                x=data_grouped['Plant'],
                y=data_grouped['Delivery Status'],
                name='Delivered Quantity',
                marker_color='lightgreen'
            )
        ])

        # Update layout for better clarity
        fig.update_layout(
            title="Ordered Quantity vs Delivered Quantity by Plant",
            barmode='group',
            xaxis_title="Plant",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )

        st.plotly_chart(fig)


        # 5. **Order Fulfillment Status by Material Number (Ordered vs Delivered)**
        st.subheader("Order Fulfillment Status by Material Number")
        # Grouping by Material Number to get Ordered and Delivered quantities
        data_grouped_material = df_260.groupby('Material Number').agg({
            'Order Quantity': 'sum',
            'Delivery Status': 'sum'
        }).reset_index()

        # Create a side-by-side bar chart comparing Order Quantity and Delivered Quantity for Material Number
        fig_material = go.Figure(data=[
            go.Bar(
                x=data_grouped_material['Material Number'],
                y=data_grouped_material['Order Quantity'],
                name='Ordered Quantity',
                marker_color='lightblue'
            ),
            go.Bar(
                x=data_grouped_material['Material Number'],
                y=data_grouped_material['Delivery Status'],
                name='Delivered Quantity',
                marker_color='lightgreen'
            )
        ])

        # Update layout for better clarity
        fig_material.update_layout(
            title="Ordered Quantity vs Delivered Quantity by Material Number",
            barmode='group',
            xaxis_title="Material Number",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )

        st.plotly_chart(fig_material)

        # 6. **Order Fulfillment Status by Supplier (Ordered vs Delivered)**
        st.subheader("Order Fulfillment Status by Supplier")
        # Grouping by Supplier to get Ordered and Delivered quantities
        data_grouped_supplier = df_260.groupby('Supplier').agg({
            'Order Quantity': 'sum',
            'Delivery Status': 'sum'
        }).reset_index()

        # Create a side-by-side bar chart comparing Order Quantity and Delivered Quantity for Supplier
        fig_supplier = go.Figure(data=[
            go.Bar(
                x=data_grouped_supplier['Supplier'],
                y=data_grouped_supplier['Order Quantity'],
                name='Ordered Quantity',
                marker_color='lightblue'
            ),
            go.Bar(
                x=data_grouped_supplier['Supplier'],
                y=data_grouped_supplier['Delivery Status'],
                name='Delivered Quantity',
                marker_color='lightgreen'
            )
        ])

        # Update layout for better clarity
        fig_supplier.update_layout(
            title="Ordered Quantity vs Delivered Quantity by Supplier",
            barmode='group',
            xaxis_title="Supplier",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )

        st.plotly_chart(fig_supplier)

        # 7. Order Fulfillment Status by Vendor Number (Ordered vs Delivered)
        st.subheader("Order Fulfillment Status by Vendor Number")
        # Grouping by Vendor Number to get Ordered and Delivered quantities
        data_grouped_vendor = df_260.groupby('Vendor Number').agg({
            'Order Quantity': 'sum',
            'Delivery Status': 'sum'
        }).reset_index()

        # Create a side-by-side bar chart comparing Order Quantity and Delivered Quantity for Vendor Number
        fig_vendor = go.Figure(data=[
            go.Bar(
                x=data_grouped_vendor['Vendor Number'],
                y=data_grouped_vendor['Order Quantity'],
                name='Ordered Quantity',
                marker_color='lightblue'
            ),
            go.Bar(
                x=data_grouped_vendor['Vendor Number'],
                y=data_grouped_vendor['Delivery Status'],
                name='Delivered Quantity',
                marker_color='lightgreen'
            )
        ])

        # Update layout for better clarity
        fig_vendor.update_layout(
            title="Ordered Quantity vs Delivered Quantity by Vendor Number",
            barmode='group',
            xaxis_title="Vendor Number",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )

        st.plotly_chart(fig_vendor)


        # Material Group 453 Analysis (Similar to Material Group 260 Analysis)
    if group_selection == "Material Group 453" and file_453:
        st.subheader("Material Group 453 Analysis")

        df_453 = preprocess_order_data(file_453)

        # 1. **Order Quantity Trends by Plant**
        st.subheader("Order Quantity Trends by Plant")
        data_grouped = df_453.groupby('Plant')['Order Quantity'].sum().reset_index()
        fig = px.bar(data_grouped, x='Plant', y='Order Quantity', title="Total Order Quantity by Plant")
        st.plotly_chart(fig)

        # 2. **Supplier Performance**
        st.subheader("Supplier Performance")
        data_grouped = df_453.groupby('Supplier')['Order Quantity'].sum().reset_index()
        fig = px.bar(data_grouped, x='Supplier', y='Order Quantity', title="Total Order Quantity by Supplier")
        st.plotly_chart(fig)

        # 3. **Material Number Consumption**
        st.subheader("Material Number Order")
        data_grouped = df_453.groupby('Material Number')['Order Quantity'].sum().reset_index()
        fig = px.bar(data_grouped, x='Material Number', y='Order Quantity', title="Order Quantity by Material Number")
        st.plotly_chart(fig)

        # 4. **Order Fulfillment Status by Plant (Ordered vs Delivered)**
        st.subheader("Order Fulfillment Status by Plant")
        data_grouped = df_453.groupby('Plant').agg({
            'Order Quantity': 'sum',
            'Delivery Status': 'sum'
        }).reset_index()

        fig = go.Figure(data=[
            go.Bar(x=data_grouped['Plant'], y=data_grouped['Order Quantity'], name='Ordered Quantity', marker_color='lightblue'),
            go.Bar(x=data_grouped['Plant'], y=data_grouped['Delivery Status'], name='Delivered Quantity', marker_color='lightgreen')
        ])
        fig.update_layout(
            title="Ordered Quantity vs Delivered Quantity by Plant",
            barmode='group',
            xaxis_title="Plant",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )
        st.plotly_chart(fig)

        # 5. **Order Fulfillment Status by Material Number (Ordered vs Delivered)**
        st.subheader("Order Fulfillment Status by Material Number")
        data_grouped_material = df_453.groupby('Material Number').agg({
            'Order Quantity': 'sum',
            'Delivery Status': 'sum'
        }).reset_index()

        fig_material = go.Figure(data=[
            go.Bar(x=data_grouped_material['Material Number'], y=data_grouped_material['Order Quantity'], name='Ordered Quantity', marker_color='lightblue'),
            go.Bar(x=data_grouped_material['Material Number'], y=data_grouped_material['Delivery Status'], name='Delivered Quantity', marker_color='lightgreen')
        ])
        fig_material.update_layout(
            title="Ordered Quantity vs Delivered Quantity by Material Number",
            barmode='group',
            xaxis_title="Material Number",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )
        st.plotly_chart(fig_material)

        # 6. **Order Fulfillment Status by Supplier (Ordered vs Delivered)**
        st.subheader("Order Fulfillment Status by Supplier")
        data_grouped_supplier = df_453.groupby('Supplier').agg({
            'Order Quantity': 'sum',
            'Delivery Status': 'sum'
        }).reset_index()

        fig_supplier = go.Figure(data=[
            go.Bar(x=data_grouped_supplier['Supplier'], y=data_grouped_supplier['Order Quantity'], name='Ordered Quantity', marker_color='lightblue'),
            go.Bar(x=data_grouped_supplier['Supplier'], y=data_grouped_supplier['Delivery Status'], name='Delivered Quantity', marker_color='lightgreen')
        ])
        fig_supplier.update_layout(
            title="Ordered Quantity vs Delivered Quantity by Supplier",
            barmode='group',
            xaxis_title="Supplier",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )
        st.plotly_chart(fig_supplier)

        # 7. **Order Fulfillment Status by Vendor Number (Ordered vs Delivered)**
        st.subheader("Order Fulfillment Status by Vendor Number")
        data_grouped_vendor = df_453.groupby('Vendor Number').agg({
            'Order Quantity': 'sum',
            'Delivery Status': 'sum'
        }).reset_index()

        fig_vendor = go.Figure(data=[
            go.Bar(x=data_grouped_vendor['Vendor Number'], y=data_grouped_vendor['Order Quantity'], name='Ordered Quantity', marker_color='lightblue'),
            go.Bar(x=data_grouped_vendor['Vendor Number'], y=data_grouped_vendor['Delivery Status'], name='Delivered Quantity', marker_color='lightgreen')
        ])
        fig_vendor.update_layout(
            title="Ordered Quantity vs Delivered Quantity by Vendor Number",
            barmode='group',
            xaxis_title="Vendor Number",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )
        st.plotly_chart(fig_vendor)



    elif group_selection == "Both" and file_260 and file_453:
        st.subheader("Material Group 260 and 453 Comparison")

        # Load data for both material groups
        df_260 = preprocess_order_data(file_260)
        df_453 = preprocess_order_data(file_453)

        # 1. **Order Quantity Trends by Plant - Comparison**
        fig_order_quantity = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Material Group 260", "Material Group 453"),
            horizontal_spacing=0.1
        )

        data_grouped_260 = df_260.groupby('Plant')['Order Quantity'].sum().reset_index()
        data_grouped_453 = df_453.groupby('Plant')['Order Quantity'].sum().reset_index()

        # Plot for Material Group 260 (left column)
        fig_order_quantity.add_trace(
            go.Bar(x=data_grouped_260['Plant'], y=data_grouped_260['Order Quantity'], name="Material Group 260", marker_color='lightblue'),
            row=1, col=1
        )

        # Plot for Material Group 453 (right column)
        fig_order_quantity.add_trace(
            go.Bar(x=data_grouped_453['Plant'], y=data_grouped_453['Order Quantity'], name="Material Group 453", marker_color='lightgreen'),
            row=1, col=2
        )

        # Update layout for Order Quantity by Plant comparison
        fig_order_quantity.update_layout(
            title="Order Quantity Trends by Plant Comparison",
            barmode='group',
            xaxis_title="Plant",
            yaxis_title="Order Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark",
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig_order_quantity)

        # 2. **Supplier Performance - Comparison**
        fig_supplier_performance = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Material Group 260", "Material Group 453"),
            horizontal_spacing=0.1
        )

        data_grouped_260_supplier = df_260.groupby('Supplier')['Order Quantity'].sum().reset_index()
        data_grouped_453_supplier = df_453.groupby('Supplier')['Order Quantity'].sum().reset_index()

        # Plot for Material Group 260 (left column)
        fig_supplier_performance.add_trace(
            go.Bar(x=data_grouped_260_supplier['Supplier'], y=data_grouped_260_supplier['Order Quantity'], name="Material Group 260", marker_color='lightblue'),
            row=1, col=1
        )

        # Plot for Material Group 453 (right column)
        fig_supplier_performance.add_trace(
            go.Bar(x=data_grouped_453_supplier['Supplier'], y=data_grouped_453_supplier['Order Quantity'], name="Material Group 453", marker_color='lightgreen'),
            row=1, col=2
        )

        # Update layout for Supplier Performance comparison
        fig_supplier_performance.update_layout(
            title="Supplier Performance Comparison",
            barmode='group',
            xaxis_title="Supplier",
            yaxis_title="Order Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark",
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig_supplier_performance)

        # 3. **Material Number Comparison**
        fig_material_number = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Material Group 260", "Material Group 453"),
            horizontal_spacing=0.1
        )

        data_grouped_260_material = df_260.groupby('Material Number')['Order Quantity'].sum().reset_index()
        data_grouped_453_material = df_453.groupby('Material Number')['Order Quantity'].sum().reset_index()

        # Plot for Material Group 260 (left column)
        fig_material_number.add_trace(
            go.Bar(x=data_grouped_260_material['Material Number'], y=data_grouped_260_material['Order Quantity'], name="Material Group 260", marker_color='lightblue'),
            row=1, col=1
        )

        # Plot for Material Group 453 (right column)
        fig_material_number.add_trace(
            go.Bar(x=data_grouped_453_material['Material Number'], y=data_grouped_453_material['Order Quantity'], name="Material Group 453", marker_color='lightgreen'),
            row=1, col=2
        )

        # Update layout for Material Number comparison
        fig_material_number.update_layout(
            title="Material Number Order Comparison",
            barmode='group',
            xaxis_title="Material Number",
            yaxis_title="Order Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark",
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig_material_number)

        # 4. **Order Fulfillment Status by Plant - Comparison**
        fig_fulfillment_status = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Material Group 260", "Material Group 453"),
            horizontal_spacing=0.1
        )

        data_grouped_260_fulfillment = df_260.groupby('Plant').agg({'Order Quantity': 'sum', 'Delivery Status': 'sum'}).reset_index()
        data_grouped_453_fulfillment = df_453.groupby('Plant').agg({'Order Quantity': 'sum', 'Delivery Status': 'sum'}).reset_index()

        # Plot for Material Group 260 (left column)
        fig_fulfillment_status.add_trace(
            go.Bar(x=data_grouped_260_fulfillment['Plant'], y=data_grouped_260_fulfillment['Order Quantity'], name="Material Group 260 - Ordered", marker_color='lightblue'),
            row=1, col=1
        )
        fig_fulfillment_status.add_trace(
            go.Bar(x=data_grouped_260_fulfillment['Plant'], y=data_grouped_260_fulfillment['Delivery Status'], name="Material Group 260 - Delivered", marker_color='lightgreen'),
            row=1, col=1
        )

        # Plot for Material Group 453 (right column)
        fig_fulfillment_status.add_trace(
            go.Bar(x=data_grouped_453_fulfillment['Plant'], y=data_grouped_453_fulfillment['Order Quantity'], name="Material Group 453 - Ordered", marker_color='lightblue'),
            row=1, col=2
        )
        fig_fulfillment_status.add_trace(
            go.Bar(x=data_grouped_453_fulfillment['Plant'], y=data_grouped_453_fulfillment['Delivery Status'], name="Material Group 453 - Delivered", marker_color='lightgreen'),
            row=1, col=2
        )

        # Update layout for Order Fulfillment Status comparison
        fig_fulfillment_status.update_layout(
            title="Order Fulfillment Status by Plant Comparison",
            barmode='group',
            xaxis_title="Plant",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark",
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig_fulfillment_status)

        # 5. **Order Fulfillment Status by Material Number - Comparison**
        fig_fulfillment_status_material = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Material Group 260", "Material Group 453"),
            horizontal_spacing=0.1
        )

        data_grouped_260_material_fulfillment = df_260.groupby('Material Number').agg({'Order Quantity': 'sum', 'Delivery Status': 'sum'}).reset_index()
        data_grouped_453_material_fulfillment = df_453.groupby('Material Number').agg({'Order Quantity': 'sum', 'Delivery Status': 'sum'}).reset_index()

        # Plot for Material Group 260 (left column)
        fig_fulfillment_status_material.add_trace(
            go.Bar(x=data_grouped_260_material_fulfillment['Material Number'], y=data_grouped_260_material_fulfillment['Order Quantity'], name="Material Group 260 - Ordered", marker_color='lightblue'),
            row=1, col=1
        )
        fig_fulfillment_status_material.add_trace(
            go.Bar(x=data_grouped_260_material_fulfillment['Material Number'], y=data_grouped_260_material_fulfillment['Delivery Status'], name="Material Group 260 - Delivered", marker_color='lightgreen'),
            row=1, col=1
        )

        # Plot for Material Group 453 (right column)
        fig_fulfillment_status_material.add_trace(
            go.Bar(x=data_grouped_453_material_fulfillment['Material Number'], y=data_grouped_453_material_fulfillment['Order Quantity'], name="Material Group 453 - Ordered", marker_color='lightblue'),
            row=1, col=2
        )
        fig_fulfillment_status_material.add_trace(
            go.Bar(x=data_grouped_453_material_fulfillment['Material Number'], y=data_grouped_453_material_fulfillment['Delivery Status'], name="Material Group 453 - Delivered", marker_color='lightgreen'),
            row=1, col=2
        )

        # Update layout for Order Fulfillment Status by Material Number comparison
        fig_fulfillment_status_material.update_layout(
            title="Order Fulfillment Status by Material Number Comparison",
            barmode='group',
            xaxis_title="Material Number",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark",
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig_fulfillment_status_material)

        # 6. **Order Fulfillment Status by Supplier - Comparison**
        fig_fulfillment_status_supplier = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Material Group 260", "Material Group 453"),
            horizontal_spacing=0.1
        )

        data_grouped_260_supplier_fulfillment = df_260.groupby('Supplier').agg({'Order Quantity': 'sum', 'Delivery Status': 'sum'}).reset_index()
        data_grouped_453_supplier_fulfillment = df_453.groupby('Supplier').agg({'Order Quantity': 'sum', 'Delivery Status': 'sum'}).reset_index()

        # Plot for Material Group 260 (left column)
        fig_fulfillment_status_supplier.add_trace(
            go.Bar(x=data_grouped_260_supplier_fulfillment['Supplier'], y=data_grouped_260_supplier_fulfillment['Order Quantity'], name="Material Group 260 - Ordered", marker_color='lightblue'),
            row=1, col=1
        )
        fig_fulfillment_status_supplier.add_trace(
            go.Bar(x=data_grouped_260_supplier_fulfillment['Supplier'], y=data_grouped_260_supplier_fulfillment['Delivery Status'], name="Material Group 260 - Delivered", marker_color='lightgreen'),
            row=1, col=1
        )

        # Plot for Material Group 453 (right column)
        fig_fulfillment_status_supplier.add_trace(
            go.Bar(x=data_grouped_453_supplier_fulfillment['Supplier'], y=data_grouped_453_supplier_fulfillment['Order Quantity'], name="Material Group 453 - Ordered", marker_color='lightblue'),
            row=1, col=2
        )
        fig_fulfillment_status_supplier.add_trace(
            go.Bar(x=data_grouped_453_supplier_fulfillment['Supplier'], y=data_grouped_453_supplier_fulfillment['Delivery Status'], name="Material Group 453 - Delivered", marker_color='lightgreen'),
            row=1, col=2
        )

        # Update layout for Order Fulfillment Status by Supplier comparison
        fig_fulfillment_status_supplier.update_layout(
            title="Order Fulfillment Status by Supplier Comparison",
            barmode='group',
            xaxis_title="Supplier",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark",
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig_fulfillment_status_supplier)

        # 7. **Order Fulfillment Status by Vendor Number - Comparison**
        fig_fulfillment_status_vendor = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Material Group 260", "Material Group 453"),
            horizontal_spacing=0.1
        )

        data_grouped_260_vendor_fulfillment = df_260.groupby('Vendor Number').agg({'Order Quantity': 'sum', 'Delivery Status': 'sum'}).reset_index()
        data_grouped_453_vendor_fulfillment = df_453.groupby('Vendor Number').agg({'Order Quantity': 'sum', 'Delivery Status': 'sum'}).reset_index()

        # Plot for Material Group 260 (left column)
        fig_fulfillment_status_vendor.add_trace(
            go.Bar(x=data_grouped_260_vendor_fulfillment['Vendor Number'], y=data_grouped_260_vendor_fulfillment['Order Quantity'], name="Material Group 260 - Ordered", marker_color='lightblue'),
            row=1, col=1
        )
        fig_fulfillment_status_vendor.add_trace(
            go.Bar(x=data_grouped_260_vendor_fulfillment['Vendor Number'], y=data_grouped_260_vendor_fulfillment['Delivery Status'], name="Material Group 260 - Delivered", marker_color='lightgreen'),
            row=1, col=1
        )

        # Plot for Material Group 453 (right column)
        fig_fulfillment_status_vendor.add_trace(
            go.Bar(x=data_grouped_453_vendor_fulfillment['Vendor Number'], y=data_grouped_453_vendor_fulfillment['Order Quantity'], name="Material Group 453 - Ordered", marker_color='lightblue'),
            row=1, col=2
        )
        fig_fulfillment_status_vendor.add_trace(
            go.Bar(x=data_grouped_453_vendor_fulfillment['Vendor Number'], y=data_grouped_453_vendor_fulfillment['Delivery Status'], name="Material Group 453 - Delivered", marker_color='lightgreen'),
            row=1, col=2
        )

        # Update layout for Order Fulfillment Status by Vendor comparison
        fig_fulfillment_status_vendor.update_layout(
            title="Order Fulfillment Status by Vendor Number Comparison",
            barmode='group',
            xaxis_title="Vendor Number",
            yaxis_title="Quantity",
            xaxis_tickangle=-45,
            template="plotly_dark",
            showlegend=True,
            height=500
        )

        st.plotly_chart(fig_fulfillment_status_vendor)




elif tabs == "Goods Receipt Analysis":
    st.title("Goods Receipt Analysis")

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

        st.write("## Dataset Overview")
        st.dataframe(df_260.head())

        # Convert date columns
        df_260["Pstng Date"] = pd.to_datetime(df_260["Pstng Date"], errors='coerce')
        df_260["SLED/BBD"] = pd.to_datetime(df_260["SLED/BBD"], errors='coerce')

        # --- 1. Time Series Analysis ---
        st.subheader("Time Series Analysis: Quantity Received Over Time")
        time_series = df_260.groupby("Pstng Date")["Quantity"].sum().reset_index()
        fig1 = px.line(time_series, x="Pstng Date", y="Quantity", title="Quantity Received Over Time")
        st.plotly_chart(fig1)

        # --- 2. Quantity Distribution ---
        st.subheader("Quantity Distribution")
        fig2 = px.histogram(df_260, x="Quantity", nbins=30, marginal="box", title="Quantity Distribution")
        st.plotly_chart(fig2)

        # --- 3. Top 10 Materials ---
        st.subheader("Top 10 Materials by Quantity Received")
        top_materials = df_260.groupby("Material Number")["Quantity"].sum().nlargest(10).reset_index()
        fig3 = px.bar(top_materials, x="Material Number", y="Quantity", title="Top 10 Materials")
        st.plotly_chart(fig3)

        # --- 4. Vendor Analysis ---
        st.subheader("Top Vendors Supplying the Most Goods")
        top_vendors = df_260.groupby("Vendor Number")["Quantity"].sum().nlargest(10).reset_index()
        fig4 = px.bar(top_vendors, x="Vendor Number", y="Quantity", title="Top 10 Vendors")
        st.plotly_chart(fig4)

        # --- 5. Plant & Site Analysis ---
        st.subheader("Quantity Received Per Plant")
        plant_quantity = df_260.groupby("Plant")["Quantity"].sum().reset_index()
        fig5 = px.bar(plant_quantity, x="Plant", y="Quantity", title="Quantity Received Per Plant")
        st.plotly_chart(fig5)

        st.subheader("Quantity Received Per Site")
        site_quantity = df_260.groupby("Site")["Quantity"].sum().reset_index()
        fig6 = px.bar(site_quantity, x="Site", y="Quantity", title="Quantity Received Per Site")
        st.plotly_chart(fig6)

        st.subheader("Quantity Received Per Batch")
        batch_quantity = df_260.groupby("Batch")["Quantity"].sum().reset_index()
        fig7 = px.bar(batch_quantity, x="Batch", y="Quantity", title="Quantity Received Per Batch")
        st.plotly_chart(fig7)

        # Allow user to choose Material Number
        material_numbers = df_260["Material Number"].unique()
        material_selection = st.selectbox("Select a Material Number for Further Analysis", material_numbers)

        # Filter the data based on selected material number
        df_material = df_260[df_260["Material Number"] == material_selection]

        # --- Material Level Analysis ---
        # --- 7. Material-Specific Time Series ---
        st.subheader(f"Time Series Analysis for Material Number {material_selection}")
        material_time_series = df_material.groupby("Pstng Date")["Quantity"].sum().reset_index()
        fig7 = px.line(material_time_series, x="Pstng Date", y="Quantity", title=f"Quantity Received Over Time for Material {material_selection}")
        st.plotly_chart(fig7)

        # --- 8. Material-Specific Batch Analysis ---
        st.subheader(f"Material-Specific Batch Analysis for Material {material_selection}")
        material_batch_analysis = df_material.groupby("Batch")["Quantity"].sum().reset_index()
        fig8 = px.bar(material_batch_analysis, x="Batch", y="Quantity", title=f"Batch Analysis for Material {material_selection}")
        st.plotly_chart(fig8)

        # --- 9. Material-Specific Vendor Analysis ---
        st.subheader(f"Material-Specific Vendor Analysis for Material {material_selection}")
        material_vendor_quantity = df_material.groupby("Vendor Number")["Quantity"].sum().reset_index()
        fig9 = px.bar(material_vendor_quantity, x="Vendor Number", y="Quantity", title=f"Vendor Performance for Material {material_selection}")
        st.plotly_chart(fig9)

        # --- 10. SLED/BBD vs Quantity Analysis for Material ---
        st.subheader(f"SLED/BBD vs Quantity Analysis for Material {material_selection}")
        fig10 = px.scatter(df_material, x="SLED/BBD", y="Quantity", title=f"SLED/BBD vs Quantity for Material {material_selection}")
        st.plotly_chart(fig10)

        # --- 11. Days to Expiry Distribution for Material ---
        df_material['Days_to_Expiry'] = (df_material["SLED/BBD"] - df_material["Pstng Date"]).dt.days
        st.subheader(f"Days to Expiry Distribution for Material {material_selection}")
        fig11 = px.histogram(df_material, x="Days_to_Expiry", nbins=30, title=f"Days to Expiry Distribution for Material {material_selection}")
        st.plotly_chart(fig11)

        # --- 12. Vendor Delivery Time Analysis for Material ---
        df_material['Days_to_Delivery'] = (df_material["SLED/BBD"] - df_material["Pstng Date"]).dt.days
        st.subheader(f"Vendor Delivery Time Analysis for Material {material_selection}")
        fig12 = px.box(df_material, x="Vendor Number", y="Days_to_Delivery", title=f"Vendor Delivery Time Efficiency for Material {material_selection}")
        st.plotly_chart(fig12)


    if group_selection == "Material Group 453" and file_453:
        st.subheader("Material Group 453 Analysis")
        df_453 = load_data(file_453)

        st.write("## Dataset Overview")
        st.dataframe(df_453.head())

        # Convert date columns
        df_453["Pstng Date"] = pd.to_datetime(df_453["Pstng Date"], errors='coerce')
        df_453["SLED/BBD"] = pd.to_datetime(df_453["SLED/BBD"], errors='coerce')

        # --- 1. Time Series Analysis ---
        st.subheader("Time Series Analysis: Quantity Received Over Time")
        time_series = df_453.groupby("Pstng Date")["Quantity"].sum().reset_index()
        fig1 = px.line(time_series, x="Pstng Date", y="Quantity", title="Quantity Received Over Time")
        st.plotly_chart(fig1)

        # --- 2. Quantity Distribution ---
        st.subheader("Quantity Distribution")
        fig2 = px.histogram(df_453, x="Quantity", nbins=30, marginal="box", title="Quantity Distribution")
        st.plotly_chart(fig2)

        # --- 3. Top 10 Materials ---
        st.subheader("Top 10 Materials by Quantity Received")
        top_materials = df_453.groupby("Material Number")["Quantity"].sum().nlargest(10).reset_index()
        fig3 = px.bar(top_materials, x="Material Number", y="Quantity", title="Top 10 Materials")
        st.plotly_chart(fig3)

        # --- 4. Vendor Analysis ---
        st.subheader("Top Vendors Supplying the Most Goods")
        top_vendors = df_453.groupby("Vendor Number")["Quantity"].sum().nlargest(10).reset_index()
        fig4 = px.bar(top_vendors, x="Vendor Number", y="Quantity", title="Top 10 Vendors")
        st.plotly_chart(fig4)

        # --- 5. Plant & Site Analysis ---
        st.subheader("Quantity Received Per Plant")
        plant_quantity = df_453.groupby("Plant")["Quantity"].sum().reset_index()
        fig5 = px.bar(plant_quantity, x="Plant", y="Quantity", title="Quantity Received Per Plant")
        st.plotly_chart(fig5)

        st.subheader("Quantity Received Per Site")
        site_quantity = df_453.groupby("Site")["Quantity"].sum().reset_index()
        fig6 = px.bar(site_quantity, x="Site", y="Quantity", title="Quantity Received Per Site")
        st.plotly_chart(fig6)

        st.subheader("Quantity Received Per Batch")
        batch_quantity = df_453.groupby("Batch")["Quantity"].sum().reset_index()
        fig7 = px.bar(batch_quantity, x="Batch", y="Quantity", title="Quantity Received Per Batch")
        st.plotly_chart(fig7)

        # Allow user to choose Material Number
        material_numbers = df_453["Material Number"].unique()
        material_selection = st.selectbox("Select a Material Number for Further Analysis", material_numbers)

        # Filter the data based on selected material number
        df_material = df_453[df_453["Material Number"] == material_selection]

        # --- Material Level Analysis ---
        # --- 7. Material-Specific Time Series ---
        st.subheader(f"Time Series Analysis for Material Number {material_selection}")
        material_time_series = df_material.groupby("Pstng Date")["Quantity"].sum().reset_index()
        fig7 = px.line(material_time_series, x="Pstng Date", y="Quantity", title=f"Quantity Received Over Time for Material {material_selection}")
        st.plotly_chart(fig7)

        # --- 8. Material-Specific Batch Analysis ---
        st.subheader(f"Material-Specific Batch Analysis for Material {material_selection}")
        material_batch_analysis = df_material.groupby("Batch")["Quantity"].sum().reset_index()
        fig8 = px.bar(material_batch_analysis, x="Batch", y="Quantity", title=f"Batch Analysis for Material {material_selection}")
        st.plotly_chart(fig8)

        # --- 9. Material-Specific Vendor Analysis ---
        st.subheader(f"Material-Specific Vendor Analysis for Material {material_selection}")
        material_vendor_quantity = df_material.groupby("Vendor Number")["Quantity"].sum().reset_index()
        fig9 = px.bar(material_vendor_quantity, x="Vendor Number", y="Quantity", title=f"Vendor Performance for Material {material_selection}")
        st.plotly_chart(fig9)

        # --- 10. SLED/BBD vs Quantity Analysis for Material ---
        st.subheader(f"SLED/BBD vs Quantity Analysis for Material {material_selection}")
        fig10 = px.scatter(df_material, x="SLED/BBD", y="Quantity", title=f"SLED/BBD vs Quantity for Material {material_selection}")
        st.plotly_chart(fig10)

        # --- 11. Days to Expiry Distribution for Material ---
        df_material['Days_to_Expiry'] = (df_material["SLED/BBD"] - df_material["Pstng Date"]).dt.days
        st.subheader(f"Days to Expiry Distribution for Material {material_selection}")
        fig11 = px.histogram(df_material, x="Days_to_Expiry", nbins=30, title=f"Days to Expiry Distribution for Material {material_selection}")
        st.plotly_chart(fig11)

        # --- 12. Vendor Delivery Time Analysis for Material ---
        df_material['Days_to_Delivery'] = (df_material["SLED/BBD"] - df_material["Pstng Date"]).dt.days
        st.subheader(f"Vendor Delivery Time Analysis for Material {material_selection}")
        fig12 = px.box(df_material, x="Vendor Number", y="Days_to_Delivery", title=f"Vendor Delivery Time Efficiency for Material {material_selection}")
        st.plotly_chart(fig12)    

    elif group_selection == "Both" and file_260 and file_453:
        st.subheader("Material Group 260 and 453 Comparison")
        
        # Load data for both files
        df_260 = load_data(file_260)
        df_453 = load_data(file_453)

        # Convert date columns
        df_260["Pstng Date"] = pd.to_datetime(df_260["Pstng Date"], errors='coerce')
        df_260["SLED/BBD"] = pd.to_datetime(df_260["SLED/BBD"], errors='coerce')

        df_453["Pstng Date"] = pd.to_datetime(df_453["Pstng Date"], errors='coerce')
        df_453["SLED/BBD"] = pd.to_datetime(df_453["SLED/BBD"], errors='coerce')

        # Create two columns for side-by-side layout
        col1, col2 = st.columns(2)

        # Left column for Material Group 260 plots
        with col1:
            # --- 1. Time Series Analysis for 260 ---
            st.subheader("Material Group 260: Time Series Analysis")
            time_series_260 = df_260.groupby("Pstng Date")["Quantity"].sum().reset_index()
            fig1_260 = px.line(time_series_260, x="Pstng Date", y="Quantity", title="Quantity Received Over Time (Material Group 260)")
            st.plotly_chart(fig1_260)

            # --- 2. Quantity Distribution for 260 ---
            st.subheader("Material Group 260: Quantity Distribution")
            fig2_260 = px.histogram(df_260, x="Quantity", nbins=30, marginal="box", title="Quantity Distribution (Material Group 260)")
            st.plotly_chart(fig2_260)

            # --- 3. Top 10 Materials for 260 ---
            st.subheader("Material Group 260: Top 10 Materials by Quantity Received")
            top_materials_260 = df_260.groupby("Material Number")["Quantity"].sum().nlargest(10).reset_index()
            fig3_260 = px.bar(top_materials_260, x="Material Number", y="Quantity", title="Top 10 Materials (Material Group 260)")
            st.plotly_chart(fig3_260)

            # --- 4. Vendor Analysis for 260 ---
            st.subheader("Material Group 260: Top Vendors Supplying the Most Goods")
            top_vendors_260 = df_260.groupby("Vendor Number")["Quantity"].sum().nlargest(10).reset_index()
            fig4_260 = px.bar(top_vendors_260, x="Vendor Number", y="Quantity", title="Top 10 Vendors (Material Group 260)")
            st.plotly_chart(fig4_260)

            # Allow user to select a Material Number for Material Group 260
            material_numbers_260 = df_260["Material Number"].unique()
            material_selection_260 = st.selectbox("Select a Material Number for Material Group 260", material_numbers_260)

        # Right column for Material Group 453 plots
        with col2:
            # --- 1. Time Series Analysis for 453 ---
            st.subheader("Material Group 453: Time Series Analysis")
            time_series_453 = df_453.groupby("Pstng Date")["Quantity"].sum().reset_index()
            fig1_453 = px.line(time_series_453, x="Pstng Date", y="Quantity", title="Quantity Received Over Time (Material Group 453)")
            st.plotly_chart(fig1_453)

            # --- 2. Quantity Distribution for 453 ---
            st.subheader("Material Group 453: Quantity Distribution")
            fig2_453 = px.histogram(df_453, x="Quantity", nbins=30, marginal="box", title="Quantity Distribution (Material Group 453)")
            st.plotly_chart(fig2_453)

            # --- 3. Top 10 Materials for 453 ---
            st.subheader("Material Group 453: Top 10 Materials by Quantity Received")
            top_materials_453 = df_453.groupby("Material Number")["Quantity"].sum().nlargest(10).reset_index()
            fig3_453 = px.bar(top_materials_453, x="Material Number", y="Quantity", title="Top 10 Materials (Material Group 453)")
            st.plotly_chart(fig3_453)

            # --- 4. Vendor Analysis for 453 ---
            st.subheader("Material Group 453: Top Vendors Supplying the Most Goods")
            top_vendors_453 = df_453.groupby("Vendor Number")["Quantity"].sum().nlargest(10).reset_index()
            fig4_453 = px.bar(top_vendors_453, x="Vendor Number", y="Quantity", title="Top 10 Vendors (Material Group 453)")
            st.plotly_chart(fig4_453)

            # Allow user to select a Material Number for Material Group 453
            material_numbers_453 = df_453["Material Number"].unique()
            material_selection_453 = st.selectbox("Select a Material Number for Material Group 453", material_numbers_453)

        # Filter data based on the selected material numbers
        df_material_260 = df_260[df_260["Material Number"] == material_selection_260]
        df_material_453 = df_453[df_453["Material Number"] == material_selection_453]

        # Create columns for material-specific analysis (optional if needed)
        col3, col4 = st.columns(2)

        # Left column for Material Group 260 material-specific analysis
        with col3:
            # --- Material-Specific Time Series for 260 ---
            st.subheader(f"Material-Specific Time Series for Material {material_selection_260}")
            material_time_series_260 = df_material_260.groupby("Pstng Date")["Quantity"].sum().reset_index()
            fig5_260 = px.line(material_time_series_260, x="Pstng Date", y="Quantity", title=f"Quantity Received Over Time (Material {material_selection_260})")
            st.plotly_chart(fig5_260)

            # --- Material-Specific Batch Analysis for 260 ---
            st.subheader(f"Material-Specific Batch Analysis for Material {material_selection_260}")
            material_batch_analysis_260 = df_material_260.groupby("Batch")["Quantity"].sum().reset_index()
            fig6_260 = px.bar(material_batch_analysis_260, x="Batch", y="Quantity", title=f"Batch Analysis for Material {material_selection_260}")
            st.plotly_chart(fig6_260)

            # --- Material-Specific Vendor Analysis for 260 ---
            st.subheader(f"Material-Specific Vendor Analysis for Material {material_selection_260}")
            material_vendor_quantity_260 = df_material_260.groupby("Vendor Number")["Quantity"].sum().reset_index()
            fig7_260 = px.bar(material_vendor_quantity_260, x="Vendor Number", y="Quantity", title=f"Vendor Performance for Material {material_selection_260}")
            st.plotly_chart(fig7_260)

            # --- SLED/BBD vs Quantity Analysis for 260 ---
            st.subheader(f"SLED/BBD vs Quantity Analysis for Material {material_selection_260}")
            fig8_260 = px.scatter(df_material_260, x="SLED/BBD", y="Quantity", title=f"SLED/BBD vs Quantity for Material {material_selection_260}")
            st.plotly_chart(fig8_260)

            # --- Days to Expiry Distribution for 260 ---
            df_material_260['Days_to_Expiry_260'] = (df_material_260["SLED/BBD"] - df_material_260["Pstng Date"]).dt.days
            st.subheader(f"Days to Expiry Distribution for Material {material_selection_260}")
            fig9_260 = px.histogram(df_material_260, x="Days_to_Expiry_260", nbins=30, title=f"Days to Expiry Distribution for Material {material_selection_260}")
            st.plotly_chart(fig9_260)

            # --- Vendor Delivery Time Analysis for 260 ---
            df_material_260['Days_to_Delivery_260'] = (df_material_260["SLED/BBD"] - df_material_260["Pstng Date"]).dt.days
            st.subheader(f"Vendor Delivery Time Analysis for Material {material_selection_260}")
            fig10_260 = px.box(df_material_260, x="Vendor Number", y="Days_to_Delivery_260", title=f"Vendor Delivery Time Efficiency for Material {material_selection_260}")
            st.plotly_chart(fig10_260)

        # Right column for Material Group 453 material-specific analysis
        with col4:
            # --- Material-Specific Time Series for 453 ---
            st.subheader(f"Material-Specific Time Series for Material {material_selection_453}")
            material_time_series_453 = df_material_453.groupby("Pstng Date")["Quantity"].sum().reset_index()
            fig5_453 = px.line(material_time_series_453, x="Pstng Date", y="Quantity", title=f"Quantity Received Over Time (Material {material_selection_453})")
            st.plotly_chart(fig5_453)

            # --- Material-Specific Batch Analysis for 453 ---
            st.subheader(f"Material-Specific Batch Analysis for Material {material_selection_453}")
            material_batch_analysis_453 = df_material_453.groupby("Batch")["Quantity"].sum().reset_index()
            fig6_453 = px.bar(material_batch_analysis_453, x="Batch", y="Quantity", title=f"Batch Analysis for Material {material_selection_453}")
            st.plotly_chart(fig6_453)

            # --- Material-Specific Vendor Analysis for 453 ---
            st.subheader(f"Material-Specific Vendor Analysis for Material {material_selection_453}")
            material_vendor_quantity_453 = df_material_453.groupby("Vendor Number")["Quantity"].sum().reset_index()
            fig7_453 = px.bar(material_vendor_quantity_453, x="Vendor Number", y="Quantity", title=f"Vendor Performance for Material {material_selection_453}")
            st.plotly_chart(fig7_453)

            # --- SLED/BBD vs Quantity Analysis for 453 ---
            st.subheader(f"SLED/BBD vs Quantity Analysis for Material {material_selection_453}")
            fig8_453 = px.scatter(df_material_453, x="SLED/BBD", y="Quantity", title=f"SLED/BBD vs Quantity for Material {material_selection_453}")
            st.plotly_chart(fig8_453)

            # --- Days to Expiry Distribution for 453 ---
            df_material_453['Days_to_Expiry_453'] = (df_material_453["SLED/BBD"] - df_material_453["Pstng Date"]).dt.days
            st.subheader(f"Days to Expiry Distribution for Material {material_selection_453}")
            fig9_453 = px.histogram(df_material_453, x="Days_to_Expiry_453", nbins=30, title=f"Days to Expiry Distribution for Material {material_selection_453}")
            st.plotly_chart(fig9_453)

            # --- Vendor Delivery Time Analysis for 453 ---
            df_material_453['Days_to_Delivery_453'] = (df_material_453["SLED/BBD"] - df_material_453["Pstng Date"]).dt.days
            st.subheader(f"Vendor Delivery Time Analysis for Material {material_selection_453}")
            fig10_453 = px.box(df_material_453, x="Vendor Number", y="Days_to_Delivery_453", title=f"Vendor Delivery Time Efficiency for Material {material_selection_453}")
            st.plotly_chart(fig10_453)









