import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px

# Function to preprocess the uploaded file
def preprocess_order_data(file):
    if file:
        data = pd.read_excel(file)
        data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df.columns = df.columns.str.strip()
        # Ensure correct data types
        data['Order Quantity'] = pd.to_numeric(data['Order Quantity'], errors='coerce')
        # data['Still to be delivered (qty)'] = pd.to_numeric(data['Still to be delivered (qty)'], errors='coerce')
        # data['Delivery Status'] = data['Order Quantity'] - data['Still to be delivered (qty)']
        return data
    return None

def generate_random_dates(df, start_date, end_date, date_col="Pstng Date"):
    """
    Generates random dates within a specified range and assigns them to a column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        date_col (str): Name of the column to store the dates.

    Returns:
        pd.DataFrame: The DataFrame with the new date column.
    """
    num_rows = len(df)
    df[date_col] = np.random.choice(pd.date_range(start_date, end_date), size=num_rows)
    return df

def overall_order_patterns(df, material_column='Material Number'):
    """
    Analyzes and visualizes overall order patterns.

    Args:
        df: The input DataFrame.
        material_column: The column containing material identifiers (default: 'Material Number').
    """
    # Frequency of Orders
    order_counts = df[material_column].value_counts().reset_index()
    order_counts.columns = [material_column, 'Order Count']
    fig_orders = px.bar(order_counts, x=material_column, y='Order Count',
                        title=f'Number of Orders per {material_column}')
    st.plotly_chart(fig_orders)

    # Volume of Orders
    material_orders = df.groupby(material_column)['Order Quantity'].sum().reset_index()
    fig_overall = px.bar(material_orders, x=material_column, y='Order Quantity',
                        title=f'Overall Order Quantity by {material_column}')
    st.plotly_chart(fig_overall)

def detect_outliers_iqr(data):
    """
    Detects outliers in a pandas Series using the IQR method.

    Args:
        data: The input pandas Series.

    Returns:
        A pandas Series containing the outliers.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

def outlier_detection(df, material_column='Material Number'):
    """
    Detects and visualizes outliers, including percentiles and highlighting high/low order quantities.

    Args:
        df: The input DataFrame.
        material_column: The column containing material identifiers (default: 'Material Number').
    """

    # Calculate outliers using IQR
    outliers = df.groupby(material_column)['Order Quantity'].apply(detect_outliers_iqr).reset_index(name='Outlier Quantity')
    outliers = outliers[[material_column, 'Outlier Quantity']]  # Remove 'level_1'

    # Calculate percentiles (e.g., 10th, 25th, 50th, 75th, 90th)
    percentiles = df.groupby(material_column)['Order Quantity'].describe(percentiles=[.1, .25, .5, .75, .9]).reset_index()

    # Merge percentiles with outliers
    outliers_with_percentiles = pd.merge(outliers, percentiles, on=material_column)

    # Add a "Type" column to indicate high or low usage
    outliers_with_percentiles['Type'] = outliers_with_percentiles.apply(
        lambda row: 'High' if row['Outlier Quantity'] > row['75%'] else ('Low' if row['Outlier Quantity'] < row['25%'] else 'Normal'), axis=1
    )

    # Color coding for "High" and "Low" in the table
    def color_coding(val):
        if val == 'High':
            color = 'red'  # Color "High" values red
        elif val == 'Low':
            color = 'blue'  # Color "Low" values blue
        else:
            color = 'black'  # Default color
        return f'color: {color}'

    # Format the table to display integers for relevant columns
    styled_table = outliers_with_percentiles.style.applymap(color_coding, subset=['Type']).format({
        'Outlier Quantity': "{:.0f}".format,
        'count': "{:.0f}".format,
        'min': "{:.0f}".format,
        '10%': "{:.0f}".format,
        '25%': "{:.0f}".format,
        '50%': "{:.0f}".format,
        '75%': "{:.0f}".format,
        '90%': "{:.0f}".format,
        'max': "{:.0f}".format
    })

    st.write(f"Outliers (IQR Method) with Percentiles for {material_column}:")
    st.write(styled_table)

    # Box Plots for Outlier Visualization
    fig_box = px.box(df, x=material_column, y='Order Quantity', title=f'Box Plot of Order Quantity by {material_column}')
    st.plotly_chart(fig_box)


def supplier_order_analysis(df):
    """
    Analyzes order patterns for each supplier.

    Args:
        df: The input DataFrame.
    """
    # Order Quantity by Supplier
    supplier_orders = df.groupby('Supplier')['Order Quantity'].sum().reset_index()
    fig_supplier = px.bar(supplier_orders, x='Supplier', y='Order Quantity', title='Order Quantity by Supplier')
    st.plotly_chart(fig_supplier)

def specific_material_analysis(df, material_column='Material Number'):
    """
    Analyzes order patterns for a specific material, including order quantity by plant, vendor, trend, and seasonal subseries.

    Args:
        df: The input DataFrame.
        material_column: The column containing material identifiers (default: 'Material Number').
    """
    selected_material = st.selectbox(f"Select a {material_column}", df[material_column].unique())
    filtered_material_data = df[df[material_column] == selected_material].copy()  # Create a copy

    # Order Quantity by Plant for Selected Material
    plant_orders = filtered_material_data.groupby('Plant')['Order Quantity'].sum().reset_index()
    fig_plant = px.bar(plant_orders, x='Plant', y='Order Quantity', title=f'Order Quantity by Plant for {selected_material}')
    st.plotly_chart(fig_plant)

    # Order Quantity by Vendor for Selected Material
    vendor_orders = filtered_material_data.groupby('Vendor Number')['Order Quantity'].sum().reset_index()
    fig_vendor = px.bar(vendor_orders, x='Vendor Number', y='Order Quantity', title=f'Order Quantity by Vendor for {selected_material}')
    st.plotly_chart(fig_vendor)

    # Time Series of Orders (Trend)
    if 'Pstng Date' in filtered_material_data.columns:
        start_date, end_date = st.date_input("Select Date Range",
                                            [filtered_material_data['Pstng Date'].min().date(),
                                             filtered_material_data['Pstng Date'].max().date()])

        filtered_time_data = filtered_material_data[(filtered_material_data['Pstng Date'].dt.date >= start_date) &
                                                    (filtered_material_data['Pstng Date'].dt.date <= end_date)]

        aggregation_level = st.selectbox("Select Aggregation Level", ['Daily', 'Weekly', 'Monthly', 'Quarterly'])

        if aggregation_level == 'Daily':
            aggregated_data = filtered_time_data.groupby('Pstng Date')['Order Quantity'].sum().reset_index()
        elif aggregation_level == 'Weekly':
            aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='W'))['Order Quantity'].sum().reset_index()
        elif aggregation_level == 'Monthly':
            aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='M'))['Order Quantity'].sum().reset_index()
        elif aggregation_level == 'Quarterly':
            aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='Q'))['Order Quantity'].sum().reset_index()

        fig_time_series = px.line(aggregated_data, x='Pstng Date', y='Order Quantity',
                                    title=f'Order Trend ({aggregation_level}) for {selected_material}')
        st.plotly_chart(fig_time_series)

        # Seasonal Subseries Plot (Monthly)
        if aggregation_level in ['Monthly', 'Quarterly']:
            filtered_time_data['Month'] = filtered_time_data['Pstng Date'].dt.month
            fig_seasonal_subseries = px.box(filtered_time_data, x='Month', y='Order Quantity',
                                                title=f'Seasonal Subseries Plot (Monthly) for {selected_material}')
            st.plotly_chart(fig_seasonal_subseries)
    else:
        st.write("No 'Pstng Date' column found for time-based analysis.")

def vendor_order_analysis(df):
    """
    Analyzes order patterns for each vendor.

    Args:
        df: The input DataFrame.
    """
    # Order Quantity by Vendor
    vendor_orders = df.groupby('Vendor Number')['Order Quantity'].sum().reset_index()
    fig_vendor = px.bar(vendor_orders, x='Vendor Number', y='Order Quantity', title='Order Quantity by Vendor')
    st.plotly_chart(fig_vendor) 


def order_trends_over_time(df, date_col="Pstng Date"):
    """
    Analyzes and visualizes order trends over time.

    Args:
        df: The input DataFrame.
        date_col: The column containing the date (default: 'Pstng Date').
    """
    # Group by date and sum the order quantity
    daily_orders = df.groupby(date_col)['Order Quantity'].sum().reset_index()

    # Create a line chart using Plotly Express
    fig = px.line(daily_orders, x=date_col, y='Order Quantity', title='Order Quantity Over Time')
    st.plotly_chart(fig)

def monthly_order_patterns(df, date_col="Pstng Date"):
    """
    Analyzes and visualizes monthly order patterns.

    Args:
        df: The input DataFrame.
        date_col: The column containing the date (default: 'Pstng Date').
    """
    # Extract month from the date column
    df['Month'] = df[date_col].dt.month_name()

    # Group by month and sum the order quantity
    monthly_orders = df.groupby('Month')['Order Quantity'].sum().reset_index()

    # Create a bar chart using Plotly Express
    fig = px.bar(monthly_orders, x='Month', y='Order Quantity', title='Total Order Quantity by Month')
    st.plotly_chart(fig)

def vendor_material_analysis(df):
    """Analyzes order patterns by vendor and material."""
    vendor_material_orders = df.groupby(['Vendor Number', 'Material Number'])['Order Quantity'].sum().reset_index()
    fig_vendor_material = px.bar(vendor_material_orders, x='Material Number', y='Order Quantity', color='Vendor Number',
                                title='Order Quantity by Vendor and Material')
    st.plotly_chart(fig_vendor_material)

def plant_order_analysis(df):
    """Analyzes order patterns by Plant."""
    plant_orders = df.groupby('Plant')['Order Quantity'].sum().reset_index()
    fig_plant = px.bar(plant_orders, x='Plant', y='Order Quantity', title='Order Quantity by Plant')
    st.plotly_chart(fig_plant)

def purchasing_document_analysis(df):
    """Analyzes order patterns by Purchasing Document."""
    purchasing_orders = df.groupby('Purchasing Document')['Order Quantity'].sum().reset_index()

    # Option 1: Data Table with Sorting
    st.write("**Order Quantity by Purchasing Document:**")
    st.dataframe(purchasing_orders.sort_values(by='Order Quantity', ascending=False))

def order_quantity_distribution(df):
    """Analyzes and visualizes the distribution of order quantities."""
    fig_histogram = px.histogram(df, x='Order Quantity', title='Distribution of Order Quantities')
    st.plotly_chart(fig_histogram)

    fig_box = px.box(df, y='Order Quantity', title='Box Plot of Order Quantities')
    st.plotly_chart(fig_box)

def material_vendor_analysis(df):
    """Analyzes vendor order quantity by material."""
    material_vendor_orders = df.groupby(['Material Number', 'Vendor Number'])['Order Quantity'].sum().reset_index()
    fig = px.bar(material_vendor_orders, x='Material Number', y='Order Quantity', color='Vendor Number', title='Order Quantity by Material and Vendor')
    st.plotly_chart(fig)

def material_plant_analysis(df):
    """Analyzes plant order quantity by material."""
    material_plant_orders = df.groupby(['Material Number', 'Plant'])['Order Quantity'].sum().reset_index()
    fig = px.bar(material_plant_orders, x='Material Number', y='Order Quantity', color='Plant', title='Order Quantity by Material and Plant')
    st.plotly_chart(fig)

def abc_analysis(df):
    """Performs ABC analysis on materials."""
    material_orders = df.groupby('Material Number')['Order Quantity'].sum().reset_index()
    material_orders = material_orders.sort_values(by='Order Quantity', ascending=False)
    material_orders['Cumulative Quantity'] = material_orders['Order Quantity'].cumsum()
    total_quantity = material_orders['Order Quantity'].sum()
    material_orders['Cumulative Percentage'] = (material_orders['Cumulative Quantity'] / total_quantity) * 100
    material_orders['Category'] = pd.cut(material_orders['Cumulative Percentage'], bins=[0, 80, 95, 100], labels=['A', 'B', 'C'])
    st.write("ABC Analysis of Materials:")
    st.dataframe(material_orders)
    # Add legend
    st.markdown("""
    **Legend:**

    * **A:** High-value items. These typically represent the top 80% of the total consumption value.
    * **B:** Medium-value items. These typically represent the next 15% of the total consumption value.
    * **C:** Low-value items. These typically represent the remaining 5% of the total consumption value.
    """)