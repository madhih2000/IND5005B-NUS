import pandas as pd
import plotly.express as px
import streamlit as st

def overall_consumption_patterns(df, material_column='Material Number'):
    """
    Analyzes and visualizes overall consumption patterns.

    Args:
        df: The input DataFrame.
        material_column: The column containing material identifiers (default: 'Material Number').
    """
    # Frequency of Transactions
    transaction_counts = df[material_column].value_counts().reset_index()
    transaction_counts.columns = [material_column, 'Transaction Count']
    fig_transactions = px.bar(transaction_counts, x=material_column, y='Transaction Count',
                              title=f'Number of Transactions per {material_column}')
    st.plotly_chart(fig_transactions)

    # Volume of Consumption
    material_consumption = df.groupby(material_column)['Quantity'].sum().reset_index()
    fig_overall = px.bar(material_consumption, x=material_column, y='Quantity',
                         title=f'Overall Consumption by {material_column}')
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
    Detects and visualizes outliers, including percentiles and highlighting high/low usage.

    Args:
        df: The input DataFrame.
        material_column: The column containing material identifiers (default: 'Material Number').
    """

    # Calculate outliers using IQR
    outliers = df.groupby(material_column)['Quantity'].apply(detect_outliers_iqr).reset_index(name='Outlier Quantity')
    outliers = outliers[[material_column, 'Outlier Quantity']]  # Remove 'level_1'

    # Calculate percentiles (e.g., 10th, 25th, 50th, 75th, 90th)
    percentiles = df.groupby(material_column)['Quantity'].describe(percentiles=[.1, .25, .5, .75, .9]).reset_index()

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
    fig_box = px.box(df, x=material_column, y='Quantity', title=f'Box Plot of Consumption by {material_column}')
    st.plotly_chart(fig_box)


def specific_material_analysis(df, material_column='Material Number'):
    """
    Analyzes consumption patterns for a specific material, including consumption by site, batch, trend, and seasonal subseries.

    Args:
        df: The input DataFrame.
        material_column: The column containing material identifiers (default: 'Material Number').
    """
    selected_material = st.selectbox(f"Select a {material_column}", df[material_column].unique())
    filtered_material_data = df[df[material_column] == selected_material].copy()  # Create a copy

    # Consumption by Site for Selected Material
    site_consumption = filtered_material_data.groupby('Site')['Quantity'].sum().reset_index()
    fig_site = px.bar(site_consumption, x='Site', y='Quantity', title=f'Consumption by Site for {selected_material}')
    st.plotly_chart(fig_site)

    # Consumption by Batch for Selected Material
    batch_consumption = filtered_material_data.groupby('Batch')['Quantity'].sum().reset_index()
    fig_batch = px.bar(batch_consumption, x='Batch', y='Quantity', title=f'Consumption by Batch for {selected_material}')
    st.plotly_chart(fig_batch)

    # Time Series of Consumption (Trend)
    start_date, end_date = st.date_input("Select Date Range",
                                        [filtered_material_data['Pstng Date'].min().date(),
                                         filtered_material_data['Pstng Date'].max().date()])

    filtered_time_data = filtered_material_data[(filtered_material_data['Pstng Date'].dt.date >= start_date) &
                                                (filtered_material_data['Pstng Date'].dt.date <= end_date)]

    aggregation_level = st.selectbox("Select Aggregation Level", ['Daily', 'Weekly', 'Monthly', 'Quarterly'])

    if aggregation_level == 'Daily':
        aggregated_data = filtered_time_data.groupby('Pstng Date')['Quantity'].sum().reset_index()
    elif aggregation_level == 'Weekly':
        aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='W'))['Quantity'].sum().reset_index()
    elif aggregation_level == 'Monthly':
        aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='M'))['Quantity'].sum().reset_index()
    elif aggregation_level == 'Quarterly':
        aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='Q'))['Quantity'].sum().reset_index()

    fig_time_series = px.line(aggregated_data, x='Pstng Date', y='Quantity',
                             title=f'Consumption Trend ({aggregation_level}) for {selected_material}')
    st.plotly_chart(fig_time_series)

    # Seasonal Subseries Plot (Monthly)
    if aggregation_level in ['Monthly', 'Quarterly']:
        filtered_time_data['Month'] = filtered_time_data['Pstng Date'].dt.month
        fig_seasonal_subseries = px.box(filtered_time_data, x='Month', y='Quantity',
                                       title=f'Seasonal Subseries Plot (Monthly) for {selected_material}')
        st.plotly_chart(fig_seasonal_subseries)


def shelf_life_analysis(df):
    """
    Analyzes and visualizes shelf life, separating finite and infinite shelf life items
    based on the assumption that items with '2100-01-01' in 'SLED/BBD' have infinite shelf life.

    Args:
        df: The input DataFrame.
    """
    df['Remaining Shelf Life (Days)'] = (df['SLED/BBD'] - df['Pstng Date']).dt.days

    # Identify infinite shelf life items (those with '2100-01-01' in 'SLED/BBD')
    infinite_shelf_life = df[df['SLED/BBD'] == pd.to_datetime('2100-01-01')]
    finite_shelf_life = df[df['SLED/BBD'] != pd.to_datetime('2100-01-01')]

    # Change 'SLED/BBD' to 'No Expiry' for infinite shelf life items
    infinite_shelf_life['SLED/BBD'] = 'No Expiry'

    # Drop 'Remaining Shelf Life (Days)' column from infinite_shelf_life
    infinite_shelf_life = infinite_shelf_life.drop(columns=['Remaining Shelf Life (Days)'], errors='ignore')

    # Visualization for finite shelf life
    if not finite_shelf_life.empty:
        fig_hist_finite = px.histogram(finite_shelf_life, x='Remaining Shelf Life (Days)',
                                        title='Distribution of Remaining Shelf Life (Days) - Finite Shelf Life')
        st.plotly_chart(fig_hist_finite)
    else:
        st.write("No items with finite shelf life in the selected data.")

    infinite_shelf_life = infinite_shelf_life.reset_index(drop = True)
    # Count of infinite shelf life items
    st.write(f"Number of Items with Infinite Shelf Life: {len(infinite_shelf_life)}")

    # Display a table with infinite shelf life items
    if not infinite_shelf_life.empty:
        st.write("Items with Infinite Shelf Life:")
        st.write(infinite_shelf_life)
    else:
        st.write("No items with infinite shelf life in the selected data.")


def vendor_consumption_analysis(df):
    """
    Analyzes consumption patterns for each vendor.

    Args:
        df: The input DataFrame.
    """
    # Consumption by Vendor
    vendor_consumption = df.groupby('Vendor Number')['Quantity'].sum().reset_index()
    fig_vendor = px.bar(vendor_consumption, x='Vendor Number', y='Quantity', title='Consumption by Vendor')
    st.plotly_chart(fig_vendor)

    # Time Series of Consumption by Vendor (Example with Monthly Aggregation)
    monthly_vendor_consumption = df.groupby(['Vendor Number', pd.Grouper(key='Pstng Date', freq='M')])['Quantity'].sum().reset_index()
    fig_vendor_time_series = px.line(monthly_vendor_consumption, x='Pstng Date', y='Quantity', color='Vendor Number',
                                     title='Monthly Consumption Trend by Vendor')
    st.plotly_chart(fig_vendor_time_series)


def location_consumption_analysis(df):
    """
    Analyzes consumption patterns at the plant and site level.

    Args:
        df: The input DataFrame.
    """
    # Consumption by Plant
    plant_consumption = df.groupby('Plant')['Quantity'].sum().reset_index()
    fig_plant = px.bar(plant_consumption, x='Plant', y='Quantity', title='Consumption by Plant')
    st.plotly_chart(fig_plant)

    # Consumption by Site
    site_consumption = df.groupby('Site')['Quantity'].sum().reset_index()
    fig_site = px.bar(site_consumption, x='Site', y='Quantity', title='Consumption by Site')
    st.plotly_chart(fig_site)


def batch_variability_analysis(df):
    """
    Analyzes the variability in consumption across different batches of the same material.

    Args:
        df: The input DataFrame.
    """
    # Calculate consumption by Material Number and Batch
    batch_consumption = df.groupby(['Material Number', 'Batch'])['Quantity'].sum().reset_index()

    # Calculate variance of consumption for each material across its batches
    batch_variance = batch_consumption.groupby('Material Number')['Quantity'].var().reset_index(name='Consumption Variance')

    # Visualize variance using a bar chart
    fig_variance = px.bar(batch_variance, x='Material Number', y='Consumption Variance',
                          title='Variance of Consumption Across Batches')
    st.plotly_chart(fig_variance)


def combined_analysis(df):
    """
    Performs combined analysis of consumption patterns by vendor and material number,
    site and material number, and batch and site.

    Args:
        df: The input DataFrame.
    """
    # Consumption by Vendor and Material Number
    vendor_material_consumption = df.groupby(['Vendor Number', 'Material Number'])['Quantity'].sum().reset_index()
    fig_vendor_material = px.bar(vendor_material_consumption, x='Material Number', y='Quantity', color='Vendor Number',
                                 title='Consumption by Vendor and Material Number')
    st.plotly_chart(fig_vendor_material)

    # Consumption by Site and Material Number
    site_material_consumption = df.groupby(['Site', 'Material Number'])['Quantity'].sum().reset_index()
    fig_site_material = px.bar(site_material_consumption, x='Material Number', y='Quantity', color='Site',
                               title='Consumption by Site and Material Number')
    st.plotly_chart(fig_site_material)
