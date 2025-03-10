import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def plot_quantity_distribution(data):
    """
    Plot the distribution of the 'Quantity' column.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Quantity' column.
    
    Returns:
    - None: Displays the plot in the Streamlit app.
    """
    fig = px.histogram(data, x='Quantity', title="Distribution of Quantities")
    st.plotly_chart(fig)

def plot_quantity_boxplot(data):
    """
    Plot a boxplot to detect outliers in the 'Quantity' column.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Quantity' column.
    
    Returns:
    - None: Displays the boxplot in the Streamlit app.
    """
    fig = px.box(data, y='Quantity', title="Boxplot of Quantity (Outlier Detection)")
    st.plotly_chart(fig)


def plot_vendor_distribution(data):
    """
    Plot the distribution of 'Vendor Number' (number of orders per vendor).
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Vendor Number' column.
    
    Returns:
    - None: Displays the bar chart in the Streamlit app.
    """
    vendor_counts = data['Vendor Number'].value_counts()
    fig = px.bar(vendor_counts, x=vendor_counts.index, y=vendor_counts.values, 
                 title="Distribution of Vendors", labels={'x': 'Vendor Number', 'y': 'Count'})
    st.plotly_chart(fig)

def plot_quantity_trends_per_vendor(data):
    """
    Plot the trends of 'Quantity' over time per 'Vendor Number' based on 'Pstng Date'.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    data_grouped = data.groupby([data['Pstng Date'].dt.to_period('M'), 'Vendor Number'])['Quantity'].sum().reset_index()

    # Convert Period back to string or datetime before plotting
    data_grouped['Pstng Date'] = data_grouped['Pstng Date'].astype(str)  # Converts '2024-01' to '2024-01'

    fig = px.line(data_grouped, x='Pstng Date', y='Quantity', color='Vendor Number', 
                  title="Quantity Trends Over Time by Vendor",
                  labels={'Pstng Date': 'Posting Date', 'Quantity': 'Total Quantity'})

    st.plotly_chart(fig)



def plot_purchase_frequency(data):
    """
    Plots the frequency of purchases over time.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    data_grouped = data.groupby(data['Pstng Date'].dt.to_period('M')).size().reset_index(name='Frequency')

    data_grouped['Pstng Date'] = data_grouped['Pstng Date'].astype(str)

    fig = px.bar(data_grouped, x='Pstng Date', y='Frequency', 
                 title="Purchase Frequency Over Time", 
                 labels={'Pstng Date': 'Posting Date', 'Frequency': 'Number of Purchases'})

    st.plotly_chart(fig)


def plot_quantity_vs_pstng_date(data):
    """
    Plot a scatter plot of 'Quantity' against 'Posting Date' with color-coding by 'Vendor Number'.
    
    Args:
    - data (pandas.DataFrame): The dataset containing 'Quantity', 'Pstng Date', and 'Vendor Number' columns.
    
    Returns:
    - None: Displays the scatter plot in the Streamlit app.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    fig = px.scatter(data, x='Pstng Date', y='Quantity', color='Vendor Number', 
                     title="Quantity vs. Posting Date", labels={'Pstng Date': 'Posting Date', 'Quantity': 'Quantity'})
    st.plotly_chart(fig)

def plot_quantity_per_plant(data):
    """
    Plot the total quantity per plant.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Plant' and 'Quantity' columns.
    
    Returns:
    - None: Displays the bar chart in the Streamlit app.
    """
    plant_counts = data.groupby('Plant')['Quantity'].sum().reset_index()
    fig = px.bar(plant_counts, x='Plant', y='Quantity', title="Quantity Per Plant", labels={'Quantity': 'Total Quantity'})
    st.plotly_chart(fig)

def plot_quantity_per_batch(data):
    """
    Plot the total quantity per batch.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Batch' and 'Quantity' columns.
    
    Returns:
    - None: Displays the bar chart in the Streamlit app.
    """
    batch_counts = data.groupby('Batch')['Quantity'].sum().reset_index()
    fig = px.bar(batch_counts, x='Batch', y='Quantity', title="Quantity Per Batch", labels={'Quantity': 'Total Quantity'})
    st.plotly_chart(fig)

def plot_sled_bbd_distribution(data):
    """
    Plot the distribution of 'SLED/BBD' dates.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'SLED/BBD' column.
    
    Returns:
    - None: Displays the histogram in the Streamlit app.
    """
    data['SLED/BBD'] = pd.to_datetime(data['SLED/BBD'])
    fig = px.histogram(data, x='SLED/BBD', title="SLED/BBD Distribution", labels={'SLED/BBD': 'SLED/BBD Date'})
    st.plotly_chart(fig)

def plot_missing_values(data):
    """
    Plot the count of missing values per column.
    
    Args:
    - data (pandas.DataFrame): The dataset to analyze for missing values.
    
    Returns:
    - None: Displays the missing values distribution bar chart in the Streamlit app.
    """
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    fig = px.bar(missing_data, x=missing_data.index, y=missing_data.values, 
                 title="Missing Data Distribution", labels={'x': 'Columns', 'y': 'Missing Values'})
    st.plotly_chart(fig)

def plot_time_based_trends(data):
    """
    Plot the trends of 'Quantity' over time based on 'Pstng Date'.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    data_grouped = data.groupby([data['Pstng Date'].dt.to_period('M').dt.to_timestamp()])['Quantity'].sum().reset_index()

    # Convert Period to string (or datetime) before plotting
    data_grouped['Pstng Date'] = data_grouped['Pstng Date'].astype(str)  # Converts '2024-01' to '2024-01'

    fig = px.line(data_grouped, x='Pstng Date', y='Quantity', 
                  title="Quantity Trends Over Time", 
                  labels={'Pstng Date': 'Posting Date', 'Quantity': 'Total Quantity'})

    st.plotly_chart(fig)

def plot_quantity_trends_per_plant(data):
    """
    Plot the trends of 'Quantity' over time per 'Plant' based on 'Pstng Date'.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Pstng Date', 'Quantity', and 'Plant' columns.
    
    Returns:
    - None: Displays the line chart in the Streamlit app.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    data_grouped = data.groupby([data['Pstng Date'].dt.to_period('M'), 'Plant'])['Quantity'].sum().reset_index()
    data_grouped['Pstng Date'] = data_grouped['Pstng Date'].astype(str)
    fig = px.line(data_grouped, x='Pstng Date', y='Quantity', color='Plant', 
                  title="Quantity Trends Over Time by Plant", labels={'Pstng Date': 'Posting Date', 'Quantity': 'Total Quantity'})
    st.plotly_chart(fig)

def plot_sled_bbd_vs_quantity(data):
    """
    Plot 'SLED/BBD' vs 'Quantity' to analyze if there is any correlation between the shelf life or batch dates and quantities.
    
    Args:
    - data (pandas.DataFrame): The dataset containing 'SLED/BBD' and 'Quantity' columns.
    
    Returns:
    - None: Displays the scatter plot in the Streamlit app.
    """
    data['SLED/BBD'] = pd.to_datetime(data['SLED/BBD'])
    fig = px.scatter(data, x='SLED/BBD', y='Quantity', 
                     title="SLED/BBD vs Quantity", 
                     labels={'SLED/BBD': 'SLED/BBD Date', 'Quantity': 'Quantity'})
    st.plotly_chart(fig)


