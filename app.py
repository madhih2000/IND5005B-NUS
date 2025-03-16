import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import plotly.figure_factory as ff
from utils import *
import consumption_utils
import order_placement_utils
import goods_receipt_utils

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
    # Upload Excel files
    uploaded_file = st.file_uploader("Upload Consumption Excel File for Analysis", type=["xlsx"])

    if uploaded_file:
        df = load_data_consumption(uploaded_file)  # Read the file

        # Check if 'Material Group' column exists
        if 'Material Group' in df.columns:
            unique_groups = df['Material Group'].astype(str).unique()  # Get unique values

            for group in unique_groups:
                st.subheader(f"Material Group {group} Analysis")

                # Filter data for the current Material Group
                df_filtered = df[df['Material Group'].astype(str) == str(group)]

                # Run analysis functions in the correct order
                df_more_filtered,top_n = consumption_utils.overall_consumption_patterns(df_filtered)
                consumption_utils.outlier_detection(df_more_filtered, top_n)
                consumption_utils.shelf_life_analysis(df_filtered)
                #consumption_utils.vendor_consumption_analysis(df_filtered)
                #consumption_utils.location_consumption_analysis(df_filtered)
                #consumption_utils.batch_variability_analysis(df_filtered)
                #consumption_utils.combined_analysis(df_filtered)
                consumption_utils.specific_material_analysis(df_filtered)

        else:
            st.error("The uploaded file does not contain a 'Material Group' column. Please check the file format.")

elif tabs == "Order Placement Analysis":
    st.title("Order Placement Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload Order Placement Excel File for Analysis", type="xlsx")

    if uploaded_file:
        df = order_placement_utils.preprocess_order_data(uploaded_file)  # Read the file

        # Check if 'Material Group' column exists
        if 'Material Group' in df.columns:
            unique_groups = df['Material Group'].astype(str).unique()  # Get unique values

            for group in unique_groups:
                st.subheader(f"Material Group {group} Analysis")

                # Filter data for the current Material Group
                df_filtered = df[df['Material Group'].astype(str) == str(group)]

                # Call the analysis functions
                df_more_filtered,top_n = order_placement_utils.overall_orderplacement_patterns(df_filtered)
                order_placement_utils.outlier_detection(df_more_filtered, top_n)
                # order_placement_utils.overall_order_patterns(df_filtered)
                # order_placement_utils.outlier_detection(df_filtered)
                # order_placement_utils.vendor_order_analysis(df_filtered)
                # order_placement_utils.order_trends_over_time(df_filtered)
                # order_placement_utils.monthly_order_patterns(df_filtered)
                # order_placement_utils.vendor_material_analysis(df_filtered)
                # order_placement_utils.plant_order_analysis(df_filtered)
                # order_placement_utils.purchasing_document_analysis(df_filtered)
                # order_placement_utils.order_quantity_distribution(df_filtered)
                # order_placement_utils.material_vendor_analysis(df_filtered)
                # order_placement_utils.supplier_order_analysis(df_filtered)
                # order_placement_utils.material_plant_analysis(df_filtered)
                # order_placement_utils.abc_analysis(df_filtered)
                order_placement_utils.specific_material_analysis(df_filtered)


    else:
        st.write("Please upload an Excel file to begin the analysis.")


elif tabs == "Goods Receipt Analysis":
    st.title("Goods Receipt Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload Goods Receipt Excel File for Analysis", type="xlsx")

    if uploaded_file:
        df = load_data_GR(uploaded_file)  # Read the file

        # Check if 'Material Group' column exists
        if 'Material Group' in df.columns:
            unique_groups = df['Material Group'].astype(str).unique()  # Get unique values

            for group in unique_groups:
                st.subheader(f"Material Group {group} Analysis")

                # Filter data for the current Material Group
                df_filtered = df[df['Material Group'].astype(str) == str(group)]

                # Call the analysis functions
                df_more_filtered,top_n = goods_receipt_utils.overall_GR_patterns(df_filtered)
                goods_receipt_utils.outlier_detection(df_more_filtered, top_n)
                goods_receipt_utils.specific_material_analysis(df_filtered)


    else:
        st.write("Please upload an Excel file to begin the analysis.")
