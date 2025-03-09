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
                consumption_utils.overall_consumption_patterns(df_filtered)
                consumption_utils.outlier_detection(df_filtered)
                consumption_utils.shelf_life_analysis(df_filtered)
                consumption_utils.vendor_consumption_analysis(df_filtered)
                consumption_utils.location_consumption_analysis(df_filtered)
                consumption_utils.batch_variability_analysis(df_filtered)
                consumption_utils.combined_analysis(df_filtered)
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
                # Generate random dates for 'Pstng Date' (if needed)
                if 'Pstng Date' not in df.columns:
                    df_filtered = order_placement_utils.generate_random_dates(df_filtered, "2024-01-01", "2024-12-31")

                # Call the analysis functions
                order_placement_utils.overall_order_patterns(df_filtered)
                order_placement_utils.outlier_detection(df_filtered)
                order_placement_utils.vendor_order_analysis(df_filtered)
                order_placement_utils.order_trends_over_time(df_filtered)
                order_placement_utils.monthly_order_patterns(df_filtered)
                order_placement_utils.vendor_material_analysis(df_filtered)
                order_placement_utils.plant_order_analysis(df_filtered)
                order_placement_utils.purchasing_document_analysis(df_filtered)
                order_placement_utils.order_quantity_distribution(df_filtered)
                order_placement_utils.material_vendor_analysis(df_filtered)
                order_placement_utils.supplier_order_analysis(df_filtered)
                order_placement_utils.material_plant_analysis(df_filtered)
                order_placement_utils.abc_analysis(df_filtered)
                order_placement_utils.specific_material_analysis(df_filtered)


    else:
        st.write("Please upload an Excel file to begin the analysis.")




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









