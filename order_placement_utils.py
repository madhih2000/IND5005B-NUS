import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to preprocess the uploaded file
def preprocess_order_data(file):
    if file:
        data = pd.read_excel(file)
        # Ensure correct data types
        data['Order Quantity'] = pd.to_numeric(data['Order Quantity'], errors='coerce')
        data['Still to be delivered (qty)'] = pd.to_numeric(data['Still to be delivered (qty)'], errors='coerce')
        data['Delivery Status'] = data['Order Quantity'] - data['Still to be delivered (qty)']
        return data
    return None