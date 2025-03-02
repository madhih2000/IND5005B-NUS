import pandas as pd
import numpy as np


# Load data
def load_data(file):
    df = pd.read_excel(file)
    df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])
    df['SLED/BBD'] = df['SLED/BBD'].fillna('No Expiry')  # Handle empty expiry dates
    return df