"""Prepare data for Plotly Dash."""
import numpy as np
import pandas as pd


def create_dataframe():
    """Create Pandas DataFrame from local CSV."""
    df = pd.read_csv("app/data/diabetes_dataset1.csv")

    return df
