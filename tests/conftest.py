"""
Pytest configuration and shared fixtures
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_train_data():
    """Create sample training data for testing."""
    data = {
        'Age': [25, 35, 45, 55, 65],
        'Tenure in Months': [12, 24, 36, 48, 60],
        'Monthly Charge': [50.0, 75.0, 100.0, 125.0, 150.0],
        'Total Charges': [600.0, 1800.0, 3600.0, 6000.0, 9000.0],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Married': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Contract': ['Month-to-Month', 'One year', 'Two year', 'Month-to-Month', 'One year'],
        'Churn': [1, 0, 0, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_drifted_data():
    """Create sample drifted data for testing."""
    data = {
        'Age': [30, 40, 50, 60, 70],  # Shifted distribution
        'Tenure in Months': [6, 12, 18, 24, 30],  # Shifted distribution
        'Monthly Charge': [60.0, 85.0, 110.0, 135.0, 160.0],
        'Total Charges': [360.0, 1020.0, 1980.0, 3240.0, 4800.0],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Married': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        'Contract': ['Month-to-Month', 'One year', 'Two year', 'Month-to-Month', 'One year'],
        'Churn': [1, 0, 0, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    return data_dir