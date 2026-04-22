"""
Unit tests for data.py module
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pipeline.data import DataManager, prepare_data_for_pipeline


class TestDataManager:
    """Test cases for DataManager class."""
    
    def test_data_manager_initialization(self):
        """Test DataManager creates correctly."""
        dm = DataManager()
        assert dm is not None
        assert dm.seed == 42
        assert dm.test_size == 0.2
    
    def test_create_binary_churn_target(self, sample_train_data):
        """Test binary target creation."""
        dm = DataManager()
        
        # Add Customer Status column
        sample_train_data['Customer Status'] = ['Churned', 'Stayed', 'Stayed', 'Churned', 'Stayed']
        
        result = dm.create_binary_churn_target(sample_train_data)
        
        assert 'Churn' in result.columns
        assert result['Churn'].dtype in [int, float]
        assert set(result['Churn'].unique()).issubset({0, 1})
        assert result['Churn'].sum() == 2  # 2 churned customers
    
    def test_simulate_drift(self, sample_train_data):  # ✅ Now inside class, 'self' works
        """Test drift simulation creates different distribution."""
        dm = DataManager()
        
        drifted = dm.simulate_drift(
            sample_train_data,
            target_col='Churn',
            mag=0.5  # ✅ Fixed: 'mag' not 'drift_magnitude'
        )
        
        # Check that drifted data has different statistics
        assert len(drifted) == len(sample_train_data)
        assert list(drifted.columns) == list(sample_train_data.columns)
        
        # Numeric columns should be different
        for col in ['Age', 'Tenure in Months', 'Monthly Charge']:
            if col in drifted.columns:
                assert not drifted[col].equals(sample_train_data[col])
    
    def test_load_data_parquet(self, sample_train_data, temp_data_dir):
        """Test loading parquet files."""
        train_path = temp_data_dir / "train.parquet"
        sample_train_data.to_parquet(train_path)
        
        dm = DataManager()
        loaded = dm.load_data(train_path)
        
        assert len(loaded) == len(sample_train_data)
        assert list(loaded.columns) == list(sample_train_data.columns)
    
    def test_load_data_csv(self, sample_train_data, temp_data_dir):
        """Test loading CSV files."""
        train_path = temp_data_dir / "train.csv"
        sample_train_data.to_csv(train_path, index=False)
        
        dm = DataManager()
        loaded = dm.load_data(train_path)
        
        assert len(loaded) == len(sample_train_data)


class TestPrepareDataForPipeline:
    """Test cases for prepare_data_for_pipeline function."""
    
    def test_prepare_data_returns_correct_types(self):
        """Test that function returns correct data types."""
        import os
        if not os.getenv('KAGGLE_USERNAME'):
            pytest.skip("Kaggle credentials not available")
        pass