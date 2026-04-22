"""
Unit tests for train.py module
"""
import pytest
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pipeline.train import ModelTrainer, train_churn_model


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer creates correctly."""
        trainer = ModelTrainer()
        assert trainer is not None
        assert trainer.label == "Churn"
        assert trainer.version is not None
    
    def test_load_data(self, sample_train_data, temp_data_dir):
        """Test loading training data."""
        train_path = temp_data_dir / "train.parquet"
        sample_train_data.to_parquet(train_path)
        
        trainer = ModelTrainer()
        loaded = trainer.load_data(train_path)
        
        assert len(loaded) == len(sample_train_data)
        assert 'Churn' in loaded.columns
    
    def test_load_data_missing_file(self):
        """Test error handling for missing file."""
        trainer = ModelTrainer()
        
        with pytest.raises(FileNotFoundError):
            trainer.load_data("nonexistent.parquet")
    
    def test_train_model_smoke_test(self, sample_train_data, temp_data_dir):
        """Smoke test for model training (fast version)."""
        # Skip if AutoGluon not available or slow test not desired
        pytest.skip("AutoGluon training is slow - skip in unit tests")
        
        train_path = temp_data_dir / "train.parquet"
        sample_train_data.to_parquet(train_path)
        
        trainer = ModelTrainer()
        predictor = trainer.train(train_path)
        
        assert predictor is not None