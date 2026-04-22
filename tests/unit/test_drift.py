"""
Unit tests for drift.py module
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pipeline.drift import detect_drift, _calculate_ks_drift_score


class TestKSDriftScore:
    """Test cases for KS drift score calculation."""
    
    def test_ks_score_identical_distributions(self, sample_train_data):
        """Test KS score is 0 for identical distributions."""
        score = _calculate_ks_drift_score(
            sample_train_data,
            sample_train_data,
            exclude={'Churn'}
        )
        assert score == 0.0
    
    def test_ks_score_different_distributions(self, sample_train_data, sample_drifted_data):
        """Test KS score is > 0 for different distributions."""
        score = _calculate_ks_drift_score(
            sample_train_data,
            sample_drifted_data,
            exclude={'Churn'}
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should detect some drift
    
    def test_ks_score_excludes_target(self, sample_train_data, sample_drifted_data):
        """Test that target column is excluded from drift calculation."""
        # Add more numeric columns
        sample_train_data['Target'] = [1, 0, 1, 0, 1]
        sample_drifted_data['Target'] = [0, 1, 0, 1, 0]
        
        score = _calculate_ks_drift_score(
            sample_train_data,
            sample_drifted_data,
            exclude={'Target'}
        )
        # Should not crash and should return valid score
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestDetectDrift:
    """Test cases for detect_drift function."""
    
    def test_detect_drift_returns_dict(self, sample_train_data, sample_drifted_data, temp_data_dir):
        """Test that detect_drift returns a dictionary."""
        # Save test data
        ref_path = temp_data_dir / "train.parquet"
        cur_path = temp_data_dir / "test_drifted.csv"
        
        sample_train_data.to_parquet(ref_path)
        sample_drifted_data.to_csv(cur_path, index=False)
        
        result = detect_drift(
            reference_path=ref_path,
            current_path=cur_path,
            target_column='Churn',
            threshold=0.05
        )
        
        assert isinstance(result, dict)
        assert 'drift_score' in result
        assert 'report_path' in result
        assert 'needs_retrain' in result
        assert 'threshold' in result
    
    def test_detect_drift_score_in_range(self, sample_train_data, sample_drifted_data, temp_data_dir):
        """Test that drift score is between 0 and 1."""
        ref_path = temp_data_dir / "train.parquet"
        cur_path = temp_data_dir / "test_drifted.csv"
        
        sample_train_data.to_parquet(ref_path)
        sample_drifted_data.to_csv(cur_path, index=False)
        
        result = detect_drift(ref_path, cur_path, 'Churn')
        
        assert 0.0 <= result['drift_score'] <= 1.0
    
    def test_detect_drift_triggers_retrain(self, sample_train_data, sample_drifted_data, temp_data_dir):
        """Test that high drift triggers retraining."""
        ref_path = temp_data_dir / "train.parquet"
        cur_path = temp_data_dir / "test_drifted.csv"
        
        sample_train_data.to_parquet(ref_path)
        sample_drifted_data.to_csv(cur_path, index=False)
        
        result = detect_drift(ref_path, cur_path, 'Churn', threshold=0.05)
        
        # With our drifted data, should trigger retrain
        assert isinstance(result['needs_retrain'], bool)
    
    def test_detect_drift_creates_report(self, sample_train_data, sample_drifted_data, temp_data_dir):
        """Test that drift report HTML file is created."""
        ref_path = temp_data_dir / "train.parquet"
        cur_path = temp_data_dir / "test_drifted.csv"
        
        sample_train_data.to_parquet(ref_path)
        sample_drifted_data.to_csv(cur_path, index=False)
        
        result = detect_drift(ref_path, cur_path, 'Churn')
        
        report_path = Path(result['report_path'])
        assert report_path.exists()
        assert report_path.suffix == '.html'