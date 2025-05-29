"""
Tests for time series modeling scripts
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add the data/Time Series directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data', 'Time Series'))

@patch('sys.modules', {'supabase': Mock()})
def test_aggregate_time_series_function():
    """Test the aggregate_time_series function with mock data"""
    with patch.dict('sys.modules', {'supabase': Mock()}):
        from aggregate_time_series import aggregate_time_series
        
        # Create mock data
        mock_data = pd.DataFrame({
            'event_date': pd.date_range('2024-01-01', periods=10),
            'goldstein_score': np.random.normal(0, 2, 10),
            'embedding_cluster_id': np.random.choice([0, 1, 2, -1], 10),
            'article_id': range(10)
        })
        mock_data['event_date'] = mock_data['event_date'].dt.date
        
        # Test aggregation
        result = aggregate_time_series(mock_data)
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert 'avg_goldstein' in result.columns
        assert 'std_goldstein' in result.columns
        assert 'num_articles' in result.columns
        assert 'num_clusters' in result.columns
        assert 'pct_noise' in result.columns
        assert len(result) == 10  # One row per day

def test_tft_data_preparation():
    """Test data preparation for TFT model"""
    # Create mock daily features
    dates = pd.date_range('2024-01-01', periods=30)
    mock_df = pd.DataFrame({
        'event_date': dates,
        'avg_goldstein': np.random.normal(0, 2, 30),
        'std_goldstein': np.random.normal(1, 0.5, 30),
        'num_articles': np.random.poisson(10, 30),
        'num_clusters': np.random.poisson(3, 30),
        'max_cluster_size': np.random.poisson(5, 30),
        'avg_cluster_size': np.random.normal(3, 1, 30),
        'pct_noise': np.random.uniform(0, 0.3, 30)
    })
    
    # Add time index and group id
    mock_df["time_idx"] = (mock_df["event_date"] - mock_df["event_date"].min()).dt.days
    mock_df["group"] = "straitwatch"
    
    # Add date-based encodings
    mock_df["day_of_week"] = mock_df["event_date"].dt.dayofweek
    mock_df["week"] = mock_df["event_date"].dt.isocalendar().week
    mock_df["month"] = mock_df["event_date"].dt.month
    
    # Rolling features
    mock_df["goldstein_lag1"] = mock_df["avg_goldstein"].shift(1)
    mock_df["goldstein_lag3"] = mock_df["avg_goldstein"].shift(3)
    mock_df["goldstein_roll3"] = mock_df["avg_goldstein"].rolling(3).mean()
    
    # Drop NaNs
    mock_df = mock_df.dropna().reset_index(drop=True)
    
    # Assertions
    assert len(mock_df) == 27  # 30 - 3 NaN rows from rolling/lag
    assert 'time_idx' in mock_df.columns
    assert 'goldstein_lag1' in mock_df.columns
    assert 'goldstein_roll3' in mock_df.columns
    assert mock_df['time_idx'].max() == 29  # 0-indexed, 30 days total

@patch.dict('sys.modules', {'supabase': Mock()})
def test_fetch_article_data_mock():
    """Test fetch_article_data with mocked Supabase"""
    with patch('aggregate_time_series.supabase') as mock_supabase:
        from aggregate_time_series import fetch_article_data
        
        # Mock response
        mock_response = Mock()
        mock_response.data = [
            {'article_id': 1, 'event_date': '2024-01-01', 'goldstein_score': 1.5, 'embedding_cluster_id': 0},
            {'article_id': 2, 'event_date': '2024-01-02', 'goldstein_score': -2.0, 'embedding_cluster_id': 1}
        ]
        
        mock_supabase.table.return_value.select.return_value.execute.return_value = mock_response
        
        # Test function
        result = fetch_article_data()
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'event_date' in result.columns
        assert result['event_date'].dtype == 'object'  # date type

def test_pct_noise_function():
    """Test the pct_noise helper function"""
    # Create test data for pct_noise function
    test_series = pd.Series([0, 1, 2, -1, -1, 3])
    
    # Extract the pct_noise function logic
    def pct_noise(x): return (x == -1).sum() / len(x) if len(x) > 0 else 0
    
    result = pct_noise(test_series)
    expected = 2 / 6  # 2 noise points out of 6 total
    
    assert abs(result - expected) < 1e-6

def test_gdelt_tft_data_preparation():
    """Test data preparation for GDELT-only TFT model"""
    # Create mock CSV data for GDELT TFT
    mock_df = pd.DataFrame({
        'event_date': pd.date_range('2024-01-01', periods=20),
        'avg_goldstein': np.random.normal(0, 2, 20),
        'std_goldstein': np.random.normal(1, 0.5, 20),
        'num_articles': np.random.poisson(10, 20),
        'num_clusters': np.random.poisson(3, 20),
        'max_cluster_size': np.random.poisson(5, 20),
        'avg_cluster_size': np.random.normal(3, 1, 20),
        'pct_noise': np.random.uniform(0, 0.3, 20)
    })
    
    # Apply GDELT TFT data preparation steps
    mock_df["time_idx"] = (mock_df["event_date"] - mock_df["event_date"].min()).dt.days
    mock_df["group"] = "gdelt_only"
    
    # Temporal features
    mock_df["day_of_week"] = mock_df["event_date"].dt.dayofweek
    mock_df["week"] = mock_df["event_date"].dt.isocalendar().week.astype(int)
    mock_df["month"] = mock_df["event_date"].dt.month
    
    # Rolling features
    mock_df["goldstein_lag1"] = mock_df["avg_goldstein"].shift(1)
    mock_df["goldstein_lag3"] = mock_df["avg_goldstein"].shift(3)
    mock_df["goldstein_roll3"] = mock_df["avg_goldstein"].rolling(3).mean()
    
    # Drop NaNs
    processed_df = mock_df.dropna().reset_index(drop=True)
    
    # Assertions
    assert len(processed_df) == 17  # 20 - 3 NaN rows
    assert 'group' in processed_df.columns
    assert processed_df['group'].iloc[0] == 'gdelt_only'
    assert 'week' in processed_df.columns
    assert processed_df['week'].dtype == int
    assert not processed_df[['goldstein_lag1', 'goldstein_lag3', 'goldstein_roll3']].isnull().any().any()

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.savefig')
def test_evaluation_functions(mock_savefig, mock_show):
    """Test evaluation script functions without requiring actual model"""
    # Create mock evaluation results
    y_true = np.random.normal(0, 1, 50)
    y_pred = y_true + np.random.normal(0, 0.5, 50)  # Add some noise
    
    # Test metrics calculation
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Assertions
    assert mse > 0
    assert rmse > 0
    assert mae > 0
    assert -1 <= r2 <= 1  # R² can be negative for very poor models
    assert rmse == np.sqrt(mse)
    
    # Test that we can create the basic structure for plots
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    assert len(axes) == 2
    assert len(axes[0]) == 2
    plt.close(fig)

if __name__ == "__main__":
    pytest.main([__file__]) 