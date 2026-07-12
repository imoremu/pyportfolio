'''
Created on 05 Jul 2025

@author: imoreno
'''

import pandas as pd
import pytest


from pyportfolio.scripts.powerquery.pq_get_irpf_report import pq_get_irpf_report, _get_config_path, \
    TEST_COLUMN_NAME
from pyportfolio.studio.portfolio_studio import PortfolioStudio

from pydatastudio.data.studio.students.student_factory import StudentFactory

from pyportfolio.columns import DATETIME, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE, COMISION, TYPE_BUY, TYPE_SELL, GPP_ALLOWABLE, \
    GPP_TOTAL

import pyportfolio.logging_setup

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Provides a sample DataFrame for testing."""
    # required_columns = [DATETIME, TRANSACTION_TYPE, TICKER, SHARES, SHARE_PRICE, COMISION] should be included
    return pd.DataFrame({
        TEST_COLUMN_NAME: ['ScenarioA', 'ScenarioA', 'ScenarioB', 'ScenarioB'],
        DATETIME: pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        TRANSACTION_TYPE: [TYPE_BUY, TYPE_SELL, TYPE_BUY, TYPE_SELL],
        TICKER: ['ABC', 'ABC', 'XYZ', 'XYZ'],
        SHARES: [10, -5, 20, -10],
        SHARE_PRICE: [100, 120, 50, 40],
        COMISION: [5, 2, 20, 10]
    })

def test_successful_run_empty_dataset():
    """Test that an empty dataset returns an empty DataFrame."""
    empty_df = pd.DataFrame(columns=[TEST_COLUMN_NAME, 'Date', 'TransactionType', 'Ticker', 'Shares', 'SharePrice', 'Comision'])
    output = pq_get_irpf_report(empty_df)
    final_df = output['LossCarryforwardLedger']
    assert final_df.empty
    assert 'Error' not in final_df.columns

def test_successful_run_single_group(sample_data):
    return pd.DataFrame({
        'Test': ['ScenarioA', 'ScenarioA', 'ScenarioB'],
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'Value': [100, 200, 300],        
    })

def test_successful_run_multiple_groups(sample_data):
    """
    Test a successful run with multiple groups, ensuring data is processed
    and concatenated correctly.
    """    
    # --- Act ---
    output = pq_get_irpf_report(sample_data)

    # --- Assert ---
    final_df = output
    
    assert len(final_df) == 2
    assert TEST_COLUMN_NAME in final_df.columns
    assert final_df.loc[0, GPP_ALLOWABLE] == 95.5
    assert final_df.loc[0, GPP_TOTAL] == 95.5
    assert final_df.loc[0, TEST_COLUMN_NAME] == 'ScenarioA'
    assert final_df.loc[1, GPP_ALLOWABLE] == 0
    assert final_df.loc[1, GPP_TOTAL] == -120
    assert final_df.loc[1, TEST_COLUMN_NAME] == 'ScenarioB'

def test_input_validation_not_a_dataframe():
    """Test that the function returns an error DataFrame for non-DataFrame input."""
    output = pq_get_irpf_report("this is not a dataframe")
    error_df = output
    assert 'Error' in error_df.columns
    assert "not a pandas DataFrame" in error_df['Error'].iloc[0]

def test_input_validation_missing_test_column(sample_data):
    """Test for error when the mandatory 'Test' column is missing."""    
    invalid_data = sample_data.drop(columns=['Test'])
    
    output = pq_get_irpf_report(invalid_data)
    
    error_df = output
    assert 'Error' in error_df.columns
    assert "Mandatory column 'Test' not found" in error_df['Error'].iloc[0]