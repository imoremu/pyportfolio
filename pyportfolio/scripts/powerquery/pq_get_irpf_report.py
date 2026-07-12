'''
Created on 05 Jul 2025

@author: imoreno
'''

import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List
import logging

# Assuming the following imports are available in the project
from pyportfolio.studio.portfolio_studio import PortfolioStudio
from pydatastudio.data.studio.students.student_factory import StudentFactory

from pydatastudio.environment.environment_config_management import get_environment_config_by_yaml_file
from pydatastudio.environment.environment_config import _EnvironmentConfig
from pydatastudio.data.studio.data_studio_constants import ENVIRONMENT_STUDENTS_KEY
# This is the function we want to test.
# For the test file, we assume it's in a module that can be imported.
# from pyportfolio.power_query.irpf_report import pq_get_irpf_report, _load_studio_config

# For demonstration, the function is included here. In a real project, you would import it.
# --- Start of code to be tested ---

PORTFOLIO_INITIAL_DATA = 'portfolio initial data'
RESEARCH_IRPF_LOSS_CARRYFORWARD_LEDGER = 'IRPF loss carry forward ledger'
TEST_COLUMN_NAME = 'Test'

def _get_config_path():

    """
    Returns the absolute path to the studio configuration YAML file.
    """
    base_dir = Path(__file__).parent.parent.parent
    return base_dir / 'config' / 'studio' / 'portfolio_env.yaml'
    

def _load_studio_config() -> _EnvironmentConfig:
    """
    Loads, parses, and validates the studio configuration from a YAML file.
    
    This helper function isolates file I/O and parsing from the main logic.
    
    Returns:
        Dict: The parsed studio configuration.
        
    Raises:
        FileNotFoundError: If the config file cannot be found.
        ValueError: If the config file is empty or missing the 'students' key.
    """
    config_path = _get_config_path()
    
    studio_config = get_environment_config_by_yaml_file(config_path, 'default')
        
    return studio_config


def pq_get_irpf_report(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a portfolio DataFrame using the pyDataStudio framework to generate
    a comprehensive IRPF report.
    """
    logger = logging.getLogger(__name__)
    try:
        # --- Configuration is now loaded via the helper method ---
        studio_config = _load_studio_config()

        # --- Input Validation ---
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Input 'dataset' is not a pandas DataFrame.")
        if dataset.empty:
            logger.warning("Input dataset is empty. Returning empty results.")
            return {'LossCarryforwardLedger': pd.DataFrame()}
        if TEST_COLUMN_NAME not in dataset.columns:
            raise ValueError(f"Mandatory column '{TEST_COLUMN_NAME}' not found in the input dataset.")

        # --- Grouped Processing ---
        all_loss_ledgers: List[pd.DataFrame] = []
        grouped = dataset.groupby(TEST_COLUMN_NAME)

        for test_name, group_df in grouped:
            logger.info(f"--- Processing Test Scenario: {test_name} ({len(group_df)} rows) ---")
            studio = PortfolioStudio()
            factory = StudentFactory(studio_config.get_config_value(ENVIRONMENT_STUDENTS_KEY))
            students = factory.create_students()
            studio.add_students(students)
            studio.add_research(PORTFOLIO_INITIAL_DATA, group_df.copy())
            
            loss_ledger_df = studio.research(RESEARCH_IRPF_LOSS_CARRYFORWARD_LEDGER)
            loss_ledger_df[TEST_COLUMN_NAME] = test_name
            
            all_loss_ledgers.append(loss_ledger_df)

        final_ledger = pd.concat(all_loss_ledgers).reset_index() if all_loss_ledgers else pd.DataFrame()
        logger.info("Successfully processed all test scenarios.")
        return final_ledger

    except (FileNotFoundError, ValueError, TypeError, yaml.YAMLError) as e:
        logger.error(f"A configuration or data validation error occurred: {e}", exc_info=True)
        return pd.DataFrame({'Error': [str(e)]})
    except Exception as e:
        logger.error(f"A critical error occurred during processing: {e}", exc_info=True)
        return pd.DataFrame({'Error': [str(e)]})