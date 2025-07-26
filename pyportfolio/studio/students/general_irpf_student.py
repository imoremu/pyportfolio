from pydatastudio.data.studio.students.abstract_data_basic_student import AbstractDataBasicStudent
from pydatastudio.data.studio.students.data_student_configuration import DataStudentConfiguration

from typing import Any
import pandas as pd

from pyportfolio.calculators.loss_carry_forward_calculator import LossCarryForwardCalculator

from pyportfolio.columns import DATETIME

from pyportfolio.studio.portfolio_studio import RESEARCH_IRPF_CAPITAL_GAIN_LOSS, RESEARCH_ANNUAL_IRPF_SUMMARY, \
    RESEARCH_IRPF_MOVEABLE_CAPITAL_INCOME

from pyportfolio.columns import GPP_ALLOWABLE, GPP_TOTAL, RCM

import logging

class GeneralIRPFStudent(AbstractDataBasicStudent):
    """
    A student class for handling general IRPF calculations, including
    annual summaries and loss carry-forward analysis.
    """

    def __init__(self, configuration: DataStudentConfiguration):
        super().__init__(configuration)
        self.logger = logging.getLogger(__name__)


    def _research_annual_irpf_summary(self, research_name: str, **attrs: Any) -> pd.DataFrame:
        """
        Aggregates detailed IRPF results into an annual summary.

        This research depends on 'IRPF Capital Gain Loss' and
        'IRPF Moveable Capital Income' being calculated first.
        """
        # 1. Get the required detailed results from the studio
        gpp_df = self.studio.research(RESEARCH_IRPF_CAPITAL_GAIN_LOSS)
        rcm_df = self.studio.research(RESEARCH_IRPF_MOVEABLE_CAPITAL_INCOME)

        combined_df = pd.concat([gpp_df, rcm_df], ignore_index=True)

        self.logger.debug(f"Combined DataFrame shape: {combined_df.shape}")
        
        # Check if DATETIME column is a date
        if not pd.api.types.is_datetime64_any_dtype(combined_df[DATETIME]):
            try:
                combined_df[DATETIME] = pd.to_datetime(combined_df[DATETIME], format='mixed', dayfirst=True)

            except Exception as e:
                self.logger.error(f"Error converting DATETIME column to datetime: {e}")

                raise ValueError(f"Could not convert internal date {combined_df[DATETIME]} column '{DATETIME}' to datetime: {e}")     
            
        # Extract year from the DATETIME column
        combined_df['Year'] = combined_df[DATETIME].dt.year

        # 3. Group by year and sum the results
        annual_summary = combined_df.groupby('Year')[[
            GPP_ALLOWABLE,
            GPP_TOTAL,
            RCM
        ]].sum(numeric_only=True).fillna(0)

        # Rename columns for clarity and compatibility with the calculator
        return annual_summary

    def _research_irpf_loss_carry_forward_ledger(self, research_name: str, **attrs: Any) -> pd.DataFrame:
        """
        Calculates the 4-year loss carry-forward ledger based on the annual IRPF summary.
        """
        annual_summary_df = self.studio.research(RESEARCH_ANNUAL_IRPF_SUMMARY)
        calculator = LossCarryForwardCalculator()
        return calculator.calculate_table(annual_summary_df)