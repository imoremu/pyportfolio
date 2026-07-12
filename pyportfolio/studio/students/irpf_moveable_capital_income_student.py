from pydatastudio.data.studio.students.abstract_data_basic_student import AbstractDataBasicStudent
from pydatastudio.data.studio.students.data_student_configuration import DataStudentConfiguration

from pyportfolio.calculators.irpf_moveable_capital_income_calculator import IRPFMoveableCapitalIncomeCalculator
from pyportfolio.transaction_manager import TransactionManager

from pyportfolio.columns import RCM

from typing import Any
import pandas as pd

class IRPFMoveableCapitalIncomeStudent(AbstractDataBasicStudent):
    """
    A student class for handling IRPF calculations.
    
    This class extends AbstractDataBasicStudent and uses irpf_moveable_capital_income_calculator to return irpf moveable capital income research.    
    """
    
    def __init__(self, configuration: DataStudentConfiguration):
        super().__init__(configuration)        

    def _research_irpf_moveable_capital_income(self, research_name: str, **attrs: Any) -> pd.DataFrame:
        """
        Calculates IRPF moveable capital income for a given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing transaction data.

        Returns:
            pd.DataFrame: A DataFrame with IRPF moveable capital income calculated.
        """
        df = self.studio.research("portfolio initial data")

        irpf_moveable_capital_income_calculator = IRPFMoveableCapitalIncomeCalculator(df)

        tm = TransactionManager(df.copy())
        tm.register_calculation(
            calculator=irpf_moveable_capital_income_calculator,
            column=RCM,
            dtype='Float64'
        )

        tm.process_all()

        result = tm.transactions

        # If result is empty, return a empty dataframe with all needed columns
        if result.empty:
            result = pd.DataFrame(columns=[RCM])

        return result
        