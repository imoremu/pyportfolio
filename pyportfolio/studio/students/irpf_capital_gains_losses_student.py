from pydatastudio.data.studio.students.abstract_data_basic_student import AbstractDataBasicStudent
from pydatastudio.data.studio.students.data_student_configuration import DataStudentConfiguration

from pyportfolio.calculators.fifo_calculator import FIFOCalculator
from pyportfolio.calculators.fifo_calculator import RESULT_FIFO_GAIN_LOSS

from pyportfolio.calculators.irpf_earnings_calculator import IrpfEarningsCalculator
from pyportfolio.transaction_manager import TransactionManager

from pyportfolio.studio.portfolio_studio import PORTFOLIO_INITIAL_DATA, RESEARCH_FIFO_EARNINGS

from pyportfolio.columns import GPP, RESULT_DEFERRED_ADJUSTMENT


from typing import Any
import pandas as pd

class IRPFCapitalGainsLossesStudent(AbstractDataBasicStudent):
    """
    A student class for handling IRPF calculations.
    
    This class extends AbstractDataBasicStudent and uses fifo_calculator and irpf_earnings_calculator to return fifo earnings and irpf capital gain loss research.
    """
    
    def __init__(self, configuration: DataStudentConfiguration):
        super().__init__(configuration)    
    
    def _research_fifo_earnings(self, research_name: str, **attrs: Any) -> pd.DataFrame:
        """
        Calculates FIFO earnings for a given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing transaction data.

        Returns:
            pd.DataFrame: A DataFrame with FIFO earnings calculated.
        """
        df = self.studio.research(PORTFOLIO_INITIAL_DATA)

        fifo_calculator = FIFOCalculator()
        tm = TransactionManager(df.copy())

        tm.register_calculation(
            calculator=fifo_calculator,
            column=RESULT_FIFO_GAIN_LOSS,
            dtype='Float64'
        )

        tm.process_all()

        return tm.transactions


    def _research_irpf_capital_gains_losses(self, research_name: str, **attrs: Any) -> pd.DataFrame:
        """
        Calculates IRPF capital gain/loss for a given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing transaction data.

        Returns:
            pd.DataFrame: A DataFrame with IRPF capital gain/loss calculated.
        """        
        df = self.studio.research(RESEARCH_FIFO_EARNINGS)

        irpf_calculator = IrpfEarningsCalculator()

        tm = TransactionManager(df.copy())
        tm.register_calculation(
            calculator=irpf_calculator
        )

        tm.process_all()

        result = tm.transactions

        # If result is empty, return a empty dataframe with all needed columns
        if result.empty:
            result = pd.DataFrame(columns=[GPP, RESULT_DEFERRED_ADJUSTMENT])


        return result