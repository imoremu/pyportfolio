import pandas as pd

from pyportfolio.transaction_manager import TransactionManager
from pyportfolio.calculators.fifo_calculator import FIFOCalculator
from pyportfolio.calculators.irpf_earnings_calculator import IrpfEarningsCalculator
from pyportfolio.columns import FIFO

fifo_result_column = FIFO # Usar constante 'FIFO Gain/Loss'
irpf_result_columns = ['IRPF - Ganancia / Pérdida Imputable', 'IRPF - Ajuste Diferido']
all_result_columns = [fifo_result_column] + irpf_result_columns

fifo_result_dtype = 'Float64'
irpf_result_dtypes = ['Float64', 'Float64']
all_result_dtypes = [fifo_result_dtype] + irpf_result_dtypes

try:
    tm = TransactionManager(dataset.copy())

    fifo_calculator = FIFOCalculator(transactions=tm.transactions)
    tm.register_calculation(
        column=fifo_result_column,
        calculator=fifo_calculator,
        dtype=fifo_result_dtype
    )

    irpf_calculator = IrpfEarningsCalculator(transactions_df=tm.transactions)
    tm.register_calculation(
        column=irpf_result_columns,
        calculator=irpf_calculator,
        dtype=irpf_result_dtypes
    )

    tm.process_all()

    results_df = tm.transactions[all_result_columns].copy()
    results_df.index = dataset.index # Reasegurar el índice

except Exception as e:
    results_df = pd.DataFrame(index=dataset.index, columns=all_result_columns)
    results_df['Error'] = str(e)

result = results_df
