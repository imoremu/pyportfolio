'''
Created on 04/07/2024

@author: Gemini
'''

import pandas as pd
from collections import deque

from pyportfolio.calculators.base_calculator import BaseTableCalculator

from pyportfolio.columns import (    
    YEAR, LOSS_YEAR, LOSS_AMOUNT,
    GP_INITIAL, GP_POST_COMP, GP_TAXABLE_BASE, GP_LOSS_AVAILABLE, GP_LOSS_CARRIED_FORWARD,
    RCM_INITIAL, RCM_POST_COMP, RCM_TAXABLE_BASE, RCM_LOSS_AVAILABLE, RCM_LOSS_CARRIED_FORWARD,
    TOTAL_TAXABLE_BASE
)


class LossCarryForwardCalculator(BaseTableCalculator):
    """
    Calculates the 4-year loss carry-forward for Spanish IRPF.

    This calculator processes annual financial results to track and apply
    net losses against future gains according to Spanish tax law. It handles
    both 'Ganancias y Pérdidas Patrimoniales' (G&P) and 'Rendimientos del
    Capital Mobiliario' (RCM) with their specific compensation rules.

    Key features:
    - Tracks losses for up to 4 years.
    - Applies losses on a First-In, First-Out (FIFO) basis.
    - Handles the 25% compensation limit between G&P and RCM pools.
    - Produces a single, detailed tax analysis table suitable for BI tools.
    """

    def __init__(self):
        """
        Initializes the calculator with annual results.

        Args:
            annual_results (pd.DataFrame): A DataFrame with a YEAR index
                and columns [GP_INITIAL, RCM_INITIAL].
        """
        super().__init__()

    def calculate_table(self, annual_results: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the loss carry-forward calculation across all years.

        Returns:
            pd.DataFrame: A single, detailed table of income and compensations per year.
        """
        if not isinstance(annual_results, pd.DataFrame):
            raise TypeError("annual_results must be a pandas DataFrame.")
        if not all(col in annual_results.columns for col in [GP_INITIAL, RCM_INITIAL]):
            raise ValueError(f"annual_results must contain '{GP_INITIAL}' and '{RCM_INITIAL}' columns.")
        if not pd.api.types.is_integer_dtype(annual_results.index.dtype) and not annual_results.index.empty:
             raise ValueError("annual_results must have an integer index representing the year.")

        annual_results = annual_results.sort_index()

        gp_losses = deque()  # Stores dicts of {LOSS_YEAR: year_of_loss, LOSS_AMOUNT: amount}
        rcm_losses = deque() # Stores dicts of {LOSS_YEAR: year_of_loss, LOSS_AMOUNT: amount}
        tax_analysis_data = []

        all_years = annual_results.index
        if all_years.empty:
            return pd.DataFrame()

        # Iterate from the first year of data to the last
        for year in range(all_years.min(), all_years.max() + 1):
            # 1. Expire old losses (older than 4 years) FIRST.
            self._expire_losses(year, gp_losses, rcm_losses)

            # 2. Capture the state of available losses at the START of the year.
            gp_loss_available = sum(loss[LOSS_AMOUNT] for loss in gp_losses)
            rcm_loss_available = sum(loss[LOSS_AMOUNT] for loss in rcm_losses)

            # 3. Process current year's results if they exist
            if year not in annual_results.index:
                tax_analysis_data.append({
                    YEAR: year, GP_INITIAL: 0, RCM_INITIAL: 0,
                    GP_LOSS_AVAILABLE: gp_loss_available, RCM_LOSS_AVAILABLE: rcm_loss_available,
                    GP_POST_COMP: 0, RCM_POST_COMP: 0,
                    GP_TAXABLE_BASE: 0, RCM_TAXABLE_BASE: 0, TOTAL_TAXABLE_BASE: 0,
                    GP_LOSS_CARRIED_FORWARD: gp_loss_available, RCM_LOSS_CARRIED_FORWARD: rcm_loss_available
                })
                continue

            initial_gp = annual_results.loc[year, GP_INITIAL]
            initial_rcm = annual_results.loc[year, RCM_INITIAL]
            
            net_gp = initial_gp
            net_rcm = initial_rcm

            # 4. Compensate within the same category first taken into account losses available from previous years            
            net_gp = self._compensate_with_pool(net_gp, gp_losses)
            net_rcm = self._compensate_with_pool(net_rcm, rcm_losses)       

            # 5.1 Compensate between categores (up to 25% limit) taken into account losses available from previous years            
            net_gp = 0.75 * net_gp + self._compensate_with_pool(0.25 * net_gp, rcm_losses)
            net_rcm = 0.75 * net_rcm + self._compensate_with_pool(0.25 * net_rcm, gp_losses)
            
            # 5.2 Compensate between categories (up to 25% limit)
            if net_gp > 0 and net_rcm < 0:
                limit = 0.25 * net_gp
                amount_to_compensate = min(abs(net_rcm), limit)
                net_gp -= amount_to_compensate
                net_rcm += amount_to_compensate

            elif net_rcm > 0 and net_gp < 0:
                limit = 0.25 * net_rcm
                amount_to_compensate = min(abs(net_gp), limit)
                net_rcm -= amount_to_compensate
                net_gp += amount_to_compensate


            # 6. Add any new net losses to the carry-forward pools
            if net_gp < 0:
                gp_losses.append({LOSS_YEAR: year, LOSS_AMOUNT: abs(net_gp)})
            if net_rcm < 0:
                rcm_losses.append({LOSS_YEAR: year, LOSS_AMOUNT: abs(net_rcm)})
            
            # 7. Record the detailed analysis data for the year
            tax_analysis_data.append({
                YEAR: year,
                GP_INITIAL: initial_gp,
                RCM_INITIAL: initial_rcm,
                GP_LOSS_AVAILABLE: gp_loss_available,
                RCM_LOSS_AVAILABLE: rcm_loss_available,
                GP_POST_COMP: net_gp,
                RCM_POST_COMP: net_rcm,
                GP_TAXABLE_BASE: max(0, net_gp),
                RCM_TAXABLE_BASE: max(0, net_rcm),
                TOTAL_TAXABLE_BASE: max(0, net_gp) + max(0, net_rcm),
                GP_LOSS_CARRIED_FORWARD: sum(loss[LOSS_AMOUNT] for loss in gp_losses),
                RCM_LOSS_CARRIED_FORWARD: sum(loss[LOSS_AMOUNT] for loss in rcm_losses)
            })
        
        analysis_df = pd.DataFrame(tax_analysis_data).set_index(YEAR)
        return analysis_df

    def _expire_losses(self, current_year: int, gp_losses: deque, rcm_losses: deque):
        """Removes losses that are more than 4 years old."""
        while gp_losses and gp_losses[0][LOSS_YEAR] < current_year - 4:
            gp_losses.popleft()

        while rcm_losses and rcm_losses[0][LOSS_YEAR] < current_year - 4:
            rcm_losses.popleft()

    def _compensate_with_pool(self, gain: float, loss_pool: deque) -> float:
        """Compensates a gain against a pool of losses (FIFO)."""
        if gain <= 0:
            return gain
        
        remaining_gain = gain
        
        while loss_pool and remaining_gain > 0:
            oldest_loss = loss_pool[0]
            amount_to_use = min(remaining_gain, oldest_loss[LOSS_AMOUNT])
            remaining_gain -= amount_to_use
            oldest_loss[LOSS_AMOUNT] -= amount_to_use
            if oldest_loss[LOSS_AMOUNT] < 1e-9:
                loss_pool.popleft()
        
        return remaining_gain


