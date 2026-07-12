'''

Created on 02/07/2025

@author: imoreno

'''

import pandas as pd
import unittest

from pyportfolio.calculators.loss_carry_forward_calculator import (
    LossCarryForwardCalculator,
    YEAR, GP_INITIAL, GP_POST_COMP, GPP_TOTAL, GP_TAXABLE_BASE, GP_LOSS_AVAILABLE, GP_LOSS_CARRIED_FORWARD,
    RCM_INITIAL, RCM_POST_COMP, RCM_TAXABLE_BASE, RCM_LOSS_AVAILABLE, RCM_LOSS_CARRIED_FORWARD,
    TOTAL_TAXABLE_BASE
)


class TestLossCarryForwardCalculator(unittest.TestCase):
    """Test suite for the LossCarryForwardCalculator class."""

    def test_initialization_errors(self):
        """Test that the class raises errors for invalid inputs."""
        calculator = LossCarryForwardCalculator()
        
        # Test for non-DataFrame input
        with self.assertRaises(TypeError):
            calculator.calculate_table("not a dataframe")
        
        # Test for missing required columns
        with self.assertRaises(ValueError):
            df = pd.DataFrame({YEAR: [2020], 'A': [1], 'B': [2]}).set_index(YEAR)
            calculator.calculate_table(df)
            
        # Test for non-integer index (e.g., string)
        with self.assertRaises(ValueError):
            df = pd.DataFrame({GP_INITIAL: [1], RCM_INITIAL: [2]}, index=['2020'])
            calculator.calculate_table(df)

    def test_simple_gp_loss_carryforward(self):
        """Test a simple G&P loss being compensated in the next year."""
        data = {YEAR: [2020, 2021], GP_INITIAL: [-1000, 1500], RCM_INITIAL: [0, 0], GPP_TOTAL: [-1200, 1500]}
        df = pd.DataFrame(data).set_index(YEAR)
        calculator = LossCarryForwardCalculator()
        analysis_table = calculator.calculate_table(df)
        
        # Check 2020 results
        self.assertAlmostEqual(analysis_table.loc[2020, GP_INITIAL], -1000)
        self.assertAlmostEqual(analysis_table.loc[2020, GPP_TOTAL], -1200)
        self.assertAlmostEqual(analysis_table.loc[2020, GP_LOSS_AVAILABLE], 0)
        self.assertAlmostEqual(analysis_table.loc[2020, GP_POST_COMP], -1000)
        self.assertAlmostEqual(analysis_table.loc[2020, GP_LOSS_CARRIED_FORWARD], 1000)

        # Check 2021 results
        self.assertAlmostEqual(analysis_table.loc[2021, GP_INITIAL], 1500)
        self.assertAlmostEqual(analysis_table.loc[2021, GPP_TOTAL], 1500)
        self.assertAlmostEqual(analysis_table.loc[2021, GP_LOSS_AVAILABLE], 1000)
        self.assertAlmostEqual(analysis_table.loc[2021, GP_POST_COMP], 500)
        self.assertAlmostEqual(analysis_table.loc[2021, GP_TAXABLE_BASE], 500)
        self.assertAlmostEqual(analysis_table.loc[2021, GP_LOSS_CARRIED_FORWARD], 0)

    def test_simple_rcm_loss_carryforward(self):
        """Test a simple RCM loss being compensated in the next year."""
        data = {YEAR: [2020, 2021], GP_INITIAL: [0, 0], RCM_INITIAL: [-500, 2000], GPP_TOTAL: [0, 0]}
        df = pd.DataFrame(data).set_index(YEAR)
        calculator = LossCarryForwardCalculator()
        analysis_table = calculator.calculate_table(df)
        
        # Check 2020 results
        self.assertAlmostEqual(analysis_table.loc[2020, RCM_LOSS_AVAILABLE], 0)
        self.assertAlmostEqual(analysis_table.loc[2020, RCM_POST_COMP], -500)
        self.assertAlmostEqual(analysis_table.loc[2020, RCM_LOSS_CARRIED_FORWARD], 500)

        # Check 2021 results
        self.assertAlmostEqual(analysis_table.loc[2021, RCM_LOSS_AVAILABLE], 500)
        self.assertAlmostEqual(analysis_table.loc[2021, RCM_POST_COMP], 1500)
        self.assertAlmostEqual(analysis_table.loc[2021, RCM_TAXABLE_BASE], 1500)
        self.assertAlmostEqual(analysis_table.loc[2021, RCM_LOSS_CARRIED_FORWARD], 0)

    def test_last_valid_year_of_loss(self):
        """Test that a loss is available on its 4th and final year."""
        data = {YEAR: [2018, 2022, 2023], GP_INITIAL: [-1000, 800, 0], RCM_INITIAL: [0, 0, 0], GPP_TOTAL: [-1000, 800, 0]}
        df = pd.DataFrame(data).set_index(YEAR)
        calculator = LossCarryForwardCalculator()
        analysis_table = calculator.calculate_table(df)

        # Check state at the start of 2022
        self.assertAlmostEqual(analysis_table.loc[2022, GP_LOSS_AVAILABLE], 1000)
        
        # Check results of 2022 calculation
        # The 800 gain is fully absorbed by the 1000 available loss.
        # The post-compensation result becomes 0, not negative.
        self.assertAlmostEqual(analysis_table.loc[2022, GP_POST_COMP], 0)
        self.assertAlmostEqual(analysis_table.loc[2022, GP_TAXABLE_BASE], 0)
        
        # The remaining 200 of the 2018 loss is carried forward.
        self.assertAlmostEqual(analysis_table.loc[2022, GP_LOSS_CARRIED_FORWARD], 200)

        # Check state at the start of 2023 (after the remaining 200 from 2018 expires)
        self.assertAlmostEqual(analysis_table.loc[2023, GP_LOSS_AVAILABLE], 0)


    def test_loss_expiration(self):
        """Test that a loss expires after 4 years."""
        data = {YEAR: [2018, 2023], GP_INITIAL: [-1000, 500], RCM_INITIAL: [0, 0], GPP_TOTAL: [-1000, 500]}
        df = pd.DataFrame(data).set_index(YEAR)
        calculator = LossCarryForwardCalculator()
        analysis_table = calculator.calculate_table(df)

        # The loss from 2018 expires at the start of 2023, so it cannot offset the gain.
        self.assertAlmostEqual(analysis_table.loc[2023, GP_LOSS_AVAILABLE], 0)
        self.assertAlmostEqual(analysis_table.loc[2023, GP_POST_COMP], 500)
        self.assertAlmostEqual(analysis_table.loc[2023, GP_TAXABLE_BASE], 500)

    def test_inter_category_compensation_from_gp_to_rcm(self):
        """Test G&P loss compensating an RCM gain (25% rule)."""
        data = {YEAR: [2020], GP_INITIAL: [-5000], RCM_INITIAL: [16000], GPP_TOTAL: [-5000]}
        df = pd.DataFrame(data).set_index(YEAR)
        calculator = LossCarryForwardCalculator()
        analysis_table = calculator.calculate_table(df)

        # RCM gain is 16000. Limit for compensation is 0.25 * 16000 = 4000.
        # G&P loss is -5000. We can use up to 4000 of it.
        self.assertAlmostEqual(analysis_table.loc[2020, RCM_POST_COMP], 12000) # 16000 - 4000
        self.assertAlmostEqual(analysis_table.loc[2020, GP_POST_COMP], -1000) # -5000 + 4000
        self.assertAlmostEqual(analysis_table.loc[2020, GP_LOSS_CARRIED_FORWARD], 1000)
        self.assertAlmostEqual(analysis_table.loc[2020, TOTAL_TAXABLE_BASE], 12000)
        
    def test_inter_category_compensation_from_rcm_to_gp(self):
        """Test RCM loss compensating a G&P gain (25% rule)."""
        data = {YEAR: [2020], GP_INITIAL: [20000], RCM_INITIAL: [-3000], GPP_TOTAL: [20000]}
        df = pd.DataFrame(data).set_index(YEAR)
        calculator = LossCarryForwardCalculator()
        analysis_table = calculator.calculate_table(df)

        # G&P gain is 20000. Limit for compensation is 0.25 * 20000 = 5000.
        # RCM loss is -3000. We can use all of it as it's less than the limit.
        self.assertAlmostEqual(analysis_table.loc[2020, GP_POST_COMP], 17000) # 20000 - 3000
        self.assertAlmostEqual(analysis_table.loc[2020, RCM_POST_COMP], 0) # -3000 + 3000
        self.assertAlmostEqual(analysis_table.loc[2020, RCM_LOSS_CARRIED_FORWARD], 0)
        self.assertAlmostEqual(analysis_table.loc[2020, TOTAL_TAXABLE_BASE], 17000)

    def test_total_taxable_base_with_both_gains(self):
        """Test Total_Taxable_Base when both G&P and RCM are positive."""
        data = {YEAR: [2020, 2021], GP_INITIAL: [-1000, 5000], RCM_INITIAL: [0, 2000], GPP_TOTAL: [-1000, 5000]}
        df = pd.DataFrame(data).set_index(YEAR)
        calculator = LossCarryForwardCalculator()
        analysis_table = calculator.calculate_table(df)

        # In 2021, G&P gain of 5000 is offset by 1000 loss from 2020.
        # RCM gain of 2000 is unaffected.
        self.assertAlmostEqual(analysis_table.loc[2021, GP_TAXABLE_BASE], 4000)
        self.assertAlmostEqual(analysis_table.loc[2021, RCM_TAXABLE_BASE], 2000)
        self.assertAlmostEqual(analysis_table.loc[2021, TOTAL_TAXABLE_BASE], 6000)

    def test_non_consecutive_years(self):
        """Test the calculator with gaps in the years."""
        data = {YEAR: [2018, 2021], GP_INITIAL: [-1000, 1200], RCM_INITIAL: [0, 0], GPP_TOTAL: [-1000, 1200]}
        df = pd.DataFrame(data).set_index(YEAR)
        calculator = LossCarryForwardCalculator()
        analysis_table = calculator.calculate_table(df)

        # Check the gap years
        self.assertAlmostEqual(analysis_table.loc[2019, GP_LOSS_AVAILABLE], 1000)
        self.assertAlmostEqual(analysis_table.loc[2019, GP_LOSS_CARRIED_FORWARD], 1000)
        self.assertAlmostEqual(analysis_table.loc[2020, GP_LOSS_AVAILABLE], 1000)
        self.assertAlmostEqual(analysis_table.loc[2020, GP_LOSS_CARRIED_FORWARD], 1000)

        # Check the year with compensation
        self.assertAlmostEqual(analysis_table.loc[2021, GP_LOSS_AVAILABLE], 1000)
        self.assertAlmostEqual(analysis_table.loc[2021, GP_POST_COMP], 200)
        self.assertAlmostEqual(analysis_table.loc[2021, GP_TAXABLE_BASE], 200)
        self.assertAlmostEqual(analysis_table.loc[2021, GP_LOSS_CARRIED_FORWARD], 0)
    
    def test_compensation_from_gpp_to_rcp_in_different_years(self):
        """Test G&P loss compensating an RCM gain (25% rule) across different years."""
        data = {
            YEAR: [2020, 2021, 2022],
            GP_INITIAL: [-5000, 0, 0],
            RCM_INITIAL: [16000, 500, 0],
            GPP_TOTAL: [-5000, 0, 0]
        }
        df = pd.DataFrame(data).set_index(YEAR)
        calculator = LossCarryForwardCalculator()
        analysis_table = calculator.calculate_table(df)

        # 2020: G&P loss of -5000, RCM 16000.
        # Limit for compensation is 0.25 * 16000 = 4000.
        # We use 4000 from G&P loss to compensate RCM gain.
        self.assertAlmostEqual(analysis_table.loc[2020, GP_POST_COMP], -1000)
        self.assertAlmostEqual(analysis_table.loc[2020, GP_LOSS_CARRIED_FORWARD], 1000)
        self.assertAlmostEqual(analysis_table.loc[2020, RCM_POST_COMP], 12000)
        self.assertAlmostEqual(analysis_table.loc[2020, RCM_LOSS_CARRIED_FORWARD], 0)

        # 2021: RCM gain of 16000. G&P loss available is 1000.
        # Limit for compensation is 0.25 * 16000 = 4000.
        # We use 1000 from G&P loss to compensate RCM gain.
        self.assertAlmostEqual(analysis_table.loc[2021, GP_LOSS_AVAILABLE], 1000)
        self.assertAlmostEqual(analysis_table.loc[2021, RCM_LOSS_AVAILABLE], 0)

        self.assertAlmostEqual(analysis_table.loc[2021, RCM_POST_COMP], 375) # 500 - 500
        self.assertAlmostEqual(analysis_table.loc[2021, GP_POST_COMP], 0) # G&P is not a gain, so it's not affected by its own pool.
        self.assertAlmostEqual(analysis_table.loc[2021, GP_LOSS_CARRIED_FORWARD], 875)

        # 2022: No G&P loss available, so RCM gain is not compensated.        
        self.assertAlmostEqual(analysis_table.loc[2022, GP_LOSS_AVAILABLE], 875)
                                                                           

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)