"""
Created on 24 Jun 2025

@author: imoreno
"""
from pydatastudio.data.studio.data_studio import DataStudio
from pydatastudio.data.studio.environment.data_studio_environment import DataStudioEnvironment



# Researches
PORTFOLIO_INITIAL_DATA = "portfolio initial data"
RESEARCH_FIFO_EARNINGS = "FIFO Earnings"
RESEARCH_IRPF_CAPITAL_GAIN_LOSS = "IRPF Capital Gains Losses"
RESEARCH_ANNUAL_IRPF_SUMMARY = "Annual IRPF Summary"
RESEARCH_IRPF_MOVEABLE_CAPITAL_INCOME = "IRPF Moveable Capital Income"
RESEARCH_IRPF_LOSS_CARRYFORWARD_LEDGER = "IRPF Loss Carryforward Ledger"

class PortfolioStudio(DataStudio):
    
    def __init__(self):
        super().__init__()       

