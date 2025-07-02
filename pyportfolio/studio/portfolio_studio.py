"""
Created on 24 Jun 2025

@author: imoreno
"""
from pydatastudio.data.studio.data_studio import DataStudio
from pydatastudio.data.studio.environment.data_studio_environment import DataStudioEnvironment

PORTFOLIO_INITIAL_DATA = "portfolio initial data"

class PortfolioStudio(DataStudio):
    
    def __init__(self):
        super().__init__()       

