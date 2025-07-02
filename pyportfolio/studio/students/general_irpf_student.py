from pydatastudio.data.studio.abstractdatabasicstudent import AbstractDataBasicStudent
from pydatastudio.data.studio.datastudentconfiguration import DataStudentConfiguration
from typing import Any
import pandas as pd

class GeneralIRPFStudent(AbstractDataBasicStudent):
    """
    A student class for handling IRPF calculations.
    
    This class extends AbstractDataBasicStudent and implements specific methods for IRPF calculations.
    """
    
    def __init__(self, configuration: DataStudentConfiguration):
        super().__init__(configuration)        