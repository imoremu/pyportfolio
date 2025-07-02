from pydatastudio.data.studio.students.abstract_data_basic_student import AbstractDataBasicStudent
from pydatastudio.data.studio.students.data_student_configuration import DataStudentConfiguration

class IRPFMoveableCapitalIncomeStudent(AbstractDataBasicStudent):
    """
    A student class for handling IRPF calculations.
    
    This class extends AbstractDataBasicStudent and implements specific methods for IRPF calculations.
    """
    
    def __init__(self, configuration: DataStudentConfiguration):
        super().__init__(configuration)        