import sys,os

def error_message_detail(error, error_detail: sys):
    """
    This function return details of error 
    """
    _, _, exc_tb = error_detail.exc_info()
    # file name
    file_name = exc_tb.tb_frame.f_code.co_filename
    # line number
    line_no = exc_tb.tb_lineno
    error_message = "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(file_name, line_no, str(error))
    return error_message

class SensorException(Exception):

    def __init__(self,error_message, error_detail:sys):
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
