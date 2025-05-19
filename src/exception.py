import sys 

def error_message_detail(error, error_detail: sys):
    """
    This function takes an error and its details and returns a formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] with error message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It provides a formatted error message when an exception occurs.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
# This class is used to handle exceptions in a more user-friendly way.
# It formats the error message to include the file name, line number, and the original error message.
# This makes it easier to debug and understand where the error occurred in the code.
# The error_message_detail function is used to extract the file name, line number, and error message from the exception.
# This is useful for logging and debugging purposes.
# The CustomException class inherits from the built-in Exception class and overrides the __init__ and __str__ methods.
# The __init__ method initializes the error message and the __str__ method returns the formatted error message

