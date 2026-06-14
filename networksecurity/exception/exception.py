import sys
from networksecurity.logging.logger import logger


class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(
            error_message, error_detail
        )

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"with error message: [{error_message}]"
        )
        return error_message

    def __str__(self):
        return self.error_message