from typing import Any, Optional
from pydantic import BaseModel


class ApiResponse(BaseModel):
    """Standardized API response model"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    status_code: int


class ResponseBuilder:
    """Builder for creating standardized API responses"""
    
    def __init__(self):
        self._success: bool = True
        self._data: Optional[Any] = None
        self._error: Optional[str] = None
        self._message: Optional[str] = None
        self._status_code: int = 200

    def set_success(self, success: bool) -> 'ResponseBuilder':
        """Set success status"""
        self._success = success
        return self

    def set_data(self, data: Any) -> 'ResponseBuilder':
        """Set response data"""
        self._data = data
        return self

    def set_error(self, error: str) -> 'ResponseBuilder':
        """Set error message"""
        self._error = error
        self._success = False
        return self

    def set_message(self, message: str) -> 'ResponseBuilder':
        """Set informational message"""
        self._message = message
        return self

    def set_status_code(self, status_code: int) -> 'ResponseBuilder':
        """Set HTTP status code"""
        self._status_code = status_code
        return self

    def build(self) -> dict:
        """Build and return the response dictionary"""
        return {
            "success": self._success,
            "data": self._data,
            "error": self._error,
            "message": self._message,
            "status_code": self._status_code
        }
    
    @staticmethod
    def success(data: Any = None, message: str = None, status_code: int = 200) -> dict:
        """Quick success response"""
        return ResponseBuilder().set_data(data).set_message(message).set_status_code(status_code).build()
    
    @staticmethod
    def error(error: str, status_code: int = 400) -> dict:
        """Quick error response"""
        return ResponseBuilder().set_error(error).set_status_code(status_code).build()