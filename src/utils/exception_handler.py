from fastapi import Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .response_builder import ResponseBuilder


async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert HTTPException to ResponseBuilder format"""
    response = ResponseBuilder.error(exc.detail, exc.status_code)
    return JSONResponse(status_code=exc.status_code, content=response)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for Pydantic validation errors"""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"][1:])  # Skip 'body'
        message = error["msg"]
        errors.append(f"{field}: {message}")
    
    error_message = "; ".join(errors) if errors else "Validation error"
    
    response = ResponseBuilder.error(error_message, status.HTTP_422_UNPROCESSABLE_ENTITY)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response
    )


def register_exception_handlers(app):
    """Register all exception handlers for the FastAPI app"""
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)