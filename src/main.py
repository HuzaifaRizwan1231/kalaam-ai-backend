from fastapi import FastAPI, Request
from .config.db import engine, Base
from .api import register_routes
from .logging import configure_logging, LogLevels
from .rate_limiter import limiter
from .utils.exception_handler import register_exception_handlers
from .middleware.auth import CurrentUser

configure_logging(LogLevels.info)

app = FastAPI()

app.state.limiter = limiter

# Register all exception handlers
register_exception_handlers(app)

""" Only uncomment below to create new tables, 
otherwise the tests will fail if not connected
"""
#Base.metadata.create_all(bind=engine)

register_routes(app)

@app.get("/")
async def read_root(current_user: CurrentUser, request: Request):
    return {"message": "Hello world"}