from fastapi import FastAPI, Request
from .config.db import engine, Base, DbSession
from .api import register_routes
from .logging import configure_logging, LogLevels
from .rate_limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

configure_logging(LogLevels.info)

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

""" Only uncomment below to create new tables, 
otherwise the tests will fail if not connected
"""
# Base.metadata.create_all(bind=engine)

register_routes(app)

@app.get("/")
async def read_root(request: Request):
    return {"message": "Hello world"}