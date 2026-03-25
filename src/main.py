import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .config.db import engine, Base
from .api import register_routes
from .logging import configure_logging, LogLevels
from .rate_limiter import limiter
from .utils.exception_handler import register_exception_handlers

configure_logging(LogLevels.info)

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url and frontend_url not in origins:
    origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter

# Register all exception handlers
register_exception_handlers(app)

""" Only uncomment below to create new tables, 
otherwise the tests will fail if not connected
"""
#Base.metadata.create_all(bind=engine)

register_routes(app)

@app.get("/")
async def read_root(request: Request):
    return {"message": "Hello world"}