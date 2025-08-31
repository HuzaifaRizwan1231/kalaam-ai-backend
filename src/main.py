from fastapi import FastAPI
from .config.db import engine, Base, DbSession
from .api import register_routes
from .logging import configure_logging, LogLevels


configure_logging(LogLevels.info)

app = FastAPI()

""" Only uncomment below to create new tables, 
otherwise the tests will fail if not connected
"""
# Base.metadata.create_all(bind=engine)

register_routes(app)