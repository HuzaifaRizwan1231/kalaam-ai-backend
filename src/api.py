# This is where we will register all of our api routes
from fastapi import FastAPI
from .routes import auth, analysis

def register_routes(app: FastAPI):
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(analysis.router, prefix="/api", tags=["analysis"])
