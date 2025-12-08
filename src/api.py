# This is where we will register all of our api routes
from fastapi import FastAPI
from .routes import auth

def register_routes(app: FastAPI):
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
