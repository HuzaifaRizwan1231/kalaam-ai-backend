# This is where we will register all of our api routes
from fastapi import FastAPI

def register_routes(app: FastAPI):
    @app.get("/")
    def read_root():
        return {"message": "Hello world"}
