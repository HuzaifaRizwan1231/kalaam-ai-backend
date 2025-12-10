from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from ..config.db import DbSession
from ..schemas.auth import UserCreate, LoginRequest
from ..controllers.auth import AuthController

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.post("/register", status_code=201)
def register(payload: UserCreate, db: DbSession):
    """Register a new user"""
    result = AuthController.register_user(payload, db)
    return result

@router.post("/token")
def login_oauth2(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DbSession
):
    """OAuth2-compatible login for Swagger UI authorization"""
    result = AuthController.authenticate_user(form_data.username, form_data.password, db)
    
    # Error: return ResponseBuilder format
    if not result.get("success"):
        return result
    
    # Success: return flat OAuth2 format (for Swagger UI)
    return result["data"]

@router.post("/login")
def login_json(payload: LoginRequest, db: DbSession):
    """JSON login endpoint for frontend - consistent ResponseBuilder format"""
    result = AuthController.authenticate_user(payload.username, payload.password, db)
    return result  # Always ResponseBuilder format
