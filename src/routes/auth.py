from typing import Annotated
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from ..config.db import DbSession
from ..schemas.auth import UserCreate
from ..controllers.auth import AuthController

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

@router.post("/register", status_code=201)
def register(payload: UserCreate, db: DbSession):
    """Register a new user"""
    result = AuthController.register_user(payload, db)
    return result

@router.post("/token")
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DbSession
):
    """Login and get access token"""
    result = AuthController.authenticate_user(form_data.username, form_data.password, db)
    return result
