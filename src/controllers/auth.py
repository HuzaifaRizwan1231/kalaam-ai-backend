from sqlalchemy.orm import Session
from ..entities.user import User
from ..schemas.auth import UserCreate
from ..utils.security import get_password_hash, verify_password, create_access_token
from ..utils.response_builder import ResponseBuilder


class AuthController:
    
    @staticmethod
    def register_user(payload: UserCreate, db: Session):
        """Register a new user"""
        existing = db.query(User).filter(User.username == payload.username).first()
        if existing:
            return ResponseBuilder.error("Username already exists", 400)
        
        user = User(
            username=payload.username,
            hashed_password=get_password_hash(payload.password)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        return ResponseBuilder.success(
            data={"id": user.id, "username": user.username},
            message="User registered successfully",
            status_code=201
        )
    
    @staticmethod
    def authenticate_user(username: str, password: str, db: Session):
        """Authenticate user and return access token in ResponseBuilder format"""
        user = db.query(User).filter(User.username == username).first()
        if not user or not verify_password(password, user.hashed_password):
            return ResponseBuilder.error("Invalid credentials", 401)
        
        access_token = create_access_token({"sub": user.username, "id": user.id})
        return ResponseBuilder.success(
            data={
                "access_token": access_token,
                "token_type": "bearer"
            },
            message="Login successful",
            status_code=200
        )
