from pydantic import BaseModel, Field, ConfigDict

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=150, description="Username for the account")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class LoginRequest(BaseModel):
    """Alternative to OAuth2PasswordRequestForm for JSON login"""
    username: str
    password: str