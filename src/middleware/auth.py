from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt.exceptions import InvalidTokenError
from ..config.db import DbSession
from ..entities.user import User
from ..utils.security import SECRET_KEY, ALGORITHM
from ..utils.response_builder import ResponseBuilder

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: DbSession
) -> User:
    """Dependency to get current authenticated user from JWT token"""

    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("id")
        
        if username is None or user_id is None:
            return ResponseBuilder.error("Could not validate credentials!",401)
            
    except InvalidTokenError:
        return ResponseBuilder.error("Invalid or expired token!",401)
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return ResponseBuilder.error("This user does not exists!",401)
    
    return user

# Type alias for cleaner usage in routes
CurrentUser = Annotated[User, Depends(get_current_user)]
