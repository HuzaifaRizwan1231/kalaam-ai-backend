from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from ..config.db import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    
    # Relationship to Analyses
    analyses = relationship("Analysis", back_populates="user")
