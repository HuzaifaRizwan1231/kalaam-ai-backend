from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..config.db import Base


class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # 'video' or 'audio'
    status = Column(String(50), default="processing")  # 'processing', 'completed', 'failed'
    transcript = Column(Text, nullable=True)
    captions = Column(JSON, nullable=True)  # JSON array of word-level captions
    wpm_data = Column(JSON, nullable=True)  # JSON array of WPM analysis
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship to User
    user = relationship("User", back_populates="analyses")
