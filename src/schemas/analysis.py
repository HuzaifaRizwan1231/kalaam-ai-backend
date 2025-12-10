from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


class CaptionWord(BaseModel):
    """Single word caption with timestamp"""
    text: str
    start: int  # milliseconds
    end: int  # milliseconds
    confidence: float


class WPMInterval(BaseModel):
    """Words per minute for a time interval"""
    start_time: float  # seconds
    end_time: float  # seconds
    word_count: int
    wpm: int


class AnalysisCreate(BaseModel):
    """Schema for creating an analysis (used internally)"""
    user_id: int
    file_name: str
    file_type: str


class AnalysisResponse(BaseModel):
    """Response schema for analysis"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    user_id: int
    file_name: str
    file_type: str
    status: str
    transcript: Optional[str] = None
    captions: Optional[List[Dict[str, Any]]] = None
    wpm_data: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AnalysisStatus(BaseModel):
    """Quick status response"""
    id: int
    status: str
    message: str
