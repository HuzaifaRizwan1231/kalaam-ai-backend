from fastapi import APIRouter, UploadFile, File, Form, Query
from typing import Optional
from ..middleware.auth import CurrentUser
from ..config.db import DbSession
from ..controllers.analysis import AnalysisController
from ..services.progress_service import ProgressService
from sse_starlette.sse import EventSourceResponse
import json

router = APIRouter()
controller = AnalysisController()

import logging


@router.post("", status_code=200)
async def analyze_file(
    current_user: CurrentUser,
    db: DbSession,
    file: UploadFile = File(..., description="Audio or video file (mp3, mp4, wav, avi) - Max 20MB"),
    topic: Optional[str] = Form(None, description="The topic to analyze coverage for"),
    audience_position: Optional[str] = Form("front", description="Audience position: front, left, right, both"),
    progress_id: Optional[str] = Query(None)
):
    logging.info(f"Received analysis request from user {current_user.id} (file: {file.filename}, progress_id: {progress_id})")
    """
    Upload and analyze audio/video file.
    Extracts audio, transcribes, calculates WPM, and semantic similarity to topic.
    Returns analysis results immediately (synchronous processing).
    """
    return await controller.create_analysis(file, current_user, db, topic, audience_position, progress_id)


@router.get("/progress/{progress_id}")
async def get_analysis_progress(progress_id: str):
    """
    Stream analysis progress for a specific progress ID.
    """
    progress_service = ProgressService()
    
    async def event_generator():
        async for data in progress_service.subscribe(progress_id):
            yield {
                "data": json.dumps(data)
            }
            
    return EventSourceResponse(event_generator())


@router.get("/{analysis_id}")
def get_analysis(
    analysis_id: int,
    current_user: CurrentUser,
    db: DbSession
):
    """
    Get analysis results by ID.
    Only returns analyses belonging to the authenticated user.
    """
    return controller.get_analysis(analysis_id, current_user, db)


@router.get("")
def get_all_analyses(
    current_user: CurrentUser,
    db: DbSession
):
    """
    Get all analyses for the authenticated user.
    """
    return controller.get_user_analyses(current_user, db)
