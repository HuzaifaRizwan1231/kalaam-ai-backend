from fastapi import APIRouter, UploadFile, File
from ..middleware.auth import CurrentUser
from ..config.db import DbSession
from ..controllers.analysis import AnalysisController

router = APIRouter()
controller = AnalysisController()


@router.post("", status_code=200)
async def analyze_file(
    current_user: CurrentUser,
    db: DbSession,
    file: UploadFile = File(..., description="Audio or video file (mp3, mp4, wav, avi) - Max 20MB")
):
    """
    Upload and analyze audio/video file.
    Extracts audio, transcribes, and calculates WPM.
    Returns analysis results immediately (synchronous processing).
    """
    return await controller.create_analysis(file, current_user, db)


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
