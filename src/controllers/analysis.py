import os, json, logging
import asyncio
import tempfile
import shutil
import time
from datetime import datetime
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from ..entities.analysis import Analysis
from ..entities.user import User
from ..services.file_processing import FileProcessingService
from ..services.filler_word_analyzer import FillerWordAnalyzer
from ..services.loudness_analyzer import LoudnessAnalyzer
from ..services.wpm_analyzer import WPMAnalyzer
from ..services.intonation_analyzer import IntonationAnalyzer
from ..services.video_analyzer import VideoAnalyzer
from ..services.topic_coverage_analyzer import TopicCoverageAnalyzer
from ..services.gesture_analyzer import GestureAnalyzer
from ..services.conclusion_generator import ConclusionGenerator
from ..services.clarity_analyzer import ClarityAnalyzer
from ..utils.response_builder import ResponseBuilder
from concurrent.futures import ProcessPoolExecutor
from ..utils.LLM_judge import prepare_gemini_input
from ..services.gemini_feedback import GeminiFeedbackService
from ..services.gemini_feedback import FinalFeedback
from ..services.progress_service import ProgressService
import functools

from ..services.analysis_orchestrator import AnalysisOrchestrator
from ..models.analysis_context import AnalysisContext
from ..utils.executors import get_cpu_executor

class AnalysisController:
    """Controller for handling file analysis operations"""

    def __init__(self):
        # Get API keys
        self.assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        
        if not self.assemblyai_key:
            raise ValueError("ASSEMBLYAI_API_KEY environment variable not set")

        # Initialize Services
        self.file_service = FileProcessingService(self.assemblyai_key)
        self.filler_analyzer = FillerWordAnalyzer()
        self.loudness_analyzer = LoudnessAnalyzer()
        self.wpm_analyzer = WPMAnalyzer()
        self.intonation_analyzer = IntonationAnalyzer()
        self.video_analyzer = VideoAnalyzer()
        self.gesture_analyzer = GestureAnalyzer()
        self.topic_analyzer = TopicCoverageAnalyzer()
        self.conclusion_generator = ConclusionGenerator()
        self.clarity_analyzer = ClarityAnalyzer()
        self.feedback_service = GeminiFeedbackService(api_key=self.gemini_key)
        self.progress_service = ProgressService()
        
        # Initialize Orchestrator
        self.orchestrator = AnalysisOrchestrator(
            self.file_service, self.filler_analyzer, self.loudness_analyzer,
            self.wpm_analyzer, self.intonation_analyzer, self.video_analyzer,
            self.gesture_analyzer, self.topic_analyzer, self.conclusion_generator,
            self.clarity_analyzer, self.feedback_service
        )

    async def create_analysis(
        self,
        file: UploadFile,
        user: User,
        db: Session,
        topic: str = None,
        audience_position: str = "front",
        progress_id: str = None,
    ):
        """Orchestrates the creation and processing of a new analysis"""
        
        # 1. Validation
        is_valid, message = self.file_service.validate_file(file)
        if not is_valid:
            return ResponseBuilder.error(message, 400)

        # 2. Database Record Initialization
        analysis = Analysis(
            user_id=user.id,
            file_name=file.filename,
            file_type="pending",
            status="processing",
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        tracking_id = progress_id or str(analysis.id)
        self.progress_service.update_progress(tracking_id, 5, "uploading")

        # 3. Pipeline Execution
        temp_dir = None
        try:
            # Prepare file and local paths
            _, _, file_type, audio_path = await self.file_service.process_file(file)
            temp_dir = os.path.dirname(audio_path)
            file_ext = os.path.splitext(file.filename)[1]
            input_path = os.path.join(temp_dir, f"input{file_ext}")

            # Create context for the orchestrator
            context = AnalysisContext(
                tracking_id=tracking_id,
                file_type=file_type,
                audio_path=audio_path,
                input_path=input_path,
                temp_dir=temp_dir,
                start_time=time.perf_counter()
            )

            # Run the heavy lifting
            await self.orchestrator.run_pipeline(context, topic, audience_position)

            # 4. Finalize Results
            analysis.status = "completed"
            analysis.file_type = file_type
            analysis.transcript = context.transcript
            analysis.captions = context.captions
            analysis.llm_judge_feedback = context.final_data["llm_judge_feedback"].model_dump_json()
            
            # Map module results back to entity
            res = context.results
            analysis.wpm_data = context.final_data["wpm_data"]
            analysis.filler_word_analysis = res["filler"]
            analysis.loudness_analysis = res["loudness"]
            analysis.head_direction_analysis = context.final_data["head_direction_analysis"]
            analysis.facial_expression_analysis = context.final_data["facial_expression_analysis"]
            analysis.posture_analysis = context.final_data["posture_analysis"]
            analysis.gesture_analysis = res["gesture"]
            analysis.intonation_analysis = res["intonation"]
            analysis.topic_coverage = res["topic"]
            analysis.clarity_analysis = res["clarity"]
            
            db.commit()
            self.progress_service.update_progress(tracking_id, 100, "complete")
            self.progress_service.remove_progress(tracking_id)

            # Prepare return data
            return_data = {**context.final_data, "analysis_id": analysis.id, "created_at": analysis.created_at.isoformat()}
            return ResponseBuilder.success(data=return_data, message="Analysis completed successfully")

        except Exception as e:
            logging.exception(f"Critical failure in analysis pipeline for {tracking_id}")
            analysis.status = "failed"
            analysis.error_message = str(e)
            db.commit()
            return ResponseBuilder.error(f"Analysis failed: {str(e)}", 500)
            
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def get_analysis(analysis_id: int, user: User, db: Session):
        """Get analysis by ID"""
        analysis = (
            db.query(Analysis)
            .filter(Analysis.id == analysis_id, Analysis.user_id == user.id)
            .first()
        )

        if not analysis:
            return ResponseBuilder.error("Analysis not found", 404)

        return ResponseBuilder.success(
            data={
                "id": analysis.id,
                "file_name": analysis.file_name,
                "file_type": analysis.file_type,
                "status": analysis.status,
                "transcript": analysis.transcript,
                "wpm_data": analysis.wpm_data,
                "head_direction_analysis": analysis.head_direction_analysis,
                "facial_expression_analysis": analysis.facial_expression_analysis,
                "filler_word_analysis": analysis.filler_word_analysis,
                "loudness_analysis": analysis.loudness_analysis,
                "posture_analysis": analysis.posture_analysis,
                "gesture_analysis": analysis.gesture_analysis,
                "clarity_analysis": (
                    analysis.clarity_analysis["clarity_score"]
                    if analysis.clarity_analysis
                    else None
                ),
                "intonation_analysis": analysis.intonation_analysis,
                "llm_judge_feedback": FinalFeedback.model_validate_json(
                    analysis.llm_judge_feedback
                ),
                "topic_coverage": analysis.topic_coverage,
                "error_message": analysis.error_message,
                "created_at": analysis.created_at.isoformat(),
                "updated_at": analysis.updated_at.isoformat(),
            },
            message="Analysis retrieved successfully",
        )

    @staticmethod
    def get_user_analyses(user: User, db: Session):
        """Get all analyses for a user"""
        analyses = db.query(Analysis).filter(Analysis.user_id == user.id).all()

        return ResponseBuilder.success(
            data={
                "analyses": [
                    {
                        "id": a.id,
                        "file_name": a.file_name,
                        "file_type": a.file_type,
                        "status": a.status,
                        "created_at": a.created_at.isoformat(),
                    }
                    for a in analyses
                ],
                "count": len(analyses),
            },
            message="Analyses retrieved successfully",
        )
