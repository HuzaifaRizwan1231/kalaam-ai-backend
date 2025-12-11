import os
import asyncio
import shutil
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from ..entities.analysis import Analysis
from ..entities.user import User
from ..services.file_processing import FileProcessingService
from ..services.filler_word_analyzer import FillerWordAnalyzer
from ..services.loudness_analyzer import LoudnessAnalyzer
from ..services.wpm_analyzer import WPMAnalyzer
from ..utils.response_builder import ResponseBuilder


class AnalysisController:
    """Controller for handling file analysis operations"""
    
    def __init__(self):
        # Get AssemblyAI API key from environment
        self.assemblyai_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.assemblyai_key:
            raise ValueError("ASSEMBLYAI_API_KEY environment variable not set")
        
        self.file_service = FileProcessingService(self.assemblyai_key)
        self.filler_analyzer = FillerWordAnalyzer()
        self.loudness_analyzer = LoudnessAnalyzer()
        self.wpm_analyzer = WPMAnalyzer()
    
    async def create_analysis(self, file: UploadFile, user: User, db: Session):
        """
        Process uploaded file (audio or video) and save analysis results to database
        Includes transcript, WPM data, filler word analysis, and loudness analysis
        """
        # Validate file
        is_valid, message = self.file_service.validate_file(file)
        if not is_valid:
            return ResponseBuilder.error(message, 400)
        
        audio_path = None
        temp_dir = None
        
        # Create initial analysis record in database
        analysis = Analysis(
            user_id=user.id,
            file_name=file.filename,
            file_type="pending",
            status="processing"
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        try:
            # Step 1: Process file and get transcription (MUST wait for this)
            transcript, captions, file_type, audio_path = await self.file_service.process_file(file)
            
            # Get temp directory from audio path for cleanup
            temp_dir = os.path.dirname(audio_path)
            
            # Step 2: Run WPM, filler, and loudness analysis IN PARALLEL
            wpm_task = asyncio.to_thread(self.wpm_analyzer.calculate_wpm, captions, 2)
            filler_task = asyncio.to_thread(self.filler_analyzer.identify_fillers, transcript)
            loudness_task = asyncio.to_thread(self.loudness_analyzer.analyze_loudness, audio_path, 1)
            
            # Wait for all three to complete
            wpm_data, filler_analysis, loudness_analysis = await asyncio.gather(
                wpm_task,
                filler_task,
                loudness_task
            )
            
            # Update analysis record with results
            analysis.file_type = file_type
            analysis.status = "completed"
            analysis.transcript = transcript
            analysis.captions = captions
            analysis.wpm_data = wpm_data
            db.commit()
            db.refresh(analysis)
            
            return ResponseBuilder.success(
                data={
                    "analysis_id": analysis.id,
                    "file_name": file.filename,
                    "file_type": file_type,
                    "transcript": transcript,
                    "wpm_data": wpm_data,
                    "filler_word_analysis": filler_analysis,
                    "loudness_analysis": loudness_analysis,
                    "created_at": analysis.created_at.isoformat()
                },
                message="Analysis completed and saved successfully",
                status_code=200
            )
        except Exception as e:
            # Update analysis status to failed
            analysis.status = "failed"
            analysis.error_message = str(e)
            db.commit()
            
            return ResponseBuilder.error(f"Analysis failed: {str(e)}", 500)
        finally:
            # Clean up temporary files
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    @staticmethod
    def get_analysis(analysis_id: int, user: User, db: Session):
        """Get analysis by ID"""
        analysis = db.query(Analysis).filter(
            Analysis.id == analysis_id,
            Analysis.user_id == user.id
        ).first()
        
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
                "error_message": analysis.error_message,
                "created_at": analysis.created_at.isoformat(),
                "updated_at": analysis.updated_at.isoformat()
            },
            message="Analysis retrieved successfully"
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
                        "created_at": a.created_at.isoformat()
                    }
                    for a in analyses
                ],
                "count": len(analyses)
            },
            message="Analyses retrieved successfully"
        )
