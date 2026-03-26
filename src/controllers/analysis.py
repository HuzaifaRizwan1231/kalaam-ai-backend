import os
import asyncio
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
from ..services.head_direction_analyzer import HeadDirectionAnalyzer
from ..services.intonation_analyzer import IntonationAnalyzer
from ..services.topic_coverage_analyzer import TopicCoverageAnalyzer
from ..utils.response_builder import ResponseBuilder
from concurrent.futures import ProcessPoolExecutor
import functools

# Global ProcessPoolExecutor for CPU-heavy tasks (Intonation, Video Analysis)
# This allows bypassing the GIL for true parallel processing
_cpu_executor = ProcessPoolExecutor(max_workers=min(os.cpu_count(), 4))

def _prosody_worker(audio_path):
    """Heavy feature extraction for intonation"""
    from ..services.intonation_analyzer import IntonationAnalyzer
    return IntonationAnalyzer().get_prosody_only(audio_path)

def _head_direction_worker(video_path):
    from ..services.head_direction_analyzer import HeadDirectionAnalyzer
    return HeadDirectionAnalyzer().analyze_video(video_path)

def _loudness_worker(audio_path):
    from ..services.loudness_analyzer import LoudnessAnalyzer
    return LoudnessAnalyzer().analyze_loudness(audio_path)


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
        self.head_direction_analyzer = HeadDirectionAnalyzer()
        self.intonation_analyzer = IntonationAnalyzer()
        self.topic_analyzer = TopicCoverageAnalyzer()
    
    async def create_analysis(self, file: UploadFile, user: User, db: Session, topic: str = None):
        """
        Process uploaded file and save analysis results in a highly parallel pipeline.
        Maximizes concurrency by starting independent analyzers while waiting for transcription.
        """
        # Validate file
        is_valid, message = self.file_service.validate_file(file)
        if not is_valid:
            return ResponseBuilder.error(message, 400)
        
        audio_path = None
        temp_dir = None
        
        # Create initial analysis record
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
            start_total = time.perf_counter()
            loop = asyncio.get_event_loop()

            async def measure_task(name, executor, func, *args):
                start = time.perf_counter()
                res = await loop.run_in_executor(executor, func, *args)
                d = time.perf_counter() - start
                print(f"[{datetime.now().strftime('%H:%M:%S')}]     -> {name} took {d:.2f}s")
                return res

            # --- PHASE 1: PRE-REQUISITES (DISK OPERATIONS) ---
            print(f"[{datetime.now().strftime('%H:%M:%S')}] --- Phase 1: Preparing media files ---")
            
            # We need to manually handle what process_file did but split it for concurrency
            import tempfile
            temp_dir = tempfile.mkdtemp()
            file_ext = os.path.splitext(file.filename)[1]
            input_path = os.path.join(temp_dir, f"input{file_ext}")
            
            with open(input_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            is_video = self.file_service.is_video_file(file.filename)
            file_type = "video" if is_video else "audio"
            audio_path = os.path.join(temp_dir, "audio.wav")
            
            if not await asyncio.to_thread(self.file_service.extract_audio, input_path, audio_path):
                raise HTTPException(status_code=500, detail="Failed to extract audio")

            # --- PHASE 2: CONCURRENT INDEPENDENT TASKS ---
            # We start all tasks that don't need transcription here.
            # This includes the "Prosody" part of intonation analysis.
            print(f"[{datetime.now().strftime('%H:%M:%S')}] --- Phase 2: Starting simultaneous cloud & local tasks ---")
            
            # 1. Broadcaster to Cloud (AssemblyAI)
            transcription_task = asyncio.create_task(asyncio.to_thread(self.file_service.transcribe_audio, audio_path))
            
            # 2. Local Prosody Extraction (HEAVY CPU, in separate process)
            prosody_task = measure_task("Prosody Extraction", _cpu_executor, _prosody_worker, audio_path)
            
            # 3. Local Loudness (Fast Thread)
            loudness_task = measure_task("Loudness", None, self.loudness_analyzer.analyze_loudness, audio_path)
            
            # 4. Local Video Analysis (HEAVY CPU, in separate process)
            head_direction_task = None
            if file_type == "video":
                head_direction_task = measure_task("Head Direction", _cpu_executor, _head_direction_worker, input_path)

            # --- PHASE 3: DEPENDENT TASKS (REQUIRES TRANSCRIPTION) ---
            print(f"[{datetime.now().strftime('%H:%M:%S')}] --- Phase 3: Waiting for transcript & scoring results ---")
            
            # Wait for Phase 2 results that ARE needed for Step 3
            transcript_obj = await transcription_task
            if not transcript_obj:
                raise HTTPException(status_code=500, detail="Transcription failed")
            
            transcript = transcript_obj.text
            captions = self.file_service.extract_captions(transcript_obj)
            
            # Wait for prosody results if not ready
            prosody_result = await prosody_task
            
            # Start dependent tasks
            wpm_task = measure_task("WPM", None, self.wpm_analyzer.calculate_wpm, captions, 2)
            filler_task = measure_task("Filler", None, self.filler_analyzer.identify_fillers, transcript)
            
            # Intonation scoring (now it's very fast because prosody_res is passed in)
            intonation_task = measure_task("Intonation Scoring", None, self.intonation_analyzer.analyze_intonation, audio_path, transcript, captions, 0.5, 0.5, prosody_result)
            
            topic_task = None
            if topic:
                topic_task = measure_task("Topic Coverage", None, self.topic_analyzer.compute_coverage, topic, transcript)

            # Wait for all remaining tasks
            dependent_tasks = [wpm_task, filler_task, intonation_task]
            if topic_task: dependent_tasks.append(topic_task)
            
            # Re-gather everything
            results = await asyncio.gather(loudness_task, head_direction_task or asyncio.sleep(0), *dependent_tasks)
            
            loudness_analysis = results[0]
            head_direction_analysis = results[1] if head_direction_task else None
            wpm_data = results[2]
            filler_analysis = results[3]
            intonation_analysis = results[4]
            topic_coverage = results[5] if topic_task else None

            total_time = time.perf_counter() - start_total
            print(f"[{datetime.now().strftime('%H:%M:%S')}] !!! Total Processing Time: {total_time:.2f}s !!!")
            
            # Update analysis record with results
            analysis.file_type = file_type
            analysis.status = "completed"
            analysis.transcript = transcript
            analysis.captions = captions
            analysis.wpm_data = wpm_data
            analysis.head_direction_analysis = head_direction_analysis
            analysis.intonation_analysis = intonation_analysis
            analysis.topic_coverage = topic_coverage
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
                    "head_direction_analysis": head_direction_analysis,
                    "intonation_analysis": intonation_analysis,
                    "topic_coverage": topic_coverage,
                    "created_at": analysis.created_at.isoformat()
                },
                message="Analysis completed successfully",
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
                "head_direction_analysis": analysis.head_direction_analysis,
                "intonation_analysis": analysis.intonation_analysis,
                "topic_coverage": analysis.topic_coverage,
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
