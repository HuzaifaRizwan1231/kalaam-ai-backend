import os
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
from ..services.head_direction_analyzer import HeadDirectionAnalyzer
from ..services.clarity_analyzer import ClarityAnalyzer
from ..services.intonation_analyzer import IntonationAnalyzer
from ..services.topic_coverage_analyzer import TopicCoverageAnalyzer
from ..services.conclusion_generator import ConclusionGenerator
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


def _head_direction_worker(video_path, audience_position="front"):
    import os

    abs_path = os.path.abspath(video_path)
    print(f"[{os.getpid()}] DEBUG: Video worker starting for: {abs_path}")
    if not os.path.exists(abs_path):
        print(f"[{os.getpid()}] ERROR: File not found at {abs_path}")
        raise ValueError(f"File not found: {abs_path}")

    print(
        f"[{os.getpid()}] DEBUG: File exists. Size: {os.path.getsize(abs_path)} bytes"
    )

    from ..services.head_direction_analyzer import HeadDirectionAnalyzer

    return HeadDirectionAnalyzer().analyze_video(
        abs_path, audience_position=audience_position
    )


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

    async def create_analysis(self, file: UploadFile, user: User, db: Session):
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

        # Create initial analysis record in database
        analysis = Analysis(
            user_id=user.id,
            file_name=file.filename,
            file_type="pending",
            status="processing",
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        try:
            start_total = time.perf_counter()
            t1 = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{t1}] --- Step 1: Extracting audio and transcribing (AssemblyAI) ---"
            )

            transcript, captions, file_type, audio_path = (
                await self.file_service.process_file(file)
            )

            t2 = datetime.now().strftime("%H:%M:%S")
            d1 = time.perf_counter() - start_total
            print(
                f"[{t2}] --- Step 1 Finished in {d1:.2f}s: Transcription received ---"
            )

            # Get temp directory from audio path for cleanup
            temp_dir = os.path.dirname(audio_path)

            # Step 2: Run analyzers IN PARALLEL
            t3 = datetime.now().strftime("%H:%M:%S")
            step2_start = time.perf_counter()

            async def measure_task(name, func, *args):
                start = time.perf_counter()
                res = await loop.run_in_executor(executor, func, *args)
                d = time.perf_counter() - start
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}]     -> {name} took {d:.2f}s"
                )
                return res

            print(
                f"[{t3}] --- Step 2: Starting parallel analyzers (WPM, Filler, Loudness, Intonation) ---"
            )

            wpm_task = measure_task("WPM", self.wpm_analyzer.calculate_wpm, captions, 2)
            filler_task = measure_task(
                "Filler", self.filler_analyzer.identify_fillers, transcript
            )
            loudness_task = measure_task(
                "Loudness", self.loudness_analyzer.analyze_loudness, audio_path, 1
            )
            intonation_task = measure_task(
                "Intonation",
                self.intonation_analyzer.analyze_intonation,
                audio_path,
                transcript,
                captions,
            )

            # Head direction analysis only makes sense for video files
            if file_type == "video":
                video_path = os.path.join(
                    temp_dir, "input" + os.path.splitext(file.filename)[1]
                )
                head_direction_task = measure_task(
                    "Head Direction",
                    self.head_direction_analyzer.analyze_video,
                    video_path,
                )
                (
                    wpm_data,
                    filler_analysis,
                    loudness_analysis,
                    intonation_analysis,
                    head_direction_analysis,
                ) = await asyncio.gather(
                    wpm_task,
                    filler_task,
                    loudness_task,
                    intonation_task,
                    head_direction_task,
                )
            else:
                wpm_data, filler_analysis, loudness_analysis, intonation_analysis = (
                    await asyncio.gather(
                        wpm_task,
                        filler_task,
                        loudness_task,
                        intonation_task,
                    )
                )
                head_direction_analysis = None

            t4 = datetime.now().strftime("%H:%M:%S")
            d2 = time.perf_counter() - step2_start
            print(f"[{t4}] --- Step 2 Finished in {d2:.2f}s: All analyses complete ---")
            total_time = time.perf_counter() - start_total
            print(f"[{t4}] !!! Total Processing Time: {total_time:.2f}s !!!")

            # Update analysis record with results
            analysis.file_type = file_type
            analysis.status = "completed"
            analysis.transcript = transcript
            analysis.captions = captions
            analysis.wpm_data = wpm_analysis
            analysis.head_direction_analysis = head_direction_analysis
            analysis.intonation_analysis = intonation_analysis
            db.commit()
            db.refresh(analysis)

            return ResponseBuilder.success(
                data={
                    "analysis_id": analysis.id,
                    "file_name": file.filename,
                    "file_type": file_type,
                    "transcript": transcript,
                    "wpm_data": wpm_analysis,
                    "filler_word_analysis": filler_analysis,
                    "loudness_analysis": loudness_analysis,
                    "head_direction_analysis": head_direction_analysis,
                    "intonation_analysis": intonation_analysis,
                    "created_at": analysis.created_at.isoformat(),
                },
                message="Analysis completed and saved successfully",
                status_code=200,
            )
        except Exception as e:
            # Update analysis status to failed
            analysis.status = "failed"
            analysis.error_message = str(e)
            db.commit()

            return ResponseBuilder.error(f"Analysis failed: {str(e)}", 500)
        finally:
            # Important: We must ensure all background tasks are handled
            # before we delete the temp directory, otherwise workers might crash
            # trying to read from a deleted location.
            all_tasks = [
                transcription_task,
                prosody_task,
                loudness_task,
                head_direction_task,
                wpm_task,
                filler_task,
                intonation_task,
                topic_task,
            ]
            running_tasks = [t for t in all_tasks if t and not t.done()]

            if running_tasks:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Cleaning up {len(running_tasks)} pending tasks before folder deletion..."
                )
                for t in running_tasks:
                    t.cancel()
                # Await cancellation completions (with return_exceptions=True to avoid raising CancelledError here)
                await asyncio.gather(*running_tasks, return_exceptions=True)

            # Clean up temporary files
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
                "intonation_analysis": analysis.intonation_analysis,
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
