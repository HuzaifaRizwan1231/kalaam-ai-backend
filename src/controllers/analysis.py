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
from ..services.intonation_analyzer import IntonationAnalyzer
from ..services.video_analyzer import VideoAnalyzer
from ..services.topic_coverage_analyzer import TopicCoverageAnalyzer
from ..services.gesture_analyzer import GestureAnalyzer
from ..services.conclusion_generator import ConclusionGenerator
from ..services.clarity_analyzer import ClarityAnalyzer
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


def _video_analysis_worker(video_path, audience_position="front"):
    import os
    from ..services.video_analyzer import VideoAnalyzer

    abs_path = os.path.abspath(video_path)
    return VideoAnalyzer().analyze_video(abs_path, audience_position=audience_position)


def _loudness_worker(audio_path):
    from ..services.loudness_analyzer import LoudnessAnalyzer

    return LoudnessAnalyzer().analyze_loudness(audio_path)


def _gesture_worker(video_path):
    from ..services.gesture_analyzer import GestureAnalyzer

    return GestureAnalyzer().analyze_gestures(video_path)


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
        self.intonation_analyzer = IntonationAnalyzer()
        self.video_analyzer = VideoAnalyzer()
        self.gesture_analyzer = GestureAnalyzer()
        self.topic_analyzer = TopicCoverageAnalyzer()
        self.conclusion_generator = ConclusionGenerator()
        self.clarity_analyzer = ClarityAnalyzer()

    async def create_analysis(
        self,
        file: UploadFile,
        user: User,
        db: Session,
        topic: str = None,
        audience_position: str = "front",
    ):
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
        transcription_task = None
        prosody_task = None
        loudness_task = None
        head_direction_task = None
        facial_expression_task = None
        wpm_task = None
        filler_task = None
        intonation_task = None
        topic_task = None
        gesture_task = None

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
            # Reconstruct original input path (assuming process_file followed the input+ext convention)
            file_ext = os.path.splitext(file.filename)[1]
            input_path = os.path.join(temp_dir, f"input{file_ext}")

            # Step 2: Run analyzers IN PARALLEL
            t3 = datetime.now().strftime("%H:%M:%S")
            step2_start = time.perf_counter()

            loop = asyncio.get_running_loop()

            async def measure_task(name, executor, func, *args):
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

            # --- PHASE 2: CONCURRENT INDEPENDENT TASKS ---
            # We start all tasks that don't need transcription here.
            # This includes the "Prosody" part of intonation analysis.
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] --- Phase 2: Starting simultaneous cloud & local tasks ---"
            )

            # 1. Broadcaster to Cloud (AssemblyAI)
            transcription_task = asyncio.create_task(
                asyncio.to_thread(self.file_service.transcribe_audio, audio_path)
            )

            # 2. Local Prosody Extraction (HEAVY CPU, in separate process)
            prosody_task = asyncio.create_task(
                measure_task(
                    "Prosody Extraction", _cpu_executor, _prosody_worker, audio_path
                )
            )

            # 3. Local Loudness (Fast Thread)
            loudness_task = asyncio.create_task(
                measure_task(
                    "Loudness",
                    None,
                    self.loudness_analyzer.analyze_loudness,
                    audio_path,
                )
            )

            # 4. Local Video Analysis (HEAVY CPU, combined in separate process)
            video_task = None
            if file_type == "video":
                video_task = asyncio.create_task(
                    measure_task(
                        "Video Analysis",
                        _cpu_executor,
                        _video_analysis_worker,
                        input_path,
                        audience_position,
                    )
                )
                gesture_task = asyncio.create_task(
                    measure_task(
                        "Gesture Analysis", _cpu_executor, _gesture_worker, input_path
                    )
                )

            # --- PHASE 3: DEPENDENT TASKS (REQUIRES TRANSCRIPTION) ---
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] --- Phase 3: Waiting for transcript & scoring results ---"
            )

            # Wait for Phase 2 results that ARE needed for Step 3
            transcript_obj = await transcription_task
            t_transcript = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{t_transcript}] --- Phase 3: Transcript received from AssemblyAI ---"
            )

            if not transcript_obj:
                raise HTTPException(status_code=500, detail="Transcription failed")

            transcript = transcript_obj.text
            captions = self.file_service.extract_captions(transcript_obj)

            # Wait for prosody results if not ready
            prosody_result = await prosody_task

            # Start dependent tasks
            wpm_task = asyncio.create_task(
                measure_task("WPM", None, self.wpm_analyzer.calculate_wpm, captions, 2)
            )
            filler_task = asyncio.create_task(
                measure_task(
                    "Filler", None, self.filler_analyzer.identify_fillers, transcript
                )
            )

            # Intonation scoring (now it's very fast because prosody_res is passed in)
            intonation_task = asyncio.create_task(
                measure_task(
                    "Intonation Scoring",
                    None,
                    self.intonation_analyzer.analyze_intonation,
                    audio_path,
                    transcript,
                    captions,
                    0.5,
                    0.5,
                    prosody_result,
                )
            )

            # Add clarity analysis task
            clarity_task = asyncio.create_task(
                measure_task(
                    "Clarity Analysis",
                    None,
                    self.clarity_analyzer.analyze_clarity,
                    audio_path,
                )
            )

            topic_task = None
            if topic:
                topic_task = asyncio.create_task(
                    measure_task(
                        "Topic Coverage",
                        None,
                        self.topic_analyzer.compute_coverage,
                        topic,
                        transcript,
                    )
                )

            # Wait for all remaining tasks
            dependent_tasks = [wpm_task, filler_task, intonation_task]
            if topic_task:
                dependent_tasks.append(topic_task)
            if gesture_task:
                dependent_tasks.append(gesture_task)

            # Re-gather everything with return_exceptions=True to avoid cascading file failures
            results = await asyncio.gather(
                loudness_task,
                video_task or asyncio.sleep(0),
                *dependent_tasks,
                clarity_task,
                return_exceptions=True,
            )

            # Unpack and handle exceptions
            def check_res(r, name):
                if isinstance(r, Exception):
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] !!! Task {name} failed: {r}"
                    )
                    return None
                return r

            loudness_analysis = check_res(results[0], "Loudness")
            video_analysis = check_res(results[1], "Video Analysis")

            head_direction_analysis = video_analysis["head"] if video_analysis else None
            facial_expression_analysis = (
                video_analysis["expression"] if video_analysis else None
            )
            posture_analysis = video_analysis["posture"] if video_analysis else None

            wpm_res = check_res(results[2], "WPM")
            wpm_analysis = {
                "intervals": wpm_res or [],
                "conclusion": "No pacing data" if not wpm_res else None,
            }

            filler_analysis = check_res(results[3], "Filler")
            if not filler_analysis:
                filler_analysis = {
                    "fillers": [],
                    "filler_counts": {},
                    "total_words": 0,
                    "filler_percentage": 0,
                }

            intonation_analysis = check_res(results[4], "Intonation")
            if not intonation_analysis:
                intonation_analysis = {
                    "emphasized_words": [],
                    "total_words": 0,
                    "total_content_words": 0,
                    "total_emphasized": 0,
                    "emphasis_percentage": 0,
                    "average_prosody_score": 0,
                    "word_scores": [],
                    "conclusion": "Scoring failed",
                }

            clarity_analysis = check_res(results[-1], "Clarity Analysis")

            topic_coverage = None
            if topic_task:
                topic_coverage = check_res(results[5], "Topic Coverage")

            gesture_analysis = None
            if gesture_task:
                gesture_analysis = check_res(
                    results[6] if topic_task else results[5], "Gesture Analysis"
                )

            # --- PHASE 4: GENERATE CONCLUSIONS ---
            # Summarize scores for humans
            intonation_analysis["conclusion"] = (
                self.conclusion_generator.get_intonation_conclusion(intonation_analysis)
            )
            wpm_analysis["conclusion"] = self.conclusion_generator.get_wpm_conclusion(
                wpm_analysis["intervals"]
            )
            loudness_analysis["conclusion"] = (
                self.conclusion_generator.get_loudness_conclusion(loudness_analysis)
            )

            if head_direction_analysis:
                head_direction_analysis["conclusion"] = (
                    self.conclusion_generator.get_eye_contact_conclusion(
                        head_direction_analysis, audience_position
                    )
                )

            if facial_expression_analysis:
                facial_expression_analysis["conclusion"] = (
                    self.conclusion_generator.get_expression_conclusion(
                        facial_expression_analysis
                    )
                )

            if topic_coverage:
                topic_coverage["conclusion"] = (
                    self.conclusion_generator.get_relevance_conclusion(topic_coverage)
                )

            if posture_analysis:
                posture_analysis["conclusion"] = (
                    self.conclusion_generator.get_posture_conclusion(posture_analysis)
                )

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
            analysis.filler_word_analysis = filler_analysis
            analysis.loudness_analysis = loudness_analysis
            analysis.head_direction_analysis = head_direction_analysis
            analysis.facial_expression_analysis = facial_expression_analysis
            analysis.posture_analysis = posture_analysis
            analysis.gesture_analysis = gesture_analysis
            analysis.intonation_analysis = intonation_analysis
            analysis.topic_coverage = topic_coverage
            analysis.clarity_analysis = clarity_analysis
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
                    "clarity_analysis": (
                        clarity_analysis["clarity_score"] if clarity_analysis else None
                    ),
                    "head_direction_analysis": head_direction_analysis,
                    "facial_expression_analysis": facial_expression_analysis,
                    "posture_analysis": posture_analysis,
                    "gesture_analysis": gesture_analysis,
                    "intonation_analysis": intonation_analysis,
                    "topic_coverage": topic_coverage,
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
                facial_expression_task,
                wpm_task,
                filler_task,
                intonation_task,
                topic_task,
                gesture_task,
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
                "facial_expression_analysis": analysis.facial_expression_analysis,
                "filler_word_analysis": analysis.filler_word_analysis,
                "loudness_analysis": analysis.loudness_analysis,
                "posture_analysis": analysis.posture_analysis,
                "gesture_analysis": analysis.gesture_analysis,
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
