import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from .progress_service import ProgressService
from ..utils.executors import get_cpu_executor
from ..models.analysis_context import AnalysisContext

class AnalysisOrchestrator:
    """
    Orchestrates the multi-stage analysis pipeline.
    Handles task scheduling, progress reporting, and result aggregation.
    """
    
    def __init__(self, 
                 file_service,
                 filler_analyzer,
                 loudness_analyzer,
                 wpm_analyzer,
                 intonation_analyzer,
                 video_analyzer,
                 gesture_analyzer,
                 topic_analyzer,
                 conclusion_generator,
                 clarity_analyzer,
                 feedback_service):
        self.file_service = file_service
        self.filler_analyzer = filler_analyzer
        self.loudness_analyzer = loudness_analyzer
        self.wpm_analyzer = wpm_analyzer
        self.intonation_analyzer = intonation_analyzer
        self.video_analyzer = video_analyzer
        self.gesture_analyzer = gesture_analyzer
        self.topic_analyzer = topic_analyzer
        self.conclusion_generator = conclusion_generator
        self.clarity_analyzer = clarity_analyzer
        self.feedback_service = feedback_service
        self.progress_service = ProgressService()

    async def run_pipeline(self, context: AnalysisContext, topic: Optional[str], audience_position: str):
        """Runs the full analysis pipeline from transcription to final feedback"""
        loop = asyncio.get_running_loop()
        cpu_executor = get_cpu_executor()

        # Helper to measure and log task execution
        async def measure_task(name, executor, func, *args):
            start = time.perf_counter()
            res = await loop.run_in_executor(executor, func, *args)
            logging.info(f"    -> {name} took {time.perf_counter() - start:.2f}s")
            return res

        all_tasks = []
        try:
            # --- PHASE 1: TRANSCRIPTION & INITIAL LOCAL TASKS ---
            self.progress_service.update_progress(context.tracking_id, 10, "extracting-audio")
            
            # Start transcription and heavy local tasks in parallel
            transcription_task = asyncio.create_task(
                asyncio.to_thread(self.file_service.transcribe_audio, context.audio_path)
            )
            prosody_task = asyncio.create_task(
                measure_task("Prosody Extraction", cpu_executor, self.intonation_analyzer.get_prosody_only, context.audio_path)
            )
            loudness_task = asyncio.create_task(
                measure_task("Loudness", None, self.loudness_analyzer.analyze_loudness, context.audio_path)
            )
            
            all_tasks.extend([transcription_task, prosody_task, loudness_task])

            video_task = None
            gesture_task = None
            if context.file_type == "video":
                video_task = asyncio.create_task(
                    measure_task("Video Analysis", cpu_executor, self.video_analyzer.analyze_video, context.input_path, 30, audience_position)
                )
                gesture_task = asyncio.create_task(
                    measure_task("Gesture Analysis", cpu_executor, self.gesture_analyzer.analyze_gestures, context.input_path)
                )
                all_tasks.extend([video_task, gesture_task])

            # --- PHASE 2: WAIT FOR TRANSCRIPT & START DEPENDENT TASKS ---
            transcript_obj = await transcription_task
            if not transcript_obj:
                raise ValueError("Transcription failed")
                
            context.transcript = transcript_obj.text
            context.captions = self.file_service.extract_captions(transcript_obj)
            self.progress_service.update_progress(context.tracking_id, 30, "analyzing-speech")

            prosody_result = await prosody_task
            
            # Start modules that depend on transcript
            wpm_task = asyncio.create_task(measure_task("WPM", None, self.wpm_analyzer.calculate_wpm, context.captions, 2))
            filler_task = asyncio.create_task(measure_task("Filler", None, self.filler_analyzer.identify_fillers, context.transcript))
            intonation_task = asyncio.create_task(
                measure_task("Intonation Scoring", None, self.intonation_analyzer.analyze_intonation, 
                             context.audio_path, context.transcript, context.captions, 0.5, 0.5, prosody_result)
            )
            clarity_task = asyncio.create_task(measure_task("Clarity Analysis", None, self.clarity_analyzer.analyze_clarity, context.audio_path))
            
            all_tasks.extend([wpm_task, filler_task, intonation_task, clarity_task])

            topic_task = None
            if topic:
                topic_task = asyncio.create_task(measure_task("Topic Coverage", None, self.topic_analyzer.compute_coverage, topic, context.transcript))
                all_tasks.append(topic_task)

            # --- PHASE 3: INCREMENTAL COMPLETION ---
            remaining_tasks = {
                loudness_task: "Loudness",
                clarity_task: "Clarity Analysis",
                wpm_task: "WPM",
                filler_task: "Filler",
                intonation_task: "Intonation",
            }
            if video_task: remaining_tasks[video_task] = "Video Analysis"
            if gesture_task: remaining_tasks[gesture_task] = "Gesture Analysis"
            if topic_task: remaining_tasks[topic_task] = "Topic Coverage"

            total = len(remaining_tasks)
            finished = 0
            for future in asyncio.as_completed(remaining_tasks.keys()):
                await future
                finished += 1
                progress = 35 + (45 * (finished / total))
                stage = "analyzing-speech"
                if progress > 50: stage = "detecting-visuals"
                if progress > 70: stage = "assessing-engagement"
                self.progress_service.update_progress(context.tracking_id, progress, stage)

            # --- PHASE 4: AGGREGATE RESULTS ---
            def get_res(t, name):
                try:
                    res = t.result()
                    if isinstance(res, Exception):
                        logging.error(f"Task {name} failed: {res}")
                        return None
                    return res
                except Exception as e:
                    logging.error(f"Task {name} raised error: {e}")
                    return None

            context.results = {
                "loudness": get_res(loudness_task, "Loudness"),
                "video": get_res(video_task, "Video Analysis") if video_task else None,
                "wpm": get_res(wpm_task, "WPM"),
                "filler": get_res(filler_task, "Filler"),
                "intonation": get_res(intonation_task, "Intonation"),
                "clarity": get_res(clarity_task, "Clarity Analysis"),
                "topic": get_res(topic_task, "Topic Coverage") if topic_task else None,
                "gesture": get_res(gesture_task, "Gesture Analysis") if gesture_task else None
            }

            # --- PHASE 5: GENERATE CONCLUSIONS & SCORES ---
            self.progress_service.update_progress(context.tracking_id, 85, "calculating-scores")
            self._generate_conclusions(context, audience_position)
            
            # --- PHASE 6: FINAL AI FEEDBACK ---
            self.progress_service.update_progress(context.tracking_id, 90, "generating-insights")
            from ..utils.LLM_judge import prepare_gemini_input
            
            context.final_data = self._build_final_data(context)
            prepared_input = prepare_gemini_input(context.final_data)
            context.final_data["llm_judge_feedback"] = self.feedback_service.generate_feedback(prepared_input)
            
            return context

        finally:
            # Atomic cleanup: ensure NO background tasks are left hanging
            # This prevents FileNotFoundError when the controller deletes the temp dir
            pending = [t for t in all_tasks if not t.done()]
            if pending:
                logging.info(f"Cleaning up {len(pending)} pending tasks in orchestrator...")
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

    def _generate_conclusions(self, context: AnalysisContext, audience_position: str):
        res = context.results
        if res["intonation"]:
            res["intonation"]["conclusion"] = self.conclusion_generator.get_intonation_conclusion(res["intonation"])
        if res["wpm"]:
            context.wpm_conclusion = self.conclusion_generator.get_wpm_conclusion(res["wpm"])
        if res["loudness"]:
            res["loudness"]["conclusion"] = self.conclusion_generator.get_loudness_conclusion(res["loudness"])
        
        video = res["video"]
        if video:
            if video.get("head"):
                video["head"]["conclusion"] = self.conclusion_generator.get_eye_contact_conclusion(video["head"], audience_position)
            if video.get("expression"):
                video["expression"]["conclusion"] = self.conclusion_generator.get_expression_conclusion(video["expression"])
            if video.get("posture"):
                video["posture"]["conclusion"] = self.conclusion_generator.get_posture_conclusion(video["posture"])
        
        if res["topic"]:
            res["topic"]["conclusion"] = self.conclusion_generator.get_relevance_conclusion(res["topic"])

    def _build_final_data(self, context: AnalysisContext) -> Dict[str, Any]:
        res = context.results
        video = res["video"]
        return {
            "file_type": context.file_type,
            "transcript": context.transcript or "",
            "wpm_data": {"intervals": res["wpm"] or [], "conclusion": getattr(context, 'wpm_conclusion', None)},
            "filler_word_analysis": res["filler"] or {},
            "loudness_analysis": res["loudness"] or {},
            "clarity_analysis": res["clarity"] or {},
            "head_direction_analysis": video["head"] if video else None,
            "facial_expression_analysis": video["expression"] if video else None,
            "posture_analysis": video["posture"] if video else None,
            "gesture_analysis": res["gesture"],
            "intonation_analysis": res["intonation"] or {},
            "topic_coverage": res["topic"] or {}
        }
