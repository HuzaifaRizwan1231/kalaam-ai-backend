import asyncio
import os
import subprocess
import mimetypes
import tempfile
import assemblyai as aai
from typing import Tuple, List, Dict, Optional
from fastapi import UploadFile, HTTPException


class FileProcessingService:
    """
    Coordinates file I/O operations including validation, conversion, 
    and transcription. Serves as the primary data ingestion layer.
    """

    # Supported audio/video formats for the analyzer
    ALLOWED_EXTENSIONS = {"mp4", "mp3", "wav", "avi", "webm", "mpeg"}
    # Cap to protect the backend from OOM or disk-full attacks (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024 

    def __init__(self, assemblyai_api_key: str):
        """Initializes the AssemblyAI client for transcription."""
        aai.settings.api_key = assemblyai_api_key
        self.transcriber = aai.Transcriber()

    @staticmethod
    def validate_file(file: UploadFile) -> Tuple[bool, str]:
        """
        Validates the incoming HTTP file stream for security and compatibility.
        Checks for:
        - Content-Type (MIME)
        - Allowed extensions
        """
        # Guard: Check file extension
        # Headers are used as the primary check (stricter than filename)
        file_ext = os.path.split(file.headers["content-type"])[1].lower()
        if file_ext not in FileProcessingService.ALLOWED_EXTENSIONS:
            return (
                False,
                f"File type {file_ext} not supported. Allowed: {', '.join(FileProcessingService.ALLOWED_EXTENSIONS)}",
            )

        return True, "Valid"

    @staticmethod
    def is_video_file(filename: str) -> bool:
        """Determines if the file contains a video track base on MIME type."""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type and mime_type.startswith("video")

    @staticmethod
    def extract_audio(input_path: str, output_path: str) -> bool:
        """
        Uses FFmpeg to extract high-quality audio (WAV) from an input file.
        FFmpeg is the industry standard for media conversion, handling 
        complex codecs better than pure-Python libraries.
        
        Args:
            input_path: Path to the original video/audio.
            output_path: Target path for the extracted audio.
        """
        try:
            # -q:a 0 = best quality
            # -map a = extract audio track only
            # -y = overwrite existing files
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-q:a",
                    "0",
                    "-map",
                    "a",
                    "-y",
                    output_path,
                ],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            # Captures standard error for debugging purpose
            print(f"FFmpeg CLI Exception: {e.stderr.decode()}")
            return False

    def transcribe_audio(self, audio_path: str) -> Optional[aai.Transcript]:
        """
        Dispatches the audio to AssemblyAI's neural transcription servers.
        Uses the 'Universal-2' model for increased accuracy.
        
        Args:
            audio_path: Path to the audio file.
        """
        try:
            # Configuration optimization:
            # - disfluencies: Captures 'um', 'uh', etc. (Critical for FillerWordAnalyzer)
            # - punctuations/text format: Increases readability for TopicCoverageAnalyzer
            config = aai.TranscriptionConfig(
                speech_models=["universal-2"],
                punctuate=True,
                format_text=True,
                disfluencies=True,
            )
            # Synchronous wait for transcript response (handled by wrapper)
            transcript = self.transcriber.transcribe(audio_path, config)

            if transcript.status == aai.TranscriptStatus.error:
                print(f"AssemblyAI Cloud Error: {transcript.error}")
                return None

            return transcript
        except Exception as e:
            print(f"Transcription Client Exception: {str(e)}")
            return None

    @staticmethod
    def extract_captions(transcript: aai.Transcript) -> List[Dict]:
        """
        Transforms AssemblyAI's word objects into a generic 
        structured list of word-level timestamps (captions).
        This serves as the core alignment data for all analysis modules.
        """
        return [
            {
                "text": word.text,
                "start": word.start,   # In Milliseconds
                "end": word.end,       # In Milliseconds
                "confidence": word.confidence,
            }
            for word in transcript.words
        ]

    async def process_file(self, file: UploadFile) -> Tuple[str, List[Dict], str, str]:
        """
        Full Pipeline (Orchestrator):
        1. Create isolated temporary workspace.
        2. Stream upload to disk.
        3. Extract audio via FFmpeg.
        4. Dispatch for transcription (AssemblyAI).
        5. Structure word-level timestamps.
        
        Returns: (transcript_text, captions, file_type, audio_path)
        """
        # Step A: Workspace setup (Temporary)
        temp_dir = tempfile.mkdtemp()  # Persistent path during this session

        try:
            # Step B: Secure Upload and I/O writing
            file_ext = os.path.splitext(file.filename)[1]
            input_path = os.path.join(temp_dir, f"input{file_ext}")
            print(f"Ingesting file into: {input_path}")

            with open(input_path, "wb") as f:
                content = await file.read()
                # Secondary size check per-session
                if len(content) > self.MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413, detail="File size exceeds the 100MB threshold"
                    )
                f.write(content)

            # Step C: Metadata analysis
            is_video = self.is_video_file(file.filename)
            file_type = "video" if is_video else "audio"

            # Step D: Media Conversion
            # Required for uniformity in analysis (Loudness, Intonation)
            audio_path = os.path.join(temp_dir, "audio.wav")
            print(f"Processing media conversion (FFmpeg)...")
            # Thread isolation: Media conversion is CPU bound; we wrap it in to_thread to keep API responsive.
            if not await asyncio.to_thread(self.extract_audio, input_path, audio_path):
                raise HTTPException(status_code=500, detail="FFmpeg extraction failure")

            # Step E: Transcription Dispatch
            print(f"Broadcasting to AssemblyAI (Cloud API)...")
            transcript = await asyncio.to_thread(self.transcriber.transcribe, audio_path)
            if not transcript:
                raise HTTPException(status_code=500, detail="Global transcription failure")

            # Step F: Structuring results
            captions = self.extract_captions(transcript)

            return transcript.text, captions, file_type, audio_path
        except Exception as e:
            # Cleanup Hook: Ensure disk space is cleared on failure.
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
