import os
import subprocess
import mimetypes
import tempfile
import assemblyai as aai
from typing import Tuple, List, Dict, Optional
from fastapi import UploadFile, HTTPException


class FileProcessingService:
    """Service for processing uploaded files and extracting audio/video"""
    
    ALLOWED_EXTENSIONS = {'.mp4', '.mp3', '.wav', '.avi'}
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB in bytes
    
    def __init__(self, assemblyai_api_key: str):
        aai.settings.api_key = assemblyai_api_key
        self.transcriber = aai.Transcriber()
    
    @staticmethod
    def validate_file(file: UploadFile) -> Tuple[bool, str]:
        """Validate uploaded file"""
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in FileProcessingService.ALLOWED_EXTENSIONS:
            return False, f"File type {file_ext} not supported. Allowed: {', '.join(FileProcessingService.ALLOWED_EXTENSIONS)}"
        
        return True, "Valid"
    
    @staticmethod
    def is_video_file(filename: str) -> bool:
        """Determine if file is video or audio"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type and mime_type.startswith("video")
    
    @staticmethod
    def extract_audio(input_path: str, output_path: str) -> bool:
        """Extract audio from video or audio file using ffmpeg"""
        try:
            subprocess.run([
                "ffmpeg", "-i", input_path,
                "-q:a", "0", "-map", "a",
                "-y",  # Overwrite output file
                output_path
            ], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr.decode()}")
            return False
    
    def transcribe_audio(self, audio_path: str) -> Optional[aai.Transcript]:
        """Transcribe audio using AssemblyAI"""
        try:
            config = aai.TranscriptionConfig(
                punctuate=True,
                format_text=True,
                disfluencies=True
            )
            transcript = self.transcriber.transcribe(audio_path, config)
            
            if transcript.status == aai.TranscriptStatus.error:
                print(f"Transcription error: {transcript.error}")
                return None
            
            return transcript
        except Exception as e:
            print(f"Transcription exception: {str(e)}")
            return None
    
    @staticmethod
    def extract_captions(transcript: aai.Transcript) -> List[Dict]:
        """Extract word-level captions with timestamps"""
        return [
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "confidence": word.confidence
            }
            for word in transcript.words
        ]
    
    async def process_file(self, file: UploadFile) -> Tuple[str, List[Dict], str, str]:
        """
        Process uploaded file and return transcript, captions, and audio path
        Returns: (transcript_text, captions, file_type, audio_path)
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()  # Don't auto-delete, caller will handle cleanup
        
        try:
            # Save uploaded file
            file_ext = os.path.splitext(file.filename)[1]
            input_path = os.path.join(temp_dir, f"input{file_ext}")
            
            with open(input_path, "wb") as f:
                content = await file.read()
                if len(content) > self.MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File size exceeds 20MB limit")
                f.write(content)
            
            # Determine file type
            is_video = self.is_video_file(file.filename)
            file_type = "video" if is_video else "audio"
            
            # Extract audio
            audio_path = os.path.join(temp_dir, "audio.wav")  # WAV format for loudness analysis
            if not self.extract_audio(input_path, audio_path):
                raise HTTPException(status_code=500, detail="Failed to extract audio")
            
            # Transcribe
            transcript = self.transcribe_audio(audio_path)
            if not transcript:
                raise HTTPException(status_code=500, detail="Transcription failed")
            
            # Extract captions
            captions = self.extract_captions(transcript)
            
            return transcript.text, captions, file_type, audio_path
        except Exception as e:
            # Clean up on error
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
