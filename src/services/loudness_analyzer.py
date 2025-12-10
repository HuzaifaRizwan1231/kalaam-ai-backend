import numpy as np
import librosa
import pyloudnorm as pyln
from typing import List, Dict


class LoudnessAnalyzer:
    """Service for analyzing audio loudness (RMS and LUFS)"""
    
    @staticmethod
    def analyze_loudness(audio_path: str, interval_duration: int = 1) -> Dict:
        """
        Analyze audio loudness over time intervals
        
        Args:
            audio_path: Path to audio file (WAV or MP3)
            interval_duration: Duration of each interval in seconds (default: 1)
            
        Returns:
            Dictionary containing loudness analysis results
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)  # Load with native sample rate
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")
        
        # Initialize loudness meter (BS.1770)
        meter = pyln.Meter(sr)
        
        # Get audio duration in seconds
        duration = len(y) / sr
        intervals = np.arange(0, duration, interval_duration)
        
        # Lists to store results
        results = []
        
        # Process each interval
        for i, start_time in enumerate(intervals):
            # Calculate start and end samples
            start_sample = int(start_time * sr)
            end_sample = int(min((start_time + interval_duration) * sr, len(y)))
            
            # Extract interval audio
            segment = y[start_sample:end_sample]
            if len(segment) == 0:
                continue
            
            # Compute RMS
            rms = np.sqrt(np.mean(segment**2))
            rms_db = float(20 * np.log10(rms)) if rms > 0 else float(-np.inf)
            
            # Compute LUFS
            try:
                lufs = float(meter.integrated_loudness(segment))
            except Exception:
                lufs = float(-np.inf)  # Handle short/quiet segments
            
            # Store results
            results.append({
                "interval": i + 1,
                "start_time": float(start_time),
                "end_time": float(min(start_time + interval_duration, duration)),
                "rms_db": rms_db,
                "lufs": lufs
            })
        
        # Calculate statistics
        valid_rms = [r["rms_db"] for r in results if r["rms_db"] != float(-np.inf)]
        valid_lufs = [r["lufs"] for r in results if r["lufs"] != float(-np.inf)]
        
        stats = {
            "average_rms_db": float(np.mean(valid_rms)) if valid_rms else None,
            "average_lufs": float(np.mean(valid_lufs)) if valid_lufs else None,
            "min_rms_db": float(np.min(valid_rms)) if valid_rms else None,
            "max_rms_db": float(np.max(valid_rms)) if valid_rms else None,
            "min_lufs": float(np.min(valid_lufs)) if valid_lufs else None,
            "max_lufs": float(np.max(valid_lufs)) if valid_lufs else None,
            "total_duration": float(duration)
        }
        
        return {
            "intervals": results,
            "statistics": stats,
            "sample_rate": int(sr),
            "interval_duration": interval_duration
        }
