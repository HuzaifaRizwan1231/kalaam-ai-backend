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
            
            # Compute RMS (use -96 dB for silence, standard for digital audio)
            rms = np.sqrt(np.mean(segment**2))
            if rms > 0 and np.isfinite(rms):
                rms_db = float(20 * np.log10(rms))
            else:
                rms_db = -96.0  # Standard value for digital silence
            
            # Compute LUFS (use -70 LUFS for silence/too short segments)
            try:
                lufs = meter.integrated_loudness(segment)
                if not np.isfinite(lufs):  # Check if lufs is -inf or nan
                    lufs = -70.0
                else:
                    lufs = float(lufs)
            except Exception:
                lufs = -70.0  # Standard LUFS for silence
            
            # Store results
            results.append({
                "interval": i + 1,
                "start_time": float(start_time),
                "end_time": float(min(start_time + interval_duration, duration)),
                "rms_db": rms_db,
                "lufs": lufs
            })
        
        # Calculate statistics (only from valid finite values)
        valid_rms = [r["rms_db"] for r in results if r["rms_db"] > -96.0]
        valid_lufs = [r["lufs"] for r in results if r["lufs"] > -70.0]
        
        stats = {
            "average_rms_db": float(np.mean(valid_rms)) if valid_rms else -96.0,
            "average_lufs": float(np.mean(valid_lufs)) if valid_lufs else -70.0,
            "min_rms_db": float(np.min(valid_rms)) if valid_rms else -96.0,
            "max_rms_db": float(np.max(valid_rms)) if valid_rms else -96.0,
            "min_lufs": float(np.min(valid_lufs)) if valid_lufs else -70.0,
            "max_lufs": float(np.max(valid_lufs)) if valid_lufs else -70.0,
            "total_duration": float(duration)
        }
        
        return {
            "intervals": results,
            "statistics": stats,
            "sample_rate": int(sr),
            "interval_duration": interval_duration
        }
