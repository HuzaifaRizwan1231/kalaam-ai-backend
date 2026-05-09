import numpy as np
import librosa
import pyloudnorm as pyln
from typing import List, Dict


class LoudnessAnalyzer:
    """
    Analyzes audio loudness levels and variation across time.
    Uses EBU R128 (Integrated Loudness) standards for consistent human-ear perceived loudness.
    """
    
    @staticmethod
    def analyze_loudness(audio_path: str, interval_duration: int = 1) -> Dict:
        """
        Computes loudness metrics (RMS and LUFS) over the audio file.
        
        LUFS (Loudness Units relative to Full Scale) is superior to peak dB 
        as it correlates more closely with human hearing perceptions.
        
        Args:
            audio_path: The file path to process.
            interval_duration: Sliding window size (default 1s for localized analysis).
        """
        try:
            # 1. Loading the Raw Signal
            # Uses the file's native sample rate (sr=None) for maximum fidelity.
            y, sr = librosa.load(audio_path, sr=None) 
        except Exception as e:
            raise Exception(f"LoudnessAnalyzer: Audio file load failed: {str(e)}")
        
        # 2. Loudness Normalization Engine (BS.1770 compliant)
        meter = pyln.Meter(sr)
        
        duration = len(y) / sr
        # Define the temporal intervals (segments)
        intervals = np.arange(0, duration, interval_duration)
        
        results = []
        
        # 3. Processing each window/interval individually
        for i, start_time in enumerate(intervals):
            start_sample = int(start_time * sr)
            end_sample = int(min((start_time + interval_duration) * sr, len(y)))
            
            segment = y[start_sample:end_sample]
            if len(segment) == 0:
                continue
            
            # 3a. RMS Calculation (Root Mean Square - mathematical average energy)
            # Digital silence floor is capped at -96dB for standard 16-bit audio.
            rms = np.sqrt(np.mean(segment**2))
            if rms > 0 and np.isfinite(rms):
                rms_db = float(20 * np.log10(rms))
            else:
                rms_db = -96.0  
            
            # 3b. LUFS Calculation (Perceptual Loudness)
            try:
                # integrated_loudness uses the K-weighting curve.
                lufs = meter.integrated_loudness(segment)
                if not np.isfinite(lufs):
                    lufs = -70.0
                else:
                    lufs = float(lufs)
            except Exception:
                lufs = -70.0  # Safe fallback for silent or ultra-short segments
            
            results.append({
                "interval": i + 1,
                "start_time": float(start_time),
                "end_time": float(min(start_time + interval_duration, duration)),
                "rms_db": rms_db,
                "lufs": lufs
            })
        
        # 4. Aggregating Global Metrics
        # We ignore segments with -96dB or -70LUFS (background silence) 
        # to prevent skewing the speaker's actual voice statistics.
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
