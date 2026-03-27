# filepath: /home/huzaifa-rizwan/Kalaam/kalaam-ai-backend/src/services/clarity_analyzer.py
import librosa
import numpy as np
from scipy.stats import variation


class ClarityAnalyzer:
    """Service for analyzing clarity of audio files"""

    def analyze_clarity(self, audio_path: str) -> dict:
        """
        Analyze the clarity of the audio file.
        :param audio_path: Path to the audio file
        :return: Dictionary containing clarity metrics
        """
        # Example logic for clarity analysis
        clarity_score = self.compute_clarity(audio_path)
        return {
            "clarity_score": clarity_score,
            "description": "Higher score indicates better clarity",
        }

    def estimate_snr(self, y):
        """
        Rough SNR estimate using signal vs residual noise
        """
        signal_power = np.mean(y**2)
        noise_power = np.mean((y - librosa.effects.harmonic(y)) ** 2)
        if noise_power == 0:
            return 50
        return 10 * np.log10(signal_power / noise_power)

    def clipping_ratio(self, y, threshold=0.99):
        """
        Percentage of samples that are clipped
        """
        clipped = np.sum(np.abs(y) >= threshold)
        return clipped / len(y)

    def normalize(self, value, min_val, max_val):
        """
        Clamp + normalize to 0–1
        """
        value = np.clip(value, min_val, max_val)
        return (value - min_val) / (max_val - min_val)

    # -----------------------------
    # Main clarity function
    # -----------------------------

    def compute_clarity(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000)

        # --- MFCC (articulation stability)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_variation = np.mean([variation(m) for m in mfcc])

        # --- Spectral features
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

        # --- Noise & distortion
        snr = self.estimate_snr(y)
        clip_ratio = self.clipping_ratio(y)

        # -----------------------------
        # Normalization (empirical ranges)
        # -----------------------------
        mfcc_score = 1 - self.normalize(mfcc_variation, 0.3, 1.5)
        centroid_score = self.normalize(centroid, 1500, 3500)
        bandwidth_score = self.normalize(bandwidth, 1000, 3000)
        flux_score = self.normalize(flux, 0.1, 1.0)
        snr_score = self.normalize(snr, 5, 30)
        clip_penalty = self.normalize(clip_ratio, 0.0, 0.05)

        # -----------------------------
        # Weighted clarity score
        # -----------------------------
        clarity = (
            0.25 * mfcc_score
            + 0.20 * centroid_score
            + 0.15 * bandwidth_score
            + 0.10 * flux_score
            + 0.25 * snr_score
            - 0.15 * clip_penalty
        )

        clarity = np.clip(clarity, 0, 1) * 100

        # -----------------------------
        # Reason analysis
        # -----------------------------
        reasons = []

        if snr < 10:
            reasons.append("High background noise")
        if centroid < 1800:
            reasons.append("Muffled or low-quality microphone")
        if mfcc_variation > 1.2:
            reasons.append("Poor articulation or slurred speech")
        if clip_ratio > 0.01:
            reasons.append("Audio clipping detected")
        if not reasons:
            reasons.append("Clear and well-articulated speech")

        return {
            "clarity_score": round(clarity, 2),
            "snr_db": round(snr, 2),
            "clipping_ratio": round(clip_ratio, 4),
            "reasons": reasons,
        }
