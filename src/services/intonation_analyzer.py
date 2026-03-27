import spacy
import librosa
import numpy as np
from typing import Dict, List, Tuple

nlp = spacy.load("en_core_web_sm")

# ---------------------------
# NLP: Content words
# ---------------------------
def _get_content_words(text: str) -> List[str]:
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and not token.is_stop
    ]

# ---------------------------
# Helpers
# ---------------------------
def _smooth(x, window=5):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="same")

def _robust_threshold(scores):
    scores = np.array(scores)
    if len(scores) == 0:
        return 0.0
    median = np.median(scores)
    mad = np.median(np.abs(scores - median)) + 1e-6
    return median + 1.0 * mad

import parselmouth

# ---------------------------
# Prosody extraction using Praat (Parselmouth)
# ---------------------------
def _get_prosody_features(audio_path: str):
    """
    Extracts fundamental frequency (F0/Pitch) and Intensity (Energy) 
    using the Praat (Boersma-CC) algorithm via Parselmouth.
    Significantly faster and more robust than librosa.pyin.
    """
    try:
        snd = parselmouth.Sound(audio_path)
    except Exception as e:
        print(f"Error loading sound in Praat: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    if snd.duration == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # High-precision pitch tracking
    # floor 75, ceiling 600 covers typical human speech range
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    
    # Calculate Intensity (Loudness) aligned with pitch
    intensity = snd.to_intensity(minimum_pitch=75, time_step=0.01)
    
    # Get raw arrays
    f0 = pitch.selected_array['frequency']
    times = pitch.xs()
    
    # Align intensity to pitch timestamps
    energy = []
    for t in times:
        try:
            val = intensity.get_value(t)
            energy.append(val if not np.isnan(val) else 0.0)
        except Exception:
            energy.append(0.0)
    
    energy = np.array(energy)
    
    # Voiced probability: Use Praat's classification (f0 > 0 means voiced)
    voiced_prob = np.where(f0 > 0, 1.0, 0.0)
    
    # Normalize Energy (Intensity)
    # Energy is in dB, shift to 0-1 range for scoring
    # Typical speech is 50-85 dB
    if len(energy) > 0 and np.max(energy) > 0:
        energy_min = np.min(energy[energy > 0]) if np.any(energy > 0) else 0
        energy_diff = (np.max(energy) - energy_min) + 1e-6
        energy_norm = (energy - energy_min) / energy_diff
        energy_norm = np.clip(energy_norm, 0, 1)
    else:
        energy_norm = energy

    # Filter/Smooth F0
    f0 = _smooth(f0)
    
    # Normalize F0
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) > 0:
        f0_norm = f0 / np.max(voiced_f0)
    else:
        f0_norm = f0
        
    return energy_norm, f0_norm, times, voiced_prob

# ---------------------------
# Main Analyzer
# ---------------------------
class IntonationAnalyzer:

    def get_prosody_only(self, audio_path: str) -> Tuple:
        """Only run the heavy feature extraction part. Can be run in parallel with transcription."""
        return _get_prosody_features(audio_path)

    def analyze_intonation(
        self,
        audio_path: str,
        transcript_text: str,
        captions: List[Dict],
        energy_weight: float = 0.5,
        pitch_weight: float = 0.5,
        precomputed_prosody: Tuple = None
    ) -> Dict:

        content_words = set(_get_content_words(transcript_text))
        
        if precomputed_prosody:
            energy, pitch, times, voiced_prob = precomputed_prosody
        else:
            energy, pitch, times, voiced_prob = _get_prosody_features(audio_path)

        if len(times) == 0:
            return {
                "emphasized_words": [],
                "total_words": len(captions),
                "total_content_words": 0,
                "total_emphasized": 0,
                "emphasis_percentage": 0.0,
                "average_prosody_score": 0.0,
                "intonation_score": 0.0,
                "intonation_label": "monotone",
                "word_scores": []
            }

        word_scores = []
        durations = [(c["end"] - c["start"]) / 1000.0 for c in captions]
        avg_duration = np.mean(durations) + 1e-6

        gaps = []
        for i in range(len(captions)):
            if i == 0:
                gaps.append(0)
            else:
                gap = (captions[i]["start"] - captions[i-1]["end"]) / 1000.0
                gaps.append(max(0, gap))

        # ---------------------------
        # Compute word-level scores
        # ---------------------------
        for i, cap in enumerate(captions):
            word = cap["text"]
            start_sec = cap["start"] / 1000.0
            end_sec = cap["end"] / 1000.0
            duration = end_sec - start_sec

            lemma = word.lower()
            is_content = lemma in content_words

            idx = np.where((times >= start_sec) & (times <= end_sec))[0]

            if len(idx) == 0:
                word_scores.append({
                    "word": word,
                    "start": round(start_sec, 3),
                    "end": round(end_sec, 3),
                    "energy": 0.0,
                    "pitch": 0.0,
                    "score": 0.0,
                    "emphasized": False,
                    "is_content_word": is_content
                })
                continue

            word_energy = float(np.mean(energy[idx]))
            word_pitch = float(np.mean(pitch[idx]))
            pitch_conf = float(np.mean(voiced_prob[idx]))

            # Use STD instead of range for more robust local movement detection
            pitch_std = float(np.std(pitch[idx]))
            
            duration_norm = duration / avg_duration
            
            # Weighted word score
            score = (
                energy_weight * word_energy +
                pitch_weight * word_pitch * pitch_conf +
                0.1 * duration_norm
            )
            
            # Penalize extremadamente flat words
            if pitch_std < 0.02:
                score *= 0.8
                
            # Penalize absolute silences
            if word_energy < 0.05:
                score *= 0.5
                
            # Pause-based boost (emphasis often follows pauses)
            if gaps[i] > 0.2:
                score += 0.05
            
            word_scores.append({
                "word": word,
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "energy": round(word_energy, 4),
                "pitch": round(word_pitch, 4),
                "pitch_delta": round(pitch_std, 4), # Rename back but stores STD
                "score": round(score, 4),
                "emphasized": False,
                "is_content_word": is_content
            })

        # ---------------------------
        # Emphasis detection
        # ---------------------------
        content_scores = [w["score"] for w in word_scores if w["is_content_word"]]

        if len(content_scores) < 10:
            k = max(1, int(0.2 * len(content_scores)))
            sorted_idx = np.argsort(content_scores)[-k:]
            threshold_indices = set(sorted_idx)

            content_idx = 0
            for w in word_scores:
                if w["is_content_word"]:
                    if content_idx in threshold_indices:
                        w["emphasized"] = True
                    content_idx += 1
        else:
            dynamic_threshold = _robust_threshold(content_scores)
            mean_score = np.mean(content_scores)
            for w in word_scores:
                if w["is_content_word"]:
                    relative = w["score"] - mean_score
                    if w["score"] > dynamic_threshold or relative > 0.1:
                        w["emphasized"] = True

        emphasized_words = [w["word"] for w in word_scores if w["emphasized"]]

        # ---------------------------
        # Intonation score (Expression level)
        # ---------------------------
        # Use raw word-level pitches (already 0.0 to 1.0 relative to speaker max)
        # and pitch deltas (movement within words).
        voiced_word_pitches = [w["pitch"] for w in word_scores if w["pitch"] > 0]
        voiced_deltas = [w["pitch_delta"] for w in word_scores if w["pitch"] > 0]
        
        if len(voiced_word_pitches) < 5:
            intonation_score = 0.0
        else:
            # Objective variance metrics
            p_std = float(np.std(voiced_word_pitches))
            p_range = float(np.max(voiced_word_pitches) - np.min(voiced_word_pitches))
            p_avg_delta = float(np.mean(voiced_deltas)) if voiced_deltas else 0.0
            
            # Energy variance
            e_vals = [w["energy"] for w in word_scores]
            e_std = float(np.std(e_vals))
            
            # Scoring formula (Weights tuned for absolute expressiveness)
            # We prioritize global variation (p_std) and volume dynamics (e_std).
            # Local jitter (p_avg_delta) is often a sign of monotone/unstable speech rather than expressivity.
            base_score = (
                0.60 * p_std + 
                0.15 * p_range + 
                0.25 * e_std
            )
            
            # Penalize local jitter (p_avg_delta). Stable, expressive speech has low-to-moderate intra-word movement.
            # Jittery vocal fry or nervous monotone often creates high local deltas.
            if p_avg_delta > 0.12:
                base_score -= (p_avg_delta - 0.12) * 0.4
            
            # Penalize monotone volume
            if e_std < 0.10:
                base_score *= 0.8
                
            # Penalty for high unvoiced ratio in content words (monotone/robotic sign)
            content_voiced_count = sum(1 for w in word_scores if w["is_content_word"] and w["pitch"] > 0)
            total_content = sum(1 for w in word_scores if w["is_content_word"])
            voiced_ratio = (content_voiced_count / total_content) if total_content > 0 else 1.0
            
            if voiced_ratio < 0.5:
                base_score *= 0.7 # Significant penalty for choppy/robotic unvoiced speech
                
            # Scale to 0-1 range. 
            intonation_score = max(0.0, min(1.0, base_score * 3.5)) 

        # Labels based on the new 0-1 scale
        if intonation_score < 0.25:
            intonation_label = "monotone"
        elif intonation_score < 0.45:
            intonation_label = "flat"
        elif intonation_score < 0.65:
            intonation_label = "moderate"
        else:
            intonation_label = "expressive"

        # ---------------------------
        # Summary metrics
        # ---------------------------
        total_content = sum(1 for w in word_scores if w["is_content_word"])
        total_emphasized = len(emphasized_words)
        emphasis_ratio = (total_emphasized / total_content * 100) if total_content > 0 else 0.0

        # Cap unrealistic emphasis
        if emphasis_ratio > 40:
            emphasis_ratio *= 0.8

        emphasis_ratio = round(emphasis_ratio, 2)

        valid_scores = [w["score"] for w in word_scores if w["is_content_word"] and w["score"] > 0]
        avg_score = round(float(np.mean(valid_scores)), 4) if valid_scores else 0.0

        return {
            "emphasized_words": emphasized_words,
            "total_words": len(captions),
            "total_content_words": total_content,
            "total_emphasized": total_emphasized,
            "emphasis_percentage": emphasis_ratio,
            "average_prosody_score": avg_score,
            "intonation_score": round(float(intonation_score), 4),
            "intonation_label": intonation_label,
            "word_scores": word_scores
        }
