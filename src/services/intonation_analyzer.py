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

# ---------------------------
# Prosody extraction using pyin
# ---------------------------
def _get_prosody_features(audio_path: str):
    # Use a consistent hop_length for all features
    HOP_LENGTH = 1024
    
    # Load at 16k for significantly faster processing
    y, sr = librosa.load(audio_path, sr=16000)

    if len(y) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Energy (Explicit hop_length to match pyin)
    energy = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    energy = _smooth(energy)
    energy_norm = energy / np.max(energy) if np.max(energy) > 0 else energy

    # Pitch (F0) using pyin
    # Use the same HOP_LENGTH
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=HOP_LENGTH
    )
    f0 = np.nan_to_num(f0)

    # Ensure all arrays match in length (sometimes pyin/rms differ by 1-2 frames due to padding)
    min_len = min(len(energy_norm), len(f0), len(voiced_prob))
    energy_norm = energy_norm[:min_len]
    f0 = f0[:min_len]
    voiced_prob = voiced_prob[:min_len]

    # Filter low-confidence pitch frames
    f0[voiced_prob < 0.6] = 0.0

    f0 = _smooth(f0)
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) > 0:
        f0_norm = f0 / np.max(voiced_f0)
    else:
        f0_norm = f0

    # Ensure times matches length exactly
    times = np.linspace(0, len(y) / sr, min_len)

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

            # Penalize tiny pitch changes
            pitch_delta = float(np.max(pitch[idx]) - np.min(pitch[idx]))

            duration_norm = duration / avg_duration

            score = (
                energy_weight * word_energy +
                pitch_weight * word_pitch * pitch_conf +
                0.1 * duration_norm
            )

            # Penalize flat pitch
            if pitch_delta < 0.1:
                score *= 0.7
            # Penalize silence
            if word_energy < 0.05:
                score *= 0.5
            # Pause-based boost
            if gaps[i] > 0.2:
                score += 0.1

            word_scores.append({
                "word": word,
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "energy": round(word_energy, 4),
                "pitch": round(word_pitch, 4),
                "pitch_delta": round(pitch_delta, 4),
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
            
            # Monotone Check: Monotone speakers typically have very low pitch std (< 0.08)
            # and low internal word movement (delta < 0.1).
            # Obama-style expressive speech has p_std > 0.15 and deltas > 0.2.
            
            # Scoring formula (Weights tuned for absolute expressiveness)
            base_score = (
                0.40 * p_std + 
                0.35 * p_avg_delta + 
                0.15 * p_range + 
                0.10 * e_std
            )
            
            # Penalty for high unvoiced ratio in content words (monotone/robotic sign)
            content_voiced_count = sum(1 for w in word_scores if w["is_content_word"] and w["pitch"] > 0)
            total_content = sum(1 for w in word_scores if w["is_content_word"])
            voiced_ratio = (content_voiced_count / total_content) if total_content > 0 else 1.0
            
            if voiced_ratio < 0.5:
                base_score *= 0.7 # Significant penalty for choppy/robotic unvoiced speech
                
            # Scale to 0-1 range. 
            # 0.3-0.4 is typically very expressive.
            intonation_score = min(1.0, base_score * 2.5)

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
