import spacy
import librosa
import numpy as np
from typing import Dict, List

nlp = spacy.load("en_core_web_sm")


def _get_content_words(text: str) -> List[str]:
    """Extract content words (NOUN, VERB, ADJ, ADV) from transcript text."""
    doc = nlp(text)
    return [token.text.lower() for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]


def _get_prosody_features(audio_path: str):
    """Return normalized amplitude (energy), pitch (f0), and time arrays."""
    y, sr = librosa.load(audio_path, sr=None)

    energy = librosa.feature.rms(y=y)[0]
    energy_norm = energy / np.max(energy) if np.max(energy) > 0 else energy

    f0 = librosa.yin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    f0 = np.nan_to_num(f0)
    f0_norm = f0 / np.max(f0) if np.max(f0) > 0 else f0

    times = np.linspace(0, len(y) / sr, len(energy))

    return energy_norm, f0_norm, times


class IntonationAnalyzer:
    """
    Service that analyzes speech intonation/emphasis using prosody features
    (energy + pitch) combined with NLP content-word detection.
    """

    def analyze_intonation(
        self,
        audio_path: str,
        transcript_text: str,
        captions: List[Dict],
        energy_weight: float = 0.5,
        pitch_weight: float = 0.5,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Analyze intonation emphasis in audio.

        Args:
            audio_path: Path to the audio file (WAV).
            transcript_text: Full transcript text for NLP processing.
            captions: Word-level captions with start/end times (ms from AssemblyAI).
            energy_weight: Weight for energy in emphasis score.
            pitch_weight: Weight for pitch in emphasis score.
            threshold: Score threshold to classify a word as emphasized.

        Returns:
            Dict with emphasized_words list, word_scores, and summary statistics.
        """
        content_words = _get_content_words(transcript_text)
        energy, pitch, times = _get_prosody_features(audio_path)

        emphasized_words = []
        word_scores = []

        for cap in captions:
            word = cap["text"]
            start_sec = cap["start"] / 1000.0
            end_sec = cap["end"] / 1000.0
            duration = end_sec - start_sec

            is_content = word.lower() in content_words

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

            score = energy_weight * word_energy + pitch_weight * word_pitch + 0.2 * duration

            is_emphasized = score > threshold and is_content

            if is_emphasized:
                emphasized_words.append(word)

            word_scores.append({
                "word": word,
                "start": round(start_sec, 3),
                "end": round(end_sec, 3),
                "energy": round(word_energy, 4),
                "pitch": round(word_pitch, 4),
                "score": round(score, 4),
                "emphasized": is_emphasized,
                "is_content_word": is_content
            })

        content_scores = [w["score"] for w in word_scores if w["is_content_word"]]

        if content_scores:
            mean_score = np.mean(content_scores)
            std_score = np.std(content_scores)
            dynamic_threshold = mean_score + 0.5 * std_score
        else:
            dynamic_threshold = threshold

        emphasized_words = [
            w["word"] for w in word_scores
            if w["is_content_word"] and w["score"] > dynamic_threshold
        ]

        # Update emphasized flag inside word_scores to match dynamic threshold
        for w in word_scores:
            if w["is_content_word"] and w["score"] > dynamic_threshold:
                w["emphasized"] = True
            else:
                w["emphasized"] = False

        pitch_values = np.array([w["pitch"] for w in word_scores])
        energy_values = np.array([w["energy"] for w in word_scores])

        # Normalize pitch & energy safely
        pitch_norm = (pitch_values - np.mean(pitch_values)) / (np.std(pitch_values) + 1e-6)
        energy_norm = (energy_values - np.mean(energy_values)) / (np.std(energy_values) + 1e-6)

        intonation_score = 0.6 * np.std(pitch_norm) + 0.4 * np.std(energy_norm)

        if intonation_score < 0.05:
            intonation_label = "monotone"
        elif intonation_score < 0.15:
            intonation_label = "flat"
        elif intonation_score < 0.3:
            intonation_label = "moderate"
        else:
            intonation_label = "expressive"

        total_content = sum(1 for w in word_scores if w["is_content_word"])
        total_emphasized = len(emphasized_words)
        emphasis_ratio = round((total_emphasized / total_content * 100), 2) if total_content > 0 else 0.0

        scores_only = [w["score"] for w in word_scores if w["is_content_word"] and w["score"] > 0]
        avg_score = round(float(np.mean(scores_only)), 4) if scores_only else 0.0

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
