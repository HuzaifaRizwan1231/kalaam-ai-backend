import os
import time
import json
import numpy as np
import assemblyai as aai
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import subprocess
import sys

# Add project root to path
sys.path.append(os.getcwd())
from src.services.intonation_analyzer import IntonationAnalyzer, _get_prosody_features

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

SAMPLES_DIR = "test_output"
MEDIA_FILES = ["bad1.mp4", "bad2.mp4", "good1.mp4", "good3.mp4"]

def extract_audio_tmp(input_path):
    output_path = input_path.replace(".mp4", ".wav")
    if os.path.exists(output_path):
        return output_path
    
    print(f"Extracting {input_path}...")
    subprocess.run([
        "ffmpeg", "-i", input_path, "-q:a", "0", "-map", "a", "-y", output_path
    ], check=True, capture_output=True)
    return output_path

def transcribe_cached(wav_path):
    cache_path = wav_path + ".json"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    
    print(f"Transcribing {wav_path}...")
    transcriber = aai.Transcriber()
    # USE THE SAME CONFIG AS THE MAIN APP
    config = aai.TranscriptionConfig(
        punctuate=True, 
        format_text=True, 
        disfluencies=True,
        speech_models=["universal-2"]
    )
    res = transcriber.transcribe(wav_path, config)
    
    data = {"text": res.text, "captions": []}
    for w in res.words:
        data["captions"].append({
            "text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence
        })
    
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data

def run_comparison():
    analyzer = IntonationAnalyzer()
    
    results = []
    
    for media_file in MEDIA_FILES:
        mp4_path = os.path.join(SAMPLES_DIR, media_file)
        if not os.path.exists(mp4_path):
             print(f"Skipping {mp4_path} - not found")
             continue
        
        wav_path = extract_audio_tmp(mp4_path)
        
        # 1. Transcription
        transcription_data = transcribe_cached(wav_path)
        
        # 2. PRAAT implementation (The standard one now)
        start = time.perf_counter()
        prosody_features = _get_prosody_features(wav_path)
        score_data = analyzer.analyze_intonation(
            wav_path, transcription_data["text"], transcription_data["captions"], 
            precomputed_prosody=prosody_features
        )
        time_taken = time.perf_counter() - start
        
        results.append({
            "media": media_file,
            "expected": "bad" if "bad" in media_file else "good",
            "score": score_data["intonation_score"],
            "label": score_data["intonation_label"],
            "time": time_taken,
            "p_std": float(np.std([w["pitch"] for w in score_data["word_scores"] if w["pitch"] > 0])),
            "p_delta": float(np.mean([w["pitch_delta"] for w in score_data["word_scores"] if w["pitch"] > 0])),
            "e_std": float(np.std([w["energy"] for w in score_data["word_scores"]]))
        })

    # Output Table
    print("\n" + "="*100)
    print(f"{'Media':<15} | {'Expected':<10} | {'Score/Label':<25} | {'P_Std':<10} | {'P_Delta':<10} | {'E_Std':<10}")
    print("-"*100)
    for r in results:
        res_str = f"{r['score']:.4f} ({r['label']})"
        print(f"{r['media']:<15} | {r['expected']:<10} | {res_str:<25} | {r['p_std']:.4f} | {r['p_delta']:.4f} | {r['e_std']:.4f}")
    print("="*100 + "\n")

if __name__ == "__main__":
    run_comparison()
