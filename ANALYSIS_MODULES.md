# Kalaam AI Analysis Modules

This document provides a detailed technical overview of the speech and video analysis modules implemented in the Kalaam AI backend.

## 🚀 High-Performance Analysis Pipeline

The analysis pipeline is designed for maximum throughput and low latency by utilizing a **Multi-Stage Parallel Architecture**.

### 1. Concurrency Model
To bypass the Python Global Interpreter Lock (GIL) and utilize all available CPU cores, the system uses a hybrid orchestration:
- **`ProcessPoolExecutor`**: Handles heavy CPU-bound math (Pitch tracking, Video processing).
- **`ThreadPoolExecutor`**: Handles I/O-bound or fast tasks (Transcription requests, Disk I/O, Loudness).
- **`asyncio`**: Orchestrates the non-blocking execution of all stages.

### 2. Execution Phases
The pipeline is split into interleaved phases to mask latency:
- **Phase 1 (Media Prep)**: Saves uploaded media and extracts a high-fidelity WAV using FFmpeg.
- **Phase 2 (Concurrent Local & Cloud processing)**: 
    - Dispatches transcription to AssemblyAI (Cloud).
    - Simultaneously starts **Prosody Extraction** (Local CPU).
    - Simultaneously starts **Video Analysis** (Local CPU).
    - Simultaneously starts **Loudness Analysis** (Local Thread).
- **Phase 3 (Dependent Tasks)**: Once the transcript returns, the system "maps" the already-extracted prosody to the words and computes **Topic Relevance** and **WPM**.

---

## 🎤 Intonation & Prosody Analyzer
Located in: `src/services/intonation_analyzer.py`

This module measures the "emotional range" and expressiveness of the speaker.

### Key Technologies
- **`librosa.pyin`**: Probabilistic YIN algorithm for high-accuracy fundamental frequency (F0) tracking.
- **`spaCy`**: Used for identifying content words (Nouns, Verbs, Adjectives) to filter out emphasis on filler words.

### Optimizations
- **16kHz Downsampling**: Audio is downsampled specifically for the pitch tracker to reduce computation by 3x without loss of voice accuracy.
- **Asynchronous Splitting**: The heavy `pyin` computation runs in parallel with transcription, so the results are ready the moment the text arrives.
- **Monotone Detection**: Uses Absolute Variability (Standard Deviation of raw pitch) instead of Z-scores to ensure monotone speakers don't receive artificially high scores from small jitters.

---

## 🎯 Topic Coverage Analyzer
Located in: `src/services/topic_coverage_analyzer.py`

Calculates how well the speaker followed their intended topic.

### Key Technologies
- **`sentence-transformers`**: Utilizes the `all-mpnet-base-v2` model for high-quality semantic embeddings.
- **Cosine Similarity**: Measures the semantic distance between the user's "Topic" and overlapping chunks of the transcript.

### How it works
1.  **Sliding Window Chunking**: Splits the transcript into overlapping segments.
2.  **Semantic Embedding**: Converts both the topic and the chunks into high-dimensional vectors.
3.  **Similarity Mapping**: Calculates the "Best Match" (Max Similarity) and "Overall Coverage" (Mean Similarity).

---

## 📈 Loudness & Energy Analyzer
Located in: `src/services/loudness_analyzer.py`

Measures vocal volume and consistency.

### Standards
- **BS.1770-4 Compliance**: Uses `pyloudnorm` to calculate **LUFS** (Loudness Units relative to Full Scale), the industry standard for perceived loudness.
- **RMS Energy**: Provides high-resolution volume envelopes for word-level emphasis detection.

---

## 👁️ Head Direction (Video) Analyzer
Located in: `src/services/head_direction_analyzer.py`

Analyzes eye contact and audience engagement using video frames.

### Key Technologies
- **MediaPipe FaceMesh**: Tracks 468+ 3D landmarks on the face in real-time.
- **PnP (Perspective-n-Point) Solver**: Uses `OpenCV` to solve the 3D pose of the head based on 2D image coordinates and a 3D generic face model.

### Features
- **3-Axis Tracking**: Measures **Yaw** (Left/Right), **Pitch** (Up/Down), and **Roll** (Tilt).
- **Dynamic Sampling**: Analyzes 1 frame per second to maintain high performance while providing a comprehensive engagement timeline.

---

## 📝 Textual Analyzers
- **Filler Word Analyzer**: Uses regex and NLP to identify disfluencies (um, ah, like, you know).
- **WPM Analyzer**: Calculates "Words Per Minute" based on word-level timestamps provided by the transcription engine.

---

## 🛠️ Infrastructure Services
- **`FileProcessingService`**: 
    - **FFmpeg Integration**: Handles robust audio extraction from various video containers.
    - **AssemblyAI Integration**: Manages word-level transcription with high accuracy for disfluencies.
- **`Exception Handler`**: A custom global middleware that ensures CORS-compliant JSON error responses, even during internal server crashes.
