"""
Kalaam AI Backend Service Layer.
Exposes all primary analysis and file-processing engines as an unified API.
Includes signal processing (Loudness, Intonation) and NLP-driven (Filler Words, Topic Coverage) metrics.
"""

from .file_processing import FileProcessingService
from .filler_word_analyzer import FillerWordAnalyzer
from .loudness_analyzer import LoudnessAnalyzer
from .wpm_analyzer import WPMAnalyzer
from .intonation_analyzer import IntonationAnalyzer
from .topic_coverage_analyzer import TopicCoverageAnalyzer
from .head_direction_analyzer import HeadDirectionAnalyzer
from .conclusion_generator import ConclusionGenerator

__all__ = [
    "FileProcessingService",
    "FillerWordAnalyzer", 
    "LoudnessAnalyzer",
    "WPMAnalyzer",
    "IntonationAnalyzer",
    "TopicCoverageAnalyzer",
    "HeadDirectionAnalyzer",
    "ConclusionGenerator"
]
