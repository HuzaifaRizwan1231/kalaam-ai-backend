from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

@dataclass
class AnalysisContext:
    """Holds intermediate and final results of the analysis pipeline"""
    tracking_id: str
    file_type: str
    audio_path: str
    input_path: str
    temp_dir: str
    transcript: Optional[str] = None
    captions: List[Dict] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=lambda: 0.0)
    
    # Final structured analysis data
    final_data: Dict[str, Any] = field(default_factory=dict)
