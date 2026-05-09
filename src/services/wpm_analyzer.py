import pandas as pd
from typing import List, Dict

class WPMAnalyzer:
    """
    Calculates Words Per Minute (WPM) from timestamp-aligned captions.
    Uses time-binning to analyze pace changes over time, rather than 
    just a single global average.
    """
    
    @staticmethod
    def calculate_wpm(captions: List[Dict], interval: int = 2) -> List[Dict]:
        """
        Calculates the instantaneous pace for fixed time intervals.
        - interval: The window size in seconds (default 2s for fine-grained resolution).
        
        Args:
            captions: List of word objects {text, start, end} from the transcriber.
            interval: The sampling window for pace calculation.
            
        Returns:
            A list of time bins with their corresponding word count and WPM value.
        """
        if not captions:
            return []
        
        # 1. Create a temporal map of words
        df = pd.DataFrame(captions)
        df["start"] = df["start"] / 1000  # Convert AssemblyAI/Whisper ms to seconds
        df["end"] = df["end"] / 1000
        
        # 2. Define the discrete time intervals (BINS)
        max_time = df["end"].max()
        time_bins = list(range(0, int(max_time) + interval, interval))
        
        # 3. Categorize each word into an interval based on its start time
        df["time_bin"] = pd.cut(df["start"], bins=time_bins, right=False)
        
        # 4. Aggregate: Count how many words appear in each bin
        wpm_data = df.groupby("time_bin", observed=False).size().reset_index(name="word_count")
        wpm_data["start_time"] = wpm_data["time_bin"].apply(lambda x: x.left)
        wpm_data["end_time"] = wpm_data["time_bin"].apply(lambda x: x.right)
        
        # 5. Scale to standard "Per Minute" metric
        # Formula: (words in interval) * (60 / seconds in interval)
        wpm_data["wpm"] = wpm_data["word_count"] * (60 / interval)
        
        # 6. Sanitize and format for JSON output
        return [
            {
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "word_count": int(row["word_count"]),
                "wpm": int(row["wpm"])
            }
            for _, row in wpm_data.iterrows()
        ]
    
    @staticmethod
    def get_average_wpm(wpm_data: List[Dict]) -> float:
        """
        Calculates the speaker's overall average pace across the entire session.
        Uses total_words / total_time * 60 for higher accuracy than averaging the means.
        """
        if not wpm_data:
            return 0.0
        
        total_words = sum(item["word_count"] for item in wpm_data)
        total_time = wpm_data[-1]["end_time"] if wpm_data else 0
        
        if total_time == 0:
            return 0.0
        
        return (total_words / total_time) * 60
    
    @staticmethod
    def get_wpm_statistics(wpm_data: List[Dict]) -> Dict:
        """
        Provides a summarized statistical overview of the speaker's tempo.
        Identifies bursts of speed or periods of hesitation (Min/Max).
        """
        if not wpm_data:
            return {
                "average_wpm": 0.0,
                "min_wpm": 0,
                "max_wpm": 0,
                "total_words": 0,
                "total_time": 0.0
            }
        
        wpm_values = [item["wpm"] for item in wpm_data]
        total_words = sum(item["word_count"] for item in wpm_data)
        total_time = wpm_data[-1]["end_time"] if wpm_data else 0
        
        return {
            "average_wpm": WPMAnalyzer.get_average_wpm(wpm_data),
            "min_wpm": min(wpm_values),
            "max_wpm": max(wpm_values),
            "total_words": total_words,
            "total_time": total_time
        }
