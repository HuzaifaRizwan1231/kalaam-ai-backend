import pandas as pd
from typing import List, Dict


class WPMAnalyzer:
    """Service for calculating Words Per Minute (WPM) from captions"""
    
    @staticmethod
    def calculate_wpm(captions: List[Dict], interval: int = 2) -> List[Dict]:
        """
        Calculate Words Per Minute for time intervals
        
        Args:
            captions: List of word-level captions with timestamps
            interval: Time interval in seconds for WPM calculation (default: 2)
            
        Returns:
            List of WPM data for each interval
        """
        if not captions:
            return []
        
        # Create DataFrame from captions
        df = pd.DataFrame(captions)
        df["start"] = df["start"] / 1000  # Convert milliseconds to seconds
        df["end"] = df["end"] / 1000
        
        # Calculate time bins
        max_time = df["end"].max()
        time_bins = list(range(0, int(max_time) + interval, interval))
        
        # Bin words by start time
        df["time_bin"] = pd.cut(df["start"], bins=time_bins, right=False)
        
        # Count words per bin
        wpm_data = df.groupby("time_bin").size().reset_index(name="word_count")
        wpm_data["start_time"] = wpm_data["time_bin"].apply(lambda x: x.left)
        wpm_data["end_time"] = wpm_data["time_bin"].apply(lambda x: x.right)
        
        # Calculate WPM (words per minute)
        wpm_data["wpm"] = wpm_data["word_count"] * (60 / interval)
        
        # Format output
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
        """Calculate average WPM from WPM data"""
        if not wpm_data:
            return 0.0
        
        total_words = sum(item["word_count"] for item in wpm_data)
        total_time = wpm_data[-1]["end_time"] if wpm_data else 0
        
        if total_time == 0:
            return 0.0
        
        return (total_words / total_time) * 60
    
    @staticmethod
    def get_wpm_statistics(wpm_data: List[Dict]) -> Dict:
        """Get statistical summary of WPM data"""
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
