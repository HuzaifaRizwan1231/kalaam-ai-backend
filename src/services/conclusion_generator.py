import numpy as np
from typing import Dict, List

class ConclusionGenerator:
    """
    Service to generate human-readable conclusions and actionable advice 
    based on the raw data from all analysis modules.
    """

    @staticmethod
    def get_intonation_conclusion(intonation_data: Dict) -> str:
        score = intonation_data.get("intonation_score", 0)
        label = intonation_data.get("intonation_label", "moderate")
        emp_percentage = intonation_data.get("emphasis_percentage", 0)
        
        if label == "monotone":
            return (
                "Your delivery is quite flat and monotone. To keep your audience engaged, "
                "try varying your pitch more—especially on key content words—and adding "
                "pauses for dramatic effect."
            )
        elif label == "flat":
            return (
                "Your voice is steady but lacks emotional range. This is good for formal "
                "reporting, but adding more energy and energy variance would make your "
                "presentation more persuasive."
            )
        elif label == "expressive":
            if emp_percentage > 30:
                 return (
                    "Excellent expressiveness! You use your voice dynamically. Just be careful "
                    "not to emphasize every word, as this can make the speech feel slightly erratic."
                )
            return (
                "Great job! Your intonation is dynamic and engaging. You are effectively "
                "using pitch and energy to highlight important points naturally."
            )
        else: # moderate
            return (
                "Your intonation is balanced. It's clear and conversational. To take it to "
                "the next level, try pushing for slightly more emphasis on your most important conclusions."
            )

    @staticmethod
    def get_eye_contact_conclusion(head_data: Dict, audience_pos: str = "front") -> str:
        if not head_data or not head_data.get("percentage_looking"):
            return "No face data was detected. Ensure you are clearly visible to the camera."
            
        looking = head_data.get("percentage_looking", 0)
        breakdown = head_data.get("direction_breakdown", {})
        
        if looking > 80:
            return (
                f"Outstanding eye contact! You spent {looking}% of the time engaging with your audience. "
                "This builds strong trust and shows confidence in your material."
            )
        elif looking > 50:
            return (
                f"Good eye contact ({looking}%). You are looking at your audience more than half the time. "
                "To improve, try to look away from your notes or the sides less frequently."
            )
        else:
            # Analyze where they ARE looking
            main_distraction = "the sides"
            if breakdown.get("LookingDown", 0) > 30:
                main_distraction = "your notes or the floor"
            elif breakdown.get("LookingUp", 0) > 30:
                main_distraction = "the ceiling"
                
            return (
                f"Your eye contact was low ({looking}%). You appeared to be distracted by {main_distraction}. "
                "Try to maintain a 'gaze-lock' with your intended audience to project authority."
            )

    @staticmethod
    def get_wpm_conclusion(wpm_data: List[Dict]) -> str:
        if not wpm_data: return "N/A"
        avg_wpm = np.mean([d["wpm"] for d in wpm_data])
        
        if avg_wpm < 110:
            return "You are speaking a bit slowly. This is great for complex topics, but for general presentations, try to increase your pace to 130-140 WPM for more energy."
        elif avg_wpm > 170:
            return "You are speaking very quickly. Your audience might struggle to follow. Try to slow down and use intentional pauses after big points."
        else:
            return f"Your pace is perfect. At {int(avg_wpm)} WPM, you are in the ideal range for clear, professional communication."

    @staticmethod
    def get_loudness_conclusion(loud_data: Dict) -> str:
        # Based on average LUFS
        lufs = loud_data.get("integrated_loudness", -20)
        if lufs < -25:
            return "Your volume is quite low. Ensure your microphone is positioned correctly and try to speak from your diaphragm to increase presence."
        elif lufs > -12:
            return "You are speaking quite loudly (borderline shouting). Try to pull back slightly to avoid overwhelming the listener."
        else:
            return "Your volume is consistent and at a comfortable level for most listeners."

    @staticmethod
    def get_relevance_conclusion(topic_data: Dict) -> str:
        if not topic_data: return "No topic analysis available."
        score = topic_data.get("overall_coverage", 0)
        if score > 0.7:
            return "You stayed perfectly on topic! Your speech correlates strongly with your intended subject matter."
        elif score > 0.4:
            return "You covered the main topic well, though you drifted into related areas occasionally. This can be good for context, but keep an eye on your focus."
        else:
            return "Your speech diverged significantly from the intended topic. Ensure your content is more tightly aligned with your goals."
