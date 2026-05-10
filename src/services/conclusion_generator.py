import numpy as np
from typing import Dict, List

class ConclusionGenerator:
    """
    Service to generate human-readable conclusions and actionable advice 
    based on raw metrics from multiple analysis engines.
    
    Transforms quantitative data (scores, percentages) into qualitative feedback
    to help users improve their public speaking.
    """

    @staticmethod
    def get_intonation_conclusion(intonation_data: Dict) -> str:
        """
        Analyzes pitch variety and specific emphasis metrics.
        Focuses on voice variation (Labels) and the amount of emphasis used.
        """
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
                "reporting, but adding more energy and variance would make your "
                "presentation more persuasive."
            )
        elif label == "expressive":
            # Guard against over-emphasis (too much vocal variety can be erratic)
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
        """
        Interprets head direction tracking data.
        Determines the percentage of time spent looking 'at' the intended audience.
        """
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
            # Analyze primary direction of distraction to provide better advice
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
        """
        Analyzes the Words Per Minute (WPM) trend.
        Ideal professional range is typically 130-150 WPM.
        """
        if not wpm_data: return "N/A"
        avg_wpm = np.mean([d["wpm"] for d in wpm_data])
        
        if avg_wpm < 110:
            return "You are speaking a bit slowly. This is great for complex topics, but for general presentations, try to increase your pace for more energy (aim for ~135 WPM)."
        elif avg_wpm > 170:
            return "You are speaking very quickly. Your audience might struggle to follow. Try to slow down and use intentional pauses after big points."
        else:
            return f"Your pace ({int(avg_wpm)} WPM) is perfect. You are in the ideal range for clear, professional communication."

    @staticmethod
    def get_loudness_conclusion(loud_data: Dict) -> str:
        """
        Analyzes audio volume levels using Integrated Loudness (LUFS).
        -23 LUFS is standard for broadcast; -18 is typical for online media.
        """
        lufs = loud_data.get("integrated_loudness", -20)
        if lufs < -25:
            return "Your volume is quite low. Ensure your microphone is positioned correctly and try to speak from your diaphragm."
        elif lufs > -12:
            return "You are speaking quite loudly (borderline shouting). Try to pull back slightly to avoid overwhelming the listener."
        else:
            return "Your volume is consistent and at a comfortable level for most listeners."

    @staticmethod
    def get_relevance_conclusion(topic_data: Dict) -> str:
        """
        Analyzes semantic alignment between the speech transcript and the declared topic.
        Uses Sentence-Transformers to compute semantic similarity.
        """
        if not topic_data: return "No topic analysis available."
        score = topic_data.get("overall_coverage", 0)
        
        if score > 0.7:
            return "You stayed perfectly on topic! Your speech correlates strongly with your intended subject matter."
        elif score > 0.4:
            return "You covered the main topic well, though you drifted into related areas occasionally. This is fine for context, but keep an eye on your focus."
        else:
            return "Your speech diverged significantly from the intended topic. Ensure your content is more tightly aligned with your goals."

    @staticmethod
    def get_expression_conclusion(expression_data: Dict) -> str:
        """
        Interprets facial expression data.
        Provides feedback on emotional connection and engagement.
        """
        if not expression_data or not expression_data.get("expression_breakdown"):
            return "No facial data detected. Ensure your face is clearly visible."
            
        breakdown = expression_data.get("expression_breakdown", {})
        smiling = breakdown.get("Smiling", 0)
        laughing = breakdown.get("Laughing", 0)
        angry = breakdown.get("Angry", 0)
        talking = breakdown.get("Talking", 0)
        neutral = breakdown.get("Neutral", 0)
        
        positive = smiling + laughing
        
        if angry > 15:
            return "You appeared frustrated or angry for a significant portion of your talk. Try to relax your brow and project a more approachable demeanor."
        elif positive > 20:
            return f"Excellent emotional engagement! You smiled or laughed for {positive:.0f}% of the time, which helps build a positive rapport with your audience."
        elif neutral > 70:
            return "Your expression was very neutral. While professional, adding a few smiles can make you appear more passionate and relatable."
        elif talking > 50:
            return "You were very focused on speaking. Don't forget to use facial expressions to emphasize your key points and connect emotionally."
        else:
            return "Your facial expressions were calm and composed throughout the presentation."

    @staticmethod
    def get_posture_conclusion(posture_data: Dict) -> str:
        """
        Interprets body posture data.
        Provides feedback on confidence, openness, and physical presence.
        """
        if not posture_data or not posture_data.get("posture_breakdown"):
            return "No posture data detected. Ensure your full body is visible to the camera."
        
        breakdown = posture_data.get("posture_breakdown", {})
        confident = breakdown.get("Confident", 0)
        slouching = breakdown.get("Slouching", 0)
        leaning = breakdown.get("Leaning", 0)
        closed = breakdown.get("Closed", 0)
        
        if confident > 70:
            return "Excellent posture! You maintained a confident, upright stance throughout your presentation. This projects authority and credibility."
        elif confident > 50:
            return f"Good posture overall ({confident:.0f}% confident). You projected a generally confident presence, though you occasionally slouched or shifted your weight. Try to maintain a stable stance."
        elif slouching > 30:
            return "You slouched frequently during your presentation. Standing tall with shoulders back will project more confidence and engage your audience better."
        elif closed > 20:
            return "Your arms were crossed or held close to your body. This can appear defensive or closed off. Try opening your arms and using gestures to appear more approachable and confident."
        elif leaning > 30:
            return "You shifted your weight and leaned frequently. Try to maintain balance and plant your feet firmly to appear more grounded and authoritative."
        else:
            return "Your posture was generally neutral. Try to maintain a more confident, upright stance to maximize your presence."
