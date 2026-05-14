from google import genai
from pydantic import BaseModel
from typing import List, Dict
from ..utils.LLM_judge import build_prompt


class FeedbackItem(BaseModel):
    category: str
    feedback_text: str

class FinalFeedback(BaseModel):
    overall_score: float
    summary: str
    strengths: List[str]
    key_issues: List[str]
    detailed_feedback: List[FeedbackItem] 
    actionable_improvements: List[str]



class GeminiFeedbackService:

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate_feedback(self, prepared_data: dict) -> FinalFeedback:
        prompt = build_prompt(prepared_data)
    
        try:

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": FinalFeedback
                }
            )
        except:
            return "No Judge feedback."

        return response.parsed