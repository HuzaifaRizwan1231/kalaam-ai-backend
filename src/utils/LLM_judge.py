def build_prompt(summary: dict) -> str:
    metrics = summary["metrics"]
    analysis = summary["analysis"]
    presentation = summary["presentation_summary"]

    return f"""
You are an expert public speaking coach.
Your task is to generate a structured, professional feedback report based ONLY on the provided analysis.
-----------------------
IMPORTANT RULES:
-----------------------
- DO NOT recompute or question the analysis
- DO NOT introduce new metrics
- ONLY use the provided conclusions and data
- Be honest, constructive, and specific
- Avoid generic advice
- Prioritize the most critical weaknesses
- Keep tone supportive, not harsh
- Make feedback actionable
-----------------------
INPUT DATA:
-----------------------
Presentation Summary:
- Topic: {presentation["topic"]}
- Duration: {presentation["duration"]} seconds

Key Metrics:
- Speaking Pace: {metrics["wpm"]} WPM → {analysis["pace"]}
- Filler Words: {metrics["filler_percentage"]}% → {analysis["fillers"]}
- Eye Contact: {metrics["eye_contact"]}% → {analysis["eye_contact"]}
- Clarity Score: {metrics["clarity_score"]} / 100
- Clarity Issues: {", ".join(analysis["clarity_issues"]) if analysis["clarity_issues"] else "None"}
- Volume: {analysis["volume"]}
- Intonation Score: {metrics["intonation_score"]} → {analysis["intonation"]}
- Gesture Score: {metrics["gesture_score"]} / 100 → {analysis["gestures"]}
- Posture: {analysis["posture"]}

Strengths:
{chr(10).join(f"- {s}" for s in summary["strengths"])}

Weaknesses:
{chr(10).join(f"- {w}" for w in summary["weaknesses"])}

Transcript Excerpt:
"{summary["transcript_excerpt"]}"
-----------------------
OUTPUT FORMAT (STRICT JSON):
-----------------------
Return ONLY valid JSON in this exact structure:
{{
  "overall_score": number (0-10),
  "summary": "A concise 2-3 sentence overall evaluation",
  "strengths": [
    "Expanded explanation of each strength with context"
  ],
  "key_issues": [
    "Most critical problems explained clearly and specifically"
  ],
  "detailed_feedback": [
    {{
      "category": "Delivery",
      "feedback_text": "Feedback on pace ({metrics["wpm"]} WPM), clarity score ({metrics["clarity_score"]}), volume ({analysis["volume"]}), and filler usage ({metrics["filler_percentage"]}%)"
    }},
    {{
      "category": "Body Language",
      "feedback_text": "Feedback on eye contact ({metrics["eye_contact"]}%), posture ({analysis["posture"]}), and gestures ({analysis["gestures"]})"
    }},
    {{
      "category": "Engagement",
      "feedback_text": "Feedback on audience connection, intonation ({analysis["intonation"]}), and expressiveness"
    }},
    {{
      "category": "Content",
      "feedback_text": "Feedback on topic relevance ({presentation["topic"]}) and overall structure"
    }}
  ],
  "actionable_improvements": [
    "Specific, concrete action the user can take to improve"
  ]
}}
-----------------------
EVALUATION GUIDELINES:
-----------------------
- Eye contact {metrics["eye_contact"]}% → {"CRITICAL issue, must address first" if metrics["eye_contact"] < 20 else "acceptable"}
- Clarity score {metrics["clarity_score"]} → {"high priority issue" if metrics["clarity_score"] < 40 else "acceptable"}
- Posture is {analysis["posture"]} → {"signals low confidence" if "closed" in analysis["posture"].lower() else "good"}
- Gestures are {analysis["gestures"]} → {"low engagement signal" if "limited" in analysis["gestures"].lower() else "good"}
- Pace {metrics["wpm"]} WPM is {analysis["pace"]} → {"leverage as strength" if analysis["pace"] == "Ideal" else "needs improvement"}
- Filler usage {metrics["filler_percentage"]}% is {analysis["fillers"]} → {"leverage as strength" if metrics["filler_percentage"] < 2 else "needs improvement"}

Focus more on fixing weaknesses than praising strengths.
Prioritize issues in this order: eye contact → clarity → posture → gestures → volume → intonation.
"""

def prepare_gemini_input(data: dict) -> dict:
    wpm_data = data.get("wpm_data") or {}
    wpm_conclusion = wpm_data.get("conclusion", "")
    
    filler_word_analysis = data.get("filler_word_analysis") or {}
    filler_pct = filler_word_analysis.get("filler_percentage", 0)
    
    loudness_analysis = data.get("loudness_analysis") or {}
    loudness_conclusion = loudness_analysis.get("conclusion", "")
    
    clarity = data.get("clarity_analysis") or {}
    head = data.get("head_direction_analysis") or {}
    posture = data.get("posture_analysis") or {}
    gesture = data.get("gesture_analysis") or {}
    intonation = data.get("intonation_analysis") or {}
    topic_coverage = data.get("topic_coverage") or {}

    # --- Derived labels ---
    def pace_label(conclusion):
        if not conclusion:
            return "Moderate"
        c = conclusion.lower()
        if "perfect" in c or "ideal" in c:
            return "Ideal"
        elif "fast" in c:
            return "Too fast"
        elif "slow" in c:
            return "Too slow"
        return "Moderate"

    def filler_label(pct):
        if pct is None:
            return "Moderate"
        if pct == 0:
            return "None"
        elif pct < 2:
            return "Very low"
        elif pct < 5:
            return "Moderate"
        return "High"

    def volume_label(conclusion):
        if not conclusion:
            return "Good"
        c = conclusion.lower()
        if "inconsistent" in c or "fluctuat" in c:
            return "Inconsistent"
        elif "loud" in c:
            return "Too loud"
        elif "quiet" in c:
            return "Too quiet"
        return "Good"

    def eye_contact_label(pct):
        if pct < 10:
            return "Very poor"
        elif pct < 40:
            return "Poor"
        elif pct < 70:
            return "Good"
        return "Excellent"

    def posture_label(breakdown):
        confident = breakdown.get("Confident", 0)
        closed = breakdown.get("Closed", 0)
        slouching = breakdown.get("Slouching", 0)
        if confident > 60:
            return "Open and confident"
        elif closed > 50 or slouching > 20:
            return "Mostly closed"
        return "Mixed"

    def gesture_label(score):
        if score >= 80:
            return "Active"
        elif score >= 60:
            return "Limited"
        return "Very limited"

    def intonation_label(label_str):
        mapping = {
            "monotone": "Flat",
            "moderate": "Expressive but slightly overdone",
            "expressive": "Highly expressive",
        }
        return mapping.get(label_str.lower(), label_str.capitalize())

    # --- Strengths & weaknesses ---
    strengths = []
    weaknesses = []

    wpm_intervals = wpm_data.get("intervals", [])
    avg_wpm = round(sum(i["wpm"] for i in wpm_intervals) / len(wpm_intervals)) if wpm_intervals else 0

    if "perfect" in wpm_conclusion.lower() or "ideal" in wpm_conclusion.lower():
        strengths.append("Excellent speaking pace")
    if filler_pct < 2:
        strengths.append("Very low filler words")
    if intonation.get("intonation_score", 0) > 0.6:
        strengths.append("Strong vocal expressiveness")
    if posture.get("posture_breakdown", {}).get("Confident", 0) > 50:
        strengths.append("Confident posture")

    eye_pct = head.get("percentage_looking", 0)
    if eye_pct < 20:
        weaknesses.append("Very poor eye contact")
    if "inconsistent" in loudness_conclusion.lower():
        weaknesses.append("Inconsistent volume")
    if posture.get("posture_breakdown", {}).get("Closed", 0) > 40:
        weaknesses.append("Closed body posture")
    if clarity.get("clarity_score", 100) < 40:
        weaknesses.append("Low audio clarity")
    if gesture.get("gesture_usage_ratio", 1) == 0:
        weaknesses.append("Very little hand gesture usage")

    # --- Transcript excerpt ---
    transcript = data.get("transcript", "")
    excerpt = transcript[:50] + "..." if len(transcript) > 50 else transcript

    # --- Duration ---
    duration = round(loudness_analysis.get("statistics", {}).get("total_duration", 0))

    return {
        "presentation_summary": {
            "duration": duration,
            "topic": topic_coverage.get("topic", "Unknown"),
        },
        "metrics": {
            "wpm": avg_wpm,
            "filler_percentage": round(filler_pct, 2),
            "eye_contact": round(eye_pct, 2),
            "clarity_score": round(clarity.get("clarity_score", 0), 2),
            "intonation_score": round(intonation.get("average_prosody_score", 0), 2),
            "gesture_score": gesture.get("presentation_gesture_score", 0),
        },
        "analysis": {
            "pace": pace_label(wpm_conclusion),
            "fillers": filler_label(filler_pct),
            "volume": volume_label(loudness_conclusion),
            "clarity_issues": clarity.get("reasons", []),
            "eye_contact": eye_contact_label(eye_pct),
            "posture": posture_label(posture.get("posture_breakdown", {})),
            "gestures": gesture_label(gesture.get("presentation_gesture_score", 0)),
            "intonation": intonation_label(intonation.get("intonation_label", "")),
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "transcript_excerpt": excerpt,
    }