import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional

class FacialExpressionAnalyzer:
    """
    Analyzes facial expressions in video streams using MediaPipe Face Mesh.
    Identifies states like Smiling, Laughing, Angry, Neutral, and Talking.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.movement_history_window = 30 # Number of frames for moving average

    def _dist(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _get_points(self, landmarks, w, h):
        return [(int(p.x * w), int(p.y * h)) for p in landmarks]

    def _facial_movement(self, prev, curr):
        if prev is None:
            return 0
        diff = 0
        for i in range(len(curr)):
            diff += self._dist(prev[i], curr[i])
        return diff / len(curr)

    def _detect_expression(self, points, avg_movement):
        # Normalization factor (face width)
        # 234: Left side of face, 454: Right side of face
        face_width = self._dist(points[234], points[454])
        if face_width == 0:
            return "Neutral"
        
        # Mouth corners (61, 291), Upper lip (13), Lower lip (14)
        mouth_width = self._dist(points[61], points[291])
        mouth_height = self._dist(points[13], points[14])
        
        # Ratios
        smile_ratio = mouth_width / face_width
        open_ratio = mouth_height / face_width
        
        # Corner height relative to upper lip (y-axis is inverted)
        corner_y = (points[61][1] + points[291][1]) / 2
        lip_y = points[13][1]
        smile_upward = (lip_y - corner_y) / face_width

        # Brow metrics (Anger detection)
        # 107, 159: Left brow/eye, 336, 386: Right brow/eye
        left_brow_eye = self._dist(points[107], points[159])
        right_brow_eye = self._dist(points[336], points[386])
        brow_ratio = (left_brow_eye + right_brow_eye) / (2 * face_width)
        
        # Expression Logic
        if smile_ratio > 0.44 or (smile_ratio > 0.38 and smile_upward > 0.02):
            if open_ratio > 0.1:
                return "Laughing"
            return "Smiling"
        elif brow_ratio < 0.07: 
            return "Angry"
        elif open_ratio > 0.05: # Mouth open but no smile
            return "Talking"
        elif avg_movement < 0.3:
            return "Neutral"
        else:
            return "Calm"

    def analyze_video(self, video_path: str, sample_every_n_frames: int = 5) -> Dict:
        """
        Processes a video file to generate a facial expression report.
        """
        import time
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            time.sleep(0.5)
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration = 1.0 / fps

        expression_counts: Dict[str, float] = {
            "Smiling": 0.0,
            "Laughing": 0.0,
            "Angry": 0.0,
            "Talking": 0.0,
            "Neutral": 0.0,
            "Calm": 0.0,
            "NoFaceDetected": 0.0
        }
        
        expression_timeline: List[Dict] = []
        movement_history = []
        prev_landmarks = None
        frame_index = 0

        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                elapsed = frame_index * frame_duration
                frame_index += 1

                # We sample more frequently for expressions as they can be fleeting
                if frame_index % sample_every_n_frames != 0:
                    continue

                h, w = frame.shape[:2]
                # Optimization: Resize for faster inference
                max_dim = 640
                if w > max_dim or h > max_dim:
                    scale = max_dim / max(w, h)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                expression = "NoFaceDetected"
                avg_movement = 0.0

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    points = self._get_points(lm, w, h)

                    # Movement calculation
                    movement = self._facial_movement(prev_landmarks, points)
                    movement_history.append(movement)
                    if len(movement_history) > self.movement_history_window:
                        movement_history.pop(0)
                    avg_movement = np.mean(movement_history)

                    # Detect expression
                    expression = self._detect_expression(points, avg_movement)
                    prev_landmarks = points

                # Accumulate time
                segment_duration = sample_every_n_frames * frame_duration
                expression_counts[expression] = expression_counts.get(expression, 0.0) + segment_duration

                expression_timeline.append({
                    "time": round(elapsed, 3),
                    "expression": expression,
                    "movement": round(float(avg_movement), 3)
                })

        cap.release()

        total_time = sum(expression_counts.values())
        
        # Percentual breakdown
        expression_breakdown = {
            exp: round((secs / total_time) * 100, 2) if total_time > 0 else 0.0
            for exp, secs in expression_counts.items()
        }

        return {
            "expression_breakdown": expression_breakdown,
            "expression_counts": {k: round(v, 2) for k, v in expression_counts.items()},
            "expression_timeline": expression_timeline,
            "total_time": round(total_time, 2)
        }
