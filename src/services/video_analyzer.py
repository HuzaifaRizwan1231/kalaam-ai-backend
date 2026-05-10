import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional

# ---------------------------
# Configurable Thresholds for Head Orientation
# ---------------------------
YAW_THRESHOLD = 35
PITCH_THRESHOLD_UP = 20
PITCH_THRESHOLD_DOWN = 15
ROLL_THRESHOLD = 20

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Outer corner of Left Eye
    (43.3, 32.7, -26.0),      # Outer corner of Right Eye
    (-28.9, -28.9, -24.1),    # Left corner of Mouth
    (28.9, -28.9, -24.1)      # Right corner of Mouth
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

def _normalize_angle(angle: float) -> float:
    angle = (angle + 180) % 360 - 180
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    return angle

def _classify_direction(yaw: float, pitch: float, roll: float) -> str:
    if abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD_DOWN and abs(roll) < ROLL_THRESHOLD:
        return "LookingAtCamera"
    if abs(yaw) >= YAW_THRESHOLD:
        return "LookingLeft" if yaw > 0 else "LookingRight"
    elif pitch >= PITCH_THRESHOLD_DOWN:
        return "LookingDown"
    elif pitch <= -PITCH_THRESHOLD_UP:
        return "LookingUp"
    elif abs(roll) >= ROLL_THRESHOLD:
        return "TiltedLeft" if roll > 0 else "TiltedRight"
    return "NotLookingAtCamera"

class VideoAnalyzer:
    """
    Combined analyzer for Head Direction (Eye Contact), Facial Expressions, and Posture.
    Reduces compute by processing both FaceMesh and Pose in the same loop.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.movement_history_window = 30

    def _dist(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def _get_points(self, landmarks, w, h):
        return [(int(p.x * w), int(p.y * h)) for p in landmarks]

    def _facial_movement(self, prev, curr):
        if prev is None: return 0
        diff = sum(self._dist(prev[i], curr[i]) for i in range(len(curr)))
        return diff / len(curr)

    def _detect_expression(self, points, avg_movement):
        face_width = self._dist(points[234], points[454])
        if face_width == 0: return "Neutral"
        
        mouth_width = self._dist(points[61], points[291])
        mouth_height = self._dist(points[13], points[14])
        
        smile_ratio = mouth_width / face_width
        open_ratio = mouth_height / face_width
        
        corner_y = (points[61][1] + points[291][1]) / 2
        lip_y = points[13][1]
        smile_upward = (lip_y - corner_y) / face_width

        left_brow_eye = self._dist(points[107], points[159])
        right_brow_eye = self._dist(points[336], points[386])
        brow_ratio = (left_brow_eye + right_brow_eye) / (2 * face_width)
        
        if smile_ratio > 0.44 or (smile_ratio > 0.38 and smile_upward > 0.02):
            return "Laughing" if open_ratio > 0.1 else "Smiling"
        elif brow_ratio < 0.07: return "Angry"
        elif open_ratio > 0.05: return "Talking"
        elif avg_movement < 0.3: return "Neutral"
        else: return "Calm"

    def _classify_posture(self, landmarks, w, h):
        """Classify body posture from MediaPipe Pose landmarks"""
        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        nose = get_point(0)
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        left_hip = get_point(23)
        right_hip = get_point(24)
        left_wrist = get_point(15)
        right_wrist = get_point(16)

        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # Slouching: nose moved significantly forward
        if nose[0] - shoulder_center[0] > 0.15 * w:
            return "Slouching"

        # Leaning: shoulder and hip misalignment
        if abs(shoulder_center[0] - hip_center[0]) > 0.1 * w:
            return "Leaning"

        # Closed: arms crossed near body
        if (np.linalg.norm(left_wrist - shoulder_center) < 0.3 * h and
            np.linalg.norm(right_wrist - shoulder_center) < 0.3 * h):
            return "Closed"

        return "Confident"

    def analyze_video(self, video_path: str, sample_every_n_frames: int = 30, audience_position: str = "front") -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            time.sleep(0.5)
            cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_duration = 1.0 / fps
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Combined] Starting analysis of {total_frames} frames (every {sample_every_n_frames})")

        # Head data
        good_contact_time = 0.0
        not_looking_time = 0.0
        direction_timeline = []
        direction_counts = {}
        good_statuses = {"LookingAtCamera"}
        if audience_position == "left": good_statuses.add("LookingLeft")
        elif audience_position == "right": good_statuses.add("LookingRight")
        elif audience_position == "both": 
            good_statuses.add("LookingLeft")
            good_statuses.add("LookingRight")

        # Expression data
        expression_counts = {k: 0.0 for k in ["Smiling", "Laughing", "Angry", "Talking", "Neutral", "Calm", "NoFaceDetected"]}
        expression_timeline = []
        movement_history = []
        prev_landmarks = None
        
        # Posture data
        posture_counts = {k: 0.0 for k in ["Confident", "Slouching", "Leaning", "Closed", "NoBodyDetected"]}
        posture_timeline = []
        
        frame_index = 0

        with self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:

            with self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.5
            ) as pose:

                while frame_index < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if not ret: break

                    elapsed = frame_index * frame_duration
                    if frame_index % 100 == 0:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Combined] Processing frame {frame_index}/{total_frames}...")

                    h, w = frame.shape[:2]
                    max_dim = 640
                    if w > max_dim or h > max_dim:
                        scale = max_dim / max(w, h)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                        h, w = frame.shape[:2]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                    pose_results = pose.process(rgb)

                    status = "NoFaceDetected"
                    expression = "NoFaceDetected"
                    yaw = pitch = roll = 0.0
                    avg_movement = 0.0
                    posture = "NoBodyDetected"

                    if results.multi_face_landmarks:
                        lms = results.multi_face_landmarks[0].landmark
                        points = self._get_points(lms, w, h)
                        
                        # 1. Head Direction (PnP)
                        image_points = np.array([(lms[idx].x * w, lms[idx].y * h) for idx in LANDMARK_IDS], dtype=np.float64)
                        focal_length = float(w)
                        cam_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float64)
                        success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, np.zeros((4, 1)))
                        if success:
                            rmat, _ = cv2.Rodrigues(rvec)
                            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvec)))
                            pitch, yaw, roll = [_normalize_angle(float(x)) for x in euler]
                            status = _classify_direction(yaw, pitch, roll)

                        # 2. Expression
                        movement = self._facial_movement(prev_landmarks, points)
                        movement_history.append(movement)
                        if len(movement_history) > self.movement_history_window: movement_history.pop(0)
                        avg_movement = np.mean(movement_history)
                        expression = self._detect_expression(points, avg_movement)
                        prev_landmarks = points

                    # 3. Posture (from pose landmarks)
                    if pose_results.pose_landmarks:
                        posture = self._classify_posture(pose_results.pose_landmarks.landmark, w, h)

                    # Accumulate segment
                    segment_duration = sample_every_n_frames * frame_duration
                    if status in good_statuses: good_contact_time += segment_duration
                    else: not_looking_time += segment_duration
                    direction_counts[status] = direction_counts.get(status, 0.0) + segment_duration
                    direction_timeline.append({"time": round(elapsed, 3), "status": status, "yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)})

                    expression_counts[expression] = expression_counts.get(expression, 0.0) + segment_duration
                    expression_timeline.append({"time": round(elapsed, 3), "expression": expression, "movement": round(float(avg_movement), 3)})

                    posture_counts[posture] = posture_counts.get(posture, 0.0) + segment_duration
                    posture_timeline.append({"time": round(elapsed, 3), "posture": posture})

                    frame_index += sample_every_n_frames

        cap.release()
        total_time_calc = good_contact_time + not_looking_time
        
        # Breakdown calcs
        head_breakdown = {d: round((s / total_time_calc) * 100, 2) if total_time_calc > 0 else 0.0 for d, s in direction_counts.items()}
        expr_breakdown = {e: round((s / total_time_calc) * 100, 2) if total_time_calc > 0 else 0.0 for e, s in expression_counts.items()}
        posture_breakdown = {p: round((s / total_time_calc) * 100, 2) if total_time_calc > 0 else 0.0 for p, s in posture_counts.items()}

        return {
            "head": {
                "audience_position": audience_position,
                "looking_time": round(good_contact_time, 2),
                "not_looking_time": round(not_looking_time, 2),
                "total_time": round(total_time_calc, 2),
                "percentage_looking": round((good_contact_time / total_time_calc * 100) if total_time_calc > 0 else 0.0, 2),
                "direction_breakdown": head_breakdown,
                "direction_timeline": direction_timeline
            },
            "expression": {
                "expression_breakdown": expr_breakdown,
                "expression_counts": {k: round(v, 2) for k, v in expression_counts.items()},
                "expression_timeline": expression_timeline,
                "total_time": round(total_time_calc, 2)
            },
            "posture": {
                "posture_breakdown": posture_breakdown,
                "posture_counts": {k: round(v, 2) for k, v in posture_counts.items()},
                "posture_timeline": posture_timeline,
                "total_time": round(total_time_calc, 2)
            }
        }
