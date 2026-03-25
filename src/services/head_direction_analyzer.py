import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List

# === Configurable thresholds ===
YAW_THRESHOLD = 35           # left-right
PITCH_THRESHOLD_UP = 20      # up
PITCH_THRESHOLD_DOWN = 15    # down (stricter)
ROLL_THRESHOLD = 20          # head tilt

# 3D reference points (generic face model in mm)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye left corner
    (43.3, 32.7, -26.0),      # Right eye right corner
    (-28.9, -28.9, -24.1),    # Left mouth corner
    (28.9, -28.9, -24.1)      # Right mouth corner
], dtype=np.float64)

# Mediapipe landmark IDs corresponding to MODEL_POINTS
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

def _normalize_angle(angle: float) -> float:
    """Normalize angles from [0..360] to [-180..180] then clamp to [-90..90]"""
    angle = (angle + 180) % 360 - 180
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return angle

def _classify_direction(yaw: float, pitch: float, roll: float) -> str:
    """Classify head direction from yaw/pitch/roll angles"""
    if (
        abs(yaw) < YAW_THRESHOLD
        and -PITCH_THRESHOLD_UP < pitch < PITCH_THRESHOLD_DOWN
        and abs(roll) < ROLL_THRESHOLD
    ):
        return "LookingAtCamera"

    if abs(yaw) >= YAW_THRESHOLD:
        return "LookingDown" if yaw > 0 else "LookingRight"
    elif pitch >= PITCH_THRESHOLD_DOWN:
        return "LookingLeft"
    elif pitch <= -PITCH_THRESHOLD_UP:
        return "LookingUp"
    elif abs(roll) >= ROLL_THRESHOLD:
        return "TiltedLeft" if roll > 0 else "TiltedRight"
    return "NotLookingAtCamera"

class HeadDirectionAnalyzer:
    """
    Service to analyze head direction in a video using OpenCV and MediaPipe FaceMesh.
    """

    def analyze_video(self, video_path: str, sample_every_n_frames: int = 30) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration = 1.0 / fps

        looking_time = 0.0
        not_looking_time = 0.0
        direction_timeline: List[Dict] = []
        direction_counts: Dict[str, float] = {}

        frame_index = 0

        # Use MediaPipe FaceMesh context manager
        with mp_face_mesh.FaceMesh(
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

                # Process only every Nth frame
                if frame_index % sample_every_n_frames != 0:
                    continue

                if frame_index % 50 == 0:
                    print(f"Processing frame {frame_index}...")

                # Resize for performance if frame is too large
                h, w = frame.shape[:2]
                max_dim = 640
                if w > max_dim or h > max_dim:
                    scale = max_dim / max(w, h)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                status = "NoFaceDetected"
                yaw = pitch = roll = 0.0

                if results.multi_face_landmarks:
                    lms = results.multi_face_landmarks[0].landmark
                    image_points = np.array(
                        [(lms[idx].x * w, lms[idx].y * h) for idx in LANDMARK_IDS],
                        dtype=np.float64
                    )

                    focal_length = float(w)
                    cam_matrix = np.array(
                        [
                            [focal_length, 0, w / 2],
                            [0, focal_length, h / 2],
                            [0, 0, 1]
                        ],
                        dtype=np.float64
                    )
                    dist_coeffs = np.zeros((4, 1))

                    success, rvec, tvec = cv2.solvePnP(
                        MODEL_POINTS, image_points, cam_matrix, dist_coeffs
                    )
                    if success:
                        rmat, _ = cv2.Rodrigues(rvec)
                        proj = np.hstack((rmat, tvec))
                        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
                        pitch, yaw, roll = [_normalize_angle(float(x)) for x in euler]
                        status = _classify_direction(yaw, pitch, roll)

                # Accumulate time
                segment_duration = sample_every_n_frames * frame_duration
                if status == "LookingAtCamera":
                    looking_time += segment_duration
                else:
                    not_looking_time += segment_duration

                direction_counts[status] = direction_counts.get(status, 0.0) + segment_duration

                direction_timeline.append({
                    "time": round(elapsed, 3),
                    "status": status,
                    "yaw": round(yaw, 2),
                    "pitch": round(pitch, 2),
                    "roll": round(roll, 2)
                })

        cap.release()

        total_time = looking_time + not_looking_time
        percentage_looking = (looking_time / total_time * 100) if total_time > 0 else 0.0

        direction_breakdown = {
            direction: round((secs / total_time) * 100, 2) if total_time > 0 else 0.0
            for direction, secs in direction_counts.items()
        }

        return {
            "looking_time": round(looking_time, 2),
            "not_looking_time": round(not_looking_time, 2),
            "total_time": round(total_time, 2),
            "percentage_looking": round(percentage_looking, 2),
            "direction_breakdown": direction_breakdown,
            "direction_timeline": direction_timeline
        }