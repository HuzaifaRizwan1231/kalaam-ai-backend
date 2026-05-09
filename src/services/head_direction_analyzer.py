import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List
from datetime import datetime

# ---------------------------
# Configurable Thresholds for Head Orientation
# ---------------------------
# Higher thresholds = more lenient (wider 'center' box).
# Fixed in degrees (Euler angles).
YAW_THRESHOLD = 35           # Left-Right head rotation
PITCH_THRESHOLD_UP = 20      # Looking up (looking away from audience)
PITCH_THRESHOLD_DOWN = 15    # Looking down (often at notes, more restrictive)
ROLL_THRESHOLD = 20          # Tilt (leaning head sideways)

# ---------------------------
# 3D Generic Face Reference Model
# ---------------------------
# A set of coordinates representing a generic human face in 3D space (in mm).
# We use these to solve the 'Perspective-n-Point' (PnP) problem, mapping 2D pixels to 3D.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (Anchor/Origin)
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Outer corner of Left Eye
    (43.3, 32.7, -26.0),      # Outer corner of Right Eye
    (-28.9, -28.9, -24.1),    # Left corner of Mouth
    (28.9, -28.9, -24.1)      # Right corner of Mouth
], dtype=np.float64)

# MediaPipe FaceMesh landmark IDs corresponding to the MODEL_POINTS above
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Initialization of Google MediaPipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh

def _normalize_angle(angle: float) -> float:
    """
    Normalizes and clamps angles from generic rotation matrices [0..360] 
    into a centered range of [-90..90] for consistent human-readable orientation.
    """
    angle = (angle + 180) % 360 - 180
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return angle

def _classify_direction(yaw: float, pitch: float, roll: float) -> str:
    """
    Directional Logic Engine.
    Maps 3D Euler angles into discrete semantic categories (LookingAtCamera, LookingDown, etc.).
    
    Yaw (Z-axis rotation): +ve = Left, -ve = Right.
    Pitch (X-axis rotation): +ve = Down, -ve = Up.
    Roll (Y-axis rotation): +ve = Tilt Left, -ve = Tilt Right.
    """
    
    # 1. Primary Eye Contact Check (Centered within thresholds)
    if (
        abs(yaw) < YAW_THRESHOLD
        and abs(pitch) < PITCH_THRESHOLD_DOWN
        and abs(roll) < ROLL_THRESHOLD
    ):
        return "LookingAtCamera"

    # 2. Sequential Classifiers for other areas
    if abs(yaw) >= YAW_THRESHOLD:
        return "LookingLeft" if yaw > 0 else "LookingRight"
    elif pitch >= PITCH_THRESHOLD_DOWN:
        return "LookingDown"
    elif pitch <= -PITCH_THRESHOLD_UP:
        return "LookingUp"
    elif abs(roll) >= ROLL_THRESHOLD:
        return "TiltedLeft" if roll > 0 else "TiltedRight"
    
    return "NotLookingAtCamera"

class HeadDirectionAnalyzer:
    """
    Tracks and analyzes eye contact and head orientation in video streams.
    Uses Computer Vision (OpenCV) and Deep Learning (MediaPipe) to detect 
    social cues like distractions or lack of audience engagement.
    """

    def analyze_video(self, video_path: str, sample_every_n_frames: int = 30, audience_position: str = "front") -> Dict:
        """
        Processes a video file to generate an engagement report.
        - sample_every_n_frames: Analysis density (default 30 = 1 check per second at 30fps).
        - audience_position: "front", "left", "right", or "both" (where the audience is).
        """
        import time
        cap = cv2.VideoCapture(video_path)
        
        # Windows-specific retry logic for I/O locks/race conditions
        if not cap.isOpened():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Initial OpenCV open failed for {video_path}. Retrying in 0.5s...")
            time.sleep(0.5)
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Total frames in video: {total_frames} (sampling every {sample_every_n_frames})")
        frame_duration = 1.0 / fps

        good_contact_time = 0.0
        not_looking_time = 0.0
        direction_timeline: List[Dict] = []
        direction_counts: Dict[str, float] = {}

        # Eye Contact Validation Logic (Dependent on setup)
        # If the 'audience' is to the left, 'LookingLeft' counts as good eye contact.
        good_statuses = {"LookingAtCamera"}
        if audience_position == "left":
            good_statuses.add("LookingLeft")
        elif audience_position == "right":
            good_statuses.add("LookingRight")
        elif audience_position == "both":
            good_statuses.add("LookingLeft")
            good_statuses.add("LookingRight")

        frame_index = 0

        # High-Performance Face Mesh context
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,   # Optimized for video (tracks between frames)
            max_num_faces=1,           # Analyze only the primary speaker
            refine_landmarks=True,      # High precision iris/lips tracking
            min_detection_confidence=0.5
        ) as face_mesh:

            while frame_index < total_frames:
                # Seek to the target frame directly to avoid decoding intermediate frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    break

                elapsed = frame_index * frame_duration
                frame_index += sample_every_n_frames

                if frame_index % 100 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Head] Processing frame {frame_index}...")

                # Optimization: Resize large 4K/1080p frames to 640px to speed up inference
                # MediaPipe doesn't need high resolution for face silhouette tracking.
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
                    # Map precise 2D landmarks to pixel coordinates
                    image_points = np.array(
                        [(lms[idx].x * w, lms[idx].y * h) for idx in LANDMARK_IDS],
                        dtype=np.float64
                    )

                    # Estimate Camera Intrinsic Matrix (approximated focal length)
                    focal_length = float(w)
                    cam_matrix = np.array(
                        [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
                        dtype=np.float64
                    )
                    dist_coeffs = np.zeros((4, 1))

                    # 3D Math: Solve Head Pose (PnP Algorithm)
                    # Finds the 3D rotation (rvec) and translation (tvec) of the head.
                    success, rvec, tvec = cv2.solvePnP(
                        MODEL_POINTS, image_points, cam_matrix, dist_coeffs
                    )
                    if success:
                        rmat, _ = cv2.Rodrigues(rvec)
                        proj = np.hstack((rmat, tvec))
                        # Transform to degrees (Euler angles)
                        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
                        pitch, yaw, roll = [_normalize_angle(float(x)) for x in euler]
                        status = _classify_direction(yaw, pitch, roll)

                # Segment Accumulation: Total time per status
                segment_duration = sample_every_n_frames * frame_duration
                if status in good_statuses:
                    good_contact_time += segment_duration
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

        # Final aggregate metrics
        total_time = good_contact_time + not_looking_time
        percentage_looking = (good_contact_time / total_time * 100) if total_time > 0 else 0.0

        # Percentual breakdown for each direction detected
        direction_breakdown = {
            direction: round((secs / total_time) * 100, 2) if total_time > 0 else 0.0
            for direction, secs in direction_counts.items()
        }

        return {
            "audience_position": audience_position,
            "looking_time": round(good_contact_time, 2),
            "not_looking_time": round(not_looking_time, 2),
            "total_time": round(total_time, 2),
            "percentage_looking": round(percentage_looking, 2),
            "direction_breakdown": direction_breakdown,
            "direction_timeline": direction_timeline
        }