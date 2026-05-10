from collections import deque
import cv2
import mediapipe as mp
import time
from collections import defaultdict

class PostureAnalyzer:
    def __init__(self):
        self.buffer = deque(maxlen=10)

    def classify_posture(self, landmarks, w, h):
        import numpy as np

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

        # Slouching
        if nose[0] - shoulder_center[0] > 0.15 * w:
            return "slouching"

        # Leaning
        if abs(shoulder_center[0] - hip_center[0]) > 0.1 * w:
            return "leaning"

        # Closed
        if (np.linalg.norm(left_wrist - shoulder_center) < 0.3 * h and
            np.linalg.norm(right_wrist - shoulder_center) < 0.3 * h):
            return "closed"

        return "confident"

    def smooth(self, posture):
        self.buffer.append(posture)
        return max(set(self.buffer), key=self.buffer.count)

    def analyze(self, video_path):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        cap = cv2.VideoCapture(video_path)

        posture_times = defaultdict(float)
        current_posture = None
        start_time = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                raw = self.classify_posture(result.pose_landmarks.landmark, w, h)
                stable = self.smooth(raw)

                now = time.time()

                if current_posture is None:
                    current_posture = stable
                    start_time = now
                elif stable != current_posture:
                    posture_times[current_posture] += now - start_time
                    current_posture = stable
                    start_time = now

        # finalize
        if current_posture:
            posture_times[current_posture] += time.time() - start_time

        total = sum(posture_times.values())

        return {
            "durations": posture_times,
            "percentages": {
                k: (v / total) * 100 if total else 0
                for k, v in posture_times.items()
            }
        }