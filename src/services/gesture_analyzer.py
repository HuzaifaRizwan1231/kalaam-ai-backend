import cv2
import mediapipe as mp
import math
from datetime import datetime
from collections import defaultdict
from typing import Dict, List


class GestureAnalyzer:
    """
    Gesture analysis service for presentation videos.

    Detects:
    - Open hand gestures
    - Pointing gestures
    - Arms crossed
    - Excessive movement
    - Low gesture usage
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def hand_is_open(self, hand_landmarks) -> bool:
        """
        Detect if palm is open using finger tip distances.
        """

        tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP,
        ]

        palm = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        extended = 0

        for tip in tips:
            finger_tip = hand_landmarks.landmark[tip]

            dist = self.calculate_distance(palm, finger_tip)

            if dist > 0.18:
                extended += 1

        return extended >= 3

    def is_pointing(self, hand_landmarks) -> bool:
        """
        Detect pointing gesture.
        Index finger extended while others folded.
        """

        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        middle_tip = hand_landmarks.landmark[
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
        ]

        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]

        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        index_dist = self.calculate_distance(index_tip, wrist)
        middle_dist = self.calculate_distance(middle_tip, wrist)
        ring_dist = self.calculate_distance(ring_tip, wrist)
        pinky_dist = self.calculate_distance(pinky_tip, wrist)

        return (
            index_dist > 0.20
            and middle_dist < 0.18
            and ring_dist < 0.18
            and pinky_dist < 0.18
        )

    def arms_crossed(self, pose_landmarks) -> bool:
        """
        Detect crossed arms using wrist/elbow positions.
        """

        left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]

        right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]

        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]

        left_cross = self.calculate_distance(left_wrist, right_elbow)
        right_cross = self.calculate_distance(right_wrist, left_elbow)

        return left_cross < 0.15 and right_cross < 0.15

    def analyze_gestures(self, video_path: str, sample_every_n_frames: int = 30) -> Dict:
        if not video_path:
            raise ValueError("Video path is required")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video")

        gesture_counts = defaultdict(int)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gesture_frames = 0
        frame_index = 0
        processed_frames = 0

        previous_left_wrist = None
        previous_right_wrist = None
        total_motion = 0

        print(f"[{datetime.now().strftime('%H:%M:%S')}] [Gestures] Starting analysis of {total_frames} frames (every {sample_every_n_frames})")

        with self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose, self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2
        ) as hands:

            while frame_index < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % 100 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [Gestures] Processing frame {frame_index}/{total_frames}...")

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                pose_results = pose.process(rgb_frame)
                hand_results = hands.process(rgb_frame)

                frame_has_gesture = False

                # -----------------------------
                # HAND GESTURE ANALYSIS
                # -----------------------------
                if hand_results.multi_hand_landmarks:

                    for hand_landmarks in hand_results.multi_hand_landmarks:

                        if self.hand_is_open(hand_landmarks):
                            gesture_counts["open_palm"] += 1
                            frame_has_gesture = True

                        if self.is_pointing(hand_landmarks):
                            gesture_counts["pointing"] += 1
                            frame_has_gesture = True

                # -----------------------------
                # POSE ANALYSIS
                # -----------------------------
                if pose_results.pose_landmarks:

                    pose_landmarks = pose_results.pose_landmarks

                    if self.arms_crossed(pose_landmarks):
                        gesture_counts["arms_crossed"] += 1

                    left_wrist = pose_landmarks.landmark[
                        self.mp_pose.PoseLandmark.LEFT_WRIST
                    ]

                    right_wrist = pose_landmarks.landmark[
                        self.mp_pose.PoseLandmark.RIGHT_WRIST
                    ]

                    if previous_left_wrist and previous_right_wrist:

                        left_motion = self.calculate_distance(
                            left_wrist, previous_left_wrist
                        )

                        right_motion = self.calculate_distance(
                            right_wrist, previous_right_wrist
                        )

                        total_motion += left_motion + right_motion

                    previous_left_wrist = left_wrist
                if frame_has_gesture:
                    gesture_frames += 1
                
                processed_frames += 1
                frame_index += sample_every_n_frames

        cap.release()

        # -----------------------------
        # FINAL ANALYSIS
        # -----------------------------

        gesture_usage_ratio = gesture_frames / processed_frames if processed_frames > 0 else 0

        average_motion = total_motion / processed_frames if processed_frames > 0 else 0

        feedback: List[str] = []

        if gesture_usage_ratio < 0.15:
            feedback.append("Very little hand gesture usage detected.")

        elif gesture_usage_ratio < 0.35:
            feedback.append("Moderate gesture usage detected.")

        else:
            feedback.append("Good hand gesture engagement detected.")

        if average_motion > 0.08:
            feedback.append("Excessive hand movement detected.")

        if gesture_counts["arms_crossed"] > 20:
            feedback.append("Frequent arms crossed posture may reduce openness.")

        score = 100

        if gesture_usage_ratio < 0.15:
            score -= 25

        if average_motion > 0.08:
            score -= 15

        if gesture_counts["arms_crossed"] > 20:
            score -= 15

        score = max(score, 0)

        return {
            "gesture_counts": dict(gesture_counts),
            "gesture_usage_ratio": round(gesture_usage_ratio, 2),
            "average_hand_motion": round(average_motion, 4),
            "presentation_gesture_score": score,
            "feedback": feedback,
        }
