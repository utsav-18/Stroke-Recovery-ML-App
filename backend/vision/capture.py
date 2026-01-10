import cv2
import mediapipe as mp
import numpy as np
from math import degrees

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def angle_between(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))

def capture_upper_limb(seconds=5):
    cap = cv2.VideoCapture(0)
    angles = []

    total_frames = seconds * 30  # ~30 FPS

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # LEFT arm (shoulder–elbow–wrist)
            shoulder = (lm[11].x, lm[11].y)
            elbow = (lm[13].x, lm[13].y)
            wrist = (lm[15].x, lm[15].y)

            angle = angle_between(shoulder, elbow, wrist)
            angles.append(angle)

    cap.release()
    return angles
