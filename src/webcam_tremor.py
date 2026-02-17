"""
webcam_tremor.py

Webcam-based tremor capture integrated with longitudinal monitoring.
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt
from src.tremor_analysis import tremor_score
from src.session_tracker import SessionTracker


# ===============================
# Configuration
# ===============================

RECORD_SECONDS = 10
SAMPLING_RATE = 30
MAX_FRAMES = RECORD_SECONDS * SAMPLING_RATE


# ===============================
# High-pass filter
# ===============================

def highpass_filter(data, cutoff=1.0, fs=30, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


# ===============================
# MediaPipe Setup
# ===============================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils


# ===============================
# Session Tracker
# ===============================

tracker = SessionTracker(baseline_sessions=3)


# ===============================
# Main Recording Loop
# ===============================

# ===============================
# Continuous Camera Mode
# ===============================

cap = cv2.VideoCapture(0)

print("\nCamera started.")
print("Press 'S' to start recording.")
print("Press 'Q' to quit.\n")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(frame, "Press S to record | Q to quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.imshow("Tremor Monitoring", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Start Recording
    if key == ord('s'):

        print("\nRecording for 10 seconds...")

        wrist_y_values = []
        frame_count = 0

        while frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[
                        mp_hands.HandLandmark.WRIST
                    ]
                    pixel_y = wrist.y * frame.shape[0]
                    wrist_y_values.append(pixel_y)

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            cv2.putText(frame, "Recording...",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2)

            cv2.imshow("Tremor Monitoring", frame)

            frame_count += 1
            cv2.waitKey(1)

        print("Recording complete.")

        if len(wrist_y_values) < 100:
            print("Not enough data captured.")
            continue

        # Signal processing
        signal = np.array(wrist_y_values)
        signal = signal - np.mean(signal)
        signal = np.diff(signal)
        signal = highpass_filter(signal, cutoff=1.0, fs=SAMPLING_RATE)

        result = tremor_score(signal, SAMPLING_RATE)
        tracker.add_session(result)
        evaluation = tracker.evaluate_session(result)

        print("\nRaw Tremor Metrics:")
        print(result)

        print("\nMonitoring Evaluation:")
        print(evaluation)


cap.release()
cv2.destroyAllWindows()