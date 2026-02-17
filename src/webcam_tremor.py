"""
webcam_tremor.py

Captures wrist motion from webcam using MediaPipe,
applies preprocessing (velocity + high-pass filter),
and runs tremor analysis.
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt
from src.tremor_analysis import tremor_score


# ===============================
# Configuration
# ===============================

RECORD_SECONDS = 15
SAMPLING_RATE = 30
MAX_FRAMES = RECORD_SECONDS * SAMPLING_RATE


# ===============================
# High-pass filter (removes drift)
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
# Webcam Capture
# ===============================

cap = cv2.VideoCapture(0)

wrist_y_values = []
frame_count = 0

print("\nRecording tremor data...")
print("Keep hand steady in front of camera.")
print("Recording for 15 seconds...\n")

while cap.isOpened() and frame_count < MAX_FRAMES:
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

            # Convert normalized coordinate to pixel scale
            pixel_y = wrist.y * frame.shape[0]

            wrist_y_values.append(pixel_y)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Tremor Capture", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

print("\nRecording complete.\n")


# ===============================
# Signal Processing
# ===============================

signal = np.array(wrist_y_values)

if len(signal) < 100:
    print("Not enough data captured.")
    exit()

# Remove DC offset
signal = signal - np.mean(signal)

# Convert position → velocity (enhances oscillation)
signal = np.diff(signal)

# Apply high-pass filter (removes slow drift)
signal = highpass_filter(signal, cutoff=1.0, fs=SAMPLING_RATE)


# ===============================
# Tremor Analysis
# ===============================

result = tremor_score(signal, SAMPLING_RATE)

print("----- Tremor Analysis Result -----")
print(f"Variance: {result['variance']:.4f}")
print(f"Amplitude: {result['amplitude']:.4f}")
print(f"Dominant Frequency: {result['dominant_frequency']:.2f} Hz")
print(f"Tremor Score: {result['tremor_score']}")
print("----------------------------------")