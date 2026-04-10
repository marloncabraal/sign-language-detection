import cv2
import numpy as np
import os
from utils import (
    mp_holistic, mediapipe_detection, draw_styled_landmarks,
    extract_keypoints, ACTIONS, SEQUENCE_LENGTH
)

DATA_PATH = 'MP_Data'
NO_SEQUENCES = 30  # how many videos to record per action this run


def setup_folders():
    """Create folders for new sequences, continuing from any existing ones."""
    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        os.makedirs(action_path, exist_ok=True)
        existing = [int(f) for f in os.listdir(action_path) if f.isdigit()]
        start = max(existing) + 1 if existing else 0
        for seq in range(start, start + NO_SEQUENCES):
            os.makedirs(os.path.join(action_path, str(seq)), exist_ok=True)
    return start


def collect():
    start_folder = setup_folders()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            for sequence in range(start_folder, start_folder + NO_SEQUENCES):
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting {action} | Video {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)
                        cv2.waitKey(1500)   # 1.5 second pause to get ready
                    else:
                        cv2.putText(image, f'Collecting {action} | Video {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done! Recorded {NO_SEQUENCES} sequences per action.")


if __name__ == '__main__':
    collect()