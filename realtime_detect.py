import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import (
    mp_holistic, mediapipe_detection, draw_styled_landmarks,
    extract_keypoints, ACTIONS, SEQUENCE_LENGTH
)

MODEL_PATH = 'action.keras'
THRESHOLD = 0.5
COLORS = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40),
                      (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


def main():
    model = load_model(MODEL_PATH)
    sequence = []
    sentence = []
    predictions = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))

                
                if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > THRESHOLD:
                        if len(sentence) == 0 or ACTIONS[np.argmax(res)] != sentence[-1]:
                            sentence.append(ACTIONS[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, ACTIONS, image, COLORS)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Sign Language Detection', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()