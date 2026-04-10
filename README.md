# Sign Language Detection

Did this as a project for the course COSC-4427-W02 - Topics in Computer Science II (Computer Vision) in Winter 2026. This is my contribution of the group work. Eventually, towards the end of the semester, me any my team members were able to expand the dataset to 50 guestures using an exsisting public database. The link to the final implementation repo and other test implementations is pasted below.

**https://github.com/tfayemi/signlang_classifier.git**

Real-time sign language gesture recognition using MediaPipe Holistic for keypoint extraction and an LSTM neural network (TensorFlow/Keras) for sequence classification. The model currently recognizes three gestures: hello, thanks, and iloveyou.

## How it works

MediaPipe Holistic extracts 1662 keypoints per frame (pose, face, and both hands). Each gesture is a sequence of 30 frames. An LSTM model is trained on these sequences to classify the gesture. At inference time, the webcam feed is processed live and predictions are overlaid on screen.

After training, the terminal displays the accuracy and the cunfusion matrix for each class.

## Project structure

```
sign-language-detection/
├── utils.py             # MediaPipe helpers, constants
├── collect_data.py      # Records training sequences from your webcam
├── train_model.py       # Trains the LSTM and saves action.keras
├── realtime_detect.py   # Runs live detection using the trained model
├── requirements.txt
└── action.keras         # Pre-trained model (included)
```

## Setup

Create a vertual env

```
pip install -r requirements.txt
```

## Running

1. Collect your own training data

   ```
   python collect_data.py
   ```

2. Train the model

   ```
   python train_model.py
   ```

3. Real-time detection

   ```
   python realtime_detect.py
   ```

This trains the LSTM and overwrites action.keras. Training logs are written to Logs/ and can be viewed with TensorBoard:

```
tensorboard --logdir Logs
```

## Adding new gestures

Edit the ACTIONS array in utils.py, then re-run collect_data.py and train_model.py.
