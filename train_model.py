import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from utils import ACTIONS, SEQUENCE_LENGTH, NUM_FEATURES

DATA_PATH = 'MP_Data'
MODEL_PATH = 'action.keras'


def load_data():
    label_map = {label: num for num, label in enumerate(ACTIONS)}
    sequences, labels = [], []
    skipped = 0

    for action in ACTIONS:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            print(f"WARNING: No data for '{action}' — skipping.")
            continue

        for sequence in sorted(os.listdir(action_path), key=int):
            seq_path = os.path.join(action_path, sequence)
            # Checks if all 30 frames exist before loading
            if not all(os.path.exists(os.path.join(seq_path, f"{i}.npy")) for i in range(SEQUENCE_LENGTH)):
                skipped += 1
                continue

            window = [np.load(os.path.join(seq_path, f"{i}.npy")) for i in range(SEQUENCE_LENGTH)]
            sequences.append(window)
            labels.append(label_map[action])

    if skipped:
        print(f"Skipped {skipped} incomplete sequences.")

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    print(f"Loaded {X.shape[0]} sequences. X shape: {X.shape}, y shape: {y.shape}")

    # Report class balance
    class_counts = y.sum(axis=0)
    for action, count in zip(ACTIONS, class_counts):
        print(f"  {action}: {int(count)} sequences")

    return X, y


def build_model():
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        LSTM(64, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='tanh'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(ACTIONS.shape[0], activation='softmax'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.summary()

    callbacks = [
        TensorBoard(log_dir='Logs'),
        EarlyStopping(monitor='loss', patience=50, restore_best_weights=True),
    ]

    model.fit(X_train, y_train, epochs=500, callbacks=callbacks)

    # Evaluate
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1)
    ypred = np.argmax(yhat, axis=1)
    print("\nConfusion matrix:")
    print(multilabel_confusion_matrix(ytrue, ypred))
    print(f"Test accuracy: {accuracy_score(ytrue, ypred):.4f}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()