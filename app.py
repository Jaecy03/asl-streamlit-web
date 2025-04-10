import streamlit as st
import numpy as np
import cv2
st.write("OpenCV version:", cv2.__version__)
import tensorflow as tf
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
from collections import deque

# Trigger dependency loading
np, cv2, tf, pickle, webrtc_streamer, VideoTransformerBase, mp, deque = load_dependencies()


# Paths
ALPHABET_MODEL_PATH = "models/asl_alphabet_model.h5"
LSTM_MODEL_PATH = "models/asl_lstm_model.h5"
ENCODER_ALPHABET_PATH = "models/label_encoder_alphabet.pkl"
ENCODER_WORD_PATH = "models/label_encoder_word.pkl"

# Load models and encoders
@st.cache_resource
def load_model_and_encoder(mode):
    if mode == "Alphabet":
        model = tf.keras.models.load_model(ALPHABET_MODEL_PATH)
        with open(ENCODER_ALPHABET_PATH, "rb") as f:
            encoder = pickle.load(f)
    else:
        model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        with open(ENCODER_WORD_PATH, "rb") as f:
            encoder = pickle.load(f)
    return model, encoder

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

# Extract features
def extract_hand_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([v for lm in hand.landmark for v in (lm.x, lm.y, lm.z)])
    return np.zeros(63)

def extract_holistic_keypoints(results):
    def flatten(landmarks, count):
        if not landmarks:
            return [0.0] * count * 3
        return [v for lm in landmarks[:count] for v in (lm.x, lm.y, lm.z)] + [0.0] * max(0, (count - len(landmarks)) * 3)

    pose = flatten(results.pose_landmarks.landmark if results.pose_landmarks else [], 33)
    face = flatten(results.face_landmarks.landmark if results.face_landmarks else [], 468)
    lh = flatten(results.left_hand_landmarks.landmark if results.left_hand_landmarks else [], 21)
    rh = flatten(results.right_hand_landmarks.landmark if results.right_hand_landmarks else [], 21)

    return np.array(pose + face + lh + rh)

# UI
st.set_page_config(page_title="ASL Recognition", layout="centered")
st.title("ASL Sign Language Recognition")
mode = st.selectbox("Choose Recognition Mode:", ["Alphabet", "Word"])
model, encoder = load_model_and_encoder(mode)

# Streamlit webcam class
class ASLRecognizer(VideoTransformerBase):
    def __init__(self):
        self.sequence = deque(maxlen=30)
        self.prediction = ""
        self.confidence = 0.0

        if mode == "Word":
            self.processor = mp_holistic.Holistic(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        else:
            self.processor = mp_hands.Hands(
                max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
            )

    def transform(self, frame):
        # Get image as array
        img = frame.to_ndarray(format="bgr24")

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.processor.process(rgb)

        if mode == "Alphabet":
            keypoints = extract_hand_keypoints(results)
            input_data = np.expand_dims(keypoints, axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            self.confidence = np.max(prediction)
            if self.confidence > 0.15:
                self.prediction = encoder.inverse_transform([np.argmax(prediction)])[0]

        else:
            keypoints = extract_holistic_keypoints(results)
            self.sequence.append(keypoints)
            if len(self.sequence) == 30:
                input_seq = np.expand_dims(self.sequence, axis=0)
                prediction = model.predict(input_seq, verbose=0)[0]
                self.confidence = np.max(prediction)
                if self.confidence > 0.5:
                    self.prediction = encoder.inverse_transform([np.argmax(prediction)])[0]

        # Always draw something so Streamlit gets output
        annotated = img.copy()
        cv2.putText(annotated, f"Prediction: {self.prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(annotated, f"Confidence: {self.confidence:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return annotated  # ✅ always return a frame!


# Start webcam stream
webrtc_streamer(
    key="asl",
    video_transformer_factory=ASLRecognizer,
    media_stream_constraints={"video": True, "audio": False}
)



