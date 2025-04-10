import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
from collections import deque

# File paths
ALPHABET_MODEL_PATH = "models/asl_alphabet_model.h5"
ALPHABET_ENCODER_PATH = "models/label_encoder_alphabet.pkl"

# Load alphabet model and encoder
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(ALPHABET_MODEL_PATH)
    with open(ALPHABET_ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

# Load model
model, encoder = load_model_and_encoder()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Extract hand keypoints
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([v for lm in hand.landmark for v in (lm.x, lm.y, lm.z)])
    return np.zeros(63)

# Streamlit UI
st.set_page_config(page_title="ASL Alphabet Recognition", layout="centered")
st.title("ASL Alphabet Recognition")
st.caption("Show an ASL hand sign to your webcam")

# Webcam prediction
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.pred = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        keypoints = extract_keypoints(results)

        input_data = np.expand_dims(keypoints, axis=0)
        prediction = model.predict(input_data, verbose=0)[0]
        confidence = np.max(prediction)

        if confidence > 0.15:
            self.pred = encoder.inverse_transform([np.argmax(prediction)])[0]

        annotated = img.copy()
        cv2.putText(annotated, f"Prediction: {self.pred}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        return annotated

# Start webcam
webrtc_streamer(
    key="asl-stream",
    video_transformer_factory=ASLTransformer,
    media_stream_constraints={"video": True, "audio": False}
)

