import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

st.set_page_config(page_title="Team Build Hub", page_icon="🤟")
st.title("🤟 AI Sign Language Detector (Fast Version)")

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

st.write("Team Build Hub - Accessibility for India")

img_file = st.camera_input("Take a photo of your hand sign")

if img_file:
    image = Image.open(img_file)
    img_array = np.array(image)
    
    # Process with MediaPipe
    results = hands.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        st.success("✅ Hand Detected!")
        # Drawing landmarks for "WOW" factor
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img_array, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        st.image(img_array, caption="AI Tracking Points")
        st.info("Impact: Helping bridge the communication gap.")
    else:
        st.warning("No hand detected. Please try again with clear lighting!")
