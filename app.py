import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

# --- TEAM BUILD HUB UI ---
st.set_page_config(page_title="Team Build Hub", page_icon="🤟")
st.title("🤟 AI Sign Language Translator")
st.subheader("Bridging the Gap with AI for Impact")

# --- LOAD YOUR AI BRAIN ---
@st.cache_resource
def load_files():
    # Load the model you sent me
    interpreter = tflite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    # Load your 9 labels (Hello, Namaste, etc.)
    with open("labels.txt", "r") as f:
        labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return interpreter, labels

interpreter, labels = load_files()

# --- CAMERA INTERFACE ---
st.write("### Step 1: Show a gesture to the camera")
img_file = st.camera_input("Take a photo of your sign")

if img_file:
    # Prepare image for your model (224x224)
    image = Image.open(img_file).convert('RGB').resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0 

    # Predict
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_index)
    result_index = np.argmax(prediction)
    
    # --- DISPLAY RESULT ---
    st.write("---")
    st.write("### Step 2: Translation")
    st.success(f"Detected Sign: **{labels[result_index]}**")
    st.info("Impact: Helping 60M+ people in India communicate better.")