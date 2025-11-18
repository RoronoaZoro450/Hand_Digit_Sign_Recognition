import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image


@st.cache_resource
def load_keras_model():

    model = load_model('sign_language_model.h5')
    return model

def preprocess_image(image):
    img = np.array(image)
    
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img, (64, 64))

    img_reshaped = img_resized.reshape(1, 64, 64, 1)
    
    return img_reshaped

# --- Main Streamlit App ---

st.title("Sign Language Digit Recognizer 🖐️")
st.write("Upload an image of a hand sign (digits 0-9) and the model will predict the digit.")

# Load the model
model = load_keras_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image and make prediction
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    # Get prediction
    prediction = model.predict(processed_image)
    st.write(prediction)
    # Get the highest confidence class
    predicted_class = np.argmax(prediction[0])
    if predicted_class == 0:
        predicted_class = "6"
    elif predicted_class == 1:
        predicted_class = "0"
    elif predicted_class == 2:
        predicted_class = "8"
    elif predicted_class == 3:
        predicted_class = "9"
    elif predicted_class == 4:
        predicted_class = "1"
    elif predicted_class == 5:
        predicted_class = "7"
    elif predicted_class == 6:
        predicted_class = "4"
    elif predicted_class == 7:
        predicted_class = "3"
    elif predicted_class == 8:
        predicted_class = "2"
    elif predicted_class == 9:
        predicted_class = "5"
    confidence = np.max(prediction[0]) * 100
    # Display the result
    st.success(f"**Predicted Digit:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

st.header("Demo Images:")

cols = st.columns(5) # Create 5 columns

for i in range(10):
    with cols[i % 5]:
        st.image(
            f"demo_img/{i}.png", 
            caption=f"{i}", 
        )

st.info("This model can only recognize digits 0-9. It is not trained on other characters")
