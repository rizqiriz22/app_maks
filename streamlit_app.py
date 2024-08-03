import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\Lenovo\\detectmask_model2.h5')

# Define the class names
class_names = ['WithMask', 'WithoutMask']  # Replace with your actual class names

# Function to preprocess the frame
def preprocess_frame(frame):
    image = cv2.resize(frame, (150, 150))  # Resize the frame to match the input size
    image = image / 255.0  # Normalize the frame
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Selamat Datang di Deteksi Masker with OpenCV")
st.title(" ")
st.write("aplikasi Deteksi masker ini adalah teknologi yang menggunakan visi komputer untuk menentukan apakah seseorang memakai masker dengan benar. Dengan memanfaatkan kekuatan OpenCV, kita dapat membuat aplikasi real-time untuk mendeteksi masker.")

# Use Streamlit's webcam input feature
frame_window = st.image([])

# Open a video capture object
cap = cv2.VideoCapture(1)

# Read frames from the webcam in a loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame
    processed_frame = preprocess_frame(frame_rgb)

    # Make predictions
    predictions = model.predict(processed_frame)
    predicted_class = class_names[np.argmax(predictions[0])]

    # Display the frame with the prediction
    frame_rgb = cv2.putText(frame_rgb, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    frame_window.image(frame_rgb)

# Release the video capture object
cap.release()
