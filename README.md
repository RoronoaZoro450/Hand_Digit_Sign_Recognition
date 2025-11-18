Deployed on Streamlit: https://hand-sign-digit-recognition.streamlit.app

Sign Language Digit Classifier (Streamlit App)

This is a simple Streamlit web app that predicts hand sign digits (0–9) using a trained deep learning model.

⭐ Features

Upload an image of a hand sign

Image is converted to grayscale and resized to 64×64

Model predicts the digit

Shows prediction

🚀 How to Run

Install dependencies:

pip install streamlit tensorflow opencv-python-headless numpy pillow


Place your trained model file in the same folder and name it:

model.h5


Run the app:

streamlit run streamlit_sign_language_app.py


Upload an image → Click Predict

📁 Project Structure
|-- model.h5
|-- streamlit_sign_language_app.py
|-- README.md

🧠 Model Details

Input shape: (64, 64, 1)

Output: 10-class softmax (digits 0–9)

Trained on sign-language digits dataset

🤝 Contribute

Feel free to fork, open issues, or improve the UI.
