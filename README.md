# Emotion-Based-Music-Recommendation-CNN

## 📌 Overview

The Emotion-Based Music Recommendation System is a deep learning-based web application that detects a user's emotional state through facial expression recognition and recommends music accordingly.

A Convolutional Neural Network (CNN) model is trained to classify emotions such as Happy, Sad, Angry, Fear, Surprise, and Neutral. Based on the detected emotion, the system suggests songs from a predefined music dataset.

---

## 🚀 Features

* Real-time facial emotion detection
* CNN-based deep learning model
* Emotion classification (Happy, Sad, Angry, Fear, Surprise, Neutral)
* Mood-based music recommendation
* Flask-based web application
* Interactive user interface

---

## 🛠️ Technologies Used

* Python
* Convolutional Neural Network (CNN)
* TensorFlow / Keras
* OpenCV
* Flask
* HTML, CSS, JavaScript

---

## 📂 Project Structure

```
Emotion-Based-Music-Recommendation-Using-CNN/
│
├── dataset/
│   ├── fer2013.csv
│   └── music_dataset.csv
│
├── models/
│   └── emotion_cnn_model.h5
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

---

## 🧠 Model Details

* Model Type: Convolutional Neural Network (CNN)
* Input Size: 48x48 grayscale images
* Output Classes: 6 emotion categories
* Optimizer: Adam
* Loss Function: Categorical Crossentropy

---

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/Emotion-Based-Music-Recommendation-Using-CNN.git
```

2. Navigate to the project directory:

```
cd Emotion-Based-Music-Recommendation-Using-CNN
```

3. Install required dependencies:

```
pip install -r requirements.txt
```

4. Run the application:

```
python app.py
```

---

## 🎯 Working Process

1. Capture image using webcam
2. Detect face using OpenCV
3. Predict emotion using trained CNN model
4. Recommend songs based on detected emotion

---

## 🔮 Future Enhancements

* Integration with Spotify API
* Mobile application support
* Improved deep learning architecture
* Real-time playlist streaming

---

## 👩‍💻 Author

Yamini
B.Sc Computer Science

