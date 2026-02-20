import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model
from tkinter import Tk, Button, Label
from PIL import Image, ImageTk

# Initialize pygame music
pygame.mixer.init()

# Load trained model
model = load_model("emotion_model.h5")

# Emotion labels (same as training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Play song function 🎵
def play_song(emotion):
    pygame.mixer.music.stop()
    song_path = f"songs/{emotion.lower()}.mp3"
    pygame.mixer.music.load(song_path)
    pygame.mixer.music.play()

# Capture with live preview 📷
def capture_emotion():
    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            status_label.config(text="❌ Camera not accessible", fg="red")
            return

        status_label.config(
            text="📷 Press SPACE to capture | ESC to cancel",
            fg="blue"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Live Camera - Press SPACE to Capture | ESC to cancel", frame)
            key = cv2.waitKey(1)

            # ESC to cancel
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                status_label.config(text="❌ Capture cancelled", fg="red")
                return

            # SPACE to capture
            if key == 32:
                cap.release()
                cv2.destroyAllWindows()
                break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            status_label.config(text="❌ No face detected", fg="red")
            #return

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]

            status_label.config(
                text=f"😊 Detected Emotion: {emotion}",
                fg="green"
            )

            play_song(emotion)
            break

        # Show captured image in GUI
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((300, 230))
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.imgtk = imgtk
        image_label.config(image=imgtk)
    
    except Exception as e:
        print("\n\nException ", e)

# ================= GUI =================

root = Tk()
root.title("Emotion Based Music Recommendation System")
root.geometry("430x550")
root.configure(bg="white")

title = Label(
    root,
    text="Emotion Based Music Player",
    font=("Arial", 16, "bold"),
    bg="white"
)
title.pack(pady=10)

image_label = Label(root, bg="white")
image_label.pack()

status_label = Label(
    root,
    text="Click the button to start",
    font=("Arial", 12),
    bg="white"
)
status_label.pack(pady=10)

capture_btn = Button(
    root,
    text="Capture Emotion",
    font=("Arial", 12, "bold"),
    bg="#4CAF50",
    fg="white",
    width=22,
    command=capture_emotion
)
capture_btn.pack(pady=15)

root.mainloop()
