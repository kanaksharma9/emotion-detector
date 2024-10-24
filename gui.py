import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import os
import datetime
import librosa
import soundfile as sf
from scipy.io import wavfile
import csv

emotion_model = load_model('emotion_model.h5')
animal_model = load_model('animal_detection_model.keras')
voice_model = load_model('voice-detection.h5')
age_model = load_model('age_detection_model.keras')

emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
animal_classes = ['Dog', 'Cat', 'Zebra', 'Lion', 'Leopard', 'Cheetah', 'Tiger', 'Bear', 
                  'Brown Bear', 'Butterfly', 'Canary', 'Crocodile', 'Polar Bear', 
                  'Bull', 'Camel', 'Crab', 'Chicken', 'Centipede', 'Cattle', 
                  'Caterpillar', 'Duck']
carnivorous_animals = ['Lion', 'Leopard', 'Cheetah', 'Tiger', 'Bear', 
                       'Brown Bear', 'Crocodile', 'Polar Bear']

def detect_emotion_from_image(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (48, 48)) 
    img_resized = img_resized.reshape(1, 48, 48, 1) / 255.0
    emotion_pred = emotion_model.predict(img_resized)
    emotion = emotion_classes[np.argmax(emotion_pred)]
    messagebox.showinfo("Emotion Detection", f"Detected Emotion: {emotion}")

def detect_animal_from_image(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized.reshape(1, 128, 128, 3) / 255.0  # Normalize
    animal_pred = animal_model.predict(img_resized)
    animal = animal_classes[np.argmax(animal_pred)]
    
    if animal in carnivorous_animals:
        messagebox.showinfo("Animal Detection", f"Detected Carnivorous Animal: {animal}")
        cv2.rectangle(img, (10, 10), (img.shape[1] - 10, img.shape[0] - 10), (0, 0, 255), 3)
    else:
        messagebox.showinfo("Animal Detection", f"Detected Animal: {animal}")
    
    # Show image with detection
    show_image(img)

def detect_animal_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    carnivorous_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (128, 128)) 
        frame_resized = frame_resized.reshape(1, 128, 128, 3) / 255.0  
        animal_pred = animal_model.predict(frame_resized)
        animal = animal_classes[np.argmax(animal_pred)]
        
        if animal in carnivorous_animals:
            carnivorous_count += 1
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 255), 3)
        
        cv2.imshow('Animal Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Carnivorous Animals", f"Number of Carnivorous Animals Detected: {carnivorous_count}")

def detect_emotion_from_voice(voice_path):
    # Check if voice is female (based on file naming convention)
    if 'female' not in voice_path:
        messagebox.showerror("Error", "Please upload a female voice.")
        return
    
    y, sr = librosa.load(voice_path)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    mfcc = mfcc.reshape(1, -1)
    emotion_pred = voice_model.predict(mfcc)
    emotion = emotion_classes[np.argmax(emotion_pred)]
    messagebox.showinfo("Voice Emotion Detection", f"Detected Emotion: {emotion}")

def detect_age_and_emotion_from_image(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (48, 48)) 
    img_resized = img_resized.reshape(1, 48, 48, 3) / 255.0 
    age_pred = age_model.predict(img_resized) 
    age = np.argmax(age_pred) + 13
    if age < 13 or age > 60:
        messagebox.showerror("Age Detection", "Not allowed")
        cv2.rectangle(img, (10, 10), (img.shape[1] - 10, img.shape[0] - 10), (0, 0, 255), 3)
    else:
        detect_emotion_from_image(img_path)
    with open('age_emotion_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([age, datetime.datetime.now()])
    show_image(img)
  
def show_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img_pil)
    label_img.config(image=img_tk)
    label_img.image = img_tk

root = tk.Tk()
root.title("Detection App")

label_img = tk.Label(root)
label_img.pack()

btn_emotion_image = tk.Button(root, text="Detect Emotion", command=lambda: detect_emotion_from_image(filedialog.askopenfilename()))
btn_emotion_image.pack()

btn_animal_image = tk.Button(root, text="Detect Animal from Image", command=lambda: detect_animal_from_image(filedialog.askopenfilename()))
btn_animal_image.pack()

btn_animal_video = tk.Button(root, text="Detect Animal from Video", command=lambda: detect_animal_from_video(filedialog.askopenfilename()))
btn_animal_video.pack()

btn_voice_emotion = tk.Button(root, text="Detect Emotion from Voice", command=lambda: detect_emotion_from_voice(filedialog.askopenfilename()))
btn_voice_emotion.pack()

btn_age_emotion = tk.Button(root, text="Detect Age and Emotion from Image", command=lambda: detect_age_and_emotion_from_image(filedialog.askopenfilename()))
btn_age_emotion.pack()

root.mainloop()
label_img = tk.Label(root)
label_img.pack()
btn_emotion_image = tk.Button(root, text="Detect Emotion from Image", command=lambda: detect_emotion_from_image(filedialog.askopenfilename()))
btn_emotion_image.pack()
btn_animal_image = tk.Button(root, text="Detect Animal from Image", command=lambda: detect_animal_from_image(filedialog.askopenfilename()))
btn_animal_image.pack()
btn_animal_video = tk.Button(root, text="Detect Animal from Video", command=lambda: detect_animal_from_video(filedialog.askopenfilename()))
btn_animal_video.pack()
btn_voice_emotion = tk.Button(root, text="Detect Emotion from Voice", command=lambda: detect_emotion_from_voice(filedialog.askopenfilename()))
btn_voice_emotion.pack()
btn_age_emotion = tk.Button(root, text="Detect Age and Emotion from Image", command=lambda: detect_age_and_emotion_from_image(filedialog.askopenfilename()))
btn_age_emotion.pack()

root.mainloop()
