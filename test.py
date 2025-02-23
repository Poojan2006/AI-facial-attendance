import cv2 
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)


imgBackground = cv2.imread("background.png")


with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


os.makedirs("Attendance", exist_ok=True)

marked_attendance = set()  # Prevent duplicate attendance in the same session

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Camera not working!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise

    faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

 
    for (x, y, w, h) in faces:
        padding = 10  # Add slight padding
        x, y, w, h = max(0, x-padding), max(0, y-padding), w+2*padding, h+2*padding
        crop_img = frame[y:y+h, x:x+w]

        resized_img = cv2.resize(crop_img, (50, 50), interpolation=cv2.INTER_LINEAR).flatten().reshape(1, -1)
        output = knn.predict(resized_img)[0]

      
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output), (x+5, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)


    imgBackground[162:162 + 480, 55:55 + 640] = frame


    cv2.imshow("Attendance System", imgBackground)

    k = cv2.waitKey(1)

    if k == ord('o'):
        if len(faces) == 0:
            speak("No face detected. Please try again.")
        else:
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w]
                resized_img = cv2.resize(crop_img, (50, 50), interpolation=cv2.INTER_LINEAR).flatten().reshape(1, -1)
                output = knn.predict(resized_img)[0]

                if output in marked_attendance:
                    speak(f"{output}, your attendance is already recorded.")
                    continue  # Skip duplicate attendance

                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                file_path = os.path.join("Attendance", f"Attendance_{date}.csv")

              
                file_exists = os.path.exists(file_path)
                with open(file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(['NAME', 'TIME'])  # Write headers
                    writer.writerow([output, timestamp])  # Write new entry

                marked_attendance.add(output)  # Prevent duplicate attendance
                speak(f"Attendance recorded successfully for {output}.")

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
