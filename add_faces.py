import cv2
import pickle
import numpy as np
import os


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
name = input("Enter Your Name: ")
capture_limit = 100
frame_skip = 5  
frame_count = 0


os.makedirs('data', exist_ok=True)

while len(faces_data) < capture_limit:
    ret, frame = video.read()
    if not ret:
        print("Error: Camera not working!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  

    faces = facedetect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        if w > 60 and h > 60: 
            padding = 10  
            x, y, w, h = max(0, x-padding), max(0, y-padding), w+2*padding, h+2*padding
            crop_img = frame[y:y+h, x:x+w]

            resized_img = cv2.resize(crop_img, (50, 50), interpolation=cv2.INTER_LINEAR)

            if frame_count % frame_skip == 0:  
                faces_data.append(resized_img)

            frame_count += 1
            cv2.putText(frame, f"Captured: {len(faces_data)}/{capture_limit}", (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    cv2.imshow("Face Capture", frame)
    if cv2.waitKey(1) == ord('q') or len(faces_data) >= capture_limit:
        break


video.release()
cv2.destroyAllWindows()


faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)


names_file = 'data/names.pkl'
faces_file = 'data/faces_data.pkl'

try:
    if os.path.exists(names_file):
        with open(names_file, 'rb') as f:
            names = pickle.load(f)
    else:
        names = []
    names.extend([name] * len(faces_data))
    
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
except Exception as e:
    print(f"Error handling names.pkl: {e}")

try:
    if os.path.exists(faces_file):
        with open(faces_file, 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
    else:
        faces = faces_data

    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)
except Exception as e:
    print(f"Error handling faces_data.pkl: {e}")

print("Face data successfully saved!")


if len(names) != len(faces):
    print(f"Warning: Number of faces ({len(faces)}) and labels ({len(names)}) do not match. Please re-run.")
