import cv2
import numpy as np
import csv
import datetime
import os

print("Starting recognizer...")

# ---------------- CAMERA AUTO DETECTION ----------------
def get_camera():
    for i in range(5):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f"Camera found at index: {i}")
            return cam
    print("❌ ERROR: No camera detected!")
    exit()

cam = get_camera()

# ---------------- LOAD HAAR CASCADE ----------------
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# ---------------- LOAD TRAINER MODEL ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# ---------------- LOAD DATASET NAMES ----------------
dataset_path = "dataset"
persons = os.listdir(dataset_path)

# ---------------- ATTENDANCE CSV ----------------
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])


print("Recognizer Started... Press 'q' to exit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ ERROR: Cannot read from camera!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)

        if confidence < 70:  # better threshold
            name = persons[label]
            text = f"{name} ({round(confidence,2)})"

            # Mark attendance
            time_now = datetime.datetime.now().strftime("%H:%M:%S")
            with open(attendance_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, time_now])
        else:
            name = "Unknown"
            text = "Unknown"

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()