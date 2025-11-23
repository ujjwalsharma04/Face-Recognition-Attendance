import cv2
import os

person_id = input("Enter person name or ID: ")

path = f"dataset/{person_id}"
os.makedirs(path, exist_ok=True)

cascade_path = r"C:\Users\Lenovo\Desktop\jupyter labs\FaceRecognition project\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cam = cv2.VideoCapture(0)
print("Capturing images... Press 'q' to stop.")

count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        filename = f"{path}/{count}.jpg"
        cv2.imwrite(filename, face)
        count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()