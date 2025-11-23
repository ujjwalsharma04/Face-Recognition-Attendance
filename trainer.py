import cv2
import os
import numpy as np

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue
    
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        faces.append(img)
        ids.append(int(1))  # 1 = person ID (simple project ke liye)

# Convert to numpy arrays
faces = np.array(faces)
ids = np.array(ids)

print("Training started…")
recognizer.train(faces, ids)
recognizer.save("trainer.yml")
print("Training completed! Model saved as trainer.yml")