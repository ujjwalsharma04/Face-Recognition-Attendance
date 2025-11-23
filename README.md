# Face-Recognition-Attendance
Automated Face Recognition Attendance System using Python &amp; OpenCV with real-time detection and CSV attendance logging.
This project captures face images, trains a recognition model, and automatically marks attendance with timestamp.  
Built for college/student attendance automation.

---

## 🚀 Features

✔️ **Real-time face recognition** using OpenCV  
✔️ **Dataset auto-folder structure** (each person gets a folder)  
✔️ **Automatic attendance saving** (`attendance.csv`)  
✔️ **Admin-only access** (password protected UI)  
✔️ **Modern Streamlit Web App UI**  
✔️ **Camera auto detection**  
✔️ **LBPH model training**  
✔️ Easily expandable for multiple students  

---

## 📂 Project Structure

FaceRecognitionProject/
│── capture_dataset.py
│── trainer.py
│── recognizer.py
│── app.py
│── haarcascade_frontalface_default.xml
│── trainer.yml
│── attendance.csv
│── dataset/
│ ├── person1/
│ ├── person2/
│ ├── ...
│── .gitignore
│── README.md


## 🛠 Requirements

Install dependencies:

pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install streamlit

📸 1. Capture Dataset
Run the script to capture images for a person:

python capture_dataset.py

It will:
Ask for person's name
Create a folder inside dataset/
Capture face images
Save automatically

🧠 2. Train the Model

python trainer.py

It will:
Load all folders in dataset/
Train LBPH model
Save model as trainer.yml

🧾 3. Run Attendance Recognizer

python recognizer.py

It will:
Detect faces live
Recognize person name
Store attendance → attendance.csv

🌐 4. Streamlit Web App
Run:
streamlit run app.py

App includes:
Login page (Admin only)
Buttons to run:
Capture Dataset
Train Model
Mark Attendance
View attendance inside the app


👨‍💻 Technologies Used
Python
OpenCV
NumPy
Streamlit
LBPH Face Recognizer
