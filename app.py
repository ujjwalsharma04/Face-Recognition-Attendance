import streamlit as st
import cv2
import os
import numpy as np
import datetime
import csv


ADMIN_PASSWORD = "admin123"
DATASET_PATH = "dataset"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAINER_FILE = "trainer.yml"
ATTENDANCE_FILE = "attendance.csv"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time"])

    time_now = datetime.datetime.now().strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, time_now])


def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    persons = os.listdir(DATASET_PATH)

    for label_id, person_name in enumerate(persons):
        folder = os.path.join(DATASET_PATH, person_name)

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(label_id)

    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_FILE)


def start_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    cam = cv2.VideoCapture(0)
    persons = os.listdir(DATASET_PATH)

    stframe = st.empty()
    st.info("Press 'Q' to stop recognition", icon="⚠️")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)

            if confidence < 70:
                name = persons[label]
                color = (0, 255, 0)
                mark_attendance(name)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()


def capture_dataset(name):
    folder = os.path.join(DATASET_PATH, name)

    if not os.path.exists(folder):
        os.makedirs(folder)

    cam = cv2.VideoCapture(0)
    count = 0

    stframe = st.empty()
    st.write("📸 Capturing 50 images…")

    while count < 50:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(folder, f"{count}.jpg"), face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        stframe.image(frame, channels="BGR")

    cam.release()
    st.success(f"🎉 Dataset created for {name}! ({count} images)")


# --------------------------------
# STREAMLIT ADVANCED UI
# --------------------------------
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🧑‍🏫",
    layout="wide",
)

# Custom Dark Theme Styling
st.markdown("""
    <style>
        .main { background-color: #0c0f0f; }
        .stButton>button {
            background: linear-gradient(90deg, #0072ff, #00c6ff);
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 18px;
        }
        .stTextInput>div>input {
            background-color: #1a1c1d;
            color: white;
        }
        .css-1d391kg { color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🧑‍🏫 FACE RECOGNITION ATTENDANCE SYSTEM")
st.subheader("Premium Modern UI • Auto Recognition • Admin Panel")


# --------------------------------
# LOGIN
# --------------------------------
st.write("## 🔐 Admin Login")
password = st.text_input("Enter Password", type="password")

if password == ADMIN_PASSWORD:

    st.sidebar.title("📌 MENU")
    menu = st.sidebar.radio(
        "",
        ["➕ Add Student", "🧠 Train Model", "🎥 Start Attendance", "📄 View Attendance"],
    )

    # -----------------------
    # ADD STUDENT
    # -----------------------
    if menu == "➕ Add Student":
        st.header("➕ Add New Student")
        name = st.text_input("Student Name")

        if st.button("Start Dataset Capture"):
            if name.strip():
                capture_dataset(name)
            else:
                st.error("⚠ Enter a name first!")

    # -----------------------
    # TRAIN MODEL
    # -----------------------
    elif menu == "🧠 Train Model":
        st.header("🧠 Train Recognition Model")
        if st.button("Train Now"):
            train_model()
            st.success("Model Trained Successfully 🎯")

    # -----------------------
    # START RECOGNITION
    # -----------------------
    elif menu == "🎥 Start Attendance":
        st.header("🎥 Real-Time Attendance")
        start_recognition()

    # -----------------------
    # VIEW ATTENDANCE
    # -----------------------
    elif menu == "📄 View Attendance":
        st.header("📄 Attendance Records")
        if os.path.exists(ATTENDANCE_FILE):
            st.download_button("⬇ Download Attendance CSV",
                               data=open(ATTENDANCE_FILE, "rb"),
                               file_name="attendance.csv")
        else:
            st.info("No attendance yet.")

else:
    st.warning("Enter correct admin password to unlock the system.")
