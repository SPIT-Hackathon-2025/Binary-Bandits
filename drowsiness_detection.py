import sys
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import time
import winsound
from scipy.spatial import distance
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

class DrowsinessDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.cap = cv2.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.eye_landmarks = {
            'left': [33, 160, 158, 133, 153, 144],
            'right': [362, 385, 387, 263, 373, 380]
        }
        self.thresh = 0.25
        self.frame_check = 20
        self.flag = 0
        self.alert_active = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def init_ui(self):
        self.label = QLabel(self)
        self.start_button = QPushButton("Start Detection")
        self.stop_button = QPushButton("Stop Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)
        self.setWindowTitle("Drowsiness Detection")
        self.setGeometry(100, 100, 500, 400)

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def start_detection(self):
        self.timer.start(30)
    
    def stop_detection(self):
        self.timer.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    def alert(self):
        if not self.alert_active:
            self.alert_active = True
            while self.flag >= self.frame_check:
                self.engine.say("ALERT")
                self.engine.runAndWait()
                winsound.Beep(1000, 500)  # Beep sound after ALERT audio
                time.sleep(0.5)
            self.alert_active = False

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                leftEye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                     face_landmarks.landmark[i].y * frame.shape[0]]
                                    for i in self.eye_landmarks['left']])
                rightEye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                      face_landmarks.landmark[i].y * frame.shape[0]]
                                     for i in self.eye_landmarks['right']])
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < self.thresh:
                    self.flag += 1
                    if self.flag >= self.frame_check:
                        cv2.putText(frame, "ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if not self.alert_active:
                            threading.Thread(target=self.alert, daemon=True).start()
                else:
                    self.flag = 0
        self.display_frame(frame)
    
    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetectionApp()
    window.show()
    sys.exit(app.exec())