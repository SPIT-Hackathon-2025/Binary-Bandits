import sys
import cv2
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

class DrowsinessDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.cap = cv2.VideoCapture(0)
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.flag = 0
        self.thresh = 0.25
        self.frame_check = 20
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

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

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 0)
        for subject in subjects:
            shape = self.predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            if ear < self.thresh:
                self.flag += 1
                if self.flag >= self.frame_check:
                    cv2.putText(frame, "ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
