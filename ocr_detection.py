import sys
import cv2
import torch
import easyocr
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from ultralytics import YOLO

class LicensePlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Plate Detection & Recognition")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setFixedSize(800, 450)
        
        self.btn_load = QPushButton("Load Video", self)
        self.btn_load.clicked.connect(self.load_video)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.btn_load)
        self.setLayout(self.layout)
        
        self.model = YOLO("best.pt")  # Load YOLOv8 model
        self.reader = easyocr.Reader(["en"])  # EasyOCR reader
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.cap = None
    
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(30)  # Process video at 30ms intervals
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            return
        
        results = self.model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                plate_crop = frame[y1:y2, x1:x2]
                text = self.recognize_plate(plate_crop)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        self.display_frame(frame)
    
    def recognize_plate(self, image):
        if image is None or image.size == 0:
            return ""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)
        return results[0][1] if results else ""
    
    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LicensePlateApp()
    window.show()
    sys.exit(app.exec())
