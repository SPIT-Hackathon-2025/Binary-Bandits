import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from tensorflow.keras.models import load_model

class PotholeDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = load_model("pothole_segmentation_model2.h5")  # Update with actual model path

    def initUI(self):
        self.setWindowTitle("Pothole Detection App")
        self.setGeometry(100, 100, 600, 400)
        
        self.label = QLabel("Upload an Image for Pothole Detection", self)
        self.label.setStyleSheet("QLabel { font-size: 16px; }")
        
        self.uploadButton = QPushButton("Upload Image", self)
        self.uploadButton.clicked.connect(self.loadImage)
        
        self.detectButton = QPushButton("Detect Potholes", self)
        self.detectButton.clicked.connect(self.processImage)
        self.detectButton.setEnabled(False)
        
        self.resultLabel = QLabel("", self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.uploadButton)
        layout.addWidget(self.detectButton)
        layout.addWidget(self.resultLabel)
        self.setLayout(layout)
        
        self.imagePath = None

    def loadImage(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if filePath:
            self.imagePath = filePath
            pixmap = QPixmap(filePath)
            self.label.setPixmap(pixmap.scaled(400, 300))
            self.detectButton.setEnabled(True)
    
    def processImage(self):
        if not self.imagePath:
            QMessageBox.warning(self, "Error", "No image selected!")
            return
        
        image = cv2.imread(self.imagePath)
        image_resized = cv2.resize(image, (400, 400))
        image_norm = image_resized / 255.0
        image_input = np.expand_dims(image_norm, axis=0)
        
        pred_mask = self.model.predict(image_input)
        pred_mask = (pred_mask > 0.2).astype(np.uint8)
        pothole_percentage = (np.sum(pred_mask.squeeze()) / pred_mask.size) * 100
        
        mask_image = (pred_mask.squeeze() * 255).astype(np.uint8)
        cv2.imwrite("output_mask.png", mask_image)
        
        self.resultLabel.setText(f"Pothole Coverage: {pothole_percentage:.2f}%")
        self.showResult()
    
    def showResult(self):
        pixmap = QPixmap("output_mask.png")
        self.label.setPixmap(pixmap.scaled(400, 300))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PotholeDetectionApp()
    window.show()
    sys.exit(app.exec_())
