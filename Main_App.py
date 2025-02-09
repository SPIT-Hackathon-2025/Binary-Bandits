import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel

class MultiFunctionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Multi-Function Detection App")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()
        
        label = QLabel("Select a function to run:")
        layout.addWidget(label)
        
        self.vehicle_btn = QPushButton("Vehicle Detection", self)
        self.vehicle_btn.clicked.connect(lambda: self.run_script("vehicle_detection.py"))
        layout.addWidget(self.vehicle_btn)

        self.pothole_btn = QPushButton("Pothole Detection", self)
        self.pothole_btn.clicked.connect(lambda: self.run_script("potholes.py"))
        layout.addWidget(self.pothole_btn)

        self.activity_btn = QPushButton("Activity Recognition", self)
        self.activity_btn.clicked.connect(lambda: self.run_script("Acc.py"))
        layout.addWidget(self.activity_btn)

        self.ocr_btn = QPushButton("License Plate Detection", self)
        self.ocr_btn.clicked.connect(lambda: self.run_script("ocr_detection.py"))
        layout.addWidget(self.ocr_btn)
        
        self.setLayout(layout)

    def run_script(self, script_name):
        try:
            subprocess.Popen([sys.executable, script_name])
        except Exception as e:
            print(f"Error launching {script_name}: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiFunctionApp()
    window.show()
    sys.exit(app.exec_())
