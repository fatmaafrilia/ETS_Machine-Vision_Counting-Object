import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt


class WelcomeScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Counting Objects")
        self.setFixedSize(500, 300)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f5;
                font-family: Arial;
            }
            QLabel {
                font-size: 22px;
                color: #333;
                font-weight: bold;
            }
            QPushButton {
                background-color: #e53935;
                color: white;
                border-radius: 6px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        layout = QVBoxLayout()
        welcome_label = QLabel("ETS MACHINE VISION")
        welcome_label.setAlignment(Qt.AlignCenter)

        start_button = QPushButton("Mulai Aplikasi")
        start_button.clicked.connect(self.open_main_window)

        layout.addStretch()
        layout.addWidget(welcome_label)
        layout.addSpacing(20)
        layout.addWidget(start_button)
        layout.addStretch()

        self.setLayout(layout)

    def open_main_window(self):
        self.main_window = ObjectDetector()
        self.main_window.show()
        self.close()


class ObjectDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎛 Deteksi Jumlah Objek (Canny + Contour + Morphology)")
        self.setFixedSize(700, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f5;
                font-family: Arial;
            }
            QLabel {
                font-size: 14px;
                color: #333;
            }
            QPushButton {
                background-color: #e53935;
                color: white;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        # Kamera & Timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Default parameters
        self.canny_low = 100
        self.canny_high = 200
        self.min_area = 1000

        # Widget tampilan gambar dan info
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background-color: #ccc; border: 2px solid #999;")
        self.info_label = QLabel("Jumlah Objek: 0")

        # Tombol Kontrol
        self.camera_button = QPushButton("Open Camera")
        self.camera_button.clicked.connect(self.start_camera)
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.image_button = QPushButton("Upload Image")
        self.image_button.clicked.connect(self.load_image)
        self.save_button = QPushButton("Save Result")
        self.save_button.clicked.connect(self.save_result)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.info_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.camera_button)
        btn_layout.addWidget(self.stop_button)
        btn_layout.addWidget(self.image_button)
        btn_layout.addWidget(self.save_button)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None

    def load_image(self):
        self.stop_camera()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            img = cv2.imread(file_path)
            self.display_and_detect(img)

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.display_and_detect(frame)

    def display_and_detect(self, frame):
        # Resize frame agar sesuai ukuran label
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        resized = cv2.resize(frame, (label_w, label_h))

        # Proses deteksi objek
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

        # Gambar bounding box
        for i, cnt in enumerate(filtered):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                resized, f"Obj {i+1}", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        self.info_label.setText(f"Jumlah Objek: {len(filtered)}")

        # Konversi ke QPixmap untuk ditampilkan
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def save_result(self):
        count_text = self.info_label.text()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Simpan Hasil", "", "Text Files (*.txt)"
        )
        if file_path:
            with open(file_path, 'w') as file:
                file.write(count_text)
            print(f"Hasil disimpan ke: {file_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    welcome = WelcomeScreen()
    welcome.show()
    sys.exit(app.exec_())
