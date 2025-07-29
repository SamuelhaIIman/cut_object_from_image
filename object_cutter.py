import sys
import numpy as np
import cv2
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


class PaintLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.last_point = QPoint()
        self.mask = None
        self.brush_size = 10
        self.pen_color = Qt.red
        self.image = None
        self.refined_mask = None

    def set_image(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.refined_mask = None
        h, w, ch = self.image.shape
        bytes_per_line = ch * w
        self.qimage = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(self.qimage))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.pixmap())
            pen = QPen(self.pen_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            painter.end()
            self.update()

            x1, y1 = self.last_point.x(), self.last_point.y()
            x2, y2 = event.pos().x(), event.pos().y()
            cv2.line(self.mask, (x1, y1), (x2, y2), color=255, thickness=self.brush_size)

            self.last_point = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def get_mask(self):
        return self.refined_mask if self.refined_mask is not None else self.mask

    def get_original_image(self):
        return self.image

    def refine_with_sam(self, sam_predictor):
        input_image = self.image.copy()
        input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

        # Load image into SAM
        sam_predictor.set_image(input_image_rgb)

        # Find all points where mask is drawn
        mask_coords = np.column_stack(np.where(self.mask > 0))

        if len(mask_coords) == 0:
            print("No painted region found.")
            return

        center_y, center_x = np.mean(mask_coords, axis=0)
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])  # foreground

        masks, scores, logits = sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )

        self.refined_mask = (masks[0].astype(np.uint8) * 255).astype(np.uint8)

        # Display the refined mask as an overlay
        overlay = input_image.copy()
        overlay[self.refined_mask == 0] = [200, 200, 200]  # faded background

        h, w, ch = overlay.shape
        qimage = QImage(overlay.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimage))
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Object Cutter - Segment Anything Enhanced")

        self.image_label = PaintLabel()
        self.load_button = QPushButton("Load Image")
        self.refine_button = QPushButton("Refine with AI")
        self.save_button = QPushButton("Cut and Save Object")

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.refine_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_button.clicked.connect(self.load_image)
        self.refine_button.clicked.connect(self.refine_object)
        self.save_button.clicked.connect(self.save_cutout)

        # Load SAM model
        checkpoint_path = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = SamPredictor(sam)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_label.set_image(file_path)

    def refine_object(self):
        self.image_label.refine_with_sam(self.predictor)

    def save_cutout(self):
        orig_img = self.image_label.get_original_image()
        mask = self.image_label.get_mask()

        alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
        rgba = np.dstack((orig_img, alpha))

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Cut Object", "", "PNG Files (*.png)")
        if save_path:
            Image.fromarray(rgba).save(save_path)
            print(f"Saved to {save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec_())
