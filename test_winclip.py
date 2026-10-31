
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from PIL import Image
from anomalib.models.image.winclip.torch_model import WinClipModel

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "transistor" / "test"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CATEGORIES = [
    "good",
    "bent_lead",
    "cut_lead",
    "damaged_case",
    "misplaced",
]


# =====================================================
# IMAGE DATASET
# =====================================================

class TransistorDataset:

    def __init__(self):
        self.images = []

        for category in CATEGORIES:
            folder = DATASET_DIR / category

            if not folder.exists():
                print(f"Warning: Missing folder: {folder}")
                continue

            for path in sorted(folder.glob("*.png")):
                self.images.append({
                    "path": path,
                    "category": category,
                })

        if not self.images:
            raise RuntimeError(
                f"No images found in {DATASET_DIR}"
            )

    def __len__(self):
        return len(self.images)

    def get(self, index):
        return self.images[index]


# =====================================================
# WINCLIP INFERENCE ENGINE
# =====================================================

class WinCLIPEngine:

    def __init__(self):
        print(f"Loading WinCLIP on {DEVICE}...")

        self.model = WinClipModel(class_name="transistor")
        self.model = self.model.to(DEVICE)
        self.model.eval()

        print("WinCLIP ready!")

    def predict(self, image_path):

        image = Image.open(image_path).convert("RGB")

        image_array = np.array(image)

        image_tensor = torch.from_numpy(
            image_array
        ).permute(2, 0, 1)

        input_tensor = self.model.transform(
            image_tensor
        ).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            prediction = self.model(input_tensor)

        score = prediction.pred_score.squeeze().item()

        anomaly_map = (
            prediction.anomaly_map
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        return image_array, score, anomaly_map


# =====================================================
# IMAGE PROCESSING
# =====================================================

def create_visuals(image, anomaly_map):

    height, width = image.shape[:2]

    # Resize anomaly map to original image dimensions
    anomaly_map = cv2.resize(
        anomaly_map,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    # Generate heatmap
    heatmap = cv2.applyColorMap(
        cv2.normalize(
            anomaly_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    # OpenCV uses BGR; convert to RGB
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(
        image,
        0.55,
        heatmap,
        0.45,
        0,
    )

    # Find candidate anomalous regions
    threshold = np.percentile(anomaly_map, 97)

    mask = (
        anomaly_map >= threshold
    ).astype(np.uint8) * 255

    # Merge nearby anomalous pixels
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Draw bounding boxes
    min_area = max(20, int(width * height * 0.0005))

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w * h < min_area:
            continue

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2,
        )

    return heatmap, overlay


# =====================================================
# QT IMAGE CONVERSION
# =====================================================

def numpy_to_pixmap(array):

    array = np.ascontiguousarray(array)

    height, width, channels = array.shape

    image = QImage(
        array.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    ).copy()

    return QPixmap.fromImage(image)


# =====================================================
# MAIN APPLICATION
# =====================================================

class TransistorInspector(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Transistor Inspector")
        self.resize(1450, 850)

        self.dataset = TransistorDataset()
        self.engine = None

        self.current_index = 0

        # Cache: index -> prediction results
        self.prediction_cache = {}

        self.setup_ui()

        # Load model
        self.status_label.setText(
            "Loading WinCLIP model..."
        )

        QApplication.processEvents()

        try:
            self.engine = WinCLIPEngine()
            self.show_image()

        except Exception as error:
            QMessageBox.critical(
                self,
                "Model Error",
                str(error),
            )

            self.status_label.setText(
                f"Model loading failed: {error}"
            )

    # -------------------------------------------------
    # USER INTERFACE
    # -------------------------------------------------

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        title = QLabel("TRANSISTOR INSPECTOR")
        title.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel(
            "WinCLIP-powered visual anomaly detection"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # Three image panels
        image_layout = QHBoxLayout()

        self.original_label = self.create_image_panel(
            "ORIGINAL IMAGE"
        )

        self.heatmap_label = self.create_image_panel(
            "ANOMALY HEATMAP"
        )

        self.overlay_label = self.create_image_panel(
            "DETECTION OVERLAY"
        )

        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.heatmap_label)
        image_layout.addWidget(self.overlay_label)

        main_layout.addLayout(image_layout)

        # Information panel
        self.info_label = QLabel("Waiting for image...")
        self.info_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.info_label.setFont(
            QFont("Arial", 13)
        )

        main_layout.addWidget(self.info_label)

        # Navigation
        navigation = QHBoxLayout()

        self.previous_button = QPushButton("← Previous")
        self.next_button = QPushButton("Next →")

        self.previous_button.clicked.connect(
            self.previous_image
        )

        self.next_button.clicked.connect(
            self.next_image
        )

        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)

        main_layout.addLayout(navigation)

        self.status_label = QLabel(
            "Use ← and → arrow keys to navigate"
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        main_layout.addWidget(self.status_label)

        central.setStyleSheet("""
            QWidget {
                background-color: #17191e;
                color: #eeeeee;
            }

            QPushButton {
                background-color: #303640;
                padding: 12px;
                border-radius: 6px;
                font-size: 14px;
            }

            QPushButton:hover {
                background-color: #465366;
            }

            QLabel {
                color: #eeeeee;
            }
        """)

    def create_image_panel(self, heading):

        container = QWidget()
        layout = QVBoxLayout(container)

        title = QLabel(heading)
        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        title.setFont(
            QFont("Arial", 11, QFont.Weight.Bold)
        )

        image_label = QLabel("No image")
        image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        image_label.setMinimumSize(300, 300)
        image_label.setStyleSheet("""
            background-color: #242830;
            border: 1px solid #444a55;
            border-radius: 8px;
        """)

        layout.addWidget(title)
        layout.addWidget(image_label)

        container.image_label = image_label

        return container

    # -------------------------------------------------
    # DISPLAY IMAGE
    # -------------------------------------------------

    def set_panel_image(self, panel, array):

        pixmap = numpy_to_pixmap(array)

        scaled = pixmap.scaled(
            panel.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        panel.image_label.setPixmap(scaled)

    def show_image(self):

        if self.engine is None:
            return

        item = self.dataset.get(self.current_index)

        self.status_label.setText(
            "Analysing image..."
        )

        QApplication.processEvents()

        try:

            # Reuse cached inference if available
            if self.current_index not in self.prediction_cache:

                image, score, anomaly_map = (
                    self.engine.predict(item["path"])
                )

                heatmap, overlay = create_visuals(
                    image,
                    anomaly_map,
                )

                self.prediction_cache[self.current_index] = (
                    image,
                    score,
                    heatmap,
                    overlay,
                )

            else:

                (
                    image,
                    score,
                    heatmap,
                    overlay,
                ) = self.prediction_cache[self.current_index]

            # Update image panels
            self.set_panel_image(
                self.original_label,
                image,
            )

            self.set_panel_image(
                self.heatmap_label,
                heatmap,
            )

            self.set_panel_image(
                self.overlay_label,
                overlay,
            )

            category = item["category"]

            self.info_label.setText(
                f"File: {item['path'].name}    |    "
                f"Actual category: {category.upper()}    |    "
                f"Anomaly score: {score:.4f}"
            )

            self.status_label.setText(
                f"Image {self.current_index + 1} "
                f"of {len(self.dataset)}    |    "
                f"Predictions cached: {len(self.prediction_cache)}"
            )

        except Exception as error:

            self.status_label.setText(
                f"Inference error: {error}"
            )

            print(f"Inference error: {error}")

    # -------------------------------------------------
    # NAVIGATION
    # -------------------------------------------------

    def next_image(self):

        self.current_index = (
            self.current_index + 1
        ) % len(self.dataset)

        self.show_image()

    def previous_image(self):

        self.current_index = (
            self.current_index - 1
        ) % len(self.dataset)

        self.show_image()

    def keyPressEvent(self, event):

        if event.key() == Qt.Key.Key_Right:
            self.next_image()

        elif event.key() == Qt.Key.Key_Left:
            self.previous_image()

        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):

        super().resizeEvent(event)

        # Refresh displayed image sizes
        if self.current_index in self.prediction_cache:
            self.show_image()


# =====================================================
# APPLICATION ENTRY POINT
# =====================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    try:
        window = TransistorInspector()
        window.show()

        sys.exit(app.exec())

    except Exception as error:

        QMessageBox.critical(
            None,
            "Application Error",
            str(error),
        )

        raise
