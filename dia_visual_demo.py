
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import EfficientAd

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
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


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "dataset" / "transistor" / "test"
MASK_DIR = BASE_DIR / "dataset" / "transistor" / "ground_truth"

RESULTS_DIR = BASE_DIR / "results" / "efficientad_s"
IMAGENET_DIR = BASE_DIR / "datasets" / "imagenette"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CATEGORIES = [
    "good",
    "bent_lead",
    "cut_lead",
    "damaged_case",
    "misplaced",
]


# =========================================================
# IMAGE UTILITIES
# =========================================================

def to_numpy(value):

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    return np.asarray(value)


def make_heatmap(anomaly_map, width, height):

    anomaly_map = np.squeeze(to_numpy(anomaly_map))

    anomaly_map = cv2.resize(
        anomaly_map.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    normalized = cv2.normalize(
        anomaly_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )

    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


# =========================================================
# STRICTER ANOMALY BOX DETECTION
# =========================================================

def find_boxes(anomaly_map, width, height):

    anomaly_map = np.squeeze(to_numpy(anomaly_map))

    anomaly_map = cv2.resize(
        anomaly_map.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    if not np.isfinite(anomaly_map).all():
        return []

    if np.ptp(anomaly_map) < 1e-8:
        return []

    # Normalise anomaly map between 0 and 1
    normalized = cv2.normalize(
        anomaly_map,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    # Only retain the strongest anomaly pixels.
    # Both conditions must be satisfied.
    percentile_threshold = np.percentile(
        normalized,
        99,
    )

    threshold = max(
        percentile_threshold,
        0.78,
    )

    binary = (
        normalized >= threshold
    ).astype(np.uint8)

    # Remove small isolated noise
    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Connect nearby suspicious pixels
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    minimum_area = max(
        30,
        int(width * height * 0.0008),
    )

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        if w * h < minimum_area:
            continue

        # Check the strongest anomaly inside this region
        region = normalized[y:y+h, x:x+w]

        if region.size == 0:
            continue

        peak_score = float(np.max(region))

        if peak_score < 0.85:
            continue

        # Add a little padding around the suspicious region
        padding = 5

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        boxes.append((
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        ))

    return boxes


def make_overlay(image, boxes):

    overlay = image.copy()

    for x, y, w, h in boxes:

        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (255, 30, 30),
            3,
        )

    return overlay


def make_pixmap(array):

    array = np.ascontiguousarray(array)

    height, width, channels = array.shape

    qimage = QImage(
        array.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    ).copy()

    return QPixmap.fromImage(qimage)


# =========================================================
# DATASET
# =========================================================

def collect_images():

    images = []

    for category in CATEGORIES:

        folder = TEST_DIR / category

        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.png")):

            images.append({
                "path": path,
                "category": category,
            })

    return images


# =========================================================
# MODEL INFERENCE
# =========================================================

def get_checkpoint():

    checkpoints = list(RESULTS_DIR.rglob("*.ckpt"))

    if not checkpoints:

        raise FileNotFoundError(
            "No EfficientAD checkpoint found in:\n"
            f"{RESULTS_DIR}\n\n"
            "Check that your training run saved a .ckpt file."
        )

    return max(
        checkpoints,
        key=lambda path: path.stat().st_mtime,
    )


def run_inference():

    checkpoint = get_checkpoint()

    print("Using checkpoint:", checkpoint)
    print("Device:", DEVICE)

    model = EfficientAd(
        model_size="small",
        imagenet_dir=IMAGENET_DIR,
    )

    engine = Engine(
        accelerator=DEVICE,
        devices=1,
        logger=False,
    )

    dataset = PredictDataset(
        path=TEST_DIR,
        image_size=(256, 256),
    )

    print("\nRunning EfficientAD inference...")
    print("Please wait...")

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(checkpoint),
        return_predictions=True,
    )

    if not predictions:

        raise RuntimeError(
            "EfficientAD did not return any predictions."
        )

    results = {}

    for prediction in predictions:

        image_path = prediction.image_path

        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0]

        image_path = Path(str(image_path)).resolve()

        results[image_path] = {
            "score": float(
                np.squeeze(to_numpy(prediction.pred_score))
            ),

            "anomaly_map": np.squeeze(
                to_numpy(prediction.anomaly_map)
            ),
        }

    print(f"Predictions received: {len(results)}")

    return results


# =========================================================
# STATIC INSPECTION CANVAS
# =========================================================

class InspectionCanvas(QWidget):

    def __init__(self):

        super().__init__()

        self.setMinimumSize(650, 480)

        self.pixmap = QPixmap()

    def set_image(self, array):

        self.pixmap = make_pixmap(array)
        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.fillRect(
            self.rect(),
            QColor("#10151e"),
        )

        painter.setPen(QColor("#718096"))
        painter.setFont(QFont("Arial", 10))

        painter.drawText(
            18,
            25,
            "DIA // VISUAL INSPECTION",
        )

        if not self.pixmap.isNull():

            scaled = self.pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2

            painter.drawPixmap(x, y, scaled)

        painter.end()


# =========================================================
# MAIN GUI
# =========================================================

class DIAWindow(QMainWindow):

    def __init__(self, images, predictions):

        super().__init__()

        self.images = images
        self.predictions = predictions

        self.current_index = 0

        self.setWindowTitle(
            "DIA - Device Inspector Aariz"
        )

        self.resize(1400, 850)

        self.setup_ui()
        self.show_current_image()

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)

        title = QLabel("D I A")

        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title.setFont(
            QFont("Arial", 25, QFont.Weight.Bold)
        )

        subtitle = QLabel(
            "DEVICE INSPECTOR AARIZ  |  EfficientAD-S"
        )

        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        root.addWidget(title)
        root.addWidget(subtitle)

        content = QHBoxLayout()

        # Static inspection image
        self.canvas = InspectionCanvas()

        content.addWidget(self.canvas, 3)

        # Right-side panels
        side = QVBoxLayout()

        self.heatmap_label = self.create_panel(
            "ANOMALY HEATMAP"
        )

        self.ground_truth_label = self.create_panel(
            "GROUND TRUTH"
        )

        side.addWidget(self.heatmap_label)
        side.addWidget(self.ground_truth_label)

        content.addLayout(side, 1)

        root.addLayout(content)

        self.info_label = QLabel(
            "Waiting for image..."
        )

        self.info_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.info_label.setFont(
            QFont("Arial", 13)
        )

        root.addWidget(self.info_label)

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

        root.addLayout(navigation)

        self.status_label = QLabel(
            "Inspection ready | Arrow keys to navigate"
        )

        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        root.addWidget(self.status_label)

        central.setStyleSheet("""
            QWidget {
                background-color: #171b23;
                color: #eeeeee;
            }

            QPushButton {
                background-color: #303947;
                padding: 12px;
                border-radius: 7px;
                font-size: 14px;
            }

            QPushButton:hover {
                background-color: #46566d;
            }

            QLabel {
                color: #eeeeee;
            }
        """)

    def create_panel(self, heading):

        container = QWidget()

        layout = QVBoxLayout(container)

        title = QLabel(heading)

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title.setFont(
            QFont("Arial", 10, QFont.Weight.Bold)
        )

        image = QLabel("No image")

        image.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        image.setMinimumSize(220, 190)

        image.setStyleSheet("""
            background-color: #252d39;
            border: 1px solid #414b5a;
            border-radius: 7px;
        """)

        layout.addWidget(title)
        layout.addWidget(image)

        container.image_label = image

        return container

    # -----------------------------------------------------
    # IMAGE DISPLAY
    # -----------------------------------------------------

    def update_panel(self, panel, array):

        pixmap = make_pixmap(array)

        scaled = pixmap.scaled(
            panel.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        panel.image_label.setPixmap(scaled)

    def show_current_image(self):

        item = self.images[self.current_index]

        path = item["path"].resolve()

        prediction = self.predictions.get(path)

        if prediction is None:

            self.status_label.setText(
                f"No prediction found for {path.name}"
            )

            return

        # Load original image
        image = np.array(
            Image.open(path).convert("RGB")
        )

        height, width = image.shape[:2]

        anomaly_map = prediction["anomaly_map"]

        # Generate heatmap
        heatmap = make_heatmap(
            anomaly_map,
            width,
            height,
        )

        # Detect suspicious regions when image is selected
        boxes = find_boxes(
            anomaly_map,
            width,
            height,
        )

        # Draw boxes on the original image
        overlay = make_overlay(
            image,
            boxes,
        )

        # Display updated inspection image
        self.canvas.set_image(overlay)

        # Update heatmap panel
        self.update_panel(
            self.heatmap_label,
            heatmap,
        )

        # Ground truth mask
        category = item["category"]

        if category == "good":

            ground_truth = np.zeros_like(image)

        else:

            mask_path = (
                MASK_DIR
                / category
                / f"{path.stem}_mask.png"
            )

            if mask_path.exists():

                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_GRAYSCALE,
                )

                mask = cv2.resize(
                    mask,
                    (width, height),
                )

                ground_truth = cv2.cvtColor(
                    mask,
                    cv2.COLOR_GRAY2RGB,
                )

            else:

                ground_truth = np.zeros_like(image)

        self.update_panel(
            self.ground_truth_label,
            ground_truth,
        )

        score = prediction["score"]

        self.info_label.setText(
            f"FILE: {path.name}     |     "
            f"DATASET CATEGORY: {category.upper()}     |     "
            f"ANOMALY SCORE: {score:.4f}     |     "
            f"BOXES: {len(boxes)}"
        )

        self.status_label.setText(
            f"Image {self.current_index + 1} "
            f"of {len(self.images)} | "
            f"{len(boxes)} suspicious regions detected"
        )

    # -----------------------------------------------------
    # NAVIGATION
    # -----------------------------------------------------

    def next_image(self):

        self.current_index = (
            self.current_index + 1
        ) % len(self.images)

        self.show_current_image()

    def previous_image(self):

        self.current_index = (
            self.current_index - 1
        ) % len(self.images)

        self.show_current_image()

    def keyPressEvent(self, event):

        if event.key() == Qt.Key.Key_Right:
            self.next_image()

        elif event.key() == Qt.Key.Key_Left:
            self.previous_image()

        else:
            super().keyPressEvent(event)


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    try:

        print("=" * 50)
        print("DIA - DEVICE INSPECTOR AARIZ")
        print("=" * 50)

        images = collect_images()

        if not images:

            raise RuntimeError(
                f"No test images found in {TEST_DIR}"
            )

        predictions = run_inference()

        window = DIAWindow(
            images,
            predictions,
        )

        window.show()

        sys.exit(app.exec())

    except Exception as error:

        QMessageBox.critical(
            None,
            "DIA Error",
            str(error),
        )

        raise
