# gui.py
import sys
import numpy as np
import cv2
import base64
import datetime
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QFileDialog, QToolBar, QAction
)
from PyQt5.QtGui import QPixmap, QPen, QColor, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF

# Set your API URL here (change later if deploying to another machine)
API_URL = "http://localhost:5000"

def encode_image(image, fmt='.png'):
    success, encoded_image = cv2.imencode(fmt, image)
    if not success:
        return None
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

def decode_image_from_b64(b64_str):
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img

class ImageViewer(QGraphicsView):
    pointSelected = pyqtSignal(QPointF)
    adjustPositive = pyqtSignal(QPointF)
    adjustNegative = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0
        self.selection_mode = False
        self.adjust_mode = False

    def wheelEvent(self, event):
        factor = 1.25
        if event.angleDelta().y() > 0:
            zoomFactor = factor
            self._zoom += 1
        else:
            zoomFactor = 1 / factor
            self._zoom -= 1

        if self._zoom < -10:
            self._zoom = -10
            return
        if self._zoom > 20:
            self._zoom = 20
            return

        self.scale(zoomFactor, zoomFactor)

    def mousePressEvent(self, event):
        if self.adjust_mode:
            scenePos = self.mapToScene(event.pos())
            if event.button() == Qt.LeftButton:
                self.adjustPositive.emit(scenePos)
            elif event.button() == Qt.RightButton:
                self.adjustNegative.emit(scenePos)
            return
        elif self.selection_mode:
            if event.button() == Qt.LeftButton:
                scenePos = self.mapToScene(event.pos())
                self.pointSelected.emit(scenePos)
                return
        super().mousePressEvent(event)

    def setSelectionMode(self, mode):
        self.selection_mode = mode
        if mode:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def setAdjustMode(self, mode):
        self.adjust_mode = mode
        if mode:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI Client for Interactive Segmentation (API-based)")
        self.current_points = []            # SAM initial points
        self.adjust_positive_points = []    # RITM positive adjustment points
        self.adjust_negative_points = []    # RITM negative adjustment points
        self.current_mask = None            # Latest mask (binary np.array)
        self.current_mask_item = None       # QGraphicsPixmapItem for mask overlay
        self.current_mask_color = None      # Unique QColor for this object
        self.object_counter = 0

        self.mask_colors = [
            QColor(255, 0, 0, 150),
            QColor(0, 255, 0, 150),
            QColor(0, 0, 255, 150),
            QColor(255, 255, 0, 150),
            QColor(255, 0, 255, 150),
            QColor(0, 255, 255, 150),
        ]

        self.scene = QGraphicsScene(self)
        self.viewer = ImageViewer(self)
        self.viewer.setScene(self.scene)
        self.setCentralWidget(self.viewer)

        self.createToolbar()

        self.viewer.pointSelected.connect(self.onPointSelected)
        self.viewer.adjustPositive.connect(self.onAdjustPositive)
        self.viewer.adjustNegative.connect(self.onAdjustNegative)

        self.image = None  # Will hold the loaded image (RGB numpy array)

    def createToolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        openAction = QAction("Open Image", self)
        openAction.triggered.connect(self.openImage)
        toolbar.addAction(openAction)

        self.selectButton = QAction("Select Points", self)
        self.selectButton.setCheckable(True)
        self.selectButton.triggered.connect(self.toggleSelectionMode)
        toolbar.addAction(self.selectButton)

        saveAction = QAction("Save Points", self)
        saveAction.triggered.connect(self.savePoints)
        toolbar.addAction(saveAction)

        samAction = QAction("Run SAM", self)
        samAction.triggered.connect(self.runSAM)
        toolbar.addAction(samAction)

        self.adjustButton = QAction("Adjust", self)
        self.adjustButton.setCheckable(True)
        self.adjustButton.triggered.connect(self.toggleAdjustMode)
        toolbar.addAction(self.adjustButton)

        finishAction = QAction("Finish", self)
        finishAction.triggered.connect(self.finishAdjustment)
        toolbar.addAction(finishAction)

    def openImage(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.bmp *.gif)"
        )
        if fileName:
            img_bgr = cv2.imread(fileName)
            if img_bgr is None:
                print("Failed to load image!")
                return
            self.image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.scene.clear()
            self.current_points = []
            self.current_mask = None
            self.current_mask_item = None
            self.adjust_positive_points = []
            self.adjust_negative_points = []
            height, width, ch = self.image.shape
            bytes_per_line = 3 * width
            qimg = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.pixmapItem = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
            self.scene.addItem(self.pixmapItem)
            self.viewer.setSceneRect(QRectF(0, 0, width, height))
            self.viewer.resetTransform()
            self.viewer._zoom = 0

    def toggleSelectionMode(self, checked):
        self.viewer.setSelectionMode(checked)
        if checked:
            self.viewer.setAdjustMode(False)
            self.adjustButton.setChecked(False)

    def toggleAdjustMode(self, checked):
        self.viewer.setAdjustMode(checked)
        if checked:
            self.viewer.setSelectionMode(False)
            self.selectButton.setChecked(False)

    def onPointSelected(self, point):
        x, y = point.x(), point.y()
        self.current_points.append((x, y))
        r = 5
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        ellipse.setPen(pen)
        ellipse.setBrush(QColor(255, 0, 0, 150))
        self.scene.addItem(ellipse)

    def savePoints(self):
        if not self.current_points:
            print("No points to save for the current object.")
            return
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        object_id = self.object_counter + 1
        default_filename = f"object_{object_id}_points_{timestamp}.txt"
        try:
            with open(default_filename, "w") as f:
                for pt in self.current_points:
                    f.write(f"{pt[0]}, {pt[1]}\n")
            print(f"Points for object {object_id} saved to {default_filename}")
        except Exception as e:
            print("Error saving points:", e)

    def runSAM(self):
        if self.image is None:
            print("No image loaded!")
            return
        if not self.current_points:
            print("No points selected for segmentation!")
            return
        # Encode image as BGR PNG for sending
        image_b64 = encode_image(cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR), fmt='.png')
        payload = {
            "image": image_b64,
            "points": self.current_points
        }
        try:
            response = requests.post(API_URL + "/run_sam", json=payload)
            if response.status_code != 200:
                print("Error from SAM server:", response.json())
                return
            result = response.json()
        except Exception as e:
            print("Error contacting SAM server:", e)
            return
        mask_b64 = result.get("mask")
        if not mask_b64:
            print("No mask returned from server")
            return
        mask_img = decode_image_from_b64(mask_b64)
        if mask_img is None:
            print("Failed to decode mask image")
            return
        mask_gray = (mask_img > 128).astype(np.uint8)
        self.current_mask = mask_gray
        # Choose a unique color
        color = self.mask_colors[self.object_counter % len(self.mask_colors)]
        self.current_mask_color = color
        r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
        colored_mask = np.zeros((mask_gray.shape[0], mask_gray.shape[1], 4), dtype=np.uint8)
        colored_mask[mask_gray > 0] = [r, g, b, a]
        qimage_mask = QImage(colored_mask.data, mask_gray.shape[1], mask_gray.shape[0],
                              4 * mask_gray.shape[1], QImage.Format_ARGB32)
        mask_pixmap = QPixmap.fromImage(qimage_mask)
        self.current_mask_item = QGraphicsPixmapItem(mask_pixmap)
        self.current_mask_item.setOpacity(1.0)
        self.scene.addItem(self.current_mask_item)
        self.object_counter += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_filename = f"object_{self.object_counter}_mask_{timestamp}.png"
        qimage_mask.save(mask_filename)
        print(f"Initial segmentation mask saved to {mask_filename}")
        self.current_points = []
        self.adjust_positive_points = []
        self.adjust_negative_points = []

    def onAdjustPositive(self, point):
        x, y = point.x(), point.y()
        print(f"Adjust positive point: ({x}, {y})")
        self.adjust_positive_points.append([x, y])
        r = 4
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)
        ellipse.setPen(pen)
        ellipse.setBrush(QColor(0, 255, 0, 150))
        self.scene.addItem(ellipse)
        self.runRITMAdjustment()

    def onAdjustNegative(self, point):
        x, y = point.x(), point.y()
        print(f"Adjust negative point: ({x}, {y})")
        self.adjust_negative_points.append([x, y])
        r = 4
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        pen = QPen(QColor(0, 0, 255))
        pen.setWidth(2)
        ellipse.setPen(pen)
        ellipse.setBrush(QColor(0, 0, 255, 150))
        self.scene.addItem(ellipse)
        self.runRITMAdjustment()

    def runRITMAdjustment(self):
        if self.current_mask is None:
            print("No initial mask to adjust.")
            return
        image_b64 = encode_image(cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR), fmt='.png')
        mask_img = (self.current_mask * 255).astype(np.uint8)
        prev_mask_b64 = encode_image(mask_img, fmt='.png')
        payload = {
            "image": image_b64,
            "prev_mask": prev_mask_b64,
            "points": {
                "foreground": self.adjust_positive_points,
                "background": self.adjust_negative_points
            }
        }
        try:
            response = requests.post(API_URL + "/run_ritm", json=payload)
            if response.status_code != 200:
                print("Error from RITM server:", response.json())
                return
            result = response.json()
        except Exception as e:
            print("Error contacting RITM server:", e)
            return
        mask_b64 = result.get("mask")
        if not mask_b64:
            print("No mask returned from RITM server")
            return
        mask_img = decode_image_from_b64(mask_b64)
        if mask_img is None:
            print("Failed to decode mask image from RITM")
            return
        mask_gray = (mask_img > 128).astype(np.uint8)
        self.current_mask = mask_gray
        color = self.current_mask_color
        r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
        colored_mask = np.zeros((mask_gray.shape[0], mask_gray.shape[1], 4), dtype=np.uint8)
        colored_mask[mask_gray > 0] = [r, g, b, a]
        qimage_mask = QImage(colored_mask.data, mask_gray.shape[1], mask_gray.shape[0],
                              4 * mask_gray.shape[1], QImage.Format_ARGB32)
        mask_pixmap = QPixmap.fromImage(qimage_mask)
        if self.current_mask_item is not None:
            self.scene.removeItem(self.current_mask_item)
        self.current_mask_item = QGraphicsPixmapItem(mask_pixmap)
        self.current_mask_item.setOpacity(1.0)
        self.scene.addItem(self.current_mask_item)
        print("RITM adjustment complete. Updated mask shape:", self.current_mask.shape)

    def finishAdjustment(self):
        if self.current_mask is None:
            print("No mask to finish.")
            return
        object_id = self.object_counter
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"object_{object_id}_final_mask_{timestamp}.png"
        color = self.current_mask_color
        r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
        colored_mask = np.zeros((self.current_mask.shape[0], self.current_mask.shape[1], 4), dtype=np.uint8)
        colored_mask[self.current_mask > 0] = [r, g, b, a]
        final_qimg = QImage(colored_mask.data, self.current_mask.shape[1], self.current_mask.shape[0],
                             4 * self.current_mask.shape[1], QImage.Format_ARGB32)
        if final_qimg.save(final_filename):
            print(f"Final adjusted mask for object {object_id} saved to {final_filename}")
        else:
            print("Failed to save final adjusted mask.")
        self.current_mask = None
        if self.current_mask_item is not None:
            self.scene.removeItem(self.current_mask_item)
            self.current_mask_item = None
        self.adjust_positive_points = []
        self.adjust_negative_points = []
        self.viewer.setAdjustMode(False)
        self.adjustButton.setChecked(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
