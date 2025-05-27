# gui.py
import sys
import numpy as np
import cv2
import base64
import datetime
import requests
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QFileDialog, QToolBar, QAction,
    QLineEdit, QLabel, QHBoxLayout, QWidget, QPushButton
)
from PyQt5.QtGui import QPixmap, QPen, QColor, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF

# Default API URL
DEFAULT_API_URL = "http://10.109.26.31:6969/"

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

def get_crop_region(points, image_shape, padding=100):
    """Calculate crop region around points with padding."""
    if not points:
        return None
    
    # Convert points to numpy array for easier calculation
    points_array = np.array(points)
    
    # Get min and max coordinates
    min_x = np.min(points_array[:, 0])
    max_x = np.max(points_array[:, 0])
    min_y = np.min(points_array[:, 1])
    max_y = np.max(points_array[:, 1])
    
    # Add padding
    min_x = max(0, min_x - padding)
    max_x = min(image_shape[1], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(image_shape[0], max_y + padding)
    
    return (int(min_x), int(min_y), int(max_x), int(max_y))

def adjust_points_to_crop(points, crop_region):
    """Adjust point coordinates relative to cropped region."""
    if not points:
        return []
    
    min_x, min_y, _, _ = crop_region
    return [[x - min_x, y - min_y] for x, y in points]

def expand_mask_to_original(mask, crop_region, original_shape):
    """Expand cropped mask back to original image size."""
    min_x, min_y, max_x, max_y = crop_region
    full_mask = np.zeros(original_shape[:2], dtype=np.uint8)
    full_mask[min_y:max_y, min_x:max_x] = mask
    return full_mask

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
        self.api_url = DEFAULT_API_URL

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
        
        # Check initial connection
        self.checkConnection()

    def createToolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Create URL input widget
        url_widget = QWidget()
        url_layout = QHBoxLayout()
        url_layout.setContentsMargins(0, 0, 0, 0)
        url_widget.setLayout(url_layout)

        url_label = QLabel("API URL:")
        self.url_input = QLineEdit(self.api_url)
        self.url_input.setMinimumWidth(200)
        
        connect_button = QPushButton("Connect")
        connect_button.clicked.connect(self.reconnect)
        
        self.connection_status = QLabel()
        self.connection_status.setMinimumWidth(80)
        
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        url_layout.addWidget(connect_button)
        url_layout.addWidget(self.connection_status)
        
        url_tool = QWidget()
        url_tool.setLayout(url_layout)
        toolbar.addWidget(url_tool)
        toolbar.addSeparator()

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

        ritmAction = QAction("Run RITM", self)
        ritmAction.triggered.connect(self.runRITMAdjustment)
        toolbar.addAction(ritmAction)

        finishAction = QAction("Finish", self)
        finishAction.triggered.connect(self.finishAdjustment)
        toolbar.addAction(finishAction)

    def checkConnection(self):
        try:
            response = requests.get(self.api_url)
            self.connection_status.setText("Connected")
            self.connection_status.setStyleSheet("color: green")
        except:
            self.connection_status.setText("Disconnected")
            self.connection_status.setStyleSheet("color: red")

    def reconnect(self):
        self.api_url = self.url_input.text()
        self.checkConnection()

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

        # Calculate crop region
        crop_region = get_crop_region(self.current_points, self.image.shape)
        if crop_region is None:
            return

        # Crop image and adjust points
        min_x, min_y, max_x, max_y = crop_region
        cropped_image = self.image[min_y:max_y, min_x:max_x]
        adjusted_points = adjust_points_to_crop(self.current_points, crop_region)

        # Encode cropped image as BGR PNG for sending
        image_b64 = encode_image(cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR), fmt='.png')
        payload = {
            "image": image_b64,
            "points": adjusted_points
        }
        try:
            response = requests.post(self.api_url + "/run_sam", json=payload)
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

        # Expand mask back to original size
        mask_gray = (mask_img > 128).astype(np.uint8)
        self.current_mask = expand_mask_to_original(mask_gray, crop_region, self.image.shape)

        # Choose a unique color
        color = self.mask_colors[self.object_counter % len(self.mask_colors)]
        self.current_mask_color = color
        r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
        colored_mask = np.zeros((self.current_mask.shape[0], self.current_mask.shape[1], 4), dtype=np.uint8)
        colored_mask[self.current_mask > 0] = [r, g, b, a]
        qimage_mask = QImage(colored_mask.data, self.current_mask.shape[1], self.current_mask.shape[0],
                              4 * self.current_mask.shape[1], QImage.Format_ARGB32)
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

    def runRITMAdjustment(self):
        if self.current_mask is None:
            print("No initial mask to adjust.")
            return
        if not self.adjust_positive_points and not self.adjust_negative_points:
            print("No adjustment points selected.")
            return

        # Calculate crop region based on adjustment points
        all_points = self.adjust_positive_points + self.adjust_negative_points
        crop_region = get_crop_region(all_points, self.image.shape)
        if crop_region is None:
            return

        # Crop image and mask, adjust points
        min_x, min_y, max_x, max_y = crop_region
        cropped_image = self.image[min_y:max_y, min_x:max_x]
        cropped_mask = self.current_mask[min_y:max_y, min_x:max_x]
        adjusted_positive = adjust_points_to_crop(self.adjust_positive_points, crop_region)
        adjusted_negative = adjust_points_to_crop(self.adjust_negative_points, crop_region)
            
        image_b64 = encode_image(cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR), fmt='.png')
        mask_img = (cropped_mask * 255).astype(np.uint8)
        prev_mask_b64 = encode_image(mask_img, fmt='.png')
        payload = {
            "image": image_b64,
            "prev_mask": prev_mask_b64,
            "points": {
                "foreground": adjusted_positive,
                "background": adjusted_negative
            }
        }
        try:
            response = requests.post(self.api_url + "/run_ritm", json=payload)
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

        # Update only the crop region in the current mask
        mask_gray = (mask_img > 128).astype(np.uint8)
        self.current_mask[min_y:max_y, min_x:max_x] = mask_gray

        color = self.current_mask_color
        r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
        colored_mask = np.zeros((self.current_mask.shape[0], self.current_mask.shape[1], 4), dtype=np.uint8)
        colored_mask[self.current_mask > 0] = [r, g, b, a]
        qimage_mask = QImage(colored_mask.data, self.current_mask.shape[1], self.current_mask.shape[0],
                              4 * self.current_mask.shape[1], QImage.Format_ARGB32)
        mask_pixmap = QPixmap.fromImage(qimage_mask)
        if self.current_mask_item is not None:
            self.scene.removeItem(self.current_mask_item)
        self.current_mask_item = QGraphicsPixmapItem(mask_pixmap)
        self.current_mask_item.setOpacity(1.0)
        self.scene.addItem(self.current_mask_item)
        print("RITM adjustment complete. Updated mask shape:", self.current_mask.shape)
        self.adjust_positive_points = []
        self.adjust_negative_points = []

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
