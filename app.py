import sys
import numpy as np
import torch
import cv2
import datetime

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QFileDialog,
    QToolBar,
    QAction,
)
from PyQt5.QtGui import QPixmap, QPen, QColor, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF

# -------------------------------
# Imports for SAM and RITM (isegm)
# -------------------------------
from isegm.utils import vis, exp
from isegm.inference import utils  # for load_is_model and find_checkpoint
from isegm.inference.evaluation import evaluate_sample_reefnet
from isegm.inference.predictors import get_predictor

# Import SAM components (install segment_anything package from https://github.com/facebookresearch/segment-anything)
from segment_anything import sam_model_registry, SamPredictor

# For backward compatibility with newer numpy versions (optional)
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

# Global constants for RITM (adjust as needed)
MODEL_THRESH = 0.6
EVAL_MAX_CLICKS = None
brs_mode = 'NoBRS'
device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# ImageViewer Class
# -------------------------------
class ImageViewer(QGraphicsView):
    # Signals for initial SAM points and adjustment clicks
    pointSelected = pyqtSignal(QPointF)       # For initial foreground points (SAM)
    adjustPositive = pyqtSignal(QPointF)        # For RITM positive clicks (left-click)
    adjustNegative = pyqtSignal(QPointF)        # For RITM negative clicks (right-click)

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
        # In adjust mode, interpret right-click as positive and left-click as negative.
        if self.adjust_mode:
            scenePos = self.mapToScene(event.pos())
            if event.button() == Qt.LeftButton:
                self.adjustPositive.emit(scenePos)
            elif event.button() == Qt.RightButton:
                self.adjustNegative.emit(scenePos)
            return
        # In selection mode, use left-click for initial SAM points.
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

# -------------------------------
# MainWindow Class
# -------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Segmentation with SAM & RITM Adjustment")

        # -------------------------------
        # Object and mask state
        # -------------------------------
        self.current_points = []            # SAM initial foreground points (list of (x,y))
        self.adjust_positive_points = []    # RITM positive clicks (right-click)
        self.adjust_negative_points = []    # RITM negative clicks (left-click)
        self.current_mask = None            # Latest binary mask (np.uint8, 0/1)
        self.current_mask_item = None       # QGraphicsPixmapItem for overlaying the mask
        self.current_mask_color = None      # Unique color (QColor) for the current object
        self.object_counter = 0             # For naming files and choosing colors

        # Predefined list of unique semi-transparent colors (RGBA)
        self.mask_colors = [
            QColor(255, 0, 0, 150),
            QColor(0, 255, 0, 150),
            QColor(0, 0, 255, 150),
            QColor(255, 255, 0, 150),
            QColor(255, 0, 255, 150),
            QColor(0, 255, 255, 150),
        ]

        # -------------------------------
        # Set up Graphics Scene and Viewer
        # -------------------------------
        self.scene = QGraphicsScene(self)
        self.viewer = ImageViewer(self)
        self.viewer.setScene(self.scene)
        self.setCentralWidget(self.viewer)

        # -------------------------------
        # Create Toolbar Buttons
        # -------------------------------
        self.createToolbar()

        # -------------------------------
        # Connect Signals from Viewer
        # -------------------------------
        self.viewer.pointSelected.connect(self.onPointSelected)
        self.viewer.adjustPositive.connect(self.onAdjustPositive)
        self.viewer.adjustNegative.connect(self.onAdjustNegative)

        # -------------------------------
        # Load Predictors for SAM and RITM
        # -------------------------------
        # SAM Predictor (for initial segmentation)
        if hasattr(sys.modules[__name__], 'sam_model_registry'):
            try:
                from segment_anything import sam_model_registry, SamPredictor
                sam_checkpoint = "../sam_vit_h_4b8939.pth"  # <<-- Update this path
                model_type = "vit_h"  # or "vit_l", "vit_b"
                device_sam = device
                
                print("Running SAM Model on device:", device_sam)
                self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                self.sam.to(device=device_sam)
                self.sam_predictor = SamPredictor(self.sam)
            except Exception as e:
                print("Error loading SAM predictor:", e)
                self.sam_predictor = None
        else:
            self.sam_predictor = None

        # RITM Predictor (for interactive adjustments via isegm)
        try:
            cfg = exp.load_config_file('./isegm/config.yml', return_edict=True)
            ritm_checkpoint = './isegm/ritm_corals.pth'  # <<-- Update this path if needed
            ritm_checkpoint = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, ritm_checkpoint)
            self.device = device
            
            print("Running RITM Model on device:", self.device)
            
            self.ritm_model = utils.load_is_model(ritm_checkpoint, self.device)
            self.ritm_predictor = get_predictor(self.ritm_model, brs_mode, self.device, prob_thresh=MODEL_THRESH)
        except Exception as e:
            print("Error loading RITM predictor:", e)
            self.ritm_predictor = None

        # -------------------------------
        # Image storage (loaded image will be stored as RGB np.array)
        # -------------------------------
        self.image = None

    def createToolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Open image button
        openAction = QAction("Open Image", self)
        openAction.triggered.connect(self.openImage)
        toolbar.addAction(openAction)

        # Toggle initial SAM selection mode button
        self.selectButton = QAction("Select Points", self)
        self.selectButton.setCheckable(True)
        self.selectButton.triggered.connect(self.toggleSelectionMode)
        toolbar.addAction(self.selectButton)

        # Save current SAM points (optional)
        saveAction = QAction("Save Points", self)
        saveAction.triggered.connect(self.savePoints)
        toolbar.addAction(saveAction)

        # Run SAM segmentation button (initial segmentation)
        samAction = QAction("Run SAM", self)
        samAction.triggered.connect(self.runSAM)
        toolbar.addAction(samAction)

        # Toggle Adjust mode button (for RITM adjustments)
        self.adjustButton = QAction("Adjust", self)
        self.adjustButton.setCheckable(True)
        self.adjustButton.triggered.connect(self.toggleAdjustMode)
        toolbar.addAction(self.adjustButton)

        # Finish button to save final adjusted mask
        finishAction = QAction("Finish", self)
        finishAction.triggered.connect(self.finishAdjustment)
        toolbar.addAction(finishAction)

    def openImage(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.bmp *.gif)"
        )
        if fileName:
            # Load image using OpenCV and convert to RGB
            img_bgr = cv2.imread(fileName)
            if img_bgr is None:
                print("Failed to load image!")
                return
            self.image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # Clear previous scene and state
            self.scene.clear()
            self.current_points = []
            self.current_mask = None
            self.current_mask_item = None
            self.adjust_positive_points = []
            self.adjust_negative_points = []
            # Display the image in the scene
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
        """Called when a SAM initial foreground point is clicked."""
        x, y = point.x(), point.y()
        self.current_points.append((x, y))
        # Draw a red marker for initial SAM points
        r = 5
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        ellipse.setPen(pen)
        ellipse.setBrush(QColor(255, 0, 0, 150))
        self.scene.addItem(ellipse)

    def savePoints(self):
        """Optional: Save the current initial SAM points to a file."""
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
        """
        Run initial segmentation using SAM based on the current_points.
        The resulting mask is overlaid (with a unique color) and saved.
        """
        if self.image is None:
            print("No image loaded!")
            return
        if self.sam_predictor is None:
            print("SAM predictor is not available.")
            return
        if not self.current_points:
            print("No points selected for segmentation!")
            return

        # Prepare the image (as loaded earlier) for SAM
        self.sam_predictor.set_image(self.image)
        input_points = np.array(self.current_points)
        input_labels = np.ones(input_points.shape[0], dtype=int)

        try:
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
        except Exception as e:
            print("Error during SAM prediction:", e)
            return

        mask = masks[0]  # use the first mask
        self.current_mask = mask.astype(np.uint8)

        # Choose a unique color for this object
        color = self.mask_colors[self.object_counter % len(self.mask_colors)]
        self.current_mask_color = color

        # Create a colored RGBA overlay from the binary mask
        r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        colored_mask[mask > 0] = [r, g, b, a]
        qimage_mask = QImage(colored_mask.data, mask.shape[1], mask.shape[0],
                             4 * mask.shape[1], QImage.Format_ARGB32)
        mask_pixmap = QPixmap.fromImage(qimage_mask)
        self.current_mask_item = QGraphicsPixmapItem(mask_pixmap)
        self.current_mask_item.setOpacity(1.0)
        self.scene.addItem(self.current_mask_item)

        # Auto-save the initial mask and points with a timestamp.
        self.object_counter += 1
        object_id = self.object_counter
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_filename = f"object_{object_id}_mask_{timestamp}.png"
        if qimage_mask.save(mask_filename):
            print(f"Initial segmentation mask for object {object_id} saved to {mask_filename}")
        else:
            print("Failed to save segmentation mask.")
        points_filename = f"object_{object_id}_points_{timestamp}.txt"
        try:
            with open(points_filename, "w") as f:
                for pt in self.current_points:
                    f.write(f"{pt[0]}, {pt[1]}\n")
            print(f"Points for object {object_id} saved to {points_filename}")
        except Exception as e:
            print("Error saving points:", e)

        # Clear the SAM points so that new clicks are for the next object.
        self.current_points = []
        # Clear any previous adjustment points.
        self.adjust_positive_points = []
        self.adjust_negative_points = []

    def onAdjustPositive(self, point):
        """Called in Adjust mode when a positive (right-click) point is added."""
        x, y = point.x(), point.y()
        print(f"Adjust positive point: ({x}, {y})")
        self.adjust_positive_points.append([x, y])
        # Draw a green marker for positive adjustment points.
        r = 4
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)
        ellipse.setPen(pen)
        ellipse.setBrush(QColor(0, 255, 0, 150))
        self.scene.addItem(ellipse)
        self.runRITMAdjustment()

    def onAdjustNegative(self, point):
        """Called in Adjust mode when a negative (left-click) point is added."""
        x, y = point.x(), point.y()
        print(f"Adjust negative point: ({x}, {y})")
        self.adjust_negative_points.append([x, y])
        # Draw a blue marker for negative adjustment points.
        r = 4
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        pen = QPen(QColor(0, 0, 255))
        pen.setWidth(2)
        ellipse.setPen(pen)
        ellipse.setBrush(QColor(0, 0, 255, 150))
        self.scene.addItem(ellipse)
        self.runRITMAdjustment()

    def runRITMAdjustment(self):
        """
        Use isegm's evaluate_sample_reefnet (RITM) to update the mask based on adjustment clicks.
        The current mask (from SAM or previous adjustment) is passed as prev_mask.
        """
        if self.current_mask is None:
            print("No initial mask to adjust.")
            return
        if self.ritm_predictor is None:
            print("RITM predictor is not available.")
            return

        # Prepare the points in the expected format.
        points = {
            "foreground": self.adjust_positive_points,
            "background": self.adjust_negative_points,
        }

        # Create a prev_mask tensor from the current mask.
        # The expected shape is [1,1,H,W] and type float32.
        prev_mask = self.current_mask.astype(np.float32)
        prev_mask = torch.tensor(prev_mask, device=self.device).unsqueeze(0).unsqueeze(0)

        try:
            clicks_list, pred, pred_mask = evaluate_sample_reefnet(
                self.image,              # Original image (RGB numpy array)
                None,                    # Additional info (unused here)
                self.ritm_predictor,     # RITM predictor
                pred_thr=MODEL_THRESH,
                max_iou_thr=None,
                max_clicks=EVAL_MAX_CLICKS,
                clicks=points,
                prev_mask=prev_mask
            )
        except Exception as e:
            print("Error during RITM adjustment:", e)
            return

        # The returned pred_mask is a boolean mask. Convert to 0/1 uint8.
        self.current_mask = pred_mask.astype(np.uint8)

        # Update the overlay with the same unique color.
        color = self.current_mask_color
        r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
        colored_mask = np.zeros((self.current_mask.shape[0], self.current_mask.shape[1], 4), dtype=np.uint8)
        colored_mask[self.current_mask > 0] = [r, g, b, a]
        qimage_mask = QImage(colored_mask.data, self.current_mask.shape[1], self.current_mask.shape[0],
                              4 * self.current_mask.shape[1], QImage.Format_ARGB32)
        mask_pixmap = QPixmap.fromImage(qimage_mask)
        # Remove previous overlay and add updated one.
        if self.current_mask_item is not None:
            self.scene.removeItem(self.current_mask_item)
        self.current_mask_item = QGraphicsPixmapItem(mask_pixmap)
        self.current_mask_item.setOpacity(1.0)
        self.scene.addItem(self.current_mask_item)
        print("RITM adjustment complete. Updated mask shape:", self.current_mask.shape)

    def finishAdjustment(self):
        """
        Save the final adjusted mask to a file (with a timestamped filename)
        and clear the adjustment state for the next object.
        """
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

        # Reset adjustment state for next object.
        self.current_mask = None
        if self.current_mask_item is not None:
            self.scene.removeItem(self.current_mask_item)
            self.current_mask_item = None
        self.adjust_positive_points = []
        self.adjust_negative_points = []
        self.viewer.setAdjustMode(False)
        self.adjustButton.setChecked(False)

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


# TODO: Integrate RITM
# TODO: Cleanup code and create requirements file
# TODO: First commit github repository
# TODO: Attmept to build executable