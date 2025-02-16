import sys
import numpy as np
import torch  # Make sure torch is installed (pip install torch)
import datetime  # For timestamped filenames

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

# Import SAM components (install segment_anything package from https://github.com/facebookresearch/segment-anything)
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    sam_model_registry = None
    SamPredictor = None


class ImageViewer(QGraphicsView):
    # Signal to emit when a point is selected
    pointSelected = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super().__init__(parent)
        # By default, enable panning via drag
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0
        self.selection_mode = False  # When True, left clicks will select points

    def wheelEvent(self, event):
        """Zoom in or out with mouse wheel."""
        factor = 1.25
        if event.angleDelta().y() > 0:
            zoomFactor = factor
            self._zoom += 1
        else:
            zoomFactor = 1 / factor
            self._zoom -= 1

        # Optionally limit zoom level:
        if self._zoom < -10:
            self._zoom = -10
            return
        if self._zoom > 20:
            self._zoom = 20
            return

        self.scale(zoomFactor, zoomFactor)

    def mousePressEvent(self, event):
        """Handle mouse press events.
        If in selection mode and left-click, record the point.
        Otherwise, use the default behavior (e.g. panning).
        """
        if self.selection_mode and event.button() == Qt.LeftButton:
            scenePos = self.mapToScene(event.pos())
            self.pointSelected.emit(scenePos)
            # Do not call the base class to avoid interfering with point selection
            return
        super().mousePressEvent(event)

    def setSelectionMode(self, mode):
        """Toggle selection mode. In selection mode, disable panning drag."""
        self.selection_mode = mode
        if mode:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer with Zoom, Pan, Multi-Object SAM Segmentation")
        # Use a separate list for points for the current object
        self.current_points = []  
        # Counter for objects processed so far (used for unique colors & filenames)
        self.object_counter = 0  
        # Predefined list of semi-transparent colors for mask overlays.
        self.mask_colors = [
            QColor(255, 0, 0, 100),
            QColor(0, 255, 0, 100),
            QColor(0, 0, 255, 100),
            QColor(255, 255, 0, 100),
            QColor(255, 0, 255, 100),
            QColor(0, 255, 255, 100),
        ]

        # Set up the graphics scene and view
        self.scene = QGraphicsScene(self)
        self.viewer = ImageViewer(self)
        self.viewer.setScene(self.scene)
        self.setCentralWidget(self.viewer)

        self.createToolbar()

        # Connect the point selection signal from the viewer
        self.viewer.pointSelected.connect(self.onPointSelected)

        # Set up SAM if available
        if sam_model_registry is not None and SamPredictor is not None:
            # Path to your SAM checkpoint file – adjust this path accordingly!
            sam_checkpoint = "../sam_vit_h_4b8939.pth"
            model_type = "vit_h"  # or "vit_l", "vit_b"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using SAM model: {model_type} on device: {device}")
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device=device)
            self.sam_predictor = SamPredictor(self.sam)
        else:
            self.sam_predictor = None

    def createToolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Button to load an image
        openAction = QAction("Open Image", self)
        openAction.triggered.connect(self.openImage)
        toolbar.addAction(openAction)

        # Toggle button for point selection mode
        self.selectButton = QAction("Select Points", self)
        self.selectButton.setCheckable(True)
        self.selectButton.triggered.connect(self.toggleSelectionMode)
        toolbar.addAction(self.selectButton)

        # Button to save current object's points to a file
        saveAction = QAction("Save Points", self)
        saveAction.triggered.connect(self.savePoints)
        toolbar.addAction(saveAction)

        # New button to run SAM segmentation on current points
        samAction = QAction("Run SAM", self)
        samAction.triggered.connect(self.runSAM)
        toolbar.addAction(samAction)

    def openImage(self):
        """Open an image file and display it in the viewer."""
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.bmp *.gif)"
        )
        if fileName:
            pixmap = QPixmap(fileName)
            # Clear any existing items and points
            self.scene.clear()
            self.current_points = []
            # Add the image to the scene
            self.pixmapItem = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmapItem)
            self.viewer.setSceneRect(QRectF(pixmap.rect()))
            # Reset any transformations (zoom/pan) applied previously
            self.viewer.resetTransform()
            self.viewer._zoom = 0

    def toggleSelectionMode(self, checked):
        """Toggle the viewer's selection mode."""
        self.viewer.setSelectionMode(checked)

    def onPointSelected(self, point):
        """Handle a point being selected on the image.
        Draw a small red circle (marker) and store its coordinates.
        """
        x = point.x()
        y = point.y()
        self.current_points.append((x, y))

        # Create a small circle (ellipse) centered at the selected point
        r = 5  # marker radius
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        ellipse.setPen(pen)
        ellipse.setBrush(QColor(255, 0, 0, 150))  # semi-transparent red fill
        self.scene.addItem(ellipse)

    def savePoints(self):
        """Save the current object's points to a text file and then clear them."""
        if not self.current_points:
            print("No points to save for the current object.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        object_id = self.object_counter + 1  # use 1-indexing for file naming
        default_filename = f"object_{object_id}_points_{timestamp}.txt"

        try:
            with open(default_filename, "w") as f:
                for pt in self.current_points:
                    f.write(f"{pt[0]}, {pt[1]}\n")
            print(f"Points for object {object_id} saved to {default_filename}")
        except Exception as e:
            print(f"Error saving points: {e}")

    def runSAM(self):
        """Run SAM segmentation on the current object's points, overlay the mask with a unique color,
        save both the segmentation mask and the points file, and then clear the points.
        """
        # Check that an image is loaded
        if not hasattr(self, 'pixmapItem'):
            print("No image loaded!")
            return

        # Check that SAM is set up
        if self.sam_predictor is None:
            print("SAM is not available. Please ensure the segment_anything package is installed and configured.")
            return

        # Ensure there are selected points
        if not self.current_points:
            print("No points selected for segmentation!")
            return

        # Convert the QPixmap image to a NumPy array (RGB)
        qimg = self.pixmapItem.pixmap().toImage().convertToFormat(QImage.Format_RGB888)
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(height * width * 3)
        img_array = np.array(ptr).reshape(height, width, 3)

        # Prepare the points and assume all are positive (foreground)
        input_points = np.array(self.current_points)  # shape: (n_points, 2)
        input_labels = np.ones(input_points.shape[0], dtype=int)

        # Set the image for SAM and predict a mask
        self.sam_predictor.set_image(img_array)
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False,  # Only return one mask; set to True for multiple masks if needed
        )
        mask = masks[0]  # Select the first (or only) mask

        # Choose a unique color for this object's overlay
        color = self.mask_colors[self.object_counter % len(self.mask_colors)]
        r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()

        # Create a colored overlay image in RGBA from the binary mask.
        # For pixels where mask==True, set them to the chosen color.
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        colored_mask[mask > 0] = [r, g, b, a]

        # Create a QImage from the colored mask. The bytesPerLine is 4*width (4 bytes per pixel).
        qimage_mask = QImage(colored_mask.data, mask.shape[1], mask.shape[0], 4 * mask.shape[1], QImage.Format_ARGB32)
        mask_pixmap = QPixmap.fromImage(qimage_mask)

        # Add the segmentation mask as a semi-transparent overlay to the scene.
        # (The overlay will display in the unique color.)
        mask_item = QGraphicsPixmapItem(mask_pixmap)
        # Opacity can be adjusted if desired; here it is set to full opacity as the mask already has alpha.
        mask_item.setOpacity(1.0)
        self.scene.addItem(mask_item)

        # Increment the object counter (so the next object gets a different color/filename)
        self.object_counter += 1
        object_id = self.object_counter
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Automatically save the segmentation mask to a file with a unique filename.
        mask_filename = f"object_{object_id}_mask_{timestamp}.png"
        if not qimage_mask.save(mask_filename):
            print("Failed to save segmentation mask.")
        else:
            print(f"Segmentation mask for object {object_id} saved to {mask_filename}")

        # Also automatically save the current object's points to a file.
        points_filename = f"object_{object_id}_points_{timestamp}.txt"
        try:
            with open(points_filename, "w") as f:
                for pt in self.current_points:
                    f.write(f"{pt[0]}, {pt[1]}\n")
            print(f"Points for object {object_id} saved to {points_filename}")
        except Exception as e:
            print(f"Error saving points: {e}")

        # Clear the current points so that new clicks are considered for a new object.
        self.current_points = []


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
