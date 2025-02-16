import sys
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
from PyQt5.QtGui import QPixmap, QPen, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF


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
        self.setWindowTitle("Image Viewer with Zoom, Pan, and Point Selection")
        self.points = []  # List to store selected points as (x, y) tuples

        # Set up the graphics scene and view
        self.scene = QGraphicsScene(self)
        self.viewer = ImageViewer(self)
        self.viewer.setScene(self.scene)
        self.setCentralWidget(self.viewer)

        self.createToolbar()

        # Connect the point selection signal from the viewer
        self.viewer.pointSelected.connect(self.onPointSelected)

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

        # Button to save selected points to a file
        saveAction = QAction("Save Points", self)
        saveAction.triggered.connect(self.savePoints)
        toolbar.addAction(saveAction)

    def openImage(self):
        """Open an image file and display it in the viewer."""
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.bmp *.gif)"
        )
        if fileName:
            pixmap = QPixmap(fileName)
            # Clear any existing items and points
            self.scene.clear()
            self.points = []
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
        Draw a small red circle and store its coordinates.
        """
        x = point.x()
        y = point.y()
        self.points.append((x, y))

        # Create a small circle (ellipse) centered at the selected point
        r = 5  # radius of the marker
        ellipse = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        ellipse.setPen(pen)
        ellipse.setBrush(QColor(255, 0, 0, 100))  # semi-transparent red fill
        self.scene.addItem(ellipse)

    def savePoints(self):
        """Save the selected points to a text (or CSV) file."""
        if not self.points:
            return  # Nothing to save if no points have been selected

        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Points", "", "Text Files (*.txt);;CSV Files (*.csv)"
        )
        if fileName:
            with open(fileName, "w") as f:
                for pt in self.points:
                    f.write(f"{pt[0]}, {pt[1]}\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
