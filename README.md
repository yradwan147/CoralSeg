# Interactive Segmentation Tool

This tool combines SAM (Segment Anything Model) and RITM (Reviving Iterative Training with Mask Guidance) for interactive image segmentation. It provides a user-friendly GUI for precise object segmentation with iterative refinement capabilities.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- CUDA 12.6 toolkit installed
- Git

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

Note: The installation might take some time due to the large dependencies. Make sure you have a stable internet connection.

## Running the Application

1. Start the server:
```bash
python server.py
```
The server will start on `http://localhost:6969` by default.

2. In a new terminal, start the GUI:
```bash
python gui.py
```

## Using the Tool

### Basic Workflow

1. **Connect to Server**
   - The GUI will attempt to connect to the server automatically
   - If needed, you can change the server URL in the toolbar
   - Click "Connect" to test the connection

2. **Load an Image**
   - Click "Open Image" in the toolbar
   - Supported formats: PNG, JPG, BMP, GIF

3. **Initial Segmentation (SAM)**
   - Click "Select Points" in the toolbar
   - Click on the image to place points on the object you want to segment
   - Click "Run SAM" to perform the initial segmentation
   - The segmented area will be highlighted with a semi-transparent color

4. **Refine Segmentation (RITM)**
   - Click "Adjust" in the toolbar
   - Left-click to add positive points (areas that should be included)
   - Right-click to add negative points (areas that should be excluded)
   - Click "Run RITM" to refine the segmentation
   - Repeat the adjustment process as needed

5. **Save Results**
   - Click "Save Points" to save the current point coordinates
   - The final mask is automatically saved when you click "Finish"
   - Files are saved with timestamps in the current directory

### Toolbar Functions

- **Open Image**: Load a new image for segmentation
- **Select Points**: Toggle point selection mode for SAM
- **Save Points**: Save current point coordinates
- **Run SAM**: Perform initial segmentation
- **Adjust**: Toggle adjustment mode for RITM
- **Run RITM**: Apply refinement to current segmentation
- **Finish**: Complete current object segmentation and save results

### Best Practices

1. **Point Selection**
   - Place points strategically around the object boundary
   - Include points inside the object for better results
   - Avoid placing points too close together

2. **RITM Adjustments**
   - Use positive points to include missed areas
   - Use negative points to exclude unwanted areas
   - Make small adjustments incrementally
   - Focus on problematic areas

3. **Performance Tips**
   - The tool automatically crops the image around points of interest when running model inference
   - Keep the crop region reasonable (not too large)
   - Clear previous points before starting a new segmentation

## Troubleshooting

1. **Connection Issues**
   - Check if the server is running
   - Verify the server URL
   - Ensure no firewall is blocking the connection

2. **Performance Issues**
   - Reduce the number of points
   - Make smaller adjustments
   - Check GPU memory usage
   - Restart the application if it becomes slow

3. **Installation Issues**
   - Ensure CUDA is properly installed
   - Try creating a fresh virtual environment
   - Check Python version compatibility

## File Formats

- **Points**: Saved as text files with coordinates
- **Masks**: Saved as PNG files with transparency
- **Timestamps**: Format: YYYYMMDD_HHMMSS

## Support

For issues and feature requests, please create an issue in the repository.