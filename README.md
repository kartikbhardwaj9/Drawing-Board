# Virtual Drawing Board with Hand Gestures and GUI

## Overview
This project is an interactive drawing application that leverages your webcam, hand gestures, and a Tkinter-based GUI to create a virtual drawing board. The app uses OpenCV for video processing, MediaPipe for hand detection, NumPy for numerical operations, and Pillow to bridge OpenCV with Tkinter. Users can select different drawing tools, change colors, adjust brush thickness, and perform undo/redo actions—all through intuitive hand gestures and on-screen controls.

## Features
- **Hand Gesture Controls:**
  - **Pen:** (Index finger up) for free drawing.
  - **Eraser:** (Index & Middle fingers up) to erase.
  - **ColorPicker:** (Thumb & Index up) to select colors from the palette.
  - **Increase/Decrease Brush Thickness:** (Index + Ring/Pinky finger up) to adjust stroke width.
  - **Shape Tools:** Draw straight lines, rectangles, and circles via designated gestures.
  - **Undo/Redo:** Remove or reapply strokes with specific gestures.
- **Graphical User Interface (GUI):**
  - A live video feed with an overlaid drawing canvas.
  - A toolbar and color palette with visual highlights.
  - A status panel that displays current tool, color, brush thickness, and instructions.
  - Buttons for saving, clearing the canvas, and exiting the application.
- **Keyboard Shortcuts:**
  - **s:** Save the current drawing as an image.
  - **x:** Clear the canvas.
  - **Esc:** Exit the application.

## Requirements
- Python 3.x
- [OpenCV-Python](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- [NumPy](https://pypi.org/project/numpy/)
- [Pillow](https://pypi.org/project/Pillow/)
- Tkinter (usually included with Python)

## Installation

### Setting Up a Virtual Environment (Optional but Recommended)
1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
