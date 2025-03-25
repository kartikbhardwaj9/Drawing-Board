import cv2
import mediapipe as mp
import numpy as np
import random
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ------------------------ Constants & Initialization ------------------------

# UI dimensions.
CANVAS_WIDTH, CANVAS_HEIGHT = 1280, 720
TOOLBAR_WIDTH = 130   # Left sidebar for tools.
COLORBOX_SIZE = 50
COLORBOX_GAP = 10

# Global drawing settings.
current_tool = "Pen"              # Default tool.
current_color = (0, 255, 0)         # Default color (green).
brush_thickness = 5               # Default thickness.
strokes = []                      # List of finalized strokes (for undo/redo).
redo_strokes = []                 # Stack for redo.
current_stroke = None             # Stroke in progress.
drawing_active = False            # Flag indicating if we are drawing.
start_point = None                # For shape tools (Line, Rect, Circle).

# Define a color palette.
color_palette = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0)
]
selected_color_index = 1  # Default index for green.

# Define an on-screen toolbar of 7 tools.
toolbar_buttons = {
    "Pen":    (10, 10, 110, 70),
    "Eraser": (10, 80, 110, 140),
    "Line":   (10, 150, 110, 210),
    "Rect":   (10, 220, 110, 280),
    "Circle": (10, 290, 110, 350),
    "Spray":  (10, 360, 110, 420),
    "Fill":   (10, 430, 110, 490)
}

# ------------------------ Gesture Definitions ------------------------
# Gestures are defined as the state of 5 fingers: [Thumb, Index, Middle, Ring, Pinky]
gesture_patterns = {
    "Pen":              [False, True, False, False, False],
    "Eraser":           [False, True, True, False, False],
    "ColorPicker":      [True, True, False, False, False],
    "IncreaseThickness":[False, True, False, True, False],
    "DecreaseThickness":[False, True, False, False, True],
    "Line":             [False, True, True, True, False],
    "Rect":             [False, True, True, False, True],
    "Circle":           [False, True, True, True, True],
    "Undo":             [False, False, True, False, False],
    "Redo":             [False, False, False, True, False]
}

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create a blank drawing canvas.
canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

# Open webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_HEIGHT)

prev_gesture = None  # Global variable to track the previous gesture

# ------------------------ Helper Functions ------------------------

def get_fingers_status(hand_landmarks, frame):
    """
    Returns a list of booleans for [thumb, index, middle, ring, pinky].
    For the thumb, compare tip and IP along the x-axis (mirrored image).
    For other fingers, check if the tip is above its pip joint.
    """
    h, w, _ = frame.shape
    fingers = []
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    fingers.append(thumb_tip.x < thumb_ip.x)
    for id in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]:
        tip = hand_landmarks.landmark[id]
        pip = hand_landmarks.landmark[id - 2]
        fingers.append(tip.y < pip.y)
    return fingers

def detect_gesture(fingers):
    """
    Check if the detected finger status matches one of our defined gestures.
    Returns the gesture name (string) or None.
    """
    for gesture, pattern in gesture_patterns.items():
        if fingers == pattern:
            return gesture
    return None

def draw_toolbar(frame):
    """
    Draw the toolbar buttons with tool names; highlight the current tool.
    """
    for tool, rect in toolbar_buttons.items():
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
        if tool == current_tool:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
        cv2.putText(frame, tool, (x1 + 5, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def draw_color_palette(frame):
    """
    Draw the color palette boxes; highlight the selected color.
    """
    start_x = TOOLBAR_WIDTH + 10
    start_y = 10
    for i, color in enumerate(color_palette):
        x1 = start_x + i * (COLORBOX_SIZE + COLORBOX_GAP)
        y1 = start_y
        x2 = x1 + COLORBOX_SIZE
        y2 = y1 + COLORBOX_SIZE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        if i == selected_color_index:
            cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 3)
    cv2.putText(frame, "Color Palette (ColorPicker Gesture)", 
                (start_x, start_y + COLORBOX_SIZE + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def draw_status_panel(frame):
    """
    Draw a semi-transparent panel at the bottom showing current settings and instructions.
    """
    overlay = frame.copy()
    panel_y1 = CANVAS_HEIGHT - 70
    cv2.rectangle(overlay, (TOOLBAR_WIDTH, panel_y1), (CANVAS_WIDTH, CANVAS_HEIGHT), (50, 50, 50), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    status_text = f"Tool: {current_tool} | Color: {current_color} | Thickness: {brush_thickness}"
    instruction_text = ("Gestures: Pen, Eraser, ColorPicker, Increase/DecreaseThickness, "
                        "Line, Rect, Circle, Undo, Redo   |   Keys: s=save, x=clear, Esc=exit")
    
    cv2.putText(frame, status_text, (TOOLBAR_WIDTH + 10, CANVAS_HEIGHT - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, instruction_text, (TOOLBAR_WIDTH + 10, CANVAS_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def redraw_canvas():
    """
    Clear and redraw the entire canvas based on the strokes list.
    (Used when undo/redo occurs.)
    """
    global canvas, strokes
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    for stroke in strokes:
        tool = stroke['tool']
        color = stroke['color']
        thickness = stroke['thickness']
        if tool in ["Pen", "Eraser"]:
            for i in range(1, len(stroke['points'])):
                cv2.line(canvas, stroke['points'][i - 1], stroke['points'][i],
                         color, thickness)
        elif tool == "Line":
            cv2.line(canvas, stroke['start'], stroke['end'], color, thickness)
        elif tool == "Rect":
            cv2.rectangle(canvas, stroke['start'], stroke['end'], color, thickness)
        elif tool == "Circle":
            center = stroke['start']
            end = stroke['end']
            radius = int(np.hypot(end[0] - center[0], end[1] - center[1]))
            cv2.circle(canvas, center, radius, color, thickness)
        elif tool == "Spray":
            # For spray, draw small random dots around each recorded point.
            for pt in stroke['points']:
                for _ in range(5):
                    offset = (random.randint(-5,5), random.randint(-5,5))
                    spray_pt = (pt[0] + offset[0], pt[1] + offset[1])
                    cv2.circle(canvas, spray_pt, 1, color, -1)
        elif tool == "Fill":
            mask = np.zeros((CANVAS_HEIGHT+2, CANVAS_WIDTH+2), np.uint8)
            cv2.floodFill(canvas, mask, stroke['point'], color)
    return

# ------------------------ Tkinter GUI Application ------------------------

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Drawing Board with GUI")
        self.root.configure(bg="#222222")
        
        # Video display panel.
        self.video_panel = tk.Label(root, bg="#000000")
        self.video_panel.pack(side="left", padx=10, pady=10)
        
        # Control frame for instructions and buttons.
        control_frame = tk.Frame(root, bg="#333333")
        control_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        instructions = (
            "Instructions:\n\n"
            "Use your hand gestures to control the drawing board.\n\n"
            "Available Gestures:\n"
            "  - Pen: [Index finger up]\n"
            "  - Eraser: [Index & Middle finger up]\n"
            "  - ColorPicker: [Thumb & Index up]\n"
            "  - Increase Thickness: [Index & Ring finger up]\n"
            "  - Decrease Thickness: [Index & Pinky up]\n"
            "  - Line: [Index, Middle, Ring up]\n"
            "  - Rect: [Index, Middle, Pinky up]\n"
            "  - Circle: [Index, Middle, Ring, Pinky up]\n"
            "  - Undo: [Only Middle finger up]\n"
            "  - Redo: [Only Ring finger up]\n\n"
            "On-screen Keys:\n"
            "  - s: Save drawing\n"
            "  - x: Clear canvas\n"
            "  - Esc: Exit application\n\n"
            "Make sure your hand is clearly visible to the webcam."
        )
        
        self.instruction_label = tk.Label(control_frame, text=instructions, justify="left",
                                          font=("Arial", 10), bg="#333333", fg="#FFFFFF")
        self.instruction_label.pack(padx=10, pady=10)
        
        # Control buttons.
        self.save_button = tk.Button(control_frame, text="Save Drawing (s)", command=self.save_drawing)
        self.save_button.pack(fill="x", padx=10, pady=5)
        self.clear_button = tk.Button(control_frame, text="Clear Canvas (x)", command=self.clear_canvas)
        self.clear_button.pack(fill="x", padx=10, pady=5)
        self.exit_button = tk.Button(control_frame, text="Exit (Esc)", command=self.on_exit)
        self.exit_button.pack(fill="x", padx=10, pady=5)
        
        self.delay = 15  # Delay in milliseconds between frames.
        self.update_frame()
        # Bind key events.
        self.root.bind("<Key>", self.key_handler)
    
    def update_frame(self):
        global current_tool, current_color, brush_thickness, strokes, redo_strokes
        global current_stroke, drawing_active, start_point, selected_color_index, canvas, prev_gesture
        
        ret, frame = cap.read()
        if not ret:
            self.root.after(self.delay, self.update_frame)
            return
        
        frame = cv2.flip(frame, 1)
        frame = draw_toolbar(frame)
        frame = draw_color_palette(frame)
        frame = draw_status_panel(frame)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        gesture = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            fingers = get_fingers_status(hand_landmarks, frame)
            gesture = detect_gesture(fingers)
            
            # Draw hand landmarks.
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(idx_tip.x * CANVAS_WIDTH), int(idx_tip.y * CANVAS_HEIGHT)
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            
            # Toolbar selection.
            if x < TOOLBAR_WIDTH:
                for tool, rect in toolbar_buttons.items():
                    x1, y1, x2, y2 = rect
                    if x1 < x < x2 and y1 < y < y2:
                        current_tool = tool
                        # End any active stroke when switching tools.
                        if current_stroke is not None:
                            strokes.append(current_stroke)
                        current_stroke = None
                        drawing_active = False
                        break
            
            # Color Palette selection.
            if gesture == "ColorPicker":
                palette_start = TOOLBAR_WIDTH + 10
                if palette_start < x < palette_start + len(color_palette)*(COLORBOX_SIZE + COLORBOX_GAP) and 10 < y < 10 + COLORBOX_SIZE:
                    index = (x - palette_start) // (COLORBOX_SIZE + COLORBOX_GAP)
                    if index < len(color_palette):
                        current_color = color_palette[index]
                        selected_color_index = index
            
            # Process gesture commands (trigger only on change).
            if gesture is not None and gesture != prev_gesture:
                if gesture == "IncreaseThickness":
                    brush_thickness += 1
                elif gesture == "DecreaseThickness":
                    brush_thickness = max(1, brush_thickness - 1)
                elif gesture == "Undo":
                    if strokes:
                        redo_strokes.append(strokes.pop())
                        redraw_canvas()
                elif gesture == "Redo":
                    if redo_strokes:
                        strokes.append(redo_strokes.pop())
                        redraw_canvas()
                elif gesture in ["Pen", "Eraser", "Line", "Rect", "Circle"]:
                    current_tool = gesture
                prev_gesture = gesture
            
            # Free-draw handling for Pen, Eraser, and Spray.
            if current_tool in ["Pen", "Eraser", "Spray"]:
                # For free-drawing, we consider a valid drawing gesture to be "Pen" or "Eraser".
                if gesture in ["Pen", "Eraser"]:
                    if current_stroke is None:
                        current_stroke = {
                            'tool': current_tool,
                            'color': current_color if current_tool != "Eraser" else (0, 0, 0),
                            'thickness': brush_thickness,
                            'points': []
                        }
                        drawing_active = True
                    # Append the current point.
                    current_stroke['points'].append((x, y))
                    if len(current_stroke['points']) > 1:
                        if current_tool == "Spray":
                            # For spray, draw random dots.
                            for _ in range(5):
                                offset = (random.randint(-5, 5), random.randint(-5, 5))
                                spray_pt = (x + offset[0], y + offset[1])
                                cv2.circle(canvas, spray_pt, 1, current_color, -1)
                        else:
                            cv2.line(canvas, current_stroke['points'][-2], current_stroke['points'][-1],
                                     current_stroke['color'], current_stroke['thickness'])
                else:
                    if current_stroke is not None:
                        strokes.append(current_stroke)
                        current_stroke = None
                        drawing_active = False
            # Shape tools: Line, Rect, Circle.
            elif current_tool in ["Line", "Rect", "Circle"]:
                if gesture == current_tool:
                    if current_stroke is None:
                        current_stroke = {
                            'tool': current_tool,
                            'color': current_color,
                            'thickness': brush_thickness,
                            'start': (x, y)
                        }
                        drawing_active = True
                    # Preview is drawn on the frame.
                    if current_tool == "Line":
                        cv2.line(frame, current_stroke['start'], (x, y), current_color, brush_thickness)
                    elif current_tool == "Rect":
                        cv2.rectangle(frame, current_stroke['start'], (x, y), current_color, brush_thickness)
                    elif current_tool == "Circle":
                        radius = int(np.hypot(x - current_stroke['start'][0], y - current_stroke['start'][1]))
                        cv2.circle(frame, current_stroke['start'], radius, current_color, brush_thickness)
                else:
                    if current_stroke is not None:
                        # Finalize the shape.
                        shape_stroke = {
                            'tool': current_tool,
                            'color': current_color,
                            'thickness': brush_thickness,
                            'start': current_stroke['start'],
                            'end': (x, y)
                        }
                        strokes.append(shape_stroke)
                        current_stroke = None
                        drawing_active = False
            # Fill tool.
            elif current_tool == "Fill":
                if gesture == "Pen":  # Use simple tap.
                    fill_stroke = {'tool': "Fill", 'point': (x, y), 'color': current_color, 'thickness': brush_thickness}
                    strokes.append(fill_stroke)
                    redraw_canvas()
        
        # Merge the drawing canvas with the frame.
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        merged = cv2.bitwise_and(frame, img_inv)
        merged = cv2.bitwise_or(merged, canvas)
        
        # Convert merged image to ImageTk format.
        img = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_panel.imgtk = imgtk
        self.video_panel.configure(image=imgtk)
        
        self.root.after(self.delay, self.update_frame)
    
    def key_handler(self, event):
        if event.char == 's':
            self.save_drawing()
        elif event.char == 'x':
            self.clear_canvas()
        elif event.keysym == 'Escape':
            self.on_exit()
    
    def save_drawing(self):
        global canvas
        cv2.imwrite("enhanced_drawing.png", canvas)
        messagebox.showinfo("Save", "Drawing saved as enhanced_drawing.png")
    
    def clear_canvas(self):
        global canvas, strokes, redo_strokes, current_stroke, drawing_active
        canvas = np.zeros((CANVAS_HEIGHT, CANVAS_HEIGHT, 3), dtype=np.uint8)
        strokes = []
        redo_strokes = []
        current_stroke = None
        drawing_active = False
    
    def on_exit(self):
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
    cap.release()
    cv2.destroyAllWindows()
