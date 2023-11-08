import face_recognition
import cv2
import json
import numpy as np
from threading import Thread, Lock
from queue import Queue
from PIL import Image, ImageTk, ImageDraw
import pygetwindow as gw
import tkinter as tk
from screeninfo import get_monitors
from PIL import ImageFont

class WebcamVideoStream:
    def __init__(self, src=1):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (grabbed, frame) = self.stream.read()
                self.lock.acquire()
                self.grabbed, self.frame = grabbed, frame
                self.lock.release()

    def read(self):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()
        return frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Load precomputed face encodings
with open('precomputed_encodings.json', 'r') as f:
    data = json.load(f)
    known_face_encodings = [np.array(d['encodings'][0]) for d in data.values()]
    known_face_names = [d['name'] for d in data.values()]

if not known_face_encodings:
    print("No known face encodings found. Check the precomputed_encodings.json file.")
    exit()

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True

# Start the video stream thread
video_stream = WebcamVideoStream(src=1).start()

# Initialize Tkinter for drawing on a specific monitor (change monitor_idx to the index of the desired monitor)
monitor_idx = 0  # for the second monitor
monitors = get_monitors()
if monitor_idx >= len(monitors):
    print(f"Monitor index {monitor_idx} is out of bounds. Only {len(monitors)} monitors detected.")
    exit()

# Use the monitor at index monitor_idx
selected_monitor = monitors[monitor_idx]

root = tk.Tk()
root.overrideredirect(True)  # Remove border and title bar

# Set the window geometry to the second monitor
root.geometry(f"+{selected_monitor.x}+{selected_monitor.y}")
root.lift()
root.wm_attributes("-topmost", True)
root.wm_attributes("-disabled", True)
root.wm_attributes("-transparentcolor", "white")

image_label = tk.Label(root)
image_label.pack()

def update_overlay(image, face_rectangles, face_names, scale_width, scale_height, window_position):
    window_x, window_y = window_position
    # Adjust the scale according to the monitor resolution
    overlay_image = Image.new("RGB", (selected_monitor.width, selected_monitor.height), "white")
    draw = ImageDraw.Draw(overlay_image)
    font = ImageFont.load_default()  # Load the default font

    for (top, right, bottom, left), name in zip(face_rectangles, face_names):
        # Scale the rectangle coordinates and adjust for the position of the window
        scaled_top = int((top * scale_height) + window_y)
        scaled_right = int((right * scale_width) + window_x)
        scaled_bottom = int((bottom * scale_height) + window_y)
        scaled_left = int((left * scale_width) + window_x)
        
        # Calculate the size of the red box (rectangle) relative to the monitor resolution
        box_padding = 20  # Adjust this value to control the size of the red box

        # Define the coordinates of the enlarged red box relative to the monitor
        enlarged_left = window_x + (scaled_left - box_padding)
        enlarged_top = window_y + (scaled_top - box_padding)
        enlarged_right = window_x + (scaled_right + box_padding)
        enlarged_bottom = window_y + (scaled_bottom + box_padding)

        # Create a rectangle using the ImageDraw rectangle method
        draw.rectangle([(enlarged_left, enlarged_top), (enlarged_right, enlarged_bottom)], outline="red", width=2)

        # Draw a red rectangle that spans the entire image
        draw.rectangle([(0, 0), (10, 10)], outline="red", width=2)

        text_bbox = draw.textbbox((0, 0), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle(((scaled_left, scaled_bottom - text_height - 10), (scaled_left + text_width + 6, scaled_bottom)), fill="red")
        draw.text((scaled_left + 3, scaled_bottom - text_height - 5), name, fill="white", font=font)
        
        # Calculate the coordinates for the middle of the monitor
        monitor_middle_x = selected_monitor.width // 2
        monitor_middle_y = selected_monitor.height // 2
        
        # Calculate the text width and height
        additional_text = "Additional Text"  # Replace with your desired text
        additional_text_bbox = draw.textbbox((0, 0), additional_text, font=font)
        additional_text_width = additional_text_bbox[2] - additional_text_bbox[0]
        additional_text_height = additional_text_bbox[3] - additional_text_bbox[1]
        
        # Calculate the position for the additional text in the middle of the monitor
        additional_text_x = monitor_middle_x - (additional_text_width // 2)
        additional_text_y = monitor_middle_y - (additional_text_height // 2)
        
        # Draw the additional text
        draw.rectangle(((additional_text_x, additional_text_y), (additional_text_x + additional_text_width, additional_text_y + additional_text_height)), fill="blue")
        draw.text((additional_text_x + 3, additional_text_y + 3), additional_text, fill="white", font=font)

    photo = ImageTk.PhotoImage(image=overlay_image)
    image_label.config(image=photo)
    image_label.image = photo  # Keep a reference!
    root.update_idletasks()
    root.update()



# Calculate scaling factors based on the camera resolution and the monitor resolution
cam_width, cam_height = video_stream.stream.get(cv2.CAP_PROP_FRAME_WIDTH), video_stream.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
scale_width = selected_monitor.width / cam_width
scale_height = selected_monitor.height / cam_height
    
try:
    while True:
        frame = video_stream.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)

        # Only process encodings if face_locations is not empty
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        else:
            face_encodings = []

        process_this_frame = not process_this_frame

        face_names = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)  # Append the name to the list

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        
        # Show the OpenCV window
        cv2.imshow('Video', frame)

        # Find the position of the 'Video' window
        video_window = gw.getWindowsWithTitle('Video')[0]
        window_x, window_y, window_width, window_height = video_window.box
        window_position = (window_x - selected_monitor.x, window_y - selected_monitor.y)

        # Calculate the position of the red square on the primary monitor
        # Adjust the coordinates if necessary to match the positions on Monitor 1
        monitor_face_locations = [(top, right, bottom, left) for (top, right, bottom, left) in face_locations]

        # Update the Tkinter overlay
        tk_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(f"Debug: Face rectangles - {monitor_face_locations}, Names - {face_names}")  # Debugging line

        update_overlay(tk_image, monitor_face_locations, face_names, scale_width, scale_height, window_position)  # Now passing the face_names as well

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    video_stream.stop()
    cv2.destroyAllWindows()
    root.destroy()
