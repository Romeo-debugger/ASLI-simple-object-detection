import cv2
import torch
import os
import threading
import pyttsx3
from jinja2 import Environment, FileSystemLoader
from http.server import BaseHTTPRequestHandler, HTTPServer

# Load the YOLOv5s model from the PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a folder for the images
if not os.path.exists('static'):
    os.makedirs('static')

# This will hold the latest frame for the video feed
latest_frame = None
detected_objects = []  # Store detected object names for TTS

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak_object(object_name):
    engine.say(object_name)
    engine.runAndWait()

def update_frame():
    global latest_frame, detected_objects
    frame_count = 0  # Frame counter
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (640, 480))

        # Process every third frame to reduce inference load
        if frame_count % 3 == 0:
            results = model(resized_frame)  # Run inference on the input image

            # Process the results
            results_df = results.pandas().xyxy[0]  # Get detections
            detected_objects = []  # Reset detected objects for the current frame

            # Draw bounding boxes on the frame
            for index, row in results_df.iterrows():
                x1, y1, x2, y2, conf, cls = row[['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']].tolist()
                if conf > 0.5:  # Threshold for detections
                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Optionally put a label
                    label = f'{model.names[int(cls)]}: {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detected_objects.append(model.names[int(cls)])  # Store detected object name

        # Save the frame as an image in the static directory
        latest_frame = frame  # Update the latest frame
        frame_count += 1  # Increment frame counter

# Start a separate thread to update frames
threading.Thread(target=update_frame, daemon=True).start()

# Function to announce detected objects continuously
def announce_objects():
    while True:
        if detected_objects:
            unique_objects = set(detected_objects)  # Get unique object names
            for obj in unique_objects:
                speak_object(obj)  # Speak each object name
                detected_objects.remove(obj)  # Remove to avoid repeated announcements

# Start a separate thread for announcing detected objects
threading.Thread(target=announce_objects, daemon=True).start()

# Jinja2 setup for serving HTML
env = Environment(loader=FileSystemLoader('.'))
template = env.from_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ProjectZETA</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #d1d0ec;
            font-family: Arial, sans-serif;
        }
        .header {
            display: flex;
            align-items: flex-start; /* Align logo slightly higher */
            justify-content: center;
            margin-bottom: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin: 0;
        }
        .logo {
            width: 50px; /* Adjusted size to better fit with title */
            height: auto;
            margin-right: 15px;
            transition: transform 0.2s; /* Add a transition for hover effect */
        }
        .logo:hover {
            transform: scale(1.1); /* Scale up on hover */
        }
        .container {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #video-feed {
            width: 640px;
            height: 480px;
            object-fit: fill;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        @media (max-width: 700px) {
            #video-feed {
                width: 100%;
                height: auto;
            }
        }
        .button {
            margin-top: 20px;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="header">
        <img src="https://tse1.mm.bing.net/th?id=OIG4.qdBKLPqhUlj4goH.5_id&pid=ImgGn" alt="Logo" class="logo">
        <h1>ProjectZETA</h1>
    </div>
    <div class="container">
        <img id="video-feed" src="/static/video_feed.jpg" alt="Webcam Feed">
    </div>
    <button class="button" onclick="location.reload();">Refresh Feed</button>
    <script>
        // Update the image source every 100ms
        setInterval(function() {
            document.getElementById('video-feed').src = "/static/video_feed.jpg?" + new Date().getTime();
        }, 100);
    </script>
</body>
</html>
''')

# Simple HTTP server to serve the HTML and video feed
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(template.render().encode('utf-8'))
        elif self.path.startswith('/static/video_feed.jpg'):
            # Serve the latest frame
            if latest_frame is not None:
                # Save the latest frame as an image
                cv2.imwrite('static/video_feed.jpg', latest_frame)
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                with open('static/video_feed.jpg', 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

# Start the HTTP server
server_address = ('', 8000)
httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
print("Starting server on port 8000...")
httpd.serve_forever()

# Release the capture when the server is stopped
cap.release()
cv2.destroyAllWindows()
