import cv2
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import threading
import pyttsx3
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

# Load EfficientDet Lite model from TensorFlow Hub
print("Loading EfficientDet Lite model from TensorFlow Hub...")
model = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

# Define class labels (COCO dataset)
LABELS = {
    1: 'person', 2: 'chair',3: 'tie',4: 'cup',5: 'pen',6: 'pencil',7: 'road',8: 'speedbreaker',
}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")

# Initialize detected objects counter
detected_objects = Counter()
announcement_lock = threading.Lock()

# Initialize text-to-speech engine
engine = pyttsx3.init('espeak')
engine.setProperty('rate', 150)

# Configure parameters
SKIP_FRAMES = 2  # Only process every 2nd frame
INPUT_SIZE = (320, 320)  # Smaller input for faster processing


def speak_objects(object_list):
    if object_list:
        message = ", ".join([f"{count} {name}" for name, count in object_list])
        print(f"TTS: {message}")

        tts_thread = threading.Thread(target=lambda: engine.say(message) or engine.runAndWait())
        tts_thread.start()


def update_frame():
    global detected_objects
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        resized_frame = cv2.resize(frame, INPUT_SIZE)
        input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.uint8)
        input_tensor = tf.expand_dims(input_tensor, 0)

        # Run inference on the model
        detections = model(input_tensor)

        # Unpack the outputs
        boxes, scores, classes, num_detections = detections
        print("Boxes shape:", boxes.shape)
        print("Scores shape:", scores.shape)
        print("Classes shape:", classes.shape)
        print("Number of detections:", num_detections.numpy())

        with announcement_lock:
            detected_objects.clear()
            for i in range(int(num_detections[0].numpy())):
                class_id = int(classes[0][i].numpy())
                score = scores[0][i].numpy()
                if score > 0.3:  # Adjust the threshold as needed
                    box = boxes[0][i].numpy()
                    y1, x1, y2, x2 = (box * [480, 640, 480, 640]).astype(int)

                    # Construct the label
                    label = f'{LABELS.get(class_id, "Object")}: {score:.2f}'

                    # Draw bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draws the rectangle
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Adds the label

                    # Count detected object
                    detected_objects[LABELS.get(class_id, "Object")] += 1

        # Encode the frame as JPEG
        _, jpeg_frame = cv2.imencode('.jpg', frame)
        yield jpeg_frame.tobytes()



def announce_objects():
    global detected_objects
    last_spoken_time = time.time()
    while True:
        current_time = time.time()
        if current_time - last_spoken_time >= 3:  # Announce every 3 seconds
            with announcement_lock:
                objects_to_announce = detected_objects.most_common()
                detected_objects.clear()

            speak_objects(objects_to_announce)
            last_spoken_time = current_time
        else:
            time.sleep(0.1)


threading.Thread(target=announce_objects, daemon=True).start()

# HTML template for video feed
HTML_TEMPLATE = '''
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
    align-items: center;
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
    <h1>ProjectZETA</h1>
</div>
<div class="container">
    <img id="video-feed" src="/video_feed" alt="Webcam Feed">
</div>
<button class="button" onclick="location.reload();">Refresh Feed</button>
</body>
</html>
'''


# HTTP server to stream the video feed
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path.startswith('/video_feed'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            for frame in update_frame():
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        return  # Suppress default logging


# Run the HTTP server
server_address = ('', 8080)
httpd = HTTPServer(server_address, VideoStreamHandler)
print("Starting server on port 8080...")
httpd.serve_forever()

# Release the capture when the server is stopped
cap.release()
cv2.destroyAllWindows()
