import cv2
import torch
import os
import threading
import pyttsx3
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from collections import Counter

# Load the YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

os.makedirs('static', exist_ok=True)

latest_frame = None
detected_objects = Counter()
announcement_lock = threading.Lock()

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)


def speak_objects(object_list):
    if object_list:
        message = ", ".join([f"{count} {name}" for name, count in object_list])
        engine.say(message)
        engine.runAndWait()


def update_frame():
    global latest_frame, detected_objects
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1
        if frame_count % 4 != 0:  # Process every 4th frame
            continue

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (320, 240))

        # Run inference
        results = model(resized_frame)

        # Clear previous detections
        with announcement_lock:
            detected_objects.clear()

        # Process results
        for det in results.xyxy[0]:
            conf = det[4].item()
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, det[:4].tolist())
                cls = int(det[5].item())
                label = f'{model.names[cls]}: {conf:.2f}'

                # Scale bounding box to original frame size
                x1, y1, x2, y2 = [
                    int(coord * (640 if i % 2 == 0 else 480) / (320 if i % 2 == 0 else 240))
                    for i, coord in enumerate([x1, y1, x2, y2])
                ]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                with announcement_lock:
                    detected_objects[model.names[cls]] += 1

        latest_frame = cv2.resize(frame, (640, 480))
        cv2.imwrite('static/video_feed.jpg', latest_frame)
        print(f"Frame updated: {time.time()}")  # Debug print


threading.Thread(target=update_frame, daemon=True).start()


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
setInterval(function() {
    document.getElementById('video-feed').src = "/static/video_feed.jpg?" + new Date().getTime();
}, 100);
</script>
</body>
</html>
'''


class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path.startswith('/static/video_feed.jpg'):
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            try:
                with open('static/video_feed.jpg', 'rb') as f:
                    self.wfile.write(f.read())
                print(f"Frame sent: {time.time()}")  # Debug print
            except FileNotFoundError:
                print("Error: video_feed.jpg not found.")
                self.send_error(404)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress default logging
        return


server_address = ('', 8000)
httpd = HTTPServer(server_address, VideoStreamHandler)
print("Starting server on port 8000...")
httpd.serve_forever()

# Release the capture when the server is stopped
cap.release()
cv2.destroyAllWindows()
