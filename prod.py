import cv2
import torch
import threading
import pyttsx3
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from collections import Counter
import gc
from queue import Queue

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

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
tts_queue = Queue()

# Frame counter for tracking frames
frame_count = 0

# Text-to-Speech worker thread to manage announcements
def tts_worker():
    while True:
        message = tts_queue.get()
        engine.say(message)
        engine.runAndWait()
        tts_queue.task_done()

# Start the TTS thread
threading.Thread(target=tts_worker, daemon=True).start()

def speak_objects(object_list):
    if object_list:
        message = ", ".join([f"{count} {name}" for name, count in object_list])
        print(f"TTS: {message}")
        tts_queue.put(message)  # Add message to the queue

def periodic_gc(frame_count, interval=5):
    """ Trigger garbage collection every 'interval' frames. """
    if frame_count % interval == 0:
        gc.collect()

def update_frame():
    """ Capture and process video frames from webcam, and detect objects. """
    global detected_objects, frame_count
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame_count += 1

        # Trigger garbage collection periodically
        periodic_gc(frame_count)

        # Process every 5th frame to reduce CPU load
        if frame_count % 5 == 0:
            resized_frame = cv2.resize(frame, (160, 120))  # Reduce resolution for efficiency

            # Run YOLO model inference in no_grad mode for memory efficiency
            with torch.no_grad():
                results = model(resized_frame)

            # Update detected objects with lock
            with announcement_lock:
                detected_objects.clear()
                for det in results.xyxy[0]:
                    conf = det[4].item()
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, det[:4].tolist())
                        cls = int(det[5].item())
                        label = f'{model.names[cls]}: {conf:.2f}'

                        # Scale bounding box to original frame size
                        x1, y1, x2, y2 = [int(coord * (640 if i % 2 == 0 else 480) / (160 if i % 2 == 0 else 120))
                                          for i, coord in enumerate([x1, y1, x2, y2])]

                        # Draw bounding box on the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Count detected object
                        detected_objects[model.names[cls]] += 1

        # Encode the frame as JPEG for HTTP streaming
        _, jpeg_frame = cv2.imencode('.jpg', frame)
        yield jpeg_frame.tobytes()

def announce_objects():
    """ Announce detected objects periodically using TTS. """
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

# Start TTS announcement thread
threading.Thread(target=announce_objects, daemon=True).start()

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

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ProjectZETA</title>
<style>
body { display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh; margin: 0; background-color: #d1d0ec; font-family: Arial, sans-serif; }
.header { display: flex; align-items: center; justify-content: center; margin-bottom: 20px; }
h1 { color: #333; text-align: center; margin: 0; }
.logo { width: 50px; height: auto; margin-right: 15px; }
.container { display: flex; align-items: center; justify-content: center; }
#video-feed { width: 640px; height: 480px; object-fit: fill; border: 2px solid #4CAF50; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
@media (max-width: 700px) { #video-feed { width: 100%; height: auto; } }
.button { margin-top: 20px; padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background-color 0.3s; }
.button:hover { background-color: #45a049; }
.hamburger { display: none; flex-direction: column; cursor: pointer; }
.hamburger div { height: 4px; width: 30px; background: #333; margin: 4px 0; }
.menu { display: none; flex-direction: column; background-color: #fff; position: absolute; top: 50px; right: 10px; border: 1px solid #ccc; border-radius: 5px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); }
.menu a { padding: 10px; text-decoration: none; color: #333; border-bottom: 1px solid #eee; }
.menu a:last-child { border-bottom: none; }
.menu a:hover { background-color: #f0f0f0; }
@media (max-width: 700px) { .hamburger { display: flex; } }
</style>
</head>
<body>
<div class="header"><img src="https://tse1.mm.bing.net/th?id=OIG4.qdBKLPqhUlj4goH.5_id&pid=ImgGn" alt="Logo" class="logo"><h1>ProjectZETA</h1></div>
<div class="container"><img id="video-feed" src="/video_feed" alt="Webcam Feed"></div>
<button class="button" onclick="location.reload();">Refresh Feed</button>
<div class="hamburger" onclick="toggleMenu()"><div></div><div></div><div></div></div>
<div class="menu" id="menu"><a href="#">Feature 1</a><a href="#">Feature 2</a><a href="#">Feature 3</a></div>
<script>function toggleMenu() { var menu = document.getElementById('menu'); menu.style.display = menu.style.display === 'flex' ? 'none' : 'flex'; }</script>
</body>
</html>
'''

# Run the HTTP server
server_address = ('', 8080)
httpd = HTTPServer(server_address, VideoStreamHandler)
print("Starting server on port 8080...")
httpd.serve_forever()

# Release the capture when the server is stopped
cap.release()
cv2.destroyAllWindows()
