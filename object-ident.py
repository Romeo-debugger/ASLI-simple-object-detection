import cv2
import numpy as np
import threading
import pyttsx3
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from collections import Counter

# Paths for class file, config, and weights
CLASS_FILE_PATH = "/home/shravan/Documents/Coding/ProjectZETA/Object_Detection_Files/coco.names"
CONFIG_PATH = "/home/shravan/Documents/Coding/ProjectZETA/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
WEIGHTS_PATH = "/home/shravan/Documents/Coding/ProjectZETA/Object_Detection_Files/frozen_inference_graph.pb"

# Load class names
with open(CLASS_FILE_PATH, "rt") as f:
    class_names = f.read().rstrip("\n").split("\n")

# Initialize DNN model
net = cv2.dnn_DetectionModel(WEIGHTS_PATH, CONFIG_PATH)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize detected objects counter
detected_objects = Counter()
announcement_lock = threading.Lock()

# Initialize text-to-speech engine
engine = pyttsx3.init('espeak')
engine.setProperty('rate', 150)


def speak_objects(object_list):
    if object_list:
        message = ", ".join([f"{name}" for name, count in object_list])
        print(f"TTS: {message}")

        tts_thread = threading.Thread(target=lambda: engine.say(message) or engine.runAndWait())
        tts_thread.start()


def get_objects(img, thres, nms):
    class_ids, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    object_info = []
    if len(class_ids) != 0:
        for class_id, confidence, box in zip(class_ids.flatten(), confs.flatten(), bbox):
            class_name = class_names[class_id - 1]
            object_info.append([box, class_name])
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            # Change font parameters for better appearance
            cv2.putText(img, class_names[class_id - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Use a different positioning for the confidence level if needed
            # cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img, object_info


def update_frame():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        result, object_info = get_objects(img, 0.6, 0.3)

        # Update detected objects
        with announcement_lock:
            for _, class_name in object_info:
                detected_objects[class_name] += 1

        # Encode the frame as JPEG
        _, jpeg_frame = cv2.imencode('.jpg', img)
        yield jpeg_frame.tobytes()


def announce_objects():
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


# HTML template with a beautiful UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ProjectZETA</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Pixelify+Sans:wght@400;700&display=swap" rel="stylesheet">
<style>
    body {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        background-color: #d1d0ec;
        font-family: 'Pixelify Sans', sans-serif;
    }
    h1 {
        color: #333;
    }
    #video-feed {
        width: 640px;
        height: 480px;
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
<h1>ProjectZETA</h1>
<img id="video-feed" src="/video_feed" alt="Webcam Feed">
<button class="button" onclick="location.reload();">Refresh Feed</button>
</body>
</html>
'''

# Run the HTTP server
server_address = ('', 8080)
httpd = HTTPServer(server_address, VideoStreamHandler)
print("Starting server on port 8080...")
httpd.serve_forever()
