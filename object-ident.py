import cv2
import pyttsx3
import threading
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
            # Draw bounding box and label
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, class_name.upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img, object_info

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

# Start a thread for object announcements
threading.Thread(target=announce_objects, daemon=True).start()

# Main loop for video capture and display
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    if not success:
        break

    # Perform object detection
    result_img, object_info = get_objects(img, 0.60, 0.3)

    # Update detected objects
    with announcement_lock:
        for _, class_name in object_info:
            detected_objects[class_name] += 1

    # Show the result in a window
    cv2.imshow("Object Detection", result_img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
