import RPi.GPIO as GPIO
import cv2
from deepface.detectors import DetectorWrapper as FaceDetector
from deepface import DeepFace
import pickle
import face_recognition
import time
from collections import defaultdict
from deepface.commons import distance as dst
import numpy as np

# GPIO Setup
solenoid_pin = 23
green_led_pin = 24
red_led_pin = 25
flash_led_pin = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(solenoid_pin, GPIO.OUT)
GPIO.setup(green_led_pin, GPIO.OUT)
GPIO.setup(red_led_pin, GPIO.OUT)
GPIO.setup(flash_led_pin, GPIO.OUT)

GPIO.output(solenoid_pin, GPIO.LOW)
GPIO.output(green_led_pin, GPIO.LOW)
GPIO.output(red_led_pin, GPIO.LOW)
GPIO.output(flash_led_pin, GPIO.LOW)

# Load face encodings
encodings_path = "./picklefiles/encodings.pickle"
print("[INFO] loading encodings + face detector...")
database = pickle.loads(open(encodings_path, "rb").read())

# Initialize webcam
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(2.0)

def get_embedding(frame):
    result = DeepFace.represent(frame, model_name="Dlib", enforce_detection=False, detector_backend="mediapipe")
    vector = result[0]["embedding"]
    vector = np.asarray(vector)
    box = list(result[0]["facial_area"].values())  
    if box:
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
    return vector, box

def unlock_door_if_recognized():
    GUIOP.output(flash_led_pin, GPIO.HIGH)  # Turn on flash
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        return

    print("Processing...")
    encoding, box = get_embedding(frame)
    if box:
        matches = face_recognition.compare_faces(database["encodings"], encoding, tolerance=0.5)
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = defaultdict(int)
            for i in matchedIdxs:
                counts[database["names"][i]] += 1

            max_name = max(counts, key=counts.get) if counts else "Unknown"

            if max_name != "Unknown":
                print(f"Access Granted to {max_name}.")
                GPIO.output(green_led_pin, GPIO.HIGH)  # Green LED ON
                GPIO.output(solenoid_pin, GPIO.HIGH)  # Unlock door
                time.sleep(5)  # Keep the door unlocked for 5 seconds
                GPIO.output(green_led_pin, GPIO.LOW)  # Green LED OFF
                GPIO.output(solenoid_pin, GPIO.LOW)  # Lock door
            else:
                print("Access Denied.")
                GPIO.output(red_led_pin, GPIO.HIGH)  # Red LED ON
                time.sleep(1)
                GPIO.output(red_led_pin, GPIO.LOW)  # Red LED OFF
        else:
            print("Face Not Recognized.")
            GPIO.output(red_led_pin, GPIO.HIGH)  # Red LED ON
            time.sleep(1)
            GPIO.output(red_led_pin, GPIO.LOW)  # Red LED OFF
    else:
        print("No face detected.")
    GPIO.output(flash_led_pin, GPIO.LOW)  # Flash OFF

try:
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):  # Use space bar to attempt unlock
            unlock_door_if_recognized()
        elif k == ord('q'):  # Press 'q' to quit
            print("Closing...")
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
    GPIO.cleanup() 


