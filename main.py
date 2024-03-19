import cv2
from deepface.detectors import DetectorWrapper as FaceDetector
from deepface import DeepFace
import pickle
import face_recognition
import time
from collections import defaultdict
from deepface.commons import distance as dst
import numpy as np

def get_embedding(frame):
    result = DeepFace.represent(frame, model_name="Dlib", enforce_detection=False, detector_backend="mediapipe")
    vector = result[0]["embedding"]
    vector = np.asarray(vector)
    box = list(result[0]["facial_area"].values())   
    if box:
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]

    return vector, box

# intial_setup = DeepFace.represent("wallpaper.jpg", model_name="Dlib", enforce_detection=False, detector_backend="mediapipe")

encodings_path = ".\\picklefiles\\encodings.pickle"
print("[INFO] loading encodings + face detector...")
database = pickle.loads(open(encodings_path, "rb").read())

# print(len(database["encodings"][0]))
# print(database["encodings"][0])

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(2.0)

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    k = cv2.waitKey(1) & 0xFF

    if k == ord(' '):

        print("Processing...")

        tic = time.time()
        encoding, box = get_embedding(frame)
        if box:
            names = []

            matches = face_recognition.compare_faces(database["encodings"], encoding, tolerance=0.5)
            # distance = dst.findEuclideanDistance(encoding,)
            distance = []
            max_name = "Unknown" # if face is not recognized, then print Unknown
            if True in matches:

                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = defaultdict(int)
                
                for i in matchedIdxs:
                    disc = dst.findEuclideanDistance(database["encodings"][i], encoding)
                    counts[database["names"][i]] += 1
                    distance.append(disc)
                    
                print(counts)
                print(distance)
                
                for name in counts:
                    if counts[name] >= 6:
                        max_name = name

            names.append(max_name)
            print("Recognized name ----- ",names)

            for (name) in (names):
                    left,top,right,bottom = box[0],box[1],box[2],box[3]

                    cv2.rectangle(frame, (left, top), (right, bottom),
                                (0, 255, 225), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,.8, (0, 255, 255), 2)
        else:
            print("No face detected.")

        toc = time.time()
        print("Time taken: ",toc-tic)

    elif k == ord('q'):
        print("Closing...")
        break

    cv2.imshow("Face Recognition", frame)

cam.release()
cv2.destroyAllWindows()



