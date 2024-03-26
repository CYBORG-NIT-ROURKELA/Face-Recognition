'''This file for capturing the headshots of the person and saving it in the dataset folder'''

import cv2
import sys
import uuid
import os 
from deepface import DeepFace
from deepface.detectors import DetectorWrapper as FaceDetector


# print(type(sys.argv))
name = input("Enter the name : ") # name passed from command line argument

currentpath = f"A:\\facial_recognition\\facial-recognition\dataset\{name}"


if not os.path.exists(currentpath):
    os.mkdir(currentpath)


cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("press space to take a photo", 500, 300)

cam = cv2.VideoCapture(1)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    save = frame.copy()
    if not ret:
        print("failed to grab frame")
        break

    k = cv2.waitKey(1)
    box = FaceDetector.detect_faces("mediapipe",frame)
    if box :
        bounding_box = box[0][1]

        '''left top right bottom'''
        bounding_box[2] =bounding_box[0] + bounding_box[2]
        bounding_box[3] =bounding_box[1] + bounding_box[3]

        left, top, right, bottom = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
        # for (left, top, right, bottom) in (bounding_box):
        # draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15


    if k & 0xFF == ord('q'):
        # q pressed
        print("closing...")
        break
    elif k%256 == 32:

        if box :
            # SPACE pressed
            img_name = "dataset/{}/image_{}.jpg".format(name, uuid.uuid4().hex)
            # print(img_name)
            status = cv2.imwrite(img_name, save)
            if status is True:
                print("{} written!".format(img_name))
            else:
                print("Image not written. Check person's folder created")
        else:
            print("No face detected")

    cv2.imshow("press space to take a photo", frame)
cam.release()
cv2.destroyAllWindows()
