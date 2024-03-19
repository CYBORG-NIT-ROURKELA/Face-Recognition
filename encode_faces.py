from imutils import paths
import face_recognition
import pickle
import cv2
import os
from deepface.detectors import DetectorWrapper as FaceDetector

print("[INFO] start processing faces...")
# imagePaths = list(paths.list_images("detectedfaces"))
imagePaths = list(paths.list_images("dataset"))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
undetectable = []
boxes=[]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):

	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	
	name = imagePath.split(os.path.sep)[1]
	 
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	frame_height, frame_width, _ = rgb.shape

	boxes = face_recognition.face_locations(rgb, model='HOG',number_of_times_to_upsample=2)
	# box = FaceDetector.detect_faces("mediapipe", cv2.imread(imagePath))
	# bounding_box = box[0][1]
	# '''left top right bottom'''
	# '''top, right, bottom, left'''
	# bounding_box[2] =bounding_box[0] + bounding_box[2]
	# bounding_box[3] =bounding_box[1] + bounding_box[3]

	# boxes[0] = bounding_box[1]
	# boxes[1] = bounding_box[2]
	# boxes[2] = bounding_box[3]
	# boxes[3] = bounding_box[0]


	encodings = face_recognition.face_encodings(rgb, boxes)

	if len(encodings) == 0:
		undetectable.append(i+1)

	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

print("know encodings : ", len(knownEncodings))
print("know names : ", len(knownNames))
print("undetectable numbers : ", len(undetectable))
print("undetectable : ", undetectable)
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()
