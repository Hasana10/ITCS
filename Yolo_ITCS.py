import numpy as np
import argparse
import time
import cv2
import os
import threading



labelsPath=os.path.sep.join(["R:\VEHICLE_COUNT\yolo-coco\coco.names"])
LABELS=open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = os.path.sep.join(["R:\VEHICLE_COUNT\yolo-coco\yolov3.weights"])
configPath = os.path.sep.join(["R:\VEHICLE_COUNT\yolo-coco\yolov3.cfg"])


print("[INFO] loading YOLO from disk...\n\n")
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

image1 = cv2.imread("Images\image5.jpg")
(H, W) = image1.shape[:2]

image2 = cv2.imread("Images\image6.jpg")
(H, W) = image2.shape[:2]

image3 = cv2.imread("Images\image7.jpg")
(H, W) = image3.shape[:2]

image4 = cv2.imread("Images\image11.jpg")
(H, W) = image4.shape[:2]

print("The images are being processed......!!!!!!")

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


blob1 = cv2.dnn.blobFromImage(image1, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob1)
layerOutputs1 = net.forward(ln)

blob2 = cv2.dnn.blobFromImage(image2, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob2)

layerOutputs2 = net.forward(ln)


blob3 = cv2.dnn.blobFromImage(image3, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob3)

layerOutputs3 = net.forward(ln)


blob4 = cv2.dnn.blobFromImage(image4, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob4)

layerOutputs4 = net.forward(ln)


boxes1 = []
confidences1 = []
classIDs1 = []
classname1 = []

boxes2 = []
confidences2 = []
classIDs2 = []
classname2 = []

boxes3 = []
confidences3 = []
classIDs3 = []
classname3 = []

boxes4 = []
confidences4 = []
classIDs4 = []
classname4 = []

list_of_vehicles = ["bicycle","car","motorbike","bus","truck"]
def get_vehicle_count(boxes, class_names):
	total_vehicle_count = 0 # total vechiles present in the image
	dict_vehicle_count = {} # dictionary with count of each distinct vehicles detected
	for i in range(len(boxes)):
		class_name = class_names[i]
		# print(i,".",class_name)
		if(class_name in list_of_vehicles):
			total_vehicle_count += 1
			

	return total_vehicle_count

for output in layerOutputs1:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > 0.5:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height

			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes1.append([x, y, int(width), int(height)])
			confidences1.append(float(confidence))
			classIDs1.append(classID)
			classname1.append(LABELS[classID])
for output in layerOutputs2:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > 0.5:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height

			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes2.append([x, y, int(width), int(height)])
			confidences2.append(float(confidence))
			classIDs2.append(classID)
			classname2.append(LABELS[classID])
for output in layerOutputs3:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > 0.5:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height

			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes3.append([x, y, int(width), int(height)])
			confidences3.append(float(confidence))
			classIDs3.append(classID)
			classname3.append(LABELS[classID])
for output in layerOutputs4:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > 0.5:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height

			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes4.append([x, y, int(width), int(height)])
			confidences4.append(float(confidence))
			classIDs4.append(classID)
			classname4.append(LABELS[classID])
total_vehicles=[0,0,0,0]
total_vehicles[0] = get_vehicle_count(boxes1, classname1)
total_vehicles[1] = get_vehicle_count(boxes2, classname2)
total_vehicles[2] = get_vehicle_count(boxes3, classname3)
total_vehicles[3] = get_vehicle_count(boxes4, classname4)
waiting_time=[0,0,0,0]
l=0
threshold=20
for i in range (0,4):
        waiting_time[i]=(int)((total_vehicles[i]/3)+10)
        if waiting_time[i]>=threshold:
                kk=(int)(i+1)
                l=l+1
                break
if l==0:
        for i in range (0,4):
                if max(total_vehicles)==total_vehicles[i]:
                        kk=i+1
                        break



print("\n\nTotal vehicle count and waiting time of all the lanes are.........\n\n")                

for i in range(0,4):
        print("Total Vehicle count in Lane",i+1," is:",total_vehicles[i])
print("\n")        
for i in range(0,4):
        print("Waiting Time of Lane",i+1," is:",waiting_time[i])        
print("\n")
print("Waitng time Threshold value is:",threshold)
print("\n")

 
import ctypes  # An included library with Python install.
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
Mbox('TRAFFIC SIGNAL OUTPUT','OPEN THE LANE '+str(kk)+' TO REDUCE THE TRAFFIC ', 1)
idxs1 = cv2.dnn.NMSBoxes(boxes1, confidences1, 0.5,0.3)
idxs2 = cv2.dnn.NMSBoxes(boxes2, confidences2, 0.5,0.3)
idxs3 = cv2.dnn.NMSBoxes(boxes3, confidences3, 0.5,0.3)
idxs4 = cv2.dnn.NMSBoxes(boxes4, confidences4, 0.5,0.3)

# ensure at least one detection exists
if len(idxs1) > 0:
	# loop over the indexes we are keeping
	for i in idxs1.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes1[i][0], boxes1[i][1])
		(w, h) = (boxes1[i][2], boxes1[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs1[i]]]
		cv2.rectangle(image1, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs1[i]], confidences1[i])
		cv2.putText(image1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
if len(idxs2) > 0:
	# loop over the indexes we are keeping
	for i in idxs2.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes2[i][0], boxes2[i][1])
		(w, h) = (boxes2[i][2], boxes2[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs2[i]]]
		cv2.rectangle(image2, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs2[i]], confidences2[i])
		cv2.putText(image2, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
if len(idxs3) > 0:
	# loop over the indexes we are keeping
	for i in idxs3.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes3[i][0], boxes3[i][1])
		(w, h) = (boxes3[i][2], boxes3[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs3[i]]]
		cv2.rectangle(image3, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs3[i]], confidences3[i])
		cv2.putText(image3, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
if len(idxs4) > 0:
	# loop over the indexes we are keeping
	for i in idxs4.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes4[i][0], boxes4[i][1])
		(w, h) = (boxes4[i][2], boxes4[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs4[i]]]
		cv2.rectangle(image4, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs4[i]], confidences4[i])
		cv2.putText(image4, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)


cv2.waitKey(0)
			
			
			






