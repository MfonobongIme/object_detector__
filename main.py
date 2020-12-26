#importing the libraries we need
import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

#extract the object names from the coco files into a list
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
#img = cv2.imread('office.jpg')

while True:
    _, img = cap.read()
    height, width, _ = img.shape #extract the width and height of image

    blob = cv2.dnn.blobFromImage(img, 1/255, (224, 224), (0, 0, 0), swapRB=True, crop=False)

    #pass the blob immage into the network
    net.setInput(blob) #sets input from the blob into the network

    output_layers_names = net.getUnconnectedOutLayersNames() #get the output layers names
    layerOutputs = net.forward(output_layers_names) #runs the forward pass and obtain the output at the output layer which we already provided the layers names

    boxes = [] #box list to extract the bounding boxes
    confidences = [] #confidence list to store the confidence

    class_ids = [] #class list which stores the predicted classes

    for output in layerOutputs: #used to extract all info from the layeroutput
        for detection in output: #used to extract the information in each of the outputs
            scores = detection[5:] #store all the 80 classes predictions starting from the 6th element till the end
            class_id = np.argmax(scores)#identify the classes that has the highest scores in scores list
            confidence = scores[class_id] #pass these elements to identify the maximum value from these scores which is the probability
            if confidence > 0.5: #set confidence threshold
                center_x = int(detection[0]*width) #multiply by width and height to rescale it back
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #since YOLO gets positions of objects from its center and opencv gets positions of objects
                #from the upper left, we need to perform this calculations for opencv to get its positions
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                #append all the information to the corresponding list
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #gets rid of redundant boxes

    font = cv2.FONT_HERSHEY_PLAIN
    colors  = np.random.uniform(0, 255, size=(len(boxes), 3))

    #create a for loop to loop over all the objects detected
    for i in indexes.flatten():

        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        #confidences = str(round(confidences[i], 2))
        color = colors [i]


        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y+20), font, 2, (255, 255, 255), 2)

    cv2.imshow('image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()