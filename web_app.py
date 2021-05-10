import streamlit as st
#from detect import detector
import cv2
import numpy as np
import time
import math
import imutils


menu = ['Welcome', 'Social distancing detector', 'Learn more!']
with st.sidebar.beta_expander("Menu", expanded=False):
    option = st.selectbox('Choose your task', menu)
if option == 'Welcome':
    st.image('data/2MCAM_logo.png')
    st.title('2[M]CAM')
    st.subheader('The one stop solution to monitor social ditancing norms in any public environent')
    st.write('Welcome to the 2MCAM web application that can accurately identify human beings that violate social distancing norms in public spaces.')

elif option == 'Social distancing detector':
    st.title('Social Distancing Detection')
    st.write('The practice of social distancing signifies maintaining a distance of 6 feet or more when in public places or simply staying at home and away from others as much as possible to help prevent spread of COVID-19.')
    st.write('A green/red bounding box is drawn over the individuals in the frame to approve or flag the distance between them. ')
    if st.button("Start"):
        cap = cv2.VideoCapture(0)
        ph = st.empty()
        #detector(cap,ph)
        labelsPath = r"dependencies/coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")

        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

        weightsPath = r"dependencies/yolov3.weights"
        configPath = r"dependencies/yolov3.cfg"

        st.write("Loading Machine Learning Model ...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        st.write("Starting Camera ...")
        #cap = cv2.VideoCapture(0)
        #ph = st.empty()

        while(cap.isOpened()):
            ret, image = cap.read()
            image = imutils.resize(image, width=800)
            (H, W) = image.shape[:2]
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()
            #print("Prediction time/frame : {:.6f} seconds".format(end - start))
            boxes = []
            confidences = []
            classIDs = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.1 and classID == 0:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
            ind = []
            for i in range(0,len(classIDs)):
                if(classIDs[i]==0):
                    ind.append(i)
            a = []
            b = []

            if len(idxs) > 0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        a.append(x)
                        b.append(y)

            distance=[]
            nsd = []
            for i in range(0,len(a)-1):
                for k in range(1,len(a)):
                    if(k==i):
                        break
                    else:
                        x_dist = (a[k] - a[i])
                        y_dist = (b[k] - b[i])
                        d = np.linalg.norm(x_dist-y_dist)
                        distance.append(d)
                        if(d <= 220):
                            nsd.append(i)
                            nsd.append(k)
                        nsd = list(dict.fromkeys(nsd))
                        #print(nsd)
            color = (0, 0, 255)
            for i in nsd:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "Red Alert"
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
            color = (0, 255, 0)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    if (i in nsd):
                        break
                    else:
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        text = 'Normal'
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

            ph.image(image,caption="Social Distancing Detector",use_column_width=True,channels='BGR')

            key = cv2.waitKey(1)
            if key==27:
                break
        cap.release()
        cv2.destroyAllWindows()
elif option=='Learn more!':
    st.title('Why 2MCAM?')
    st.image('data/img1.jpg')
    st.write('2MCAM is a user-friendly ML web application and is designed to detect possible violations of social distancing norms.')
    st.write('Violation of physical distancing norms (1.5 - 2 meters) is looked down upon in the era of the covid19 pandemic. This is why we have a solution for you: 2MCAM, the web application that can instantaneously detect and flag such violations.')
    st.image('data/2feet.jpeg')
    st.write('Our application has a light-weight frontend and heavily tested backend waiting to be used. Discover the working of our application by navigating to the social distancing detector section.')
