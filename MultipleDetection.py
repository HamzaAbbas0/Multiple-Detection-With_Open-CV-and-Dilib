# python Detection-Project --webcam webcam_index

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import json
from tkinter import *
from tkinter import messagebox
try:
    video=cv2.VideoCapture(0)
    if video.isOpened() == False:
        raise Exception
    #print(video.isOpened())
except:
    print("camera is not connected")
    messagebox.showinfo(title="retry", message="Camera Is Not Avaiable")

# Coordinates Of Rectangles
x, y = (140, 115)
w, h = (250, 250)

#In code use variables
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 1
sleep = 20
YAWN_THRESH = 20
COUNTER = 0
# NotePad File Work
colorcode ={'red':(0,0,255), "green":(0,255,0),"pink":(244,194,194),"purple":(128,0,128),"white":(255,255,255),"black":(0,0,0)}
try:
    with open("variables_files.txt", "r") as f:
        line = f.readlines()
        print(line[0])

        jtopy = json.dumps(line[0])
        variables = json.loads(line[0])
        print(type(line))
        print(type(variables))
        print(line)
except :
    print("notepad file is missing")
    messagebox.showinfo(title="error", message="NotePad Files Missing")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

print("-> Loading the predictor and detector...")

#Models Of Detections
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
smile_cascade=cv2.CascadeClassifier("haarcascade_smile.xml")
print("-> Starting Video Stream")
#vs = VideoStream(src=args["webcam"]).start()
# vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)
try:
    while True:
        boolean,frame = video.read()
        #frame = imutils.resize(frame, width=640) #for resizing the screen
        if boolean == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
            image = cv2.rectangle(frame, (x, y), (x + w, y + h), color=colorcode[variables["boxcolorred"]], thickness=3)
            if len(rects) == 0:
                #cv2.rectangle(frame,(x,y),(w -45, h + 136),(0,255,0),cv2.FILLED)
                cv2.putText(frame,text="NOT-DETECTED", org=(w-45, h + 136),fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=0.6, color=(0, 0, 255), thickness=1)

            else:
                #cv2.rectangle(frame,(x,y),(h-45, y + 136), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, text="DETECTED", org=(w -45, h + 136), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6,color=(0, 255, 0), thickness=1)
                image = cv2.rectangle(frame, (x, y), (x + w, y + h), color=colorcode[variables["boxcolorgreen"]], thickness=3)
            for (fx, fy, fw, fh) in rects:
                rect = dlib.rectangle(int(fx), int(fy), int(fx + fw), int(fy + fh))
                img = cv2.putText(frame, text=variables["face"], org=(x, y - 5), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=0.5, color=colorcode[variables["textcolor"]], thickness=1)
                #image1 = cv2.rectangle(frame, (x, y), (x + w, y + h), color=colorcode[variables["boxcolorgreen"]],thickness=3)

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                eye = final_ear(shape)
                ear = eye[0]
                leftEye = eye[1]
                rightEye = eye[2]

                distance = lip_distance(shape)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= 1 and COUNTER <=30 :
                        img = cv2.putText(frame, text=variables["closedeyes"], org=(x, y - 45), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=0.5, color=colorcode[variables["textcolor"]], thickness=1)
                    elif COUNTER >= 30:
                        cv2.putText(frame, text=variables["sleeping"], org=(x, y - 45), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=0.5, color=colorcode[variables["textcolor"]], thickness=1)
                        print(COUNTER, "eyes value check")
                    print(COUNTER,"check")
                else:
                    COUNTER = 0
                if (distance > YAWN_THRESH):
                    img = cv2.putText(frame, text=variables["laughing"], org=(x, y - 25), fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=0.5, color=colorcode[variables["textcolor"]], thickness=1)

                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Laughing: {:.2f}".format(distance), (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Smile Detection
            smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=16, minSize=(25, 25))
            for sx, sy, sw, sh in smile:
                if len(smile) > 1:
                    img1 = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
                    img = cv2.putText(frame, text=variables["smile"], org=(x, y - 65), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                      fontScale=0.5, color=colorcode[variables["textcolor"]], thickness=1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key ==(27):
            break
except :
    messagebox.showerror(title="error", message="Unable To Do Detection")


