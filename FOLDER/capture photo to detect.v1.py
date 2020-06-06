#!/usr/bin/env python
# coding: utf-8

#  Go through the code first before anything :)
#   This code mainly lets you use your WEBCAM TO TAKE PHOTOS and store them in the DESIRED LOCATION

#    It is been built to take 1 Photo at a single run of code

import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while 1:
    ret,img = cap.read()
    faces = face_cascade.detectMultiScale(img)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0))
        roi_color = img[y:y+h , x:x+w]
        ro_color = cv2.resize(roi_color,(100,100))

        face_file_name = r"C:\Users\SHREYAS\Desktop\New folder\data\1st" + str(y) + ".jpg"   #  change the name of SHREYAS to ur pc name so that code can work fine
        cv2.imwrite(face_file_name, ro_color)
        
    cv2.imshow('img',img)
    break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):   # window exits when q is pressed
        break
    
cap.release()
cv2.destroyAllWindows()

