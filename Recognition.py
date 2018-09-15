import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('../FaceRecognition/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("../FaceRecognition/Recognize/trainingdata.yml")
id=0
#https://docs.opencv.org/3.4.1/d0/de1/group__core.html
#font=cv2.cv.InitFont(cv2.FONT_HERSHEY_SIMPLEX,5,1,0,4)
#fontface = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontface = cv2.FONT_HERSHEY_DUPLEX
fontscale = 1
fontcolor = (255, 255, 0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        #locy = int(img.shape[0]/2) # the text location will be in the middle
        #locx = int(img.shape[1]/2) #           of the frame for this example
        if(id==1):
            id="VARUN"
        else:
            id="UNKNOWN"
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
        cv2.putText(img, str(id), (x,y+h), fontface, fontscale, fontcolor) 
    cv2.imshow('img',img)
    print(str(id))
    
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()
