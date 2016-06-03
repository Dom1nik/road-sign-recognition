# -*- coding: utf-8 -*-
"""
Created on Sat May 28 12:22:07 2016

@author: dominik
"""

import numpy as np
import cv2
import Tkinter 
import tkMessageBox
from tkFileDialog import askopenfilename
import Image, ImageTk
import cPickle as pickle

# rectangle color and stroke
color = (0,0,255)       # reverse of RGB (B,G,R) - weird
strokeWeight = 1        # thickness of outline
matchCount=0
threshold=30


root = Tkinter.Tk()
fileurl=''


#used for loading of template keypoints and descriptors file
def unpickle_keypoints(array):
    
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)
    
def check(img,kp,des,x,y,height,width):
    cnt=0    
    yheight=y+height
    xwidth=x+width
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    cropImg = img[y:yheight, x:xwidth]
                                     
    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create()
    
    
    # find the keypoints and descriptors with SURF
    kp1=kp
    des1=des
    kp2, des2 = surf.detectAndCompute(cropImg,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    #make sure that number of features in both test and query image
    #is greater than or equal tonumber of nearest neighbours in 
    #knn match
    
    if len(kp1)>=2 and len(kp2)>=2:
        matches = flann.knnMatch(des1,des2,k=2)        
        
        # Need to draw only good matches, so create a mask
        #matchesMask = [[0,0] for i in xrange(len(matches))]
        
        cnt=len(matches)      
    return cnt



def askopenfile():
    global fileurl
    Tkinter.Tk().withdraw() 
    filename = askopenfilename()
    fileurl=filename
    
def detect():
    
    # rectangle color and stroke
    color = (0,0,255)       # reverse of RGB (B,G,R) - weird
    strokeWeight = 1        # thickness of outline
    #matchCount=0
    # set window name
    windowName = "Object Detection"
    cap = cv2.VideoCapture('2')
    
    while(cap.isOpened()):
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
            # load an image to search for faces
            
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = frame
            cv2.imshow('frame',img)
    
            # load detection file (various files for different views and uses)
            
            cascade1 = cv2.CascadeClassifier("trained_models/dataALLtest/cascade.xml")
    
    
            # preprocessing, as suggested by: http://www.bytefish.de/wiki/opencv/object_detection
            # img_copy = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
            # gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray)
            
            # detect objects, return as list
            rects1 = cascade1.detectMultiScale(img)
    
    # display until escape key is hit
    #while True:
        
            # get a list of rectangles
            for x,y, width,height in rects1:
                cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
          
            # display!
            cv2.imshow(windowName, img)
           
    
    # if esc key is hit, quit!
    cap.release()
    cv2.destroyAllWindows()
    
def recognize():
    # set window name
    #windowName = "Object Detection"
    font = cv2.FONT_HERSHEY_SIMPLEX 
    keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ) )
    cap=cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('{}'.format(fileurl))
    cap.set(3,352)
    cap.set(4,240)
    
    flag=[0,0,0,0,0,0]
    
    while(cap.isOpened()):
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if not np.any(frame):
                break

            flag=[0,0,0,0,0,0]
            # load an image to search for faces
            
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = frame
            cv2.imshow('frame',img)
    
            # load detection file (various files for different views and uses)
            cascade1 = cv2.CascadeClassifier("trained_models/datapretjV2/cascade.xml")
            cascade2 = cv2.CascadeClassifier("trained_models/data60kmhV3/cascade.xml")
            cascade3 = cv2.CascadeClassifier("trained_models/datastopV2/cascade.xml")
            cascade4 = cv2.CascadeClassifier("trained_models/data80kmhV3/cascade.xml")
            cascade5 = cv2.CascadeClassifier("trained_models/datatriangleV2/cascade.xml")
            cascade6 = cv2.CascadeClassifier("trained_models/datakrivismjerV2/cascade.xml")
    
    
            # preprocessing, as suggested by: http://www.bytefish.de/wiki/opencv/object_detection
            # img_copy = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
            # gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray)
            
            # detect objects, return as list
            rects1 = cascade1.detectMultiScale(img,minSize=(30,30))
            rects2 = cascade2.detectMultiScale(img,minSize=(50,50))
            rects3 = cascade3.detectMultiScale(img,minSize=(30,30))
            rects4 = cascade4.detectMultiScale(img,minSize=(30,30))
            rects5 = cascade5.detectMultiScale(img,minSize=(30,30))
            rects6 = cascade6.detectMultiScale(img,minSize=(30,30))
            
            kp1, des1 = unpickle_keypoints(keypoints_database[1])

            if len(rects1)>0:
                #match1=cv2.imread("B32_pretjecanje.jpg")
                #cv2.imshow("match1",match1)
                for x,y, width,height in rects1:
                    cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                    match1=check(img,kp1,des1,x,y,width,height)
                    if match1>threshold:
                        flag[0]=1
                        match1=0                        
                        
            kp2, des2 = unpickle_keypoints(keypoints_database[2])

            if len(rects2)>0:
                #match2=cv2.imread("670V60_0.png")
                #cv2.imshow("match2",match2)
                for x,y, width,height in rects2:
                    cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                    match2=check(img,kp2,des2,x,y,width,height)
                    if match2>threshold:
                        flag[1]=1
                        match2=0                        

            kp3, des3 = unpickle_keypoints(keypoints_database[3])           

            if len(rects3)>0:
                #match3=cv2.imread("stop-znak.png")
                #cv2.imshow("match3",match3)
                for x,y, width,height in rects3:
                    cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                    match3=check(img,kp3,des3,x,y,width,height)
                    if match3>threshold:
                        flag[2]=1
                        match3=0                        

            kp4, des4 = unpickle_keypoints(keypoints_database[5])        

            if len(rects4)>0:
                #match4=cv2.imread("80kmh.jpg")
                #cv2.imshow("match4",match4)
                for x,y, width,height in rects4:
                    cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                    match4=check(img,kp4,des4,x,y,width,height)
                    if match4>threshold:
                        flag[3]=1                               
                        match4=0                        

            kp5, des5 = unpickle_keypoints(keypoints_database[4])        

            if len(rects5)>0:
                #match5=cv2.imread("2669_4.jpg")
                #cv2.imshow("match5",match5)
                for x,y, width,height in rects5:
                    cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                    match5=check(img,kp5,des5,x,y,width,height)
                    if match5>threshold:
                        flag[4]=1
                        match5=0                        

            
            kp6, des6 = unpickle_keypoints(keypoints_database[0])                   

            if len(rects6)>0:
                #match6=cv2.imread("index.jpeg")
                #cv2.imshow("match",match6)
                for x,y, width,height in rects6:
                    cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                    match6=check(img,kp6,des6,x,y,width,height)
                    if match6>threshold:
                        flag[5]=1
                        match6=0                        
            
            if flag[0]==1:
                cv2.putText(img,'noovertaking!',(10,205), font, 1, (200,255,155), 1, cv2.LINE_AA)
                
            if flag[1]==1:
                cv2.putText(img,'60kmh!',(10,180), font, 1, (200,255,155), 1, cv2.LINE_AA)
                
            if flag[2]==1:
                cv2.putText(img,'Stop!',(10,165), font, 1, (200,255,155), 1, cv2.LINE_AA)
                
            if flag[3]==1:
                cv2.putText(img,'80kmh!',(10,150), font, 1, (200,255,155), 1, cv2.LINE_AA)
                
            if flag[4]==1:
                cv2.putText(img,'triangle!',(10,135), font, 1, (200,255,155), 1, cv2.LINE_AA)
                
            if flag[5]==1:
                cv2.putText(img,'wrongdir!',(10,120), font, 1, (200,255,155), 1, cv2.LINE_AA)
                
            # display!
            cv2.imshow('frame', img)
           
    
    # if esc key is hit, quit!
    cap.release()
    cv2.destroyAllWindows()
    #exit()

def quitScript():
    root.destroy()

A = Tkinter.Button(root, text ="Browse", command = askopenfile)
B = Tkinter.Button(root, text ="Detect", command = detect)
C = Tkinter.Button(root, text ="Recognize", command = recognize)
D = Tkinter.Button(root, text ="Quit", command = quitScript)


A.pack()
B.pack()
C.pack()
D.pack()

root.mainloop()    
    
