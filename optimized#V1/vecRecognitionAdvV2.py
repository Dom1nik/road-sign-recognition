# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 14:41:22 2016

@author: dominik

working script - road sign detection

IMPLEMENTED:
+tkinter menu (1)browse,2)detection,3)recognize,4)quit)
+1)browse for local video input
+2)detection of 42 road signs using one trained model for all (no multiprocessing)
+3)detection and recognition of 6 road signs (text trigger on match) (multiprocessing) -> stop/wrong dir/60kmh limit/80kmh limit/triangle/no overtaking

ADDITIONAL:
+multiprocessing - each process takes one frame (total 4) and applies model detection (multiprocess function: "nProcs" )
+line 243/244 --> change video input from local file to live cam stream
+line 249/250 --> change resoolution of came input frame

PROBLEM SOLVED:
+ create-kill process loop (resources wasting) --> implemented pool with number of worker processes (4) 
+ that have while(True) loop implemented inside them (waiting for a input frame) --> same worker processes
+ are continuosly used in a loop until key 'q' pressed
 
"""

import numpy as np
import cv2
import Tkinter 
from tkFileDialog import askopenfilename
from multiprocessing import Manager, Pool
import cPickle as pickle
#import time
 
# rectangle color and stroke
color = (0,0,255)       # reverse of RGB (B,G,R) - weird
strokeWeight = 1        # thickness of outline
matchCount=0
threshold=30
POISON_PILL = "STOP"

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
    
    
#function called for multicore processing (called from function recognize)
def nProcs(frame,keypoints_database,threshold,flag):

    while True:
        # block until something is placed on the queue
        new_value = frame.get() 

        # check to see if we just got the poison pill
        if new_value == POISON_PILL:
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # load an image to search for signs
        img = new_value

        # each sign has its own trained model
        cascade1 = cv2.CascadeClassifier("trained_models/datapretjV2/cascade.xml")
        cascade2 = cv2.CascadeClassifier("trained_models/data60kmhV3/cascade.xml")
        cascade3 = cv2.CascadeClassifier("trained_models/datastopV2/cascade.xml")
        cascade4 = cv2.CascadeClassifier("trained_models/data80kmhV3/cascade.xml")
        cascade5 = cv2.CascadeClassifier("trained_models/datatriangleV2/cascade.xml")
        cascade6 = cv2.CascadeClassifier("trained_models/datakrivismjerV2/cascade.xml")
        
        # object detection using models, return as list
        rects1 = cascade1.detectMultiScale(img,minSize=(30,30))
        rects2 = cascade2.detectMultiScale(img,minSize=(30,30))
        rects3 = cascade3.detectMultiScale(img,minSize=(30,30))
        rects4 = cascade4.detectMultiScale(img,minSize=(30,30))
        rects5 = cascade5.detectMultiScale(img,minSize=(30,30))
        rects6 = cascade6.detectMultiScale(img,minSize=(30,30))
           
        kp1, des1 = unpickle_keypoints(keypoints_database[1])
        
        if len(rects1)>0:
            #znak zabrane pretjecanja
            #template="znakcina.png"
            for x,y, width,height in rects1:
                #cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                match1=check(img,kp1,des1,x,y,width,height)              
                 
                if match1>threshold:
                    flag[0]=1
                    
        kp2, des2 = unpickle_keypoints(keypoints_database[2])
              
        if len(rects2)>0:
            #znak 60km/h
            #template="111.png"
            for x,y, width,height in rects2:
                #cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                match2=check(img,kp2,des2,x,y,width,height)                  
                
                if match2>threshold:
                    flag[1]=1 
                    
        kp3, des3 = unpickle_keypoints(keypoints_database[3])           
                    
        if len(rects3)>0:
            #znak stop
            #template="stop-znak.png"
            for x,y, width,height in rects3:
                #cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                match3=check(img,kp3,des3,x,y,width,height)
                
                if match3>threshold:                                
                    flag[2]=1     
                    
        kp4, des4 = unpickle_keypoints(keypoints_database[5])        
                    
        if len(rects4)>0:
            #znak 80km/h
            #template="2222.png"
            for x,y, width,height in rects4:
                #cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                match4=check(img,kp4,des4,x,y,width,height)                 
               
                if match4>threshold:
                    flag[3]=1   
                    
        kp5, des5 = unpickle_keypoints(keypoints_database[4])        
                            
        if len(rects5)>0:
            #znak trokut (izricita naredba)
            #template="triangle.png"
            for x,y, width,height in rects5:
                #cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                match5=check(img,kp5,des5,x,y,width,height)                  
               
                if match5>threshold:
                    flag[4]=1       
                    
        kp6, des6 = unpickle_keypoints(keypoints_database[0])                   
                    
        if len(rects6)>0:
            #znak nedozvoljenog smjera
            #template="stop1.png"
            for x,y, width,height in rects6:
                #cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                match6=check(img,kp6,des6,x,y,width,height)
                
                if match6>threshold:
                    flag[5]=1  
                    
                    
#function for surf and flann calculations (called from function nProcs)
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
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                cnt=cnt+1
    
    return cnt

#tkinter base function
def askopenfile():
    global fileurl
    Tkinter.Tk().withdraw() 
    filename = askopenfilename()
    fileurl=filename


#function for detection of 42 signs using one common trained model for all signs (no multiprocessing)
def detect():
    # rectangle color and stroke
    color = (0,0,255)       # reverse of RGB (B,G,R) - weird
    strokeWeight = 1        # thickness of outline
    
    #cap = cv2.VideoCapture('{}'.format(fileurl))    
    cap=cv2.VideoCapture(0)
    
    while(cap.isOpened()):
            ret, frame = cap.read()
            
            #if q pressed, break loop, jump to cap release and close all windows / force quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
            # load an image to search for faces
            img = frame
            cv2.imshow('frame',img)
    
            # load detection file (various files for different views and uses)
            
            cascade1 = cv2.CascadeClassifier("trained_models/dataALL/cascade.xml")
        
            # detect objects, return as list
            rects1 = cascade1.detectMultiScale(img)
    
            # get a list of rectangles
            for x,y, width,height in rects1:
                cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
        
    cap.release()
    cv2.destroyAllWindows()

#function for detection and recognition of 6 signs using separated model for each sign (multiprocessing)
def recognize():
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    manager=Manager()  
    flag=manager.list([0,0,0,0,0,0])
           
    keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ) )
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('{}'.format(fileurl))
    
    #params (came only): width; height    
    cap.set(3, 352)
    cap.set(4, 240)
    
    f1 = manager.Queue()
    f2 = manager.Queue()
    f3 = manager.Queue() 
    f4 = manager.Queue()
    #flag=manager.Queue([0,0,0,0,0,0])

        
    _, frame1 = cap.read() 
    _, frame2 = cap.read() 
    _, frame3 = cap.read() 
    _, frame4 = cap.read()
    
    pool=Pool(processes=4)  
    pool.apply_async(nProcs, args=(f1,keypoints_database,threshold,flag))
    pool.apply_async(nProcs, args=(f2,keypoints_database,threshold,flag))
    pool.apply_async(nProcs, args=(f3,keypoints_database,threshold,flag))
    pool.apply_async(nProcs, args=(f4,keypoints_database,threshold,flag))
     
    while(cap.isOpened()):

        _, frame1 = cap.read() 
        _, frame2 = cap.read() 
        _, frame3 = cap.read() 
        _, frame4 = cap.read() 
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
         
        #nprocs=4
        f1.put(frame1)
        #time.sleep(0.050) 
        f2.put(frame2)
        #time.sleep(0.050)            
        f3.put(frame3)
        #time.sleep(0.050) 
        f4.put(frame4)
        #time.sleep(0.050) 
        
        
        if flag[0]==1:
            cv2.putText(frame1,'noovertaking!',(10,205), font, 1, (200,255,155), 1, cv2.LINE_AA)
            flag[0]=0

        if flag[1]==1:
            cv2.putText(frame1,'60kmh!',(10,180), font, 1, (200,255,155), 1, cv2.LINE_AA)
            flag[1]=0
            
        if flag[2]==1:
            cv2.putText(frame1,'Stop!',(10,165), font, 1, (200,255,155), 1, cv2.LINE_AA)
            flag[2]=0

        if flag[3]==1:
            cv2.putText(frame1,'80kmh!',(10,150), font, 1, (200,255,155), 1, cv2.LINE_AA)
            flag[3]=0
       
        if flag[4]==1:
            cv2.putText(frame1,'triangle!',(10,135), font, 1, (200,255,155), 1, cv2.LINE_AA)
            flag[4]=0
        
        if flag[5]==1:
            cv2.putText(frame1,'wrongdir!',(10,120), font, 1, (200,255,155), 1, cv2.LINE_AA)
            flag[5]=0
        
        cv2.imshow('output',frame1)
   
    f1.put(POISON_PILL)
    f2.put(POISON_PILL)
    f3.put(POISON_PILL)
    f4.put(POISON_PILL)
    # wait for them to exit
    pool.terminate()
    pool.join()
    
    # if q key is hit, jump to this line! force quit! 
    cap.release()
    cv2.destroyAllWindows()

#button "quit" function - quitting program
def quitScript():
    root.destroy()



#tkinter menu
A = Tkinter.Button(root, text ="Browse", command = askopenfile)
B = Tkinter.Button(root, text ="Detect", command = detect)
C = Tkinter.Button(root, text ="Recognize", command = recognize)
D = Tkinter.Button(root, text ="Quit", command = quitScript)

A.pack()
B.pack()
C.pack()
D.pack()

root.mainloop()    
    
