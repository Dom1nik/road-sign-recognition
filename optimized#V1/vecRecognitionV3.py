# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 19:02:24 2016

@author: domo
"""
import numpy as np
import cv2
import Tkinter 
from tkFileDialog import askopenfilename
from multiprocessing import Manager, Pool
import cPickle as pickle

# rectangle color and stroke
color = (0,0,255)       # reverse of RGB (B,G,R) - weird
strokeWeight = 1        # thickness of outline
matchCount=0
threshold=30

root = Tkinter.Tk()
filename=''
POISON_PILL="STOP"

def rectsCalc1(f,flag,keypoints_database):
    #print "ja"
    cascade1 = cv2.CascadeClassifier("trained_models/datapretjV2/cascade.xml")
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    detector = cv2.FeatureDetector_create("SURF")
    descriptor = cv2.DescriptorExtractor_create("SURF")
    

    while(True):
        img=f.get()
        if img==POISON_PILL:
            break
        
        rects1=cascade1.detectMultiScale(img,minSize=(30,30))
                
        kp, des = unpickle_keypoints(keypoints_database[1])
        #rects1=[]
        if len(rects1)>0:
            for x,y, width,height in rects1:
                #cv2.rectangle(frame, (x,y), (x+width, y+height), color, strokeWeight)
                match1=check(detector,descriptor,flann,img,kp,des,x,y,width,height)
                if match1>threshold:   
                    #flag[0]=1
                    print "noovertaking!"                    
                    #match1=0
        
    
def rectsCalc3(f,flag,keypoints_database):

    cascade3 = cv2.CascadeClassifier("trained_models/datastopV2/cascade.xml")
    
     # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    detector = cv2.FeatureDetector_create("SURF")
    descriptor = cv2.DescriptorExtractor_create("SURF")
        
    
    while(True):
        img=f.get()
        
        if img==POISON_PILL:
            break        
        rects3=cascade3.detectMultiScale(img,minSize=(30,30))
        
        kp, des = unpickle_keypoints(keypoints_database[3])           
        #rects3=[]
        if len(rects3)>0:

            for x,y, width,height in rects3:
                #cv2.rectangle(frame, (x,y), (x+width, y+height), color, strokeWeight)
                match3=check(detector,descriptor,flann,img,kp,des,x,y,width,height)
                if match3>threshold:
                    #flag[2]=1
                    print "stop!"
                    #match3=0  
        
        
def rectsCalc5(f,flag,keypoints_database):
     # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    detector = cv2.FeatureDetector_create("SURF")
    descriptor = cv2.DescriptorExtractor_create("SURF")
    

    cascade5 = cv2.CascadeClassifier("trained_models/datatriangleV2/cascade.xml")
    while(True):
        img=f.get()
        if img==POISON_PILL:
            break    
        rects5=cascade5.detectMultiScale(img,minSize=(30,30))
        #rects5=[]        
        kp, des = unpickle_keypoints(keypoints_database[4])        
                        
        if len(rects5)>0:
            for x,y, width,height in rects5:
                #cv2.rectangle(frame, (x,y), (x+width, y+height), color, strokeWeight)
                match5=check(detector,descriptor,flann,img,kp,des,x,y,width,height)
                if match5>threshold:
                    #flag[4]=1
                    print "triangle!"
                    #match5=0                        

    
def rectsCalc6(f,flag,keypoints_database):
    cascade6 = cv2.CascadeClassifier("trained_models/datakrivismjerV2/cascade.xml")     
    detector = cv2.FeatureDetector_create("SURF")
    descriptor = cv2.DescriptorExtractor_create("SURF")
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    while(True):

        img=f.get()
        if img==POISON_PILL:
            break    
        
        rects6=cascade6.detectMultiScale(img,minSize=(30,30))
        kp, des = unpickle_keypoints(keypoints_database[0])                   
        #rects6=[]
        if len(rects6)>0:
            for x,y, width,height in rects6:
                #cv2.rectangle(frame, (x,y), (x+width, y+height), color, strokeWeight)
                match6=check(detector,descriptor,flann,img,kp,des,x,y,width,height)
                print match6
                if match6>threshold:
                    #flag[5]=1
                    print "wrongdir!"
                    #match6=0       

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
    
    
    
def printSign(flag):
    
    if flag[0]:
        #cv2.putText(frame,'noovertaking!',(10,205), font, 1, (200,255,155), 1, cv2.LINE_AA)
        print 'noovertaking'    
                
    if flag[2]:
        #cv2.putText(frame,'Stop!',(10,165), font, 1, (200,255,155), 1, cv2.LINE_AA)
        print 'stop'     
  
    if flag[4]:
        #cv2.putText(frame,'triangle!',(10,135), font, 1, (200,255,155), 1, cv2.LINE_AA)
        print 'triangle'

    if flag[5]:
        #cv2.putText(frame,'wrongdir!',(10,120), font, 1, (200,255,155), 1, cv2.LINE_AA)
        print 'wrongdir'

    
def check(detector,descriptor,flann,img,kp1,des1,x,y,height,width):
    cnt=0    
    yheight=y+height
    xwidth=x+width
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    cropImg = img[y:yheight, x:xwidth]
                                     
    # Initiate SIFT detector
    #surf = cv2.xfeatures2d.SURF_create()
    
    kp2 = detector.detect(cropImg)
    kp2, des2 = descriptor.compute(cropImg, kp2)
      
    #make sure that number of features in both test and query image
    #is greater than or equal tonumber of nearest neighbours in 
    #knn match
    
    if len(kp1)>=2 and len(kp2)>=2:
        matches = flann.knnMatch(des1,des2,k=2)        
        cnt=len(matches) 
        
    return cnt

def askopenfile():
    global filename
    Tkinter.Tk().withdraw() 
    filename = askopenfilename()
    

#function for detection of 42 signs using one common trained model for all signs (no multiprocessing)
def detect():
    # rectangle color and stroke
    color = (0,0,255)       # reverse of RGB (B,G,R) - weird
    strokeWeight = 1        # thickness of outline
    
    cap = cv2.VideoCapture('{}'.format(filename))    
    #cap=cv2.VideoCapture(0)
    
    while(cap.isOpened()):
            ret, frame = cap.read()
            
            #if q pressed, break loop, jump to cap release and close all windows / force quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
            # load an image to search for faces
            img = frame
    
            # load detection file (various files for different views and uses)
            
            cascade1 = cv2.CascadeClassifier("trained_models/dataALL/cascade.xml")
        
            # detect objects, return as list
            rects1 = cascade1.detectMultiScale(img)
    
            # get a list of rectangles
            for x,y, width,height in rects1:
                cv2.rectangle(img, (x,y), (x+width, y+height), color, strokeWeight)
                
            cv2.imshow('frame',img)

        
    cap.release()
    cv2.destroyAllWindows()

    

def recognize():
    
    manager=Manager()
    
    flag=manager.list([0,0,0,0,0,0])
    
    f=manager.Queue()
    keypoints_database = pickle.load( open( "keypoints_database.p", "rb" ) )
    
    # load detection file (various files for different views and uses)
    #cap=cv2.VideoCapture(0)
    cap = cv2.VideoCapture('{}'.format(filename))
    ret, frame = cap.read()

    cap.set(3,320)
    cap.set(4,240)
              
    pool=Pool(processes=4)
    pool.apply_async(rectsCalc1, args=(f,flag,keypoints_database))    
    pool.apply_async(rectsCalc3, args=(f,flag,keypoints_database))    
    pool.apply_async(rectsCalc5, args=(f,flag,keypoints_database))    
    pool.apply_async(rectsCalc6, args=(f,flag,keypoints_database))  
    

    while(cap.isOpened()):
            ret, frame = cap.read()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if not np.any(frame):
                break

            #flag=[0,0,0,0,0,0]
            #frame=frame[100:700, 100:500]

            f.put(frame)
            
            #printSign(flag)
            
            # display! 
            cv2.imshow('frame', frame)
           
    
    f.put(POISON_PILL)
    pool.terminate()
    pool.join()
    # if esc key is hit, quit!
    cap.release()
    cv2.destroyAllWindows()
    quitScript()

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
    

