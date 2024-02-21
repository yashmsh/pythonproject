import cv2
import numpy as np
import json
import time
import argparse
import sys


hello = str(sys.argv[1])

covifor = [[132, 100, 100],[152, 255, 255]]
desrem = [[80, 100, 100],[100, 255, 255]]



if hello == "covifor":
	lowerBound=np.array(covifor[0])
	upperBound=np.array(covifor[1])

elif hello == 'desrem':
	lowerBound=np.array(desrem[0])
	upperBound=np.array(desrem[1])


cam= cv2.imread('remdi5.jpeg')
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

font = cv2.FONT_HERSHEY_DUPLEX

img=cv2.resize(cam,(250,550))

#convert BGR to HSV
imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# create the Mask
mask=cv2.inRange(imgHSV,lowerBound,upperBound)
#morphology
maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

maskFinal=maskClose
conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img,conts,-1,(255,0,0),3)

n_contours = len(conts)
print(n_contours)

for i in range(len(conts)):
    x,y,w,h=cv2.boundingRect(conts[i])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)

if n_contours==0:
    print("This is fake: Color Not Similar to the Original")

else:
    print("This is not fake")

 
cv2.imshow("cam",img)
cv2.waitKey(0)
