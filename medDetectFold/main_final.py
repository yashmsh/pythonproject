import cv2
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import os
import pickle as pkl
import pytesseract 
from spellchecker import SpellChecker
import re 
import sys

final_count = []
warning = []
misspelled1=[]
main_text=[]

#rx symbol detection module

net = cv2.dnn.readNet("yolov3_custom_last1.weights", "yolov3_custom.cfg")
classes = []
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


img = cv2.imread("remdi1.jpeg")
img = cv2.resize(img, None, fx=0.9, fy=0.9)
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 1, color, 1)

if boxes == 0:
	warning.append("No Rx Sybmol found")
	final_count.append('0')
else:
	warning.append("Rx Sybmol was found on the package")
	final_count.append('1')
	

#spelling checker module

large = cv2.imread('finalImage1.jpg')
rgb = cv2.pyrDown(large)
rgb_orig = rgb.copy()
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)




# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel = np.ones((5, 5), np.uint8)
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
font = cv2.FONT_HERSHEY_COMPLEX
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

# using RETR_EXTERNAL instead of RETR_CCOMP
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours[0])


mask = np.zeros(bw.shape, dtype=np.uint8)

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.45 and w > 8 and h > 8:
        cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        detected_img = rgb_orig[y:y+h, x:x+w]
        pkl.dump(detected_img, open(f"{idx}_detect", "wb" )) 


# for p in box:
#   pt = (p[0],p[1])
#   print(pt)
#   cv2.circle(rgb,pt,5,(200,0,0),2)

rc= cv2.minAreaRect(contours[5])  
box = cv2.boxPoints(rc)

cv2.imshow('rect',rgb)
cv2.waitKey(0)




for i in range(1,len(contours)):
    try:
    	path = '/Users/maheshjain/Desktop/Brain/Saved/'
    	one_detect = pkl.load(open( f'{i}_detect', "rb" ))
    	cv2.imwrite(os.path.join(path , f'{i}_detect.jpg'),one_detect)
    	cv2.imshow('test',one_detect)
    	img_cv = cv2.imread(os.path.join(path , f'{i}_detect.jpg'))
    	img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    	grayImage = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    	text_final = pytesseract.image_to_string(grayImage)
    	text_final = re.sub(r'[^\w\s]','',text_final)
    	text_final = text_final.replace('_', '')
    	text_final=text_final.split()
    	main_text.append(text_final)
    	i= i+1
    	cv2.waitKey(1)


        

    except:
        pass

output = []



def reemovNestings(main_text):
    for i in main_text:
        if type(i) == list:
            reemovNestings(i)
        else:
            output.append(i)
  
reemovNestings(main_text)



spell = SpellChecker()
spell.word_frequency.load_words(['remdesivir', 'mgvial', 'lyophilized','mg','ml','singledose','imylan','mylan','heaven'])
misspelled = spell.unknown(output)
#print(output)


if len(misspelled) == 0:
    final_count.append('1')
    warning.append('No spelling errors were found')
else:
    final_count.append('0')
    warning.append('Spelling error were found on the package')
    misspelled1.append(misspelled)
    print("These items are misspelled", misspelled1)




# color detection module

sys_arg = str(sys.argv[1])

covifor = [[132, 100, 100],[152, 255, 255]]
desrem = [[80, 100, 100],[100, 255, 255]]



for text in output:
    if text == "covifor" or sys_arg=="covifor":
    	lowerBound=np.array(covifor[0])
    	upperBound=np.array(covifor[1])

    elif text == 'desrem' or sys_arg=="desrem":
        lowerBound=np.array(desrem[0])
        upperBound=np.array(desrem[1])
    else:
        lowerBound=np.array(desrem[0])
        upperBound=np.array(desrem[1])



kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

font = cv2.FONT_HERSHEY_DUPLEX

img=cv2.resize(large,(250,550))

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
    warning.append("This is fake: Color Not Similar to the Original")
    final_count.append('0')

else:
    warning.append("Color Similar to the Original")
    final_count.append('1')



final_count_length = len(final_count)
number = final_count.count("1")

percentage_certainty = (number/final_count_length)*100



print("Percetage Certainty of Originality:", percentage_certainty)
print(warning) 





