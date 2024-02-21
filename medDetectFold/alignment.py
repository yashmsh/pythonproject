import cv2
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import os
import pickle as pkl
import pytesseract 
from spellchecker import SpellChecker
import re 
from imutils import perspective


pt1=[]
final = []

large = cv2.imread('5.png')
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


#area = cv2.contourArea(contours)



mask = np.zeros(bw.shape, dtype=np.uint8)

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.45 and w > 8 and h > 8:
        cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

for j in range(0,len(contours)):
    abc = cv2.contourArea(contours[j])
    if abc > 2000:
      rc= cv2.minAreaRect(contours[j])
      box = cv2.boxPoints(rc)
      box = perspective.order_points(box) 
      for p in box:
          pt = (p[0],p[1])
          cv2.circle(rgb,pt,5,(200,0,0),2)
          tb = (p[0])
          pt1.append(tb)

main_coord = pt1[::4]
res = [ele for ele in main_coord if ele > 0]


a1 = res[0]
b = a1 + 5
c = a1 - 5

res1 = all(c < ele1 < b for ele1 in res)
print(res1)



cv2.imshow("rgb",rgb)
cv2.waitKey(0)

