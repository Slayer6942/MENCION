import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascade.xml')
img = cv2.imread('Imagen_2.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.35,1)

