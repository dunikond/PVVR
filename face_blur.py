import numpy as np
import cv2
import matplotlib.pyplot as plt
  
image = 'faces/000000001296.jpg' 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #nacteni xml souboru s daty pro rozpoznani obliceje 

# nacteni obrazku v opencv, BGR 
image = cv2.imread(image)
  
# prevod BGR do grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# vykresledni puvodni fotky

face_data = face_cascade.detectMultiScale(gray_image, 1.3, 5) 
  
# vyklesleni obdelniku okolo oblasti zajmu (roi)
for (x, y, w, h) in face_data:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) #vykresleni ctverce o ro
    roi = image[y:y+h, x:x+w]
    # aplikace filtru 
    roi = cv2.GaussianBlur(roi, (23, 23), 30)
    # vlozeni rozmazaneho obrazku do originalniho obrazku
    image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
  
  
# vykresleni vysledku
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.style.use('seaborn')
    plt.show()

a = 8
