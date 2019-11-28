# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:03:56 2019

@author: shree111
"""

# import numpy as np
from keras.models import model_from_json
import cv2
import operator
import pyautogui

IMG_SIZE = 64

# pyautogui
screenWidth, screenHeight = pyautogui.size()
pyautogui.moveTo(screenWidth/2, screenHeight/2)
    
def performTask(action):
    move_num = 10
    scroll = 40
    currentMouseX, currentMouseY = pyautogui.position()
    count_action  = 0
    if action == "MOVE LEFT":
        pyautogui.press('left') 
        #pyautogui.moveTo(currentMouseX - move_num, currentMouseY)
    
    elif action == "MOVE RIGHT":
        pyautogui.press('right') 
        #pyautogui.moveTo(currentMouseX + move_num, currentMouseY)
          
    elif action == "ZOOM OUT":
        pyautogui.scroll(-scroll)
        #pyautogui.moveTo(currentMouseX , currentMouseY + move_num)
    
    elif action == "ZOOM IN":
        pyautogui.scroll(scroll)
        #pyautogui.moveTo(currentMouseX, currentMouseY - move_num)
    
        
    print(action)

# Loading the model
json_file = open("model-bw-6-class.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw-6-class.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories = {0: 'ZOOM OUT', 1: 'ZOOM IN', 2: 'MOVE LEFT', 3: 'MOVE RIGHT', 4:'MOVE UP', 5: 'MOVE DOWN'}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    #print(frame.shape)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)) 
    roi2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi2, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("output", test_image)
    
    # Batch of 1
    result = loaded_model.predict(roi.reshape(1, IMG_SIZE, IMG_SIZE, 3))
    prediction = {'ZOOM OUT': result[0][0], 
                  'ZOOM IN': result[0][1], 
                  'MOVE LEFT': result[0][2],
                  'MOVE RIGHT': result[0][3],
                  'MOVE UP': result[0][4],
                  'MOVE DOWN': result[0][5],
                  }
    
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    print(prediction[0][0])
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)    
    cv2.imshow("Real Frame", frame)
    performTask(str(prediction[0][0]))
    
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()