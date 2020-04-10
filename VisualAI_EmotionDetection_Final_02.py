#!/usr/bin/env python
try:
	import tkinter
	from tkinter import ttk
	from tkinter import *
except ImportError:
	import Tkinter
	from Tkinter import ttk
	from Tkinter import *

import cv2
import PIL.Image, PIL.ImageTk

import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# load model
model = model_from_json(open("fer.json", "r").read())
# load weights
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class App:
	def __init__(self, window, window_title, video_source=0):
		self.window = window
		self.window.title(window_title)
		self.video_source = video_source
		
		# open video source (by default this will try to open the computer webcam)
		self.vid = MyVideoCapture(self.video_source)
				
		frame0 = Frame(self.window, width=800, height=600, bd=1)
		frame0.pack()

		frame1 = Frame(frame0, bd=2, relief=RAISED)
		frame1.pack(expand=1, fill=X, pady=10, padx=5)
		
		canvas1 = Canvas(frame1, bg='yellow', width=800, height=20)
		canvas1.pack()
		
		self.canvas = tkinter.Canvas(frame1, width=800, height=600)
		self.canvas.pack()
					
		canvas1.create_text(400, 10, text='NonLutte - Facial Expression Recognition App', font=('verdana', 20, 'bold'))
		frame1.pack(expand=1, fill=X, pady=10, padx=5)		

# 		
# 		# Create a canvas that can fit the above video source size
# 		#self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
# 		self.canvas = tkinter.Canvas(window, width = 800, height = 600)
		
		btn = tkinter.Button(self.window, text="Close", command=self.window.destroy)
		btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)			
		
		self.pb = ttk.Progressbar(self.window, orient="horizontal", length=750, mode="determinate", value=0)
		self.pb.pack()	
		
		# After it is called once, the update method will be automatically called every delay milliseconds
		self.delay = 15
		self.update()
		
		self.window.mainloop()
	
	def update(self):
		# Get a frame from the video source
		ret, frame = self.vid.get_expression()
		
		if ret:
			self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
			self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

			self.pb['value'] = float(np.random.randint(0, 100 + 1))
		
		self.window.after(self.delay, self.update)
		

class MyVideoCapture:
	
	def __init__(self, video_source=0):
		# Open the video source
		self.vid = cv2.VideoCapture(video_source)
		
		if not self.vid.isOpened():
			raise ValueError("Unable to open video source", video_source)
	
		# Get video source width and height
		self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
	

	def get_expression(self):
		while True:

			cap = cv2.VideoCapture(0)
			ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
		    
			if not ret:
				continue
			
			gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
		
			faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
		
			for (x, y, w, h) in faces_detected:
				cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
				roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
				roi_gray = cv2.resize(roi_gray, (48, 48))
				img_pixels = image.img_to_array(roi_gray)
				img_pixels = np.expand_dims(img_pixels, axis=0)
				img_pixels /= 255
		
				predictions = model.predict(img_pixels)
		
		        # find max indexed array
				max_index = np.argmax(predictions[0])						
				
				#self.cv2.create_text(400, 10, text=max_index, font=('verdana', 20, 'bold'))

				emotions = ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
				predicted_emotion = emotions[max_index]
		
				cv2.putText(test_img, predicted_emotion, (int(x+20), int(y-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		
			resized_img = cv2.resize(test_img, (800, 600))
			return (ret, cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))	
		
	
	# Release the video source when the object is destroyed
	def __del__(self):
		if self.vid.isOpened():
			self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "NonLutte - Facial Expression Recognition App")





