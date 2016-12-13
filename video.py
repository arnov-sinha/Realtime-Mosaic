"""
This file is used for capturing video from webcams.

@author: Arnov
"""

from threading import Thread
import cv2

class video:
	def __init__(self, src=0):
		'''
		initialize the video camera stream
		'''
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		'''
		initialize the variable used to indicate if the thread should
		be stopped
		'''
		self.stopped = False

	def start(self):
		'''
		start reading from the video stream
		'''
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		'''
		keeps the frames updated for the stream
		'''
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		'''
		return most recently read frame
		'''
		return self.frame
