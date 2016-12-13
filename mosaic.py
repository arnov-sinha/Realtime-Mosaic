"""
This file does the actual mosaicing of the two streams.

@author: Arnov
"""

import video
from mosaicBuilder import mosaic
from motion import motion
import imutils
import time
import cv2




leftVideo = video.video(src=0).start() #The webcam attached to the laptop
rightVideo = video.video(src=1).start() #The inbuilt camera of the laptop

panImage = mosaic()
motion = motion(area=500)

time.sleep(2.0)

while True:
    l = leftVideo.read()
    r = rightVideo.read()
    left = imutils.resize(l, width=400)
    right = imutils.resize(r, width=400)

    result = panImage.pack(left, right)

    if result is None:
        print("Something went wrong, no results :(")
        break

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    smoothGray = cv2.GaussianBlur(gray, (21, 21), 0)
    loc = motion.motionLocator(smoothGray)

    cv2.imshow("Result", result)
    cv2.imshow("Left Frame", left)
    cv2.imshow("Right Frame", right)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
