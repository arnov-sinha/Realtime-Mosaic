"""
This file contains the class used to mosaic two images together.

@author: Arnov
"""
import cv2

class motion:
    def __init__(self, thresh=5, frameWeight=0.5, area=5000):
        self.thresh = thresh
        self.frameWeight = frameWeight
        self.area = area
        self.avg = None

    def motionLocator(self, image):
        '''
        This function is used to find the locations of motion
        Parameters: image: frame in question
        '''
        loc = []
        if self.avg == None:
            self.avg = image.astype("float")
            return loc

        cv2.accumulateWeighted(image, self.avg, self.frameWeight)
        delta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))
        currentThresh = cv2.threshold(delta, self.thresh, 255, cv2.THRESH_BINARY)[1]
        currentThresh = cv2.dilate(currentThresh, None, iterations=2)

        contour = cv2.findContours(currentThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[1]

        for i in contour:
            if (cv2.contourArea(i) > self.area):
                loc.append(i)
        return loc
