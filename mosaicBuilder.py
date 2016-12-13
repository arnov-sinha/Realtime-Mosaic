"""
This file contains the class used to mosaic two images together.

@author: Arnov
"""
import numpy
import cv2

class mosaic:
    def __init__(self):
        self.initH = None
    '''
    contains the methods for blending two images
    '''

    def findFeatures(self, img):
		'''
        Finds features in the image using DOG and finds interest points by \
        detecting gaussian blobs
        '''
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		detector = cv2.FeatureDetector_create("SIFT") # Can use FAST or MSER
		kps = detector.detect(gray)

		extractor = cv2.DescriptorExtractor_create("SIFT") #SIFT or SURF
		(kps, features) = extractor.compute(gray, kps)

		kps = numpy.float32([i.pt for i in kps])

		return (kps, features)

    def findHomography(self, dA,dB, featuresA,featuresB, ratio, thresh):
        '''
        This function matches descriptors and returns the homography.
        dA,dB: set of descriptor keypoints
        featuresA,featuresB: features
        ratio: Lowes ratio matching value; not needed if Flann is used or
        with Bruteforce method if crossCheck=True in the second parameter
        thresh: RANSAC threshold
        '''
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        initial = matcher.knnMatch(featuresA, featuresB, 2)# k no of points
        final = []

        for i in initial:
            #Lowes ratio test
            if len(i) == 2 and (i[0].distance < i[1].distance * ratio):
				final.append((i[0].trainIdx, i[0].queryIdx)) #not sure

        if len(final) > 4: #homography calculator
            a = numpy.float32([dA[i] for (_ ,i) in final])
            b = numpy.float32([dB[i] for (i, _) in final])

            (H,status) = cv2.findHomography(a,b,cv2.RANSAC,thresh)

            return (final, H, status)
        print("Homography could not be calculated as the no. of descriptor \
        points is less than the minimum")
        return None

    def pack(self, image1, image2):
        '''
        Blends and pack the two images into one
        parameters:
        image1 is the left image
        image2 is the right image
        thresh is the RANSAC threshold
        '''
        ratio=0.75 #D.Lowe's ratio for feature matching
        thresh=4.0

        if self.initH is None:
			dA, featuresA = self.findFeatures(image1)
			dB, featuresB = self.findFeatures(image2)

			match = self.findHomography(dA, dB, featuresA, featuresB, ratio, thresh)

			if match is None:
				return None

			self.initH = match[1]
        result = cv2.warpPerspective(image1, self.initH,(image1.shape[1] + image2.shape[1], image1.shape[0]))
        result[0:image2.shape[0], 0:image2.shape[1]] = image2

        return result
