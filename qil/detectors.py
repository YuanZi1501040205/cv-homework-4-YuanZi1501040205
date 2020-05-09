# Import python libraries
import numpy as np
import cv2
from qil.background_model import BGModel
import copy

class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self, bgmodel, debug):
        """Initialize variables used by Detectors class
        Args:
            bgmodel is a string that specifies the background modeling approach
            'mean' for mean
            'median for median
            'gaussian' for Gaussian
            'mog' for Mixture of Gaussians, and
            'knn' for K-Nearest Neighbor
            debug is a flag (0 or 1) to be used to display/visualize results
            set debug value to 1 to see output images
        Return:
            None
        """
        self.debug = debug
        self.bgmodel = BGModel(bgmodel)
        self.previous_frames = []

        # Start implementation here

    def Detect(self, frame):
        """Detect objects in video frame
            Please note that this should not require any classifier-based object detection
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """
        # Start implementation here and make sure to return vector of object centroids in a frame
        frame = copy.copy(frame)
        fgMask = BGModel.compute_fgmask(self, frame)

        # morphological operation to erase the noise resulted by background subtraction
        # kernel = np.ones((1, 1), np.uint8)
        # fgMask = cv2.erode(fgMask, kernel, iterations=2)
        # kernel = np.ones((2, 2), np.uint8)
        # fgMask = cv2.dilate(fgMask, kernel, iterations=)
        #convert fgMas to CV_8UC1 format! very important otherwise can not use find contours function
        fgMask = cv2.normalize(src=fgMask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # Find contours, sort, get the largest contours convert them to bounding box and calculate their centroids
        contours, hierarchy = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        target_contours = []
        bbx = []
        centers = []
        # get the largest 10 contours for 6 people in the video if contours less than 10 then get the all contours
        # limit track max ability to 10 objects
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(sorted_contours) < 10: # ability arguments
            n = len(sorted_contours)
        else:
            n = 10
        # n = len(sorted_contours)
        for i in range(n):
            c = sorted_contours[i]
            target_contours.append(c)
            bbx.append(cv2.boundingRect(c))
            x, y, w, h = bbx[-1]
            fgMask = cv2.rectangle(fgMask, (x, y), (x + w, y + h), (125, 125, 125), 2)
            centers.append([[int(x + 0.5 * w)], [int(y + 0.5 * h)]])
        # cv2.imshow('KNN_detection_result', fgMask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return(centers)
