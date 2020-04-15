# Import python libraries
import numpy as np
import cv2
from qil.background_model import BGModel


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
        self.fgbg = BGModel(bgmodel)

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
