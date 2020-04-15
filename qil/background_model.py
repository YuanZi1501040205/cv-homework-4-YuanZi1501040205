# Import python libraries
import numpy as np
import cv2

class BGModel(object):
    """ This class defines background modeling and computation of foreground objects
    Attributes:
        None
    """

    def __init__(self, bgmodel):
        """Initialize variables necessary for different models for background modeling
        Args:
            bgmodel is a string that specifies the background modeling approach
            'mean' for mean
            'median for median
            'gaussian' for Gaussian
            'mog' for Mixture of Gaussians, and
            'knn' for K-Nearest Neighbor
        Return:
            None
        """
        # Start implementation here

    def compute_fgmask(self, frame):
        """Method that implements calculation of the foreground objects
        Args:
            frame is the current input image from the video
            frame is expected to be gray scale since all approaches in this class are
            using gray scale images for background modeling
        Return:
            foreground image
            foreground image should be gray scale images with higher pixel values to represent foreground objects
            """
        # Start implementation here and make sure to return foreground image


