# Import python libraries
import numpy as np
import cv2
from scipy.stats import norm

global previous_frames # variable to store the previous frames for background model calculation

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
        if bgmodel == "mean":
            backSub = createBackgroundSubtractorMEAN()
        elif bgmodel == "median":
            backSub = createBackgroundSubtractorMEDIAN()
        elif bgmodel == "gaussian":
            backSub = createBackgroundSubtractorGAUSSIAN()
        elif bgmodel == "mog":
            backSub = cv2.createBackgroundSubtractorMOG2()
        elif bgmodel == "knn":
            backSub = cv2.createBackgroundSubtractorKNN()
        else:
            print("Background model assign Error!")
            pass



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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgMask = BGModel.backSub.apply(frame)
        return(fgMask)

def createBackgroundSubtractorMEAN():
    def apply(frame):
        """Method that implements calculation of the foreground objects with Mean filter method
        Args:
            n is the number of previous frames to calculate the background
            threshold is the threshold of difference to determine if the pixel belong to foreground
        Return:
            foreground image
            foreground image should be gray scale images with higher pixel values 255 to represent foreground objects
            """
        # tuning args
        n = 10 #the number of previous frames to calculate the background
        threshold = 10 #set the threshold of difference to determine if the pixel belong to foreground

        w, h = frame.shape[0], frame.shape[1]
        frame = frame.astype(int)
        previous_frames = []

        # read the previous frames and store them in a n query
        if np.shape(previous_frames)[0] < n:
            previous_frames.append(frame)
        elif np.shape(previous_frames)[0] == n:
             previous_frames.pop(0)
             previous_frames.append(frame)
        else:
            print("Previous frames storage overflow!")
            pass

        # calculate the background with mean filter
        if np.shape(previous_frames)[0] < n:
            bg = np.zeros([w, h])
        elif np.shape(previous_frames)[0] == n:
            bg = np.zeros([w, h])
            for i in range(n):
                bg = bg + np.array(previous_frames)[i]
            bg = bg / n

        # compare the frame and background to calculate the foreground with threshold
        fgMask = np.zeros([w, h])
        for i in range(w):
            for j in range(h):
                if abs(frame[i][j] - bg[i][j]) > threshold:
                    fgMask[i][j] = 255
                else:
                    fgMask[i][j] = 0

        return(fgMask)



def createBackgroundSubtractorMEDIAN():
    def apply(frame):
        """Method that implements calculation of the foreground objects with Median filter method
        Args:
            n is the number of previous frames to calculate the background
            threshold is the threshold of difference to determine if the pixel belong to foreground
        Return:
            foreground image
            foreground image should be gray scale images with higher pixel values 255 to represent foreground objects
            """
        # tuning args
        n = 10 #the number of previous frames to calculate the background
        threshold = 10 #set the threshold of difference to determine if the pixel belong to foreground

        w, h = frame.shape[0], frame.shape[1]
        frame = frame.astype(int)
        previous_frames = []

        # read the previous frames and store them in a n query
        if np.shape(previous_frames)[0] < n:
            previous_frames.append(frame)
        elif np.shape(previous_frames)[0] == n:
             previous_frames.pop(0)
             previous_frames.append(frame)
        else:
            print("Previous frames storage overflow!")
            pass

        # calculate the background with median filter
        if np.shape(previous_frames)[0] < n:
            bg = np.zeros([w, h])
        elif np.shape(previous_frames)[0] == n:
            bg = np.zeros([w, h])
            for i in range(w):
                for j in range(h):
                    median_query = []
                    for k in range(n):
                        median_query.append(previous_frames[k][i][j])
                    bg[i][j] = np.median(median_query)

        # compare the frame and background to calculate the foreground with threshold
        fgMask = np.zeros([w, h])
        for i in range(w):
            for j in range(h):
                if abs(frame[i][j] - bg[i][j]) > threshold:
                    fgMask[i][j] = 255
                else:
                    fgMask[i][j] = 0

        return(fgMask)

def createBackgroundSubtractorGAUSSIAN():
    def apply(frame):
        w, h = frame.shape[0], frame.shape[1]
        # read previous frames
        previous_frames = []
        frame_count = 0
        cap = cv2.VideoCapture("QIL_orig.mp4")
        while (True):
            # Capture frame-by-frame
            ret, inside_frame = cap.read()
            if ret:
                inside_frame = cv2.cvtColor(inside_frame, cv2.COLOR_BGR2GRAY)
                previous_frames.append(inside_frame)
                frame_count += 1
                # Display the original frame
                cv2.imshow('Original', inside_frame)

                # Slower the FPS
                cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        # fit gaussian for each pixel and compare each pixel with the gaussian distribution, if frame's difference associate with the mean value over the 1 standard deviation consider it as object
        fgMask = np.zeros([w, h])
        for i in range(w):
            for j in range(h):
                pixel = []
                for k in range(frame_count):
                    pixel.append(previous_frames[k][i][j])
                mu, std = norm.fit(pixel)
                threshold = std # For the normal distribution, the values less than one standard deviation away from the mean account for 68.27% of the set;
                if abs(frame[i][j] - mu) > threshold:
                    fgMask[i][j] = 255

        return(fgMask)

