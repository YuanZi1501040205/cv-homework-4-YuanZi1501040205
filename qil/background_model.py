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
        self.bgmodel = bgmodel
        self.previous_frames = []

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
        model_flag = self.bgmodel.bgmodel
        if model_flag == "mean":
            fgMask = createBackgroundSubtractorMEAN(frame, self.previous_frames)
        elif model_flag == "median":
            fgMask = createBackgroundSubtractorMEDIAN(frame, self.previous_frames)
        elif model_flag == "gaussian":
            fgMask = createBackgroundSubtractorGAUSSIAN(frame, self.previous_frames)
        elif model_flag == "mog":
            fgMask = cv2.createBackgroundSubtractorMOG2().apply(frame)
        elif model_flag == "knn":
            fgMask = cv2.createBackgroundSubtractorKNN().apply(frame)
        else:
            print("Background model assign Error!")
            pass
        return(fgMask)

def createBackgroundSubtractorMEAN(frame, previous_frames):
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



def createBackgroundSubtractorMEDIAN(frame, previous_frames):
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

def createBackgroundSubtractorGAUSSIAN(frame, previous_frames):
    w, h = frame.shape[0], frame.shape[1]
    # read previous frames

    # method A: use all frames in the video to build the background model
    # frame_count = 0
    # cap = cv2.VideoCapture("QIL_orig.mp4")
    # while (True):
    #     # Capture frame-by-frame
    #     ret, inside_frame = cap.read()
    #     if ret:
    #         inside_frame = cv2.cvtColor(inside_frame, cv2.COLOR_BGR2GRAY)
    #         previous_frames.append(inside_frame)
    #         frame_count += 1
    #         # Display the original frame
    #         # cv2.imshow('Original', inside_frame)
    #         #
    #         # # Slower the FPS
    #         # cv2.waitKey(1)
    #     else:
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    # method two, use the n previous frames to build the background model
    n = 10 # the number of previous frames to calculate the background
    frame_count = n
    # read the previous frames and store them in a n query
    if np.shape(previous_frames)[0] < n:
        previous_frames.append(frame)
    elif np.shape(previous_frames)[0] == n:
         previous_frames.pop(0)
         previous_frames.append(frame)
    else:
        print("Previous frames storage overflow!")
        pass
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

# # %%debug read all videos
# import numpy as np
# import cv2
# from scipy.stats import norm
# global previous_frames  # variable to store the previous frames for background model calculation
# previous_frames = []
# frame_count = 0
# cap = cv2.VideoCapture("QIL_orig.mp4")
# while (True):
#     # Capture frame-by-frame
#     ret, inside_frame = cap.read()
#     if ret:
#         inside_frame = cv2.cvtColor(inside_frame, cv2.COLOR_BGR2GRAY)
#         previous_frames.append(inside_frame)
#         frame_count += 1
#         # Display the original frame
#         cv2.imshow('Original', inside_frame)
#
#         # Slower the FPS
#         cv2.waitKey(1)
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()
# #%%
# n = 10
# i = 0
# previous_f = []
# frame = previous_frames[20]
# w, h = frame.shape[0], frame.shape[1]
# for i in range(20):
#     # read the previous frames and store them in a n query
#     if np.shape(previous_f)[0] < n:
#         previous_f.append(previous_frames[i])
#     elif np.shape(previous_f)[0] == n:
#          previous_f.pop(0)
#          previous_f.append(previous_frames[i])
#     else:
#         print("Previous frames storage overflow!")
#         pass
# # %%
# # calculate the background with median filter
# if np.shape(previous_f)[0] < n:
#     bg = np.zeros([w, h])
# elif np.shape(previous_f)[0] == n:
#     bg = np.zeros([w, h])
#     for i in range(w):
#         for j in range(h):
#             median_query = []
#             for k in range(n):
#                 median_query.append(previous_f[k][i][j])
#             bg[i][j] = np.median(median_query)
# # %%
# # compare the frame and background to calculate the foreground with threshold
# threshold = 16
# fgMask = np.zeros([w, h])
# for i in range(w):
#     for j in range(h):
#         if abs(frame[i][j] - bg[i][j]) > threshold:
#             fgMask[i][j] = 255
#         else:
#             fgMask[i][j] = 0
# # %%
# cv2.imshow('fgMask', fgMask); cv2.waitKey(0); cv2.destroyAllWindows()
# # %%
# kernel = np.ones((1, 1), np.uint8)
# fgMask = cv2.erode(fgMask, kernel, iterations=8)
# kernel = np.ones((2, 2), np.uint8)
# fgMask = cv2.dilate(fgMask, kernel, iterations=3)
# # %% bounding box
# # find the contours and calculate the bounding boxs for objects
# fgMask = cv2.normalize(src=fgMask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# # %%
# cv2.imshow('fgMask', fgMask); cv2.waitKey(0); cv2.destroyAllWindows()
#
# # %%
# contours, hierarchy = cv2.findContours(fgMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# target_contours = []
# bbx = []
# centers = []
# # get the largest 6 contours for 6 people in the video
# sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
# for i in range(6):
#     c = sorted_contours[i]
#     target_contours.append(c)
#     bbx.append(cv2.boundingRect(c))
#     x, y, w, h = bbx[-1]
#     fgMask = cv2.rectangle(fgMask, (x, y), (x + w, y + h), (125, 125, 125), 2)
#     centers.append([int(x + 0.5*w), int(y + 0.5*h)])
# cv2.imshow('fgMask', fgMask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("Number of Contours found = " + str(len(contours)))
# print("centroids " + str(centers))
# # %%