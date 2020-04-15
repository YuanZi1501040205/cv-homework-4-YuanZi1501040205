# CV_Homework_4
HomeWork-4: Object Detection and Tracking
Due Date: 11:59 PM, 6th May 2020

Object detection and tracking is a common requirement for a range of applications in computer vision.  In this assignment, we will leverage the broad understanding we have developed throughout this semester, specifically as it related to background modeling for foreground object detection, Kalman filtering for prediction of object location, and jointly, for object tracking.  

This homework assignment will focus on the design of an object tracking solution that is implemented under a 2 module framework; a module for detection of foreground objects, and a module for tracking individual detected objects while ensuring consistent object id association using distance as a metric. 

You are given a solution template that includes the main function that implements the overall object tracking solution as shown above.  In addition, the Track module is also available that already includes implementation of the Kalman filter that uses the centroid of detected objects as its state parameter.  The focus of this assignment will be on implementing the Detect module.  In doing so, you will also be required to implement appropriate Background Modeling approach to perform Background Subtraction and Foreground Object Detection.  

Specific assignment steps are:
1.	In the given BGModel() class (background_model.py), you are to implement the following background subtraction methods:
  a.	Background model estimated by computing the mean value of observed image intensity for each pixel
  b.	Background model estimated by computing the median of observed image intensity for each pixel
  c.	Background model estimated by modeling each pixel’s intensity as a Gaussian (mean and variance) 
  d.	Background model estimated by modeling each pixel’s intensity as a Mixture of Gaussians (can use cv2.createBackgroundSubtractorMOG2())
  e.	Background model estimated by modeling each pixel’s intensity based on Nearest Neighbor estimate of observed intensities (can use cv2. createBackgroundSubtractorKNN())

2.	Note that the specific computation of the foreground image after background subtraction should be performed in the compute_fgmask() method

3.	In the given Detectors() class (detectors.py), you are to implement approach to detect objects for tracking and their centroid values.  The detector should allow for use of one of the background subtraction methods and get the foreground image.  This should be further processed to detect objects worthy of tracking.  Your choice of processing to detect objects and compute their centroids should be implemented in the Detect() method within the class. (Please note that this does not require classifier based detection of objects)

4.	Write a report outlining your approach for background subtraction and object detection.

Instruction:
1.	Submit code on Github
2.	All coding must be done using python
3.	Use tools available in OpenCV.

Submission Instruction:
1.	Submit only the source code files.
2.	The TA will have the OpenCV setup in his development environment and will paste your code and run it. Do not use any third-party libraries as the TA will not have access to it. 
Include a readme file to describe any required set up, like location of image files that was hard coded, or any other information that is required to run the code
