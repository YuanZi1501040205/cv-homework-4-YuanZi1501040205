# Import python libraries
import numpy as np
import cv2
import copy
from qil.detectors import Detectors
from qil.tracker import Tracker
import argparse


def main():
    """Main function for multi object tracking
    Args:
        -v name of video file
        -b name of background model to be used for object detection
        The background models to be made available include mean, median,
        Gaussian, mixture of Gaussians, and K-Nearest Neighbor
        The specific string to be used to specify the background modeling approach are
        'mean' for mean
        'median for median
        'gaussian' for Gaussian
        'mog' for Mixture of Gaussians, and
        'knn' for K-Nearest Neighbor
    Return:
        None
    Example usage:
        python cv_hw4.py -v QIL_orig.mp4 -b knn -d 0
    """

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
                    help="name of input video file")
    ap.add_argument("-b", "--bgmodel", required=True,
                    help="type of background modeling (mean, median, gaussian, mog, or knn")
    ap.add_argument("-d", "--debug", type=int, default=0,
                    help="debug flag (set to 1 to display images)")

    args = vars(ap.parse_args())

    # Debug value to decide if images are displayed or not
    debug = args["debug"]

    # Create opencv video capture object
    cap = cv2.VideoCapture(args["video"])

    # Create Object Detector
    detector = Detectors(args["bgmodel"],debug)

    # Create Object Tracker
    # dist_thresh: 15 - Search within this distance to find match
    # max_frames_to_skip: 30 - delete objects with cannot match for these number of frames
    # max_trace_length: 1000 - maximum length of a trajectory
    # trackIdCount: 1 - starting count for objects
    tracker = Tracker(15, 30, 1000, 1)

    # Variables initialization
    skip_frame_count = 0
    track_colors = np.random.uniform(0, 255, size=(10000, 3))
    frame_count = 0
    min_track_length = 50

    pause = False

    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            # Make copy of original frame
            orig_frame = copy.copy(frame)
            last_frame = copy.copy(frame)
            if frame != None:
                print("frame ! = None")
            # Detect and return centroids of the objects in the frame
            centers = detector.Detect(frame)

            # If centroids are detected then track them
            if (len(centers) > 0):

                # Track object using Kalman Filter
                tracker.Update(centers)

                # For identified object tracks draw tracking line if object is tracked for
                # longer then min_track_length frames
                # Use various colors to indicate different track_id
                for i in range(len(tracker.tracks)):
                    if (len(tracker.tracks[i].trace) > min_track_length):
                        for j in range(len(tracker.tracks[i].trace)-1):
                            # Draw trace line
                            x1 = tracker.tracks[i].trace[j][0][0]
                            y1 = tracker.tracks[i].trace[j][1][0]
                            x2 = tracker.tracks[i].trace[j+1][0][0]
                            y2 = tracker.tracks[i].trace[j+1][1][0]
                            clr = tracker.tracks[i].track_id % 9999
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                     track_colors[clr], 2)

                # Display the resulting tracking frame
                if debug == 1:
                    cv2.imshow('Tracking', frame)

            # Display the original frame
            if debug == 1:
                cv2.imshow('Original', orig_frame)

            # Slower the FPS
            cv2.waitKey(1)

            # Check for key strokes
            k = cv2.waitKey(1) & 0xff

            # if the `q` key was pressed, break from the loop
            if k == ord("q"):
                break
            if k == 112:  # 'p' has been pressed. this will pause/resume the code.
                pause = not pause
                if (pause is True):
                    print("Code is paused. Press 'p' to resume..")
                    print("Object ID: {}: {}".format(tracker.tracks[i].track_id, len(tracker.tracks[i].trace)))
                    while (pause is True):
                        # stay in this loop until
                        key = cv2.waitKey(30) & 0xff
                        if key == 112:
                            pause = False
                            print("Resume code..!!")
                            break
        else:
            # For identified object tracks draw tracking line if object is tracked for
            # longer then min_track_length frames
            # Use various colors to indicate different track_id
            tracked_object_count = 0
            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > min_track_length):
                    tracked_object_count += 1
                    for j in range(len(tracker.tracks[i].trace) - 1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0][0]
                        y1 = tracker.tracks[i].trace[j][1][0]
                        x2 = tracker.tracks[i].trace[j + 1][0][0]
                        y2 = tracker.tracks[i].trace[j + 1][1][0]
                        clr = tracker.tracks[i].track_id % 9999
                        cv2.line(last_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)

            text1 = "Total Objects Detected and Tracked: {}".format(tracked_object_count)
            cv2.putText(last_frame, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            text2 = "Total Video Frames: {}".format(frame_count)
            cv2.putText(last_frame, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            text3 = "Background Model: {}".format(args["bgmodel"])
            cv2.putText(last_frame, text3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Write the result image
            result_file = "output/TrackingResult-"+args["bgmodel"]+".png"
            cv2.imwrite(result_file, last_frame)
            print(text1+"\n"+text2+"\n"+text3)

            # Display the resulting tracking frame
            if debug == 1:
                cv2.imshow('All Tracked Objects', last_frame)
                cv2.waitKey(0)
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()