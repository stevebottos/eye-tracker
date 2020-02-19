"""
To set up the environment, run these commands (through Anaconda terminal):

conda create -n EyeGaze python=3.6
pip install numpy scipy matplotlib scikit-learn cmake
conda install -c conda-forge opencv
pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f


NOTES:
    1) I'd like to try to optimize this so that it can report quicker
"""

import cv2
import dlib
import numpy as np
import time


# This is just to pass to the slider since it needs a function
def nothing(x):
    pass

def getPoints(gray, face):
    pts = predictor(gray, face)

    # I'll bring in the points on the edges of the eyes so that there's more whitespace
    rightpts = np.array([(pts.part(36).x, pts.part(36).y),
                         (pts.part(37).x, pts.part(37).y),
                         (pts.part(38).x, pts.part(38).y),
                         (pts.part(39).x, pts.part(39).y),
                         (pts.part(40).x, pts.part(40).y),
                         (pts.part(41).x, pts.part(41).y)], np.int32)

    leftpts = np.array([(pts.part(42).x, pts.part(42).y),
                        (pts.part(43).x, pts.part(43).y),
                        (pts.part(44).x, pts.part(44).y),
                        (pts.part(45).x, pts.part(45).y),
                        (pts.part(46).x, pts.part(46).y),
                        (pts.part(47).x, pts.part(47).y)], np.int32)

    min_x_left = np.min(leftpts[:, 0]) - 30
    max_x_left = np.max(leftpts[:, 0]) + 30
    min_y_left = np.min(leftpts[:, 1]) - 30
    max_y_left = np.max(leftpts[:, 1]) + 30
    leftbox = [min_y_left,max_y_left, min_x_left,max_x_left]

    min_x_right = np.min(rightpts[:, 0]) - 30
    max_x_right = np.max(rightpts[:, 0]) + 30
    min_y_right = np.min(rightpts[:, 1]) - 30
    max_y_right = np.max(rightpts[:, 1]) + 30
    rightbox = [min_y_right,max_y_right, min_x_right,max_x_right]

    return leftpts, leftbox, rightpts, rightbox

def getProcessingCaptures(gray, leftpts, rightpts):

    mask = np.zeros((gray.shape), np.uint8)
    cv2.polylines(mask, [leftpts, rightpts], True, 255, 2)
    cv2.fillPoly(mask, [leftpts, rightpts], 255)
    eyes_only = cv2.bitwise_and(gray, gray, mask=mask)

    # change black to white
    eyes_only[eyes_only == 0] = 255

    return eyes_only

def trimImage(img, lb, rb):


    left = img[lb[0]:lb[1], lb[2]:lb[3]]
    right = img[rb[0]: rb[1], rb[2]: rb[3]]

    return left, right

# Midpoint coordinates for the images
laptop = 1
if laptop == 1:
    height = 768
    width = 1366
else:
    height = 1080
    width = 1920

midline = int(height/2)
centerline = int(width/2)

# Initializer values
cap = cv2.VideoCapture(1)  # 0 = laptop webcam, 1 = logitech cam
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# The windows for the sliders
cv2.namedWindow('Left Eye Processed')
cv2.createTrackbar('Left Eye Threshold', 'Left Eye Processed', 0, 255, nothing)
cv2.createTrackbar('Left Eye Erode', 'Left Eye Processed', 0, 10, nothing)
cv2.createTrackbar('Left Eye Dilate', 'Left Eye Processed', 0, 10, nothing)

cv2.namedWindow('Right Eye Processed')
cv2.createTrackbar('Right Eye Threshold', 'Right Eye Processed', 0, 255, nothing)
cv2.createTrackbar('Right Eye Threshold', 'Right Eye Processed', 0, 255, nothing)
cv2.createTrackbar('Right Eye Erode', 'Right Eye Processed', 0, 10, nothing)
cv2.createTrackbar('Right Eye Dilate', 'Right Eye Processed', 0, 10, nothing)


# Blob detector initialization
detector_params = cv2.SimpleBlobDetector_Params()

# Tell the detector to throw away any small regions
detector_params.filterByArea = True
detector_params.minArea = 100
detector_params.minThreshold = 10
detector_params.maxThreshold = 200
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

while True:
    # Timing the program to make sure we're not getting too complex for tracking purposes
    start = time.time()
    # Capture the full camera image
    _, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detects the number of faces in the shot, might not need since this is designed for one person
    faces = face_detector(gray)

    # For every face detected... (going to optimize for just one)
    for face in faces:

        # Get the points to draw around the eye and the dimensions of the preview boxes
        leftpts, leftbox, rightpts, rightbox = getPoints(gray, face)
        eyes_only = getProcessingCaptures(gray, leftpts, rightpts)
        lefteye, righteye = trimImage(gray, leftbox, rightbox)
        lefteye_only, righteye_only = trimImage(eyes_only, leftbox, rightbox)

        # LEFT EYE CALIBRATION PARAMS
        left_thresh = cv2.getTrackbarPos('Left Eye Threshold', 'Left Eye Processed')
        left_erode_iter = cv2.getTrackbarPos('Left Eye Erode', 'Left Eye Processed')
        left_dilate_iter = cv2.getTrackbarPos('Left Eye Dilate', 'Left Eye Processed')
        # left_blur = cv2.getTrackbarPos('Left Eye Blur', 'Left Eye Processed')
        _, left_processed = cv2.threshold(lefteye_only, left_thresh, 255, cv2.THRESH_BINARY)
        left_processed = cv2.erode(left_processed, None, iterations=left_erode_iter)
        left_processed = cv2.dilate(left_processed, None, iterations=left_dilate_iter)
        left_processed = cv2.medianBlur(left_processed, 5)

        # RIGHT EYE CALIBRATION PARAMS
        right_thresh = cv2.getTrackbarPos('Right Eye Threshold', 'Right Eye Processed')
        right_erode_iter = cv2.getTrackbarPos('Right Eye Erode', 'Right Eye Processed')
        right_dilate_iter = cv2.getTrackbarPos('Right Eye Dilate', 'Right Eye Processed')
        # right_blur = cv2.getTrackbarPos('Right Eye Blur', 'Right Eye Processed')
        _, right_processed = cv2.threshold(righteye_only, right_thresh, 255, cv2.THRESH_BINARY)
        right_processed = cv2.erode(right_processed, None, iterations=right_erode_iter)
        right_processed = cv2.dilate(right_processed, None, iterations=right_dilate_iter)
        right_processed = cv2.medianBlur(right_processed, 5)


        keypoints_left = blob_detector.detect(left_processed)
        keypoints_right = blob_detector.detect(right_processed)
        left_w_keypts = cv2.drawKeypoints(lefteye, keypoints_left, True, (0, 255, 0),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        right_w_keypts = cv2.drawKeypoints(righteye, keypoints_right, True, (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Show the images
        cv2.imshow("Left Eye", left_w_keypts)
        cv2.imshow("Right Eye", right_w_keypts)
        cv2.imshow("Left Eye Processed", left_processed)
        cv2.imshow("Right Eye Processed", right_processed)

        # Positioning
        cv2.moveWindow("Left Eye", centerline+40, midline-200)
        cv2.moveWindow("Right Eye", centerline-350, midline-200)
        cv2.moveWindow("Left Eye Processed", centerline+40, midline)
        cv2.moveWindow("Right Eye Processed", centerline-350, midline);

    key = cv2.waitKey(1)
    if key == 32:  # 32 is space key on my computer.
        break

    end = time.time()
    print(end-start)

cap.release()
cv2.destroyAllWindows()