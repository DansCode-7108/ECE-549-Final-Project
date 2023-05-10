#!/usr/bin/env python3
"""
Script Name: image_match.py
Contributors: Daniel Lorenzo, Neet Maru
Last Modified: 5/8/2023
Reference Program URL: https://github.com/jayeshbhole/Sign-Detection-OpenCV/blob/master/fedup.py
Description: image_match.py was designed for the ECE 549 Advanced Robotics final project. The original script, written
by Jayesh Bhole, was designed to identify matches between signs captured in a video stream and a set of reference
images. However, the original program was not designed to work in the ROS environment.
Modifications: This redesigned program was adapted to the ROS environment using rospy, cv_bridge, and the Image and
String messages formats. Like other ROS nodes, this program can now support inter-process communication via
subscribers and publishers.
"""

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

bridge = CvBridge()

# Difference Variable
minDiff = 60000
minSquareArea = 5000
match = -1

# Frame width & Height
w = 640
h = 480

# Font Type
font = cv2.FONT_HERSHEY_SIMPLEX

signs_path = "/home/ece448_2018/catkin_ws/src/chapter5_tutorials/signs/"

# Reference Images Display name & Original Name
ReferenceImages = ["forward.png", "stop.png", "uturn.png", "left_arrow.png", "right_arrow.png"]
ReferenceTitles = ["forward.png", "stop.png", "uturn.png", "left_arrow.png", "right_arrow.png"]


# define class for References Images
class Symbol:
    """
    Class: Symbol
    Purpose: Container for reference image data
    Parameters: None
    """
    def __init__(self):
        self.img = 0
        self.name = 0

symbol = [Symbol() for i in range(len(ReferenceImages))]


def dist_dir(pts):
    """
    Function: dist_dir
    Purpose: Calculates distances between corners of sign and
    Parameters: pts (rectangle vertice coordinates)
    Returns: distance, direction ()
    """
    rect = order_points(pts)
    markerSize = rect[1][0] - rect[0][0]

    if markerSize == 0:
        distance = 0
    else:
        distance = int((10 * 1200) / markerSize)

    direction = 640 - ((rect[0][0] + rect[1][0] + rect[2][0] + rect[3][0]) // 4)
    return distance, direction


def readRefImages():
    """
    Function: readRefImages
    Purpose: Converts all reference images to cv2 objects
             Each cv2 object is saved in a Symbol object with its reference title
    Parameters: None
    Returns: None
    """
    for count in range(len(ReferenceImages)):
        image = cv2.imread(signs_path + ReferenceImages[count], cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        symbol[count].img = cv2.resize(image, (w//2, h//2), interpolation=cv2.INTER_AREA)
        symbol[count].name = ReferenceTitles[count]


def order_points(pts):
    """
    Function: order_points
    Purpose: Sorts the coordinates of each point in pts
    Parameters: pts
    Returns: rect (rectangle vertex coordinates in order: top-left, top-right, bottom-right, bottom-left)
    """

    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    Function: four_point_transform
    Purpose: Correct for skewed perspective of sign
    Parameters: image, pts (cv2 image, coordinates of corners of skewed rectangular border)
    Returns: warped (undistorted version of image)
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    maxWidth = w//2
    maxHeight = h//2

    dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)  # Perspective transform matrix
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # Applied perspective transform matrix

    return warped


def auto_canny(image, sigma=0.33):
    """
    Function: auto_canny
    Purpose: Detects edges in image
    Parameters: image, sigma (cv2 image, deviation from median intensity of image used for upper and lower thresholds)
    Returns: edged (edge detected version of image)
    """
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def resize_and_threshold_warped(image):
    """
    Function: resize_and_threshold_warped
    Purpose: Creates strong contrast between edges and surroundings. Effectively masks out non-edge details
    Parameters: image
    Returns: warped_processed (high contrast version of input image)
    """
    warped_new_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(warped_new_gray, (5, 5), 0)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur)
    threshold = (min_val + max_val)/2

    ret, warped_processed = cv2.threshold(warped_new_gray, threshold, 255, cv2.THRESH_BINARY)
    warped_processed = cv2.resize(warped_processed, (320, 240))

    return warped_processed


processed_images = [0, 0, 0, 0]
instruction = ""

def process_image(cv_image):
    """
    Function: process_image
    Purpose: To identify matches between signs in the video stream and the references images available to the program
    Parameters: cv_image (cv2 formatted image)
    Returns: None, Displays processed images alongside original image
    """
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = auto_canny(gray)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.005*cv2.arcLength(cnt, True), True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)

            if area > minSquareArea:
                cv2.drawContours(cv_image, [approx], 0, (0, 0, 255), 2)
                warped = four_point_transform(cv_image, approx.reshape(4, 2))
                warped_eq = resize_and_threshold_warped(warped)

                pts = approx.reshape(4, 2)
                di_st, di_rn = dist_dir(pts)

                for i in range(len(ReferenceImages)):
                    diffImg = cv2.bitwise_xor(warped_eq, symbol[i].img)
                    diff = cv2.countNonZero(diffImg)

                    if diff < minDiff:
                        match = i

                        cv2.putText(cv_image, symbol[i].name,
                                    tuple(approx.reshape(4, 2)[0]),
                                    font, 1, (200, 0, 255), 2, cv2.CV_AA)
                        diff = minDiff
                        break
		cv2.putText(cv_image, str(di_rn), (100, 100), font, 2, (0, 0, 255), 2, cv2.CV_AA)


		cv2.imshow("Corrected Perspective", warped_eq)
        cv2.imshow("Matching Operation", diffImg)
    cv2.imshow("Contours", edges)
    cv2.imshow("Main Frame", cv_image)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()

# Creates 4 cv2 windows to show the input image, the image after processing, and the matching sign images.
cv2.namedWindow("Main Frame", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Matching Operation", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Corrected Perspective", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Contours", cv2.WINDOW_AUTOSIZE)


def image_callback(img_msg):
    """
    Function: image_callback
    Purpose: Converts incoming ROS image message to cv2 object
    Parameters: img_msg
    Returns: None
    """

    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridgeError: {0}".format(e))

    process_image(cv_image)


def image_match():
    """
    Function: image_match
    Purpose: Initializes ROS node, configures publishers and subscribers, and waits for incoming ROS messages
    Parameters: None
    Returns: None
    """

    rospy.init_node('image_match', anonymous=True)
    rospy.Subscriber("/camera/image_rect_color", Image, image_callback, queue_size=1)
    rate = rospy.Rate(10) # 10hz

    readRefImages()

    while not rospy.is_shutdown():
        pub = rospy.Publisher('sign_data', String, queue_size=10)
	    pub.publish("instruction, size")
        rate.sleep()


if __name__ == '__main__':
    image_match()


