#!/usr/bin/env python3
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

bridge = CvBridge()

def image_callback(image_data):
    try:
        cv_image = bridge.imgmsg_to_cv2(image_data, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # Define the Region of Interest (ROI)
    height, width, _ = cv_image.shape
    roi_top = int(height * 0.75)  # Adjust to focus on the lower quarter of the image
    roi_bottom = height
    roi_left = 0
    roi_right = width

    # Crop the image to the ROI
    cv_image_roi = cv_image[roi_top:roi_bottom, roi_left:roi_right]

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image_roi, cv2.COLOR_BGR2GRAY)

    # Thresholding to get a binary image
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    twist = Twist()

    if contours:
        # Find the largest contour and its center
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + roi_top
        else:
            cx, cy = 0, 0

        # Determine the deviation from the center of the image
        deviation = cx - (width / 2)

        # Control for following the line
        twist.linear.x = 1  # Linear velocity (m/s)
        twist.angular.z = -float(deviation) / 130  # Angular velocity (rad/s)
    else:
        # No line found, rotate in place
        twist.linear.x = 0.03
        twist.angular.z = 0.60  # Adjust angular velocity as needed

    # Publish the velocity
    velocity_publisher.publish(twist)

    # Display the processed video
    display_size = (int(width * 0.5), int((height) * 0.3))  # 50% of the original size
    resized_image = cv2.resize(cv_image_roi, display_size)
    cv2.imshow("Processed View", resized_image)
    cv2.waitKey(1)


def main():
    rospy.init_node('line_follower')

    global velocity_publisher
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    image_subscriber = rospy.Subscriber('/rrbot/camera1/image_raw', Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
