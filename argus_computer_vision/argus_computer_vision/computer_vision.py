import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from sensor_msgs.msg import CompressedImage, Image

from argus_computer_vision.color_detection import ColorDetector
from argus_computer_vision.aruco_detection import ArUcoDetection
from argus_computer_vision.intersection_detection import LineStateClassifier

class ComputerVision(Node):
    """
    """
    def __init__(self) -> None:
        super().__init__("computer_vision")
        self.get_logger().info("Initializing Computer vision Node")
        self.declare_all_parameters()
        self.bridge = CvBridge()

        self.compressed = False
        if self.compressed:
            self.frame_sub = self.create_subscription(CompressedImage, "/argus/frame_pub/cam_frame/compressed", self.frame_callback, qos_profile_system_default)
        else:
            self.frame_sub = self.create_subscription(Image, "/argus/frame_pub/cam_frame", self.frame_callback, qos_profile_system_default)
        # self.feedback_pub  = self.create_publisher(MultiWeedDetectDebug, "~/feedback", qos_profile_sensor_data)

        self.classifier = LineStateClassifier(self.get_logger())
        self.line_detector = ColorDetector(self.get_logger())
        self.aruco_detector = ArUcoDetection(self.get_logger())
        

    def cleanup(self):
        return

    def __publish_main_topics(self, msgs):
        self.get_logger().info(msgs)
        return

    def frame_callback(self, frame):
        msgs = 'hello'
        if self.compressed:
            cv_frame = self.bridge.compressed_imgmsg_to_cv2(frame, desired_encoding='passthrough')
        else:
            cv_frame = self.bridge.imgmsg_to_cv2(frame, desired_encoding='passthrough')
        self.__publish_main_topics(msgs)

    def declare_all_parameters(self):
        # self.declare_parameter()
        return


def main(args=None):
    rclpy.init(args=args)
    node = ComputerVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.cleanup()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
