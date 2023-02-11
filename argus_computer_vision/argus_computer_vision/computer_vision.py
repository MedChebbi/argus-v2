import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from sensor_msgs.msg import CompressedImage, Image

from argus_computer_vision.color_detection import ColorDetector
from argus_computer_vision.aruco_detection import ArUcoDetection
from argus_computer_vision.line_state_classification import LineStateClassifier

from argus_computer_vision.display import Display

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
        
        if self.get_parameter("cv_debug").value:
            self.debug_frame_pub  = self.create_publisher(Image, "~/debug_frame", qos_profile_system_default)

        self.line_detector = ColorDetector(self.get_logger(),{
                    'detection_mode': 0,
                    'color_space': self.get_parameter('line_detection.color_space').value,
                    'color_min_range': self.get_parameter('line_detection.color_min_range').value,
                    'color_max_range': self.get_parameter('line_detection.color_max_range').value,
                    'max_area': self.get_parameter('line_detection.max_area').value,
                    'min_area': self.get_parameter('line_detection.min_area').value,
                    })
        self.line_state_active = self.get_parameter('line_state_classification.activated').value
        self.classifier = None
        if self.line_state_active:
            self.classifier = LineStateClassifier(self.get_logger(),{
                'on_edge': self.get_parameter('line_state_classification.on_edge').value,
                'model_path': self.get_parameter('line_state_classification.model_path').value,
                'input_shape': self.get_parameter('line_state_classification.input_shape').value,
                'class_names': self.get_parameter('line_state_classification.class_names').value,
                'threshold': self.get_parameter('line_state_classification.threshold').value,
                'queue_size': self.get_parameter('line_state_classification.queue_size').value,
            })

        self.aruco_detector = ArUcoDetection(self.get_logger(),
                                             aruco_type=self.get_parameter('aruco_detection.aruco_type').value)
        

    def cleanup(self):
        return

    def __publish_main_topics(self, msgs):
        self.get_logger().info(msgs)
        return

    def frame_callback(self, frame):
        debug_frame_msg = Image()
        if self.compressed:
            cv_frame = self.bridge.compressed_imgmsg_to_cv2(frame, desired_encoding='passthrough')
        else:
            cv_frame = self.bridge.imgmsg_to_cv2(frame, desired_encoding='passthrough')

        debug_frame = cv_frame.copy()

        detected, error, ang = self.line_detector.detect(cv_frame, debug=True)
        debug_frame = Display.display_line(debug_frame, self.line_detector, error, ang)

        corners, ids = self.aruco_detector.detect(cv_frame, debug=True)
        debug_frame = Display.display_aruco_markers(debug_frame, corners, ids)

        debug_frame_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='rgb8')
        self.debug_frame_pub.publish(debug_frame_msg)
        # self.__publish_main_topics(msgs)

    def declare_all_parameters(self):
        self.declare_parameter("cv_debug", True)

        # Line detection parameters
        self.declare_parameter("line_detection.color_space", 0)
        self.declare_parameter("line_detection.color_min_range", [0, 0, 0])
        self.declare_parameter("line_detection.color_max_range", [25, 25, 25])
        self.declare_parameter("line_detection.max_area", 0.4)
        self.declare_parameter("line_detection.min_area", 0.01)

        # Line state classification parameters
        self.declare_parameter("line_state_classification.activated",  False)
        self.declare_parameter("line_state_classification.on_edge",  False)
        self.declare_parameter("line_state_classification.model_path",  "")
        self.declare_parameter("line_state_classification.input_shape",  [64, 64, 1])
        self.declare_parameter("line_state_classification.class_names",  ['straight', 'x', 'T', 'left', 'right', 'end'])
        self.declare_parameter("line_state_classification.threshold",  0.65)
        self.declare_parameter("line_state_classification.queue_size",  5)

        # Aruco detector parameters
        self.declare_parameter("aruco_detection.aruco_type",  'DICT_4X4_1000')


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
