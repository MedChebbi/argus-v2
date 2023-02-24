import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from sensor_msgs.msg import CompressedImage, Image
from rcl_interfaces.msg import SetParametersResult

from argus_computer_vision.color_detection import ColorDetector
from argus_computer_vision.aruco_detection import ArUcoDetection
from argus_computer_vision.line_state_classification import LineStateClassifier
from argus_computer_vision.display import Display

from argus_interfaces.msg import LineInfo
from argus_interfaces.srv import GetArucoID


class ComputerVision(Node):
    """
    """
    def __init__(self) -> None:
        super().__init__("computer_vision")
        self.get_logger().info("Initializing Computer vision Node")
        self.bridge = CvBridge()

        self.declare_all_parameters()

        self.compressed = False
        if self.compressed:
            self.frame_sub = self.create_subscription(CompressedImage, "/argus/frame_pub/cam_frame/compressed", self.frame_callback, qos_profile_system_default)
        else:
            self.frame_sub = self.create_subscription(Image, "/argus/frame_pub/cam_frame", self.frame_callback, qos_profile_system_default)
        
        self.line_info_pub = self.create_publisher(LineInfo, "~/line_info", qos_profile_sensor_data)
        self.aruco_det_srv = self.create_service(GetArucoID, '~/get_aruco_id', self.get_aruco_id)
        
        if self.get_parameter("cv_debug").value:
            self.debug_frame_pub  = self.create_publisher(Image, "~/debug_frame", qos_profile_system_default)

        self.dynamic_params = {
                            'cv_debug': self.get_parameter("cv_debug"),
                            'line_detection.max_area': self.get_parameter('line_detection.max_area').value,
                            'line_detection.min_area': self.get_parameter('line_detection.min_area').value,
                            'line_detection.color_max_range': self.get_parameter('line_detection.color_max_range').value,
                            'line_detection.color_min_range': self.get_parameter('line_detection.color_min_range').value,
                            'line_state_classification.activated': self.get_parameter('line_state_classification.activated').value,
                            'line_detection.thr': self.get_parameter('line_detection.thr').value,
                            }

        self.line_detector = ColorDetector(self.get_logger(), {
                            'detection_mode': 0,
                            'color_space': self.get_parameter('line_detection.color_space').value,
                            'color_min_range': self.get_parameter('line_detection.color_min_range').value,
                            'color_max_range': self.get_parameter('line_detection.color_max_range').value,
                            'max_area': self.get_parameter('line_detection.max_area').value,
                            'min_area': self.get_parameter('line_detection.min_area').value,
                            })
        self.classifier = None
        if self.get_parameter('line_state_classification.activated').value:
            self.classifier = LineStateClassifier(self.get_logger(), {
                    'on_edge': self.get_parameter('line_state_classification.on_edge').value,
                    'model_path': self.get_parameter('line_state_classification.model_path').value,
                    'class_names': self.get_parameter('line_state_classification.class_names').value,
                    'threshold': self.get_parameter('line_state_classification.threshold').value,
                    'queue_size': self.get_parameter('line_state_classification.queue_size').value,
                    })

        self.aruco_detector = ArUcoDetection(self.get_logger(),
                                             aruco_type=self.get_parameter('aruco_detection.aruco_type').value)
        
        self.add_on_set_parameters_callback(self.fine_tune_params)
        self._aruco_ids = None


    def frame_callback(self, frame):
        """
        """
        line_state = ''
        if self.compressed:
            cv_frame = self.bridge.compressed_imgmsg_to_cv2(frame, desired_encoding='passthrough')
        else:
            cv_frame = self.bridge.imgmsg_to_cv2(frame, desired_encoding='bgr8')

        debug_frame = cv_frame.copy()

        detected, error, ang = self.line_detector.detect(cv_frame, params=self.get_line_params(self.dynamic_params), debug=True)
        if self.dynamic_params["line_state_classification.activated"]:
            line_state, _ = self.classifier.predict(self.line_detector.mask, debug=True)
        corners, self._aruco_ids = self.aruco_detector.detect(cv_frame, debug=True)
        
        if self.dynamic_params["cv_debug"]:
            debug_frame_msg = Image()
            debug_frame = Display.display_line(debug_frame, self.line_detector, error, ang)
            debug_frame = Display.display_line_state(debug_frame, line_state)
            debug_frame = Display.display_aruco_markers(debug_frame, corners, self._aruco_ids)
            debug_frame_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self.debug_frame_pub.publish(debug_frame_msg)

        self.__publish_main_topics(detected, error, ang, line_state)


    def get_line_params(self, params):
        new_params = dict()
        for k, v in params.items():
            if k.split('.')[0] == 'line_detection':
                new_params[k.split('.')[-1]] = v
        return new_params


    def fine_tune_params(self, params=None):
        if params is not None:
            for param in params:
                if param.name in self.dynamic_params.keys():
                   self.dynamic_params[param.name] = param.value
            return SetParametersResult(successful=True)


    def get_aruco_id(self, request, response):
        """
        Gets the Aruco ID from an Empty service call
        """
        response.valid = False
        response.id = 0
        if self._aruco_ids is not None:
            response.valid = True
            response.id = int(self._aruco_ids[0])

        return response


    def __publish_main_topics(self, detected, error, ang, state):
        """
        """
        line_info_msg = LineInfo()
        line_info_msg.detected = detected
        line_info_msg.error = error
        line_info_msg.angle = ang
        line_info_msg.state = state
        self.line_info_pub.publish(line_info_msg)


    def declare_all_parameters(self):
        self.declare_parameter("cv_debug", True)

        # Line detection parameters
        self.declare_parameter("line_detection.color_space", 0)
        self.declare_parameter("line_detection.color_min_range", [0, 0, 0])
        self.declare_parameter("line_detection.color_max_range", [80, 80, 80])
        self.declare_parameter("line_detection.max_area", 0.4)
        self.declare_parameter("line_detection.min_area", 0.01)
        self.declare_parameter("line_detection.thr", 50)

        # Line state classification parameters
        self.declare_parameter("line_state_classification.activated",  False)
        self.declare_parameter("line_state_classification.on_edge",  False)
        self.declare_parameter("line_state_classification.model_path",  "")
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

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
