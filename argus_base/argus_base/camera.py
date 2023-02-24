from enum import IntEnum
import cv2
from cv_bridge import CvBridge

from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default


class Resolution(IntEnum):
    LOW_240p = 0 # 320×240
    VGA_480p = 1 # 640x480
    VGA_WIDE_480p = 2 # 864x480
    HD_720p = 3 # 1280×720
    FULL_HD_1080p = 4 # 1920×1080

class FramePub(Node):

    def __init__(self):
        super().__init__('frame_pub')

        # Declare custom quality of service
        self.custom_qos = qos_profile_sensor_data
        self.custom_qos.depth = 1

        self.declare_all_parameters()

        self.use_compressed = self.get_parameter('compressed').value
        if self.use_compressed:
            self.frame_msg = CompressedImage()
            self.image_pub_ = self.create_publisher(CompressedImage,
                                                    "~/cam_frame/compressed",
                                                    qos_profile_system_default)
        else:
            self.frame_msg = Image()
            self.image_pub_ = self.create_publisher(Image,
                                                    "~/cam_frame",
                                                    qos_profile_system_default)

        self._timer = self.create_timer(1.0 / self.get_parameter("rate").value, self.timer_callback)
        self.bridge = CvBridge()
        try:
            self.cap = cv2.VideoCapture(2)
            self.configure_cam()
            self.add_on_set_parameters_callback(self.configure_cam)
        except Exception as e:
            self.get_logger().error(f"Failed to connect to camera {e}")


    def timer_callback(self):
        """
        """
        if (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret:
                # self.get_logger().info(f"frame shape: {frame.shape}")
                if self.use_compressed:
                    self.frame_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
                else:
                    self.frame_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.frame_msg.header.stamp = self.get_clock().now().to_msg()
                self.frame_msg.header.frame_id = 'camera'  
                
                self.image_pub_.publish(self.frame_msg)
                
            else:
                self.get_logger().warning("Failed to grab frame")
                
    def declare_all_parameters(self):
        self.declare_parameter("compressed", True)
        self.declare_parameter("rate", 15)
        self.declare_parameter("resolution", 2)
        self.declare_parameter("brightness", 50)
        self.declare_parameter("sharpness", 50)
        self.declare_parameter("saturation", 50)
        self.declare_parameter("auto_exposure", 1)
        self.declare_parameter("exposure", 400)
        self.declare_parameter("gain", 50)
        
    def configure_resolution(self, resolution):
        
        if resolution == Resolution.LOW_240p:
            width, height = 320, 240
        elif resolution == Resolution.VGA_480p:
            width, height = 640, 480
        elif resolution == Resolution.VGA_WIDE_480p:
            width, height = 864, 480
        elif resolution == Resolution.HD_720p:
            width, height = 1280, 720
        elif resolution == Resolution.FULL_HD_1080p:
            width, height = 1920, 1080
        
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    
    def configure_cam(self, params=None):
        params_of_interest = {
                "resolution": None,
                "brightness": cv2.CAP_PROP_BRIGHTNESS,
                "sharpness": cv2.CAP_PROP_SHARPNESS,
                "saturation": cv2.CAP_PROP_SATURATION,
                "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
                "exposure": cv2.CAP_PROP_EXPOSURE,
                "gain": cv2.CAP_PROP_GAIN,
            }
        if params is not None:
            for param in params:
                if param.name in params_of_interest.keys():
                    if params_of_interest[param.name] is None:
                        self.configure_resolution(param.value)
                        self.get_logger().info(f"camera-control: resolution is set to {Resolution(param.value).name}")
                    else:
                        self.cap.set(params_of_interest[param.name], param.value)
                        val = self.cap.get(params_of_interest[param.name])
                        self.get_logger().info(f"camera-control of {param.name}: {param.value}: {val}")
            return SetParametersResult(successful=True)

        else:
            for k, v in params_of_interest.items():
                if k == 'resolution':
                    resolution = self.get_parameter(k).value
                    self.configure_resolution(resolution)
                    self.get_logger().info(f"camera-control: resolution is set to: {Resolution(resolution).name}")
                else:
                    val = self.get_parameter(k).value
                    self.get_logger().info(f"camera-control: {k} is set to: {val}")
                    self.cap.set(v, val)
        

def main(args=None):
    rclpy.init(args=args)
    frame_pub = FramePub()
    rclpy.spin(frame_pub)
    frame_pub.cap.release()
    frame_pub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
