import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default


class FramePub(Node):

    def __init__(self, rate):
        super().__init__('frame_pub')

        # Declare custom quality of service
        self.custom_qos = qos_profile_sensor_data
        self.custom_qos.depth = 1
        self.video = False
        self.reverse = False
        self.use_compressed = True
        if self.use_compressed:
            self.image_pub_ = self.create_publisher(CompressedImage,
                                                    "~/frame/compressed",
                                                    qos_profile_system_default)
        else:
            self.image_pub_ = self.create_publisher(Image,
                                                    "~/frame",
                                                    qos_profile_system_default)

        self._timer = self.create_timer(1.0 / rate, self.timer_callback)
        self.seq = 0
        self.bridge = CvBridge()
        if self.video:
            video_dir = '/home/mohamed/Trabotyx/data/obj_det/tracking_outdoor_4.mp4'
            self.video_cap = cv2.VideoCapture(video_dir)
            self.frame_idx = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
            self.start = 0
            self.finish = self.frame_idx
            if self.reverse:
                self.start = int(self.frame_idx)
                self.finish = 0
            self.seq = self.start
        else:
            self.video_cap = cv2.VideoCapture(0)

    def timer_callback(self):
        
        if self.use_compressed:
            frame_msg = CompressedImage()
        else:
            frame_msg = Image()

        if (self.video_cap.isOpened()):
            if self.video:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.seq)
            ret, frame = self.video_cap.read()
            if self.video:
                if self.reverse:
                    self.seq -=1
                else:
                    self.seq +=1

            if ret:
                #frame = cv2.resize(frame, (1240, 720))
                if self.use_compressed:
                    frame_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
                else:
                    frame_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                frame_msg.header.stamp = self.get_clock().now().to_msg()
                frame_msg.header.frame_id = 'camera'
                #print(self.seq)         
                
                self.image_pub_.publish(frame_msg)
                if self.video:
                    if (not self.reverse) and self.seq >= self.finish:
                        self.seq = self.start
                        print("Video ended!!")
                    if self.reverse and self.seq <= self.finish:
                        self.seq = self.start
                        print("Video ended!!")
            
            else:
                if self.video:
                    self.seq = self.start
                    print("Video ended!!")
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.seq)


def main(args=None):
    rclpy.init(args=args)
    frame_pub = FramePub(15.0)
    rclpy.spin(frame_pub)
    frame_pub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
