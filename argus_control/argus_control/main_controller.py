import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult

from argus_control.pid_controller import PID

from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

from argus_interfaces.msg import LineInfo


class Controller(Node):
    """
    """
    ANG_PROP = 0.025
    def __init__(self):
        super().__init__("controller")
        self.get_logger().info("Initializing Main controller Node")
        self.declare_all_parameters()
        self.line_info_sub = self.create_subscription(LineInfo, "/argus/computer_vision/line_info", self.line_cb, qos_profile_sensor_data)
        self.cmd_vel_sub = self.create_subscription(Twist, '/argus/teleop_keyboard/cmd_vel', self.tele_vel_cb, qos_profile_sensor_data)
        self.auto_mode_sub = self.create_subscription(Bool, '/argus/cmd_auto_steer', self.auto_mode_cb, qos_profile_sensor_data)
        
        self.vel_pub = self.create_publisher(Twist, '/argus/cmd_vel', 10)
        self.pid_param = {"KP": self.get_parameter("KP").value,
                          "KI": self.get_parameter("KI").value,
                          "KD": self.get_parameter("KI").value,
                          "S" : self.get_parameter("S").value,
                          "bias": self.get_parameter("bias").value}

        max_ang_speed = self.get_parameter("max_ang_speed").value
        self.pid_controller = PID(self.pid_param['KP'], self.pid_param['KI'], self.pid_param['KD'],
                                  sample_time=0.01, setpoint=0, bias=self.get_parameter("bias").value,
                                  output_limits=(-max_ang_speed, max_ang_speed))
        
        self.line_info = LineInfo()
        self.vel = Twist()
        self.auto_mode = True
        self._stop_states = ['T', 'x', 'left', 'right']
        self.add_on_set_parameters_callback(self.tune_pid_params)

    def step(self):
        """
        """
        if self.auto_mode:
            self.line_follow()
        self.vel_pub.publish(self.vel)

    def line_follow(self):
        """
        """
        if self.line_info.detected == True:
            # if self.line_info.state in self._stop_states:
            #     if self.line_info.state == "right":
            #         self.vel.linear.x = 0.08
            #         self.vel.angular.z = -0.7
            #     else:
            #         self.vel.linear.x = 0.0
            #         self.vel.angular.z = 0.0
            #         self.vel_pub.publish(self.vel)
            #         print("Intersection!!")
            #         time.sleep(0.5)
            #         self.vel.linear.x = self.pid_param["S"]
            #         self.vel.angular.z = 0.0
            #         self.vel_pub.publish(self.vel)
            #         time.sleep(1.0)
            # else:
            self.vel.linear.x = self.pid_param["S"]
            self.vel.angular.z = self.pid_controller(self.line_info.error) + self.angle_correction(self.line_info.angle)
            
            self.get_logger().info(f"after pid robot speed: {self.vel}\n")
        else:
            self.vel.linear.x = -0.02
            if self.line_info.error >= 0.0:
                self.vel.angular.z = -0.7
            else:
                self.vel.angular.z = 0.7

    def tune_pid_params(self, params=None):
        """
        """
        if params is not None:
            for param in params:
                if param.name in self.pid_param.keys():
                   self.pid_param[param.name] = param.value
            
            self.pid_controller.tunings = [self.pid_param["KP"], self.pid_param["KI"],
                                           self.pid_param["KD"], self.pid_param['bias']]
            return SetParametersResult(successful=True)

    def angle_correction(self, angle):
        def clamp(num, min, max):
            return min if num < min else max if num > max else num
        return clamp(-angle * self.ANG_PROP, min=-1.0, max=1.0)

    def line_cb(self, data):
        """
        """
        self.line_info = data

    def auto_mode_cb(self, data):
        """
        """
        self.auto_mode = data.data

    def tele_vel_cb(self, vel):
        """
        """
        self.vel = vel

    def declare_all_parameters(self):
        """
        """
        self.declare_parameter("KP", 2.0)
        self.declare_parameter("KI", 0.0)
        self.declare_parameter("KD", 0.23)
        self.declare_parameter("S", 0.1)
        self.declare_parameter("bias", 0.05)
        self.declare_parameter("max_ang_speed", 2.5)
    
    def cleanup(self):
        self.auto_mode == False
        self.vel.linear.x = 0.0
        self.vel.angular.z = 0.0
        self.vel_pub.publish(self.vel)

def main(args=None):
    rclpy.init(args=args)
    node = Controller()
    # start spinning
    while rclpy.ok():
        try:
            rclpy.spin_once(node, timeout_sec=0.005)
            node.step()
        except KeyboardInterrupt:
            break
    
    node.cleanup()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()