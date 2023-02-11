# Copyright 2011 Brown University Robotics.
# Copyright 2017 Open Source Robotics Foundation, Inc.
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import sys

import std_msgs.msg
import geometry_msgs.msg
import rclpy

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty


msg = """
This node takes keypresses from the keyboard and publishes them
as Twist messages. It works best with an AZERTY keyboard layout.
---------------------------
Moving around:
   a    z    e
   q    s    d
        x    

s or space bar : stop

i/k : increase/decrease only linear speed by 10%
o/l : increase/decrease only angular speed by 10%

b/n : upper/lower camera orientation
m : To switch between manual mode and auto steer mode.

CTRL-C to quit
"""

MAX_LIN_VEL = 0.5
MAX_ANG_VEL = 1.0

LIN_VEL_STEP_SIZE = 0.05
ANG_VEL_STEP_SIZE = 0.1

# Camera orientation in rad
CAM_POS_UP = -1.0
CAM_POS_DOWN = 1.0

moveBindings = {
    'z': (1, 0, 0, 0),
    'x': (-1, 0, 0, 0),
    'q': (0, 0, 0, 1),
    'd': (0, 0, 0, -1),
    'a': (1, 0, 0, 1),
    'e': (1, 0, 0, -1),
    's': (0, 0, 0, 0),
}

speedBindings = {
    'i': (1 + LIN_VEL_STEP_SIZE, 1),
    'k': (1 - LIN_VEL_STEP_SIZE, 1),
    'o': (1, 1 + ANG_VEL_STEP_SIZE),
    'l': (1, 1 - ANG_VEL_STEP_SIZE),
}


def getKey(settings):
    if sys.platform == 'win32':
        # getwch() returns a string on Windows
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        # sys.stdin.read() returns a string on Linux
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)


def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def vels(speed, turn):
    return 'currently:\tspeed %s\tturn %s ' % (speed, turn)

def make_simple_profile(output, input, slop):
    if input > output:
        output = min(input, output + slop)
    elif input < output:
        output = max(input, output - slop)
    else:
        output = input

    return output

def constrain(input_vel, low_bound, high_bound):
    if input_vel < low_bound:
        input_vel = low_bound
    elif input_vel > high_bound:
        input_vel = high_bound
    else:
        input_vel = input_vel

    return input_vel

def check_linear_limit_velocity(velocity):
    return constrain(velocity, 0, MAX_LIN_VEL)


def check_angular_limit_velocity(velocity):
    return constrain(velocity, 0, MAX_ANG_VEL)

def main():
    settings = saveTerminalSettings()

    rclpy.init()

    node = rclpy.create_node('teleop_keyboard')
    pub = node.create_publisher(geometry_msgs.msg.Twist, '/argus/cmd_vel', 10)
    camera_pos_pub = node.create_publisher(std_msgs.msg.Float64, '/argus/camera/cmd_pos', 2)
    auto_steer_pub = node.create_publisher(std_msgs.msg.Bool, '/argus/cmd_auto_steer', 2)

    auto_steer = False
    camera_pos = 0.0
    speed = 0.3
    turn = 1.0
    control_linear_velocity = 0.0
    control_angular_velocity = 0.0
    x = 0.0
    th = 0.0
    status = 0.0

    try:
        print(msg)
        print(vels(speed, turn))
        while True:
            key = getKey(settings)
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                th = moveBindings[key][3]
            elif key in speedBindings.keys():
                speed = check_linear_limit_velocity(speed * speedBindings[key][0])
                turn = check_angular_limit_velocity(turn * speedBindings[key][1])
                print(vels(speed, turn))
                if (status == 14):
                    print(msg)
                status = (status + 1) % 15
            elif key == ' ' or key == 's':
                x = 0.0
                th = 0.0
                control_linear_velocity = 0.0
                control_angular_velocity = 0.0
            elif key == 'm':
                auto_steer = not auto_steer
                print(f" Auto steer mode: {auto_steer}")
            elif key == 'b':
                camera_pos = CAM_POS_UP
            elif key == 'n':
                camera_pos = CAM_POS_DOWN
            else:
                if (key == '\x03'):
                    break

            twist = geometry_msgs.msg.Twist()
            control_linear_velocity = make_simple_profile(
                control_linear_velocity,
                x * speed,
                (LIN_VEL_STEP_SIZE / 2.0))

            control_angular_velocity = make_simple_profile(
                control_angular_velocity,
                th * turn,
                (ANG_VEL_STEP_SIZE / 2.0))

            twist.linear.x = control_linear_velocity
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = th * turn
            pub.publish(twist)

            camera_pos_msg = std_msgs.msg.Float64()
            camera_pos_msg.data = camera_pos
            camera_pos_pub.publish(camera_pos_msg)

            auto_steer_msg = std_msgs.msg.Bool()
            auto_steer_msg.data = auto_steer
            auto_steer_pub.publish(auto_steer_msg)

    except Exception as e:
        print(e)

    finally:
        twist = geometry_msgs.msg.Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)

        restoreTerminalSettings(settings)


if __name__ == '__main__':
    main()