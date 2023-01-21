import os
import yaml

import launch
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch descriptions/actions generator
    """
    node = Node(
        package='argus_base',
        executable='frame_pub_main',
        name='frame_pub',
        namespace='argus',
        output='screen')

    os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '[{time}] [{severity}] [{name}] - {message}'
    ld = launch.LaunchDescription()
    ld.add_action(SetEnvironmentVariable('RCUTILS_CONSOLE_LINE_BUFFERED', '1'))
    ld.add_action(SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"))
    ld.add_action(node)

    return ld