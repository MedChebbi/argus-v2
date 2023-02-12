import os
import yaml

import launch
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch descriptions/actions generator
    """
    ld = launch.LaunchDescription()
    params_file = os.path.join(get_package_share_directory('argus_computer_vision'), 'config', 'default.yaml')
    params_file = yaml.safe_load(params_file)
    node = Node(
        package='argus_computer_vision',
        executable='computer_vision_main',
        name='computer_vision',
        namespace='argus',
        parameters=[params_file],
        output='screen')

    os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '[{time}] [{severity}] [{name}] - {message}'
    ld.add_action(SetEnvironmentVariable('RCUTILS_CONSOLE_LINE_BUFFERED', '1'))
    ld.add_action(SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"))
    ld.add_action(node)

    return ld