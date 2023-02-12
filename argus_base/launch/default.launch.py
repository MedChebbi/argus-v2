import os
import yaml

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch descriptions/actions generator
    """
    params_file = os.path.join(get_package_share_directory('argus_base'),'config', 'default.yaml')
    
    ld = launch.LaunchDescription()
    params_file = yaml.safe_load(params_file)
    os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '[{time}] [{severity}] [{name}] - {message}'
    node = Node(
        package='argus_base',
        executable='frame_pub_main',
        name='frame_pub',
        namespace='argus',
        parameters=[params_file],
        output='screen')
    ld.add_action(SetEnvironmentVariable('RCUTILS_CONSOLE_LINE_BUFFERED', '1'))
    ld.add_action(SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"))
    ld.add_action(node)

    return ld