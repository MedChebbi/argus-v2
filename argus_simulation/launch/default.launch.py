import os
import launch
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = os.path.join(
        FindPackageShare('argus_simulation').find("argus_simulation"), 'models')

    gazebo = ExecuteProcess(
        cmd=[[
            'ign gazebo ',
            PathJoinSubstitution([
                FindPackageShare('argus_simulation'),
                'worlds',
                'simple_world.sdf '
            ]),
            '--render-engine ogre2 '
        ]],
        shell=True
    )
    
    bridge_cmd_vel = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist'],
    )
    bridge_camera_cmd_pos = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/camera/cmd_pos@std_msgs/msg/Float64@ignition.msgs.Double'],
    )
    bridge_camera = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/camera@sensor_msgs/msg/Image@ignition.msgs.Image'],
    )
    return launch.LaunchDescription([
        gazebo,
        bridge_cmd_vel,
        bridge_camera,
        bridge_camera_cmd_pos
    ])
