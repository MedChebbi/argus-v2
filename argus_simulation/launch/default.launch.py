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
                'robolympics_world_0.sdf '
            ]),
            '--render-engine ogre2 '
        ]],
        shell=True
    )
    
    bridge_cmd_vel = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/argus/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist'],
    )
    bridge_camera_cmd_pos = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/argus/camera/cmd_pos@std_msgs/msg/Float64@ignition.msgs.Double'],
    )
    bridge_camera = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/argus/frame_pub/cam_frame@sensor_msgs/msg/Image@ignition.msgs.Image'],
    )
    bridge_odom = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/argus/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry'],
    )
    return launch.LaunchDescription([
        gazebo,
        bridge_cmd_vel,
        bridge_camera,
        bridge_camera_cmd_pos,
        bridge_odom
    ])
