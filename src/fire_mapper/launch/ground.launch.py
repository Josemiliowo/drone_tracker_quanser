"""Ground station (laptop): mocap -> TF, target map, drone-2 dispatcher, Foxglove bridge.

Bench (Motive VRPN):          ros2 launch fire_mapper ground.launch.py mocap_server:=<Motive PC IP>
Flight (bytes from Simulink): ros2 launch fire_mapper ground.launch.py mocap_source:=tcp   (Simulink frame: Z up, X fwd, Y left)

Both publish /mocap/<RigidBodyName>/pose, which mocap_tf_bridge converts to the ROS map frame.
vrpn needs ros-humble-vrpn-mocap and VRPN enabled in Motive (port 3883).
tcp: format/port in config/params.yaml (tcp_pose_bridge).
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, LaunchConfigurationEquals as Eq
from launch.substitutions import LaunchConfiguration as LC, PythonExpression
from launch_ros.actions import Node


def check_args(context):
    src = LC('mocap_source').perform(context)
    if src not in ('vrpn', 'tcp', 'none'):
        raise RuntimeError(f"mocap_source:={src!r} is not valid - use vrpn, tcp or none "
                           f"(no pose receiver would start and the drone would never move)")
    return []


def generate_launch_description():
    params = os.path.join(get_package_share_directory('fire_mapper'), 'config', 'params.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('mocap_tf', default_value='true',
                              description='false if the mocap driver already broadcasts TF'),
        DeclareLaunchArgument('foxglove', default_value='true'),
        DeclareLaunchArgument('up_axis', default_value='auto',
                              description="up axis of incoming poses; auto = y for vrpn (Motive), z for tcp (Simulink)"),
        DeclareLaunchArgument('mocap_source', default_value='vrpn',
                              description="vrpn = Motive VRPN (bench), tcp = bytes from Simulink (flight), none"),
        OpaqueFunction(function=check_args),
        DeclareLaunchArgument('mocap_server', default_value='192.168.1.100',
                              description='IP of the PC running Motive'),

        Node(package='vrpn_mocap', executable='client_node', name='vrpn_mocap', namespace='mocap',
             parameters=[{'server': LC('mocap_server'), 'port': 3883, 'frame_id': 'mocap',
                          'update_freq': 120.0, 'refresh_freq': 1.0,
                          'use_vrpn_timestamps': False}],  # ROS time of the laptop (chrony server)
             condition=Eq('mocap_source', 'vrpn'), output='screen'),
        Node(package='fire_mapper', executable='tcp_pose_bridge', name='tcp_pose_bridge',
             parameters=[params], condition=Eq('mocap_source', 'tcp'), output='screen'),

        Node(package='fire_mapper', executable='mocap_tf_bridge', name='mocap_tf_bridge',
             parameters=[params, {'up_axis': PythonExpression([
                 "'", LC('up_axis'), "' if '", LC('up_axis'), "' != 'auto' else ('z' if '",
                 LC('mocap_source'), "' == 'tcp' else 'y')"])}], condition=IfCondition(LC('mocap_tf')), output='screen'),
        # tcp (Simulink): map detections only with poses sent while isTracking = 1;
        # vrpn: Motive only streams tracked poses, so the normal pose topic is fine
        Node(package='fire_mapper', executable='drone_link', name='drone_link',
             parameters=[params, {'pose_topic': PythonExpression([
                 "'/mocap/RigidBody_Drone1/pose_valid' if '", LC('mocap_source'),
                 "' == 'tcp' else '/drone1/mocap_pose'"])}],
             output='screen'),
        Node(package='fire_mapper', executable='drone_markers', name='drone_markers', output='screen'),
        Node(package='fire_mapper', executable='target_map', name='target_map', parameters=[params],
             remappings=[('target_observations', '/drone1/target_observations')], output='screen'),
        Node(package='fire_mapper', executable='target_dispatcher', name='target_dispatcher',
             parameters=[params], output='screen'),
        Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
             parameters=[{'port': 8765, 'send_buffer_limit': 50000000}],
             condition=IfCondition(LC('foxglove'))),
    ])
