"""Drone 1 (Jetson): RealSense + camera mount TF + YOLO + localizer + cloud throttle.

    ros2 launch fire_mapper drone1.launch.py model:=/models/best.engine

Default cam_x/y/z were measured with a mocap marker on the D435 left-IR lens (2026-11-23),
lens (-0.003982, 0.007765, -0.256875), pivot (-0.000601, 0.073703, 0.006560) in Motive X,Y(up),Z.
The camera IS the drone's front, so the offset is 26.3 cm straight forward (side offset 0.3 mm -> 0),
z: marker (12 mm) lay on the floor, lens 4 cm above its bottom -> lens Y = 0.007765 - 0.006 + 0.040
= 0.041765, z = 0.041765 - 0.073703 = -0.0319. Rigid-body yaw with camera along +X: -1.6 deg (no offset). base_link x must point at the camera: see body_yaw_offsets_deg.

cam_x/y/z = MEASURED position of the D435 (its camera_link, ~centre of the front glass / the
screw hole) relative to the mocap rigid-body pivot of drone 1, in drone 1's body axes
(x forward, y left, z up). The camera points straight down: pitch = +90 deg.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('fire_mapper')
    params = os.path.join(pkg, 'config', 'params.yaml')
    rs_launch = os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')

    args = [
        DeclareLaunchArgument('model', default_value='/models/best.engine'),
        DeclareLaunchArgument('cam_x', default_value='0.2634'),
        DeclareLaunchArgument('cam_y', default_value='0.0'),
        DeclareLaunchArgument('cam_z', default_value='-0.0319'),
        DeclareLaunchArgument('cam_pitch', default_value='1.5707963'),  # +90 deg = looking down
        DeclareLaunchArgument('cam_yaw', default_value='0.0'),          # rotate if image "up" isn't drone forward
    ]

    # Topic names for realsense-ros 4.x: /drone1/camera/... ; frames: camera_link, camera_color_optical_frame
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        launch_arguments={
            'camera_namespace': 'drone1',
            'camera_name': 'camera',
            'align_depth.enable': 'true',
            'pointcloud.enable': 'true',
            'decimation_filter.enable': 'true',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            # older realsense-ros versions call these rgb_camera.profile / depth_module.profile
            'rgb_camera.color_profile': '640,480,30',
            'depth_module.depth_profile': '640,480,30',
        }.items(),
    )

    cam_tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='cam_mount_tf',
        arguments=['--x', LC('cam_x'), '--y', LC('cam_y'), '--z', LC('cam_z'),
                   '--roll', '0', '--pitch', LC('cam_pitch'), '--yaw', LC('cam_yaw'),
                   '--frame-id', 'drone1/base_link', '--child-frame-id', 'camera_link'],
    )

    yolo = Node(package='fire_mapper', executable='yolo_detector', name='yolo_detector',
                namespace='drone1', parameters=[params, {'model': LC('model')}], output='screen')
    loc = Node(package='fire_mapper', executable='detection_localizer', name='detection_localizer',
               namespace='drone1', parameters=[params], output='screen')
    throttle = Node(package='fire_mapper', executable='cloud_throttle', name='cloud_throttle',
                    namespace='drone1', parameters=[params],
                    remappings=[('cloud_in', 'camera/depth/color/points'), ('cloud_out', 'cloud')])

    return LaunchDescription(args + [realsense, cam_tf, yolo, loc, throttle])
