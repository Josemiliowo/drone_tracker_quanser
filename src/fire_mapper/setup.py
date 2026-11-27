from glob import glob
from setuptools import setup

package_name = 'fire_mapper'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/foxglove', glob('foxglove/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='YOLO fire/tap detection -> map coordinates -> goals for drone 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mocap_tf_bridge = fire_mapper.mocap_tf_bridge:main',
            'yolo_detector = fire_mapper.yolo_detector:main',
            'detection_localizer = fire_mapper.detection_localizer:main',
            'target_map = fire_mapper.target_map:main',
            'target_dispatcher = fire_mapper.target_dispatcher:main',
            'cloud_throttle = fire_mapper.topic_throttle:main',
            'drone_markers = fire_mapper.drone_markers:main',
            'tcp_pose_bridge = fire_mapper.tcp_pose_bridge:main',
            'drone_link = fire_mapper.drone_link:main',
        ],
    },
)
