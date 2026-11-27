"""Simple 3D drone models for Foxglove/RViz, attached to each drone's mocap frame.

Publishes one latched MarkerArray (/drone_models) with frame_locked markers, so they follow
TF map -> <name>/base_link. Approximate QDrone 2: 50 x 50 cm with props, X frame, D435 on the front edge.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

COLORS = {'drone1': (1.0, 0.55, 0.0), 'drone2': (0.0, 0.8, 0.9)}


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


class DroneMarkers(Node):
    def __init__(self):
        super().__init__('drone_markers')
        self.declare_parameter('bodies', ['drone1', 'drone2'])
        self.declare_parameter('camera_body', 'drone1')
        self.declare_parameter('cam_xyz', [0.2634, 0.0, -0.0319])  # same as drone1.launch.py
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, '/drone_models', latched)
        self.publish()

    def m(self, frame, ns, mid, mtype, xyz, scale, rgba, q=(0.0, 0.0, 0.0, 1.0)):
        k = Marker()
        k.header.frame_id = frame
        k.header.stamp = self.get_clock().now().to_msg()
        k.ns, k.id, k.type, k.action = ns, mid, mtype, Marker.ADD
        k.frame_locked = True
        k.pose.position.x, k.pose.position.y, k.pose.position.z = map(float, xyz)
        k.pose.orientation.x, k.pose.orientation.y, k.pose.orientation.z, k.pose.orientation.w = q
        k.scale.x, k.scale.y, k.scale.z = map(float, scale)
        k.color.r, k.color.g, k.color.b, k.color.a = map(float, rgba)
        return k

    def publish(self):
        g = lambda n: self.get_parameter(n).value
        arr = MarkerArray()
        for name in g('bodies'):
            f = f'{name}/base_link'
            r, gr, b = COLORS.get(name, (0.8, 0.8, 0.8))
            dark = (0.15, 0.15, 0.15, 1.0)
            i = 0
            arr.markers.append(self.m(f, name, i, Marker.CUBE, (0, 0, -0.02), (0.16, 0.16, 0.07), (r, gr, b, 1.0)))
            for yaw in (math.pi / 4, -math.pi / 4):  # X arms
                i += 1
                arr.markers.append(self.m(f, name, i, Marker.CUBE, (0, 0, 0), (0.43, 0.025, 0.02), dark, yaw_quat(yaw)))
            for sx in (1, -1):  # rotor discs
                for sy in (1, -1):
                    i += 1
                    arr.markers.append(self.m(f, name, i, Marker.CYLINDER, (0.152 * sx, 0.152 * sy, 0.02),
                                              (0.2, 0.2, 0.004), (0.7, 0.7, 0.7, 0.35)))
            i += 1  # forward arrow
            arr.markers.append(self.m(f, name, i, Marker.ARROW, (0.05, 0, 0.03), (0.2, 0.025, 0.025), (r, gr, b, 1.0)))
            i += 1
            label = self.m(f, name, i, Marker.TEXT_VIEW_FACING, (0, 0, 0.18), (0, 0, 0.08), (1, 1, 1, 1))
            label.text = name
            arr.markers.append(label)
            if name == g('camera_body'):
                i += 1  # D435: 90 x 25 x 25 mm, long side across the drone, lens facing down
                arr.markers.append(self.m(f, name, i, Marker.CUBE, g('cam_xyz'), (0.025, 0.09, 0.025),
                                          (0.3, 0.3, 0.35, 1.0)))
                i += 1  # view direction (down) cue
                cx, cy, cz = g('cam_xyz')
                arr.markers.append(self.m(f, name, i, Marker.ARROW, (cx, cy, cz - 0.015), (0.12, 0.015, 0.015),
                                          (0.2, 0.6, 1.0, 0.8), (0.0, 0.7071068, 0.0, 0.7071068)))
        self.pub.publish(arr)
        self.get_logger().info(f'published drone models for {g("bodies")} on /drone_models')


def main():
    rclpy.init()
    rclpy.spin(DroneMarkers())


if __name__ == '__main__':
    main()
