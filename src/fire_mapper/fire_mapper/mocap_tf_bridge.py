"""OptiTrack rigid-body poses -> ROS map frame (Z-up) -> TF map -> <name>/base_link.

Works with any mocap driver that publishes one geometry_msgs/PoseStamped per rigid body
(natnet_ros2, mocap4r2, vrpn_mocap, a Quanser bridge, ...). Point `pose_topic_fmt` at it.

Motive is left untouched (Quanser's Simulink stack depends on its settings). If Motive streams
Y-up, poses are converted here: x_map = x_m, y_map = -z_m, z_map = y_m.

Per rigid body it publishes:
  TF  map -> <name>/base_link
  /<name>/mocap_pose  (PoseStamped in map, already converted; used by target_dispatcher)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

from fire_mapper.geometry import mocap_to_map


class MocapTfBridge(Node):
    def __init__(self):
        super().__init__('mocap_tf_bridge')
        self.declare_parameter('bodies', ['drone1', 'drone2'])                    # ROS names
        self.declare_parameter('mocap_names', ['RigidBody_Drone1', 'RigidBody_Drone2'])  # Motive names
        self.declare_parameter('pose_topic_fmt', '/mocap/{mocap}/pose')  # {mocap} = Motive name, {name} = ROS name
        # 'y' (Motive) or 'z' (Simulink). dynamic typing: YAML/launch turn a bare y into True
        self.declare_parameter('up_axis', 'y', ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('body_yaw_offsets_deg', [0.0, 0.0])  # per body, if nose != rigid-body +X
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('restamp', False)  # true if the driver stamps with Motive's clock

        g = lambda n: self.get_parameter(n).value
        self.map_frame, self.restamp = g('map_frame'), g('restamp')
        up = g('up_axis')
        self.up = 'y' if up is True else str(up).lower()
        if self.up not in ('y', 'z'):
            raise ValueError(f"up_axis must be 'y' or 'z', got {up!r}")
        names, mocap = g('bodies'), g('mocap_names')
        offsets = list(g('body_yaw_offsets_deg')) + [0.0] * len(names)
        if len(mocap) != len(names):
            raise ValueError('bodies and mocap_names must have the same length')

        self.br = TransformBroadcaster(self)
        self._subs, self._pubs = [], {}
        for name, mname, off in zip(names, mocap, offsets):
            topic = g('pose_topic_fmt').format(mocap=mname, name=name)
            self._pubs[name] = self.create_publisher(PoseStamped, f'/{name}/mocap_pose', 10)
            yaw = float(off) * 3.141592653589793 / 180.0
            self._subs.append(self.create_subscription(
                PoseStamped, topic, lambda m, n=name, y=yaw: self.on_pose(m, n, y), qos_profile_sensor_data))
            self.get_logger().info(f'{topic} ({mname}) -> TF {self.map_frame} -> {name}/base_link '
                                   f'[up={self.up}, yaw offset={off} deg]')

    def on_pose(self, msg: PoseStamped, name: str, yaw_off: float):
        p, o = msg.pose.position, msg.pose.orientation
        pos, q = mocap_to_map([p.x, p.y, p.z], [o.x, o.y, o.z, o.w], self.up, yaw_off)
        stamp = self.get_clock().now().to_msg() if self.restamp else msg.header.stamp

        t = TransformStamped()
        t.header.stamp, t.header.frame_id = stamp, self.map_frame
        t.child_frame_id = f'{name}/base_link'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, pos)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = map(float, q)
        self.br.sendTransform(t)

        out = PoseStamped()
        out.header = t.header
        out.pose.position.x, out.pose.position.y, out.pose.position.z = map(float, pos)
        out.pose.orientation = t.transform.rotation
        self._pubs[name].publish(out)


def main():
    rclpy.init()
    rclpy.spin(MocapTfBridge())


if __name__ == '__main__':
    main()
