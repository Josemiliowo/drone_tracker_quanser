"""Re-publish a PointCloud2 at a lower rate (keeps WiFi free; Foxglove decay builds the map)."""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2


class CloudThrottle(Node):
    def __init__(self):
        super().__init__('cloud_throttle')
        self.declare_parameter('rate', 2.0)
        self.period = 1.0 / self.get_parameter('rate').value
        self.last = None
        self.pub = self.create_publisher(PointCloud2, 'cloud_out', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, 'cloud_in', self.cb, qos_profile_sensor_data)

    def cb(self, msg):
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        if self.last is None or t - self.last >= self.period:
            self.last = t
            self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(CloudThrottle())


if __name__ == '__main__':
    main()
