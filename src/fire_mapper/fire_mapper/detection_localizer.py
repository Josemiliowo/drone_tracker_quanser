"""Bounding box + aligned depth + tf2  ->  target position in the map frame.

For every detection:
  1. median of valid depth in the centre of the box (D435 aligned_depth_to_color, uint16 mm)
  2. deproject the box centre with the color intrinsics -> point in camera optical frame
  3. tf2 lookup optical frame -> map at the IMAGE timestamp (mocap gives map -> base_link,
     static TF gives base_link -> camera_link, realsense gives camera_link -> optical)
  4. if depth is invalid: intersect the pixel ray with the plane z = fallback_plane_z
Publishes vision_msgs/Detection3DArray in `map`.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D, ObjectHypothesisWithPose
from tf2_ros import Buffer, TransformListener, TransformException

from fire_mapper.geometry import make_T, deproject, ray_plane, robust_depth


class DetectionLocalizer(Node):
    def __init__(self):
        super().__init__('detection_localizer')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('depth_topic', 'camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('info_topic', 'camera/color/camera_info')
        self.declare_parameter('depth_min', 0.3)
        self.declare_parameter('depth_max', 3.0)
        self.declare_parameter('bbox_shrink', 0.5)
        self.declare_parameter('use_fallback', True)
        self.declare_parameter('fallback_plane_z', 0.05)  # default plane height (m, map) for other classes
        # per-class plane height for the ray/plane fallback: the camera looks down, so the
        # box centre is roughly on the object's top part (~0.75 x object height)
        self.declare_parameter('fallback_classes', ['fire', 'tap'])
        self.declare_parameter('fallback_heights', [0.08, 0.12])
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('tf_timeout', 0.1)

        g = lambda n: self.get_parameter(n).value
        self.map_frame = g('map_frame')
        self.dmin, self.dmax, self.shrink = g('depth_min'), g('depth_max'), g('bbox_shrink')
        self.use_fallback, self.plane_z = g('use_fallback'), g('fallback_plane_z')
        self.plane_by_class = dict(zip(g('fallback_classes'), g('fallback_heights')))
        self.tf_timeout = Duration(seconds=g('tf_timeout'))

        self.bridge = CvBridge()
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(Detection3DArray, 'target_observations', 10)

        subs = [
            message_filters.Subscriber(self, Detection2DArray, 'detections'),
            message_filters.Subscriber(self, Image, g('depth_topic'), qos_profile=qos_profile_sensor_data),
            message_filters.Subscriber(self, CameraInfo, g('info_topic'), qos_profile=qos_profile_sensor_data),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(subs, queue_size=30, slop=g('sync_slop'))
        self.sync.registerCallback(self.on_sync)
        self.n_depth = self.n_fallback = self.n_drop = 0
        self.create_timer(10.0, self.report)

    def lookup(self, frame, stamp):
        tf = self.tf_buffer.lookup_transform(self.map_frame, frame, Time.from_msg(stamp), self.tf_timeout)
        tr, q = tf.transform.translation, tf.transform.rotation
        return make_T([tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w])

    def on_sync(self, dets: Detection2DArray, depth_msg: Image, info: CameraInfo):
        if not dets.detections:
            return
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        frame = info.header.frame_id  # camera_color_optical_frame (depth is aligned to color)
        try:
            T = self.lookup(frame, dets.header.stamp)
        except TransformException as e:
            self.n_drop += len(dets.detections)
            self.get_logger().warn(f'TF {self.map_frame}<-{frame} failed: {e}', throttle_duration_sec=2.0)
            return

        depth = self.bridge.imgmsg_to_cv2(depth_msg)
        depth_m = depth.astype(np.float32) * (0.001 if depth.dtype == np.uint16 else 1.0)

        out = Detection3DArray()
        out.header.stamp = dets.header.stamp
        out.header.frame_id = self.map_frame
        for d in dets.detections:
            u, v = d.bbox.center.position.x, d.bbox.center.position.y
            x0, x1 = u - d.bbox.size_x / 2, u + d.bbox.size_x / 2
            y0, y1 = v - d.bbox.size_y / 2, v + d.bbox.size_y / 2
            z = robust_depth(depth_m, x0, y0, x1, y1, self.shrink, self.dmin, self.dmax)
            if z is not None:
                p = (T @ np.append(deproject(u, v, z, fx, fy, cx, cy), 1.0))[:3]
                src = 'depth'
                self.n_depth += 1
            elif self.use_fallback:
                cls = d.results[0].hypothesis.class_id
                p = ray_plane(T, u, v, fx, fy, cx, cy, self.plane_by_class.get(cls, self.plane_z))
                src = 'plane'
                if p is None:
                    self.n_drop += 1
                    continue
                self.n_fallback += 1
            else:
                self.n_drop += 1
                continue

            o = Detection3D()
            o.header = out.header
            o.id = src  # how the position was obtained: 'depth' or 'plane'
            o.bbox.center.position.x, o.bbox.center.position.y, o.bbox.center.position.z = map(float, p)
            o.bbox.center.orientation.w = 1.0
            h = ObjectHypothesisWithPose()
            h.hypothesis = d.results[0].hypothesis
            h.pose.pose = o.bbox.center
            o.results.append(h)
            out.detections.append(o)
        if out.detections:
            self.pub.publish(out)

    def report(self):
        self.get_logger().info(
            f'observations: depth={self.n_depth} plane={self.n_fallback} dropped={self.n_drop}')


def main():
    rclpy.init()
    rclpy.spin(DetectionLocalizer())


if __name__ == '__main__':
    main()
