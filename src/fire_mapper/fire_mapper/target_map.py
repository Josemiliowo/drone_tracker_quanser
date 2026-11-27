"""Merge per-frame observations into a map of confirmed targets (fires / taps).

Observation within merge_radius of an existing target of the same class -> running mean.
Otherwise -> new candidate. Candidate becomes confirmed after min_hits observations.
Publishes /targets (Detection3DArray, map), /targets/markers (MarkerArray) and saves YAML.
"""
import os

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

CLASS_COLORS = {'fire': (1.0, 0.2, 0.1), 'tap': (0.1, 0.4, 1.0)}


class Target:
    def __init__(self, tid, cls, p, score):
        self.id, self.cls = tid, cls
        self.sum = np.array(p, dtype=float)
        self.hits = 1
        self.best_score = score

    @property
    def pos(self):
        return self.sum / self.hits

    def add(self, p, score):
        self.sum += p
        self.hits += 1
        self.best_score = max(self.best_score, score)


class TargetMap(Node):
    def __init__(self):
        super().__init__('target_map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('merge_radius', 0.25)
        self.declare_parameter('min_hits', 5)
        self.declare_parameter('max_hits', 200)  # cap so old observations keep some weight
        self.declare_parameter('marker_size', 0.15)
        self.declare_parameter('save_path', os.path.expanduser('~/fire_drone_targets.yaml'))
        self.declare_parameter('publish_rate', 2.0)

        g = lambda n: self.get_parameter(n).value
        self.frame, self.radius, self.min_hits = g('map_frame'), g('merge_radius'), g('min_hits')
        self.max_hits, self.size = g('max_hits'), g('marker_size')
        self.save_path = os.path.expanduser(g('save_path'))
        self.targets: list[Target] = []
        self.next_id = 0
        self.dirty = False

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Detection3DArray, '/targets', latched)
        self.pub_m = self.create_publisher(MarkerArray, '/targets/markers', latched)
        self.create_subscription(Detection3DArray, 'target_observations', self.on_obs, 50)
        self.create_service(Trigger, '/targets/clear', self.on_clear)
        self.create_timer(1.0 / g('publish_rate'), self.publish)

    def on_obs(self, msg: Detection3DArray):
        for d in msg.detections:
            c = d.bbox.center.position
            p = np.array([c.x, c.y, c.z])
            cls = d.results[0].hypothesis.class_id
            score = d.results[0].hypothesis.score
            best, best_dist = None, self.radius
            for t in self.targets:
                if t.cls == cls:
                    dist = np.linalg.norm(t.pos[:2] - p[:2])  # match in the floor plane
                    if dist < best_dist:
                        best, best_dist = t, dist
            if best is None:
                self.targets.append(Target(self.next_id, cls, p, score))
                self.next_id += 1
            elif best.hits < self.max_hits:
                was = best.hits >= self.min_hits
                best.add(p, score)
                if not was and best.hits >= self.min_hits:
                    x, y, z = best.pos
                    self.get_logger().info(f'CONFIRMED {cls} #{best.id} at ({x:.2f}, {y:.2f}, {z:.2f})')
            self.dirty = True

    def confirmed(self):
        return [t for t in self.targets if t.hits >= self.min_hits]

    def publish(self):
        now = self.get_clock().now().to_msg()
        arr = Detection3DArray()
        arr.header.stamp, arr.header.frame_id = now, self.frame
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for t in self.confirmed():
            x, y, z = map(float, t.pos)
            d = Detection3D()
            d.header = arr.header
            d.id = str(t.id)
            d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z = x, y, z
            d.bbox.center.orientation.w = 1.0
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = t.cls, float(t.best_score)
            h.pose.pose = d.bbox.center
            d.results.append(h)
            arr.detections.append(d)

            col = CLASS_COLORS.get(t.cls, (0.8, 0.8, 0.8))
            m = Marker()
            m.header = arr.header
            m.ns, m.id, m.type = 'targets', t.id, Marker.CYLINDER
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = self.size
            m.scale.z = self.size
            m.color.r, m.color.g, m.color.b = col
            m.color.a = 0.9
            markers.markers.append(m)
            txt = Marker()
            txt.header = arr.header
            txt.ns, txt.id, txt.type = 'labels', t.id, Marker.TEXT_VIEW_FACING
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = x, y, z + 0.25
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.12
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f'{t.cls} #{t.id} ({t.hits})'
            markers.markers.append(txt)
        self.pub.publish(arr)
        self.pub_m.publish(markers)
        if self.dirty:
            self.save()
            self.dirty = False

    def save(self):
        data = [{'id': t.id, 'class': t.cls, 'x': round(float(t.pos[0]), 3),
                 'y': round(float(t.pos[1]), 3), 'z': round(float(t.pos[2]), 3), 'hits': t.hits}
                for t in self.confirmed()]
        tmp = self.save_path + '.tmp'
        with open(tmp, 'w') as f:
            yaml.safe_dump({'frame': self.frame, 'targets': data}, f, sort_keys=False)
        os.replace(tmp, self.save_path)

    def on_clear(self, req, resp):
        self.targets.clear()
        self.dirty = True
        resp.success, resp.message = True, 'target map cleared'
        return resp


def main():
    rclpy.init()
    rclpy.spin(TargetMap())


if __name__ == '__main__':
    main()
