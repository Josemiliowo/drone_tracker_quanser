"""YOLO (Ultralytics) detector: color image -> vision_msgs/Detection2DArray.

Runs on the Jetson. Give it a TensorRT .engine (export on the Jetson itself):
    yolo export model=best.pt format=engine half=True imgsz=640
A .pt also works (slower).
"""
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

COLORS = [(0, 0, 255), (255, 128, 0), (0, 255, 0), (255, 0, 255)]


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.declare_parameter('model', 'best.engine')
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('image_topic', 'camera/color/image_raw')
        self.declare_parameter('annotated_rate', 5.0)  # Hz, 0 disables
        # rename the model's class names (the model says fuego/toma, the rest of the code uses fire/tap)
        self.declare_parameter('rename_from', ['fuego', 'toma'])
        self.declare_parameter('rename_to', ['fire', 'tap'])

        from ultralytics import YOLO  # imported late so --help etc. work without it
        model_path = self.get_parameter('model').value
        self.model = YOLO(model_path, task='detect')
        rename = dict(zip(self.get_parameter('rename_from').value, self.get_parameter('rename_to').value))
        self.names = {i: rename.get(n, n) for i, n in self.model.names.items()}
        self.conf = self.get_parameter('conf').value
        self.imgsz = self.get_parameter('imgsz').value
        rate = self.get_parameter('annotated_rate').value
        self.annot_period = 1.0 / rate if rate > 0 else None
        self.last_annot = 0.0
        self.bridge = CvBridge()

        self.pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.pub_img = self.create_publisher(CompressedImage, 'detections/image/compressed', 2)
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.on_image,
                                 qos_profile_sensor_data)
        self.get_logger().info(f'Loaded {model_path}, classes: {self.model.names} -> published as {self.names}')

    def on_image(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]

        out = Detection2DArray()
        out.header = msg.header  # keep the image stamp: the localizer matches on it
        for box, cls, score in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                   res.boxes.conf.tolist()):
            x0, y0, x1, y1 = box
            d = Detection2D()
            d.header = msg.header
            d.bbox.center.position.x = (x0 + x1) / 2
            d.bbox.center.position.y = (y0 + y1) / 2
            d.bbox.size_x = x1 - x0
            d.bbox.size_y = y1 - y0
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = self.names[int(cls)]
            h.hypothesis.score = float(score)
            d.results.append(h)
            out.detections.append(d)
        self.pub.publish(out)

        now = time.monotonic()
        if self.annot_period and now - self.last_annot >= self.annot_period:
            self.last_annot = now
            self.publish_annotated(msg, img, res)

    def publish_annotated(self, msg, img, res):
        for box, cls, score in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                   res.boxes.conf.tolist()):
            x0, y0, x1, y1 = map(int, box)
            col = COLORS[int(cls) % len(COLORS)]
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            cv2.putText(img, f'{self.names[int(cls)]} {score:.2f}', (x0, max(0, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        c = CompressedImage()
        c.header = msg.header
        c.format = 'jpeg'
        c.data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()
        self.pub_img.publish(c)


def main():
    rclpy.init()
    rclpy.spin(YoloDetector())


if __name__ == '__main__':
    main()
