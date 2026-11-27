#!/usr/bin/env python3
"""QDrone 2 (drone 1) fire/tap detector -- plain Python 3.6, no ROS.

RealSense D435 (color + depth aligned to color) -> YOLO11 (TensorRT engine) -> for every box:
pixel centre, box size, robust depth -> sent to the laptop as one JSON line per frame over TCP.
A small JPEG with the boxes drawn is sent a few times per second for Foxglove.
The laptop (ROS 2 node `drone_link`) does the camera -> map transform with the mocap pose.

On the drone:
    python3 fire_detector.py --laptop 192.168.2.XX --engine ~/fire_drone_ws/models/best.engine
Build the engine once on the drone (TensorRT 8.2, ~10-20 min):
    /usr/src/tensorrt/bin/trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 --workspace=1024
Test on the laptop without a camera or GPU:
    python3 fire_detector.py --laptop 127.0.0.1 --onnx best.onnx --images some_folder/

Never blocks on the network: if the laptop is away, frames are simply not sent and it reconnects.
"""
import argparse
import base64
import glob
import json
import os
import socket
import sys
import time

import cv2
import numpy as np

NAMES = ['fuego', 'toma']            # model class order (see the .onnx metadata)
RENAME = {'fuego': 'fire', 'toma': 'tap'}


# --------------------------------------------------------------------------- model
def letterbox(img, size=640):
    """Resize keeping aspect ratio and pad to size x size (like Ultralytics). Returns img, scale, pad."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, px, py


def preprocess(img_bgr, size=640):
    lb, r, px, py = letterbox(img_bgr, size)
    x = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0   # BGR->RGB, HWC->CHW
    return np.ascontiguousarray(x[None]), r, px, py


def postprocess(out, r, px, py, img_w, img_h, conf=0.5, iou=0.45):
    """YOLO11 raw output (1, 4+nc, N) -> list of (cls, score, x0, y0, x1, y1) in image pixels."""
    p = out[0].T                                   # (N, 4+nc)
    scores = p[:, 4:]
    cls = scores.argmax(1)
    sc = scores[np.arange(len(cls)), cls]
    keep = sc >= conf
    if not keep.any():
        return []
    p, cls, sc = p[keep], cls[keep], sc[keep]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    x0 = np.clip((cx - w / 2 - px) / r, 0, img_w)
    y0 = np.clip((cy - h / 2 - py) / r, 0, img_h)
    x1 = np.clip((cx + w / 2 - px) / r, 0, img_w)
    y1 = np.clip((cy + h / 2 - py) / r, 0, img_h)
    dets = []
    for c in np.unique(cls):                       # NMS per class
        idx = np.where(cls == c)[0]
        boxes = [[float(x0[i]), float(y0[i]), float(x1[i] - x0[i]), float(y1[i] - y0[i])] for i in idx]
        kept = cv2.dnn.NMSBoxes(boxes, [float(sc[i]) for i in idx], conf, iou)
        for k in np.array(kept).reshape(-1):
            i = idx[int(k)]
            dets.append((int(c), float(sc[i]), float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i])))
    return dets


class TrtModel(object):
    """TensorRT 8.2 engine runner. Uses torch CUDA tensors as GPU buffers (no pycuda needed)."""

    def __init__(self, path):
        import tensorrt as trt
        import torch
        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        with open(path, 'rb') as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('could not load engine ' + path)
        self.ctx = self.engine.create_execution_context()
        self.bufs, self.out_idx = [], None
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = torch.float16 if self.engine.get_binding_dtype(i) == trt.float16 else torch.float32
            t = torch.empty(shape, dtype=dtype, device='cuda')
            self.bufs.append(t)
            if self.engine.binding_is_input(i):
                self.in_idx = i
            else:
                self.out_idx = i
        self.stream = torch.cuda.Stream()

    def __call__(self, x):
        torch = self.torch
        with torch.cuda.stream(self.stream):
            self.bufs[self.in_idx].copy_(torch.from_numpy(x))
            self.ctx.execute_async_v2([int(b.data_ptr()) for b in self.bufs], self.stream.cuda_stream)
            out = self.bufs[self.out_idx].float().cpu().numpy()
        return out


class OnnxModel(object):
    """CPU fallback for testing on the laptop (onnxruntime)."""

    def __init__(self, path):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.name = self.s.get_inputs()[0].name

    def __call__(self, x):
        return self.s.run(None, {self.name: x})[0]


# --------------------------------------------------------------------------- depth
def robust_depth(depth_m, x0, y0, x1, y1, shrink=0.5, dmin=0.3, dmax=3.0):
    """Median of valid depth (m) in the central `shrink` part of the box, or None."""
    h, w = depth_m.shape
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * shrink / 2.0, (y1 - y0) * shrink / 2.0
    a, b = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    c, d = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    patch = depth_m[c:d, a:b]
    vals = patch[(patch > dmin) & (patch < dmax)]
    if vals.size < 10:
        return None
    return float(np.median(vals))


# --------------------------------------------------------------------------- sources
class RealSense(object):
    def __init__(self, w=640, h=480, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        cprof = prof.get_stream(rs.stream.color).as_video_stream_profile()
        dprof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        i = cprof.get_intrinsics()
        self.K = [i.fx, i.fy, i.ppx, i.ppy]
        # where the color camera sits relative to the depth (left IR) camera = camera_link origin
        e = cprof.get_extrinsics_to(dprof)
        self.color_in_depth = list(e.translation)   # metres, depth optical frame (x right, y down, z fwd)
        for _ in range(15):                          # let auto exposure settle
            self.pipe.wait_for_frames()

    def read(self):
        fs = self.align.process(self.pipe.wait_for_frames())
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        if not c or not d:
            return None
        if c.get_frame_timestamp_domain() == self.rs.timestamp_domain.global_time:
            t = c.get_timestamp() / 1000.0             # host clock at capture
        else:
            t = time.time()
        depth_m = np.asanyarray(d.get_data()).astype(np.float32) * self.scale
        return t, np.asanyarray(c.get_data()), depth_m


class ImageFolder(object):
    """Testing without a camera: loops over images, constant fake depth."""

    def __init__(self, folder, depth=1.5):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.files = sorted(sum([glob.glob(os.path.join(folder, e)) for e in exts], []))
        if not self.files:
            raise RuntimeError('no images in ' + folder)
        img = cv2.imread(self.files[0])
        h, w = img.shape[:2]
        self.K = [w * 0.95, w * 0.95, w / 2.0, h / 2.0]   # roughly D435 color FOV
        self.color_in_depth = [0.015, 0.0, 0.0]
        self.depth, self.i = depth, 0

    def read(self):
        img = cv2.imread(self.files[self.i % len(self.files)])
        self.i += 1
        time.sleep(0.1)
        return time.time(), img, np.full(img.shape[:2], self.depth, np.float32)


# --------------------------------------------------------------------------- network
class Link(object):
    """Newline-delimited JSON to the laptop. Never blocks the detection loop for long."""

    def __init__(self, host, port):
        self.addr, self.sock, self.last_try = (host, port), None, 0.0

    def send(self, obj):
        if self.sock is None:
            if time.time() - self.last_try < 2.0:
                return False
            self.last_try = time.time()
            try:
                self.sock = socket.create_connection(self.addr, timeout=0.3)
                self.sock.settimeout(0.05)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print('connected to laptop %s:%d' % self.addr)
            except OSError:
                self.sock = None
                return False
        try:
            obj['t_send'] = time.time()                 # lets the laptop estimate the clock offset
            self.sock.sendall((json.dumps(obj, separators=(',', ':')) + '\n').encode())
            return True
        except OSError:
            print('laptop disconnected, retrying')
            self.sock.close()
            self.sock = None
            return False


def draw(img, dets):
    for c, s, x0, y0, x1, y1, z in dets:
        col = (0, 0, 255) if c == 0 else (255, 128, 0)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
        label = '%s %.2f %s' % (RENAME.get(NAMES[c], NAMES[c]), s, '%.2fm' % z if z else 'no depth')
        cv2.putText(img, label, (int(x0), max(12, int(y0) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--laptop', required=True, help='laptop IP (runs ros2 drone_link)')
    ap.add_argument('--port', type=int, default=18020)
    ap.add_argument('--engine', help='TensorRT engine (on the drone)')
    ap.add_argument('--onnx', help='ONNX model, CPU (laptop testing)')
    ap.add_argument('--images', help='folder of test images instead of the RealSense')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--jpeg-rate', type=float, default=5.0, help='annotated images per second (0 = off)')
    ap.add_argument('--jpeg-width', type=int, default=424, help='width of the annotated image sent')
    args = ap.parse_args()
    if not (args.engine or args.onnx):
        ap.error('give --engine (drone) or --onnx (laptop test)')

    model = TrtModel(args.engine) if args.engine else OnnxModel(args.onnx)
    src = ImageFolder(args.images) if args.images else RealSense()
    link = Link(args.laptop, args.port)
    print('model ready, K=%s, color offset from depth cam=%s' % (src.K, src.color_in_depth))

    n, t_fps, last_jpeg = 0, time.time(), 0.0
    while True:
        frame = src.read()
        if frame is None:
            continue
        t_cap, img, depth_m = frame
        h, w = img.shape[:2]
        x, r, px, py = preprocess(img)
        raw = model(x)
        dets = []
        for c, s, x0, y0, x1, y1 in postprocess(raw, r, px, py, w, h, args.conf):
            dets.append((c, s, x0, y0, x1, y1, robust_depth(depth_m, x0, y0, x1, y1)))

        msg = {'type': 'det', 't_cap': t_cap, 'w': w, 'h': h, 'K': src.K, 'ext': src.color_in_depth,
               'dets': [{'cls': RENAME.get(NAMES[c], NAMES[c]), 'score': round(s, 3),
                         'u': round((x0 + x1) / 2, 1), 'v': round((y0 + y1) / 2, 1),
                         'bw': round(x1 - x0, 1), 'bh': round(y1 - y0, 1),
                         'z': None if z is None else round(z, 4)} for c, s, x0, y0, x1, y1, z in dets]}
        link.send(msg)

        if args.jpeg_rate > 0 and time.time() - last_jpeg >= 1.0 / args.jpeg_rate:
            last_jpeg = time.time()
            small = draw(img.copy(), dets)
            k = args.jpeg_width / float(w)
            small = cv2.resize(small, (args.jpeg_width, int(h * k)))
            ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                link.send({'type': 'jpeg', 't_cap': t_cap, 'data': base64.b64encode(buf.tobytes()).decode()})

        n += 1
        if time.time() - t_fps >= 5.0:
            print('%.1f fps, last frame: %d detections' % (n / (time.time() - t_fps), len(dets)))
            n, t_fps = 0, time.time()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
