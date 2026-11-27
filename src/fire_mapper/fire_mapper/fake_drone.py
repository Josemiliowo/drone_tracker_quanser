"""Fake drones for testing the laptop side without hardware (no OptiTrack, Simulink, or QDrone needed).

Sends exactly what the real system sends:
  * poses for drone 1 and drone 2 as UDP packets of 14 doubles (like host/quarc_relay.py) -> tcp_pose_bridge
  * detections + a drawn camera image over TCP (like drone/fire_detector.py) -> drone_link
Drone 1 flies a slow circle at 1.5 m over 2 fires and 1 tap, with the D435 looking down. Drone 2 stays still.

    ros2 launch fire_mapper ground.launch.py mocap_source:=tcp      # terminal 1
    ros2 run fire_mapper fake_drone                                  # terminal 2
Then in Foxglove: drones moving, dots under drone 1, red/blue cylinders appearing at the objects.
`ros2 service call /targets/send std_srvs/srv/Trigger` then prints the matrix for Simulink.
"""
import argparse
import base64
import json
import math
import socket
import struct
import time

import cv2
import numpy as np


# objects on the floor (map / Simulink frame: x fwd, y left, z up) -- the "truth" the map should find
OBJECTS = [('fire', 1.0, 0.5, 0.10), ('fire', -0.6, -1.0, 0.10), ('tap', 0.4, -1.3, 0.16)]
SIZE = {'fire': 0.12, 'tap': 0.10}                    # m, rough object width for the box size

W, H, FX, FY, CX, CY = 640, 480, 615.0, 615.0, 320.0, 240.0
CAM_XYZ = (0.2634, 0.0, -0.0319)                      # same as config (drone_link cam_xyz)
EXT = (0.015, 0.0, 0.0)                               # color camera offset in the depth optical frame
OPT_IN_LINK = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


T_BASE_OPT = T_from(rpy(0, math.pi / 2, 0), CAM_XYZ) @ T_from(OPT_IN_LINK, OPT_IN_LINK @ np.array(EXT))


def drone1_pose(t, radius=1.2, height=1.5, speed=0.3):
    a = speed * t / radius
    x, y = radius * math.cos(a), radius * math.sin(a)
    yaw = a + math.pi / 2                             # nose along the direction of travel
    return x, y, height, yaw


def detections(x, y, z, yaw):
    """Project the objects into drone 1's camera -> fire_detector.py style detections."""
    T_map_opt = T_from(rpy(0, 0, yaw), (x, y, z)) @ T_BASE_OPT
    T_opt_map = np.linalg.inv(T_map_opt)
    dets = []
    for cls, ox, oy, oh in OBJECTS:
        p = T_opt_map @ np.array([ox, oy, oh, 1.0])     # top of the object, in the optical frame
        if p[2] < 0.3:
            continue
        u, v = FX * p[0] / p[2] + CX, FY * p[1] / p[2] + CY
        b = FX * SIZE[cls] / p[2]
        if b / 2 < u < W - b / 2 and b / 2 < v < H - b / 2:
            dets.append({'cls': cls, 'score': 0.9, 'u': round(u, 1), 'v': round(v, 1), 'bw': round(b, 1),
                         'bh': round(b, 1), 'z': round(float(p[2]) + np.random.normal(0, 0.01), 4),
                         'yellow': 0.25 if cls == 'fire' else None})
    return dets


def image(dets):
    img = np.full((H, W, 3), (70, 110, 80), np.uint8)             # green-grey floor
    cv2.putText(img, 'FAKE DRONE (test)', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for d in dets:
        u, v, r = int(d['u']), int(d['v']), int(d['bw'] / 2)
        col = (0, 0, 255) if d['cls'] == 'fire' else (255, 128, 0)
        cv2.circle(img, (u, v), r, (30, 30, 200) if d['cls'] == 'fire' else (200, 200, 200), -1)
        if d['cls'] == 'fire':
            cv2.circle(img, (u, v), max(2, r // 3), (40, 180, 240), -1)   # yellow centre
        cv2.rectangle(img, (u - r, v - r), (u + r, v + r), col, 2)
        cv2.putText(img, '%s %.2f %.2fm' % (d['cls'], d['score'], d['z']), (u - r, max(12, v - r - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    img = cv2.resize(img, (424, 318))
    return base64.b64encode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--laptop', default='127.0.0.1', help='PC running ground.launch.py')
    ap.add_argument('--pose-port', type=int, default=18010)
    ap.add_argument('--det-port', type=int, default=18020)
    ap.add_argument('--no-detections', action='store_true', help='only send poses')
    args = ap.parse_args()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tcp, last_try = None, 0.0
    t0, last_det, last_img, last_print = time.time(), 0.0, 0.0, 0.0
    print('fake drones -> %s (poses UDP %d, detections TCP %d). Objects: %s'
          % (args.laptop, args.pose_port, args.det_port, OBJECTS))
    while True:
        now = time.time()
        t = now - t0
        x, y, z, yaw = drone1_pose(t)
        d1 = (x, y, z, 0.0, 0.0, yaw, 1.0)                       # x y z roll pitch yaw isTracking
        d2 = (-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)                # drone 2 parked
        udp.sendto(struct.pack('<14d', *(d1 + d2)), (args.laptop, args.pose_port))

        if not args.no_detections and now - last_det >= 1 / 30.0:
            last_det = now
            if tcp is None and now - last_try > 2.0:
                last_try = now
                try:
                    tcp = socket.create_connection((args.laptop, args.det_port), timeout=0.5)
                    print('connected to drone_link')
                except OSError:
                    tcp = None
            if tcp is not None:
                dets = detections(x, y, z, yaw)
                msgs = [{'type': 'det', 't_cap': now, 'w': W, 'h': H, 'K': [FX, FY, CX, CY], 'ext': list(EXT),
                         'dets': dets, 'rejected': 0}]
                if now - last_img >= 0.2:
                    last_img = now
                    msgs.append({'type': 'jpeg', 't_cap': now, 'data': image(dets)})
                try:
                    for m in msgs:
                        m['t_send'] = time.time()
                        tcp.sendall((json.dumps(m) + '\n').encode())
                except OSError:
                    print('drone_link disconnected, retrying')
                    tcp.close()
                    tcp = None
        if now - last_print >= 5.0:
            last_print = now
            print('t=%5.1f s  drone1 at (%.2f, %.2f, %.2f) yaw %.0f deg, %d objects in view'
                  % (t, x, y, z, math.degrees(yaw) % 360, len(detections(x, y, z, yaw))))
        time.sleep(0.01)                                          # 100 Hz poses


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
