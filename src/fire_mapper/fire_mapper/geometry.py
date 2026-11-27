"""Pure-numpy geometry helpers (no ROS imports, so they can be unit-tested anywhere)."""
import numpy as np


def quat_to_rot(x, y, z, w):
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R):
    """3x3 rotation matrix -> quaternion (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def make_T(t, q):
    """Translation (3,) + quaternion (x,y,z,w) -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(*q)
    T[:3, 3] = t
    return T


def yaw_only(T):
    """Keep translation and heading of T, drop roll/pitch (drone 2's odom frame is level)."""
    yaw = np.arctan2(T[1, 0], T[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    out[:3, 3] = T[:3, 3]
    return out


def deproject(u, v, z, fx, fy, cx, cy):
    """Pixel (u, v) at depth z (metres, along optical axis) -> point in optical frame."""
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def ray_plane(T_map_cam, u, v, fx, fy, cx, cy, plane_z):
    """Intersect the pixel ray with the horizontal plane z = plane_z in map.

    T_map_cam: 4x4 transform of the optical frame in map. Returns the map point or None.
    """
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    o = T_map_cam[:3, 3]
    d = T_map_cam[:3, :3] @ d_cam
    if abs(d[2]) < 1e-6:
        return None
    s = (plane_z - o[2]) / d[2]
    if s <= 0:
        return None
    return o + s * d


def robust_depth(depth_m, x0, y0, x1, y1, shrink=0.5, dmin=0.3, dmax=3.0):
    """Median of valid depth inside the central `shrink` fraction of the bbox.

    depth_m: HxW float array in metres (0 = invalid). Returns float or None.
    """
    h, w = depth_m.shape
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * shrink / 2, (y1 - y0) * shrink / 2
    a, b = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    c, d = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    patch = depth_m[c:d, a:b]
    vals = patch[(patch > dmin) & (patch < dmax)]
    if vals.size < 10:
        return None
    return float(np.median(vals))


# Motive Y-up -> ROS Z-up: rotate +90 deg about X.  x_ros = x_m, y_ros = -z_m, z_ros = y_m
YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]])


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mocap_to_map(p, q, up_axis='y', body_yaw_offset=0.0):
    """Convert a mocap pose (position p, quaternion q xyzw) to the ROS map frame (Z-up).

    The world AND the rigid body's local axes are converted the same way, so a rigid body
    created with the drone nose along Motive +X gives base_link x = nose with offset 0.
    body_yaw_offset (rad) rotates base_link about its up axis if the rigid body's local
    axes don't line up with the drone (nose = +x, left = +y).
    Returns (position (3,), quaternion xyzw (4,)).
    """
    p = np.asarray(p, dtype=float)
    R = quat_to_rot(*q)
    if up_axis.lower() == 'y':
        C = YUP_TO_ZUP
        p = C @ p
        R = C @ R @ C.T
    R = R @ rotz(body_yaw_offset)
    return p, rot_to_quat(R)
