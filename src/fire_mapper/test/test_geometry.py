"""Check the full camera->map chain for a downward camera on the drone edge (pure numpy)."""
import numpy as np

from fire_mapper.geometry import (make_T, rot_to_quat, quat_to_rot, deproject, ray_plane,
                                  robust_depth, yaw_only)


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


# drone at (1, 2, 1.5), yaw 90 deg (facing +y in map)
T_map_base = T_from(rpy(0, 0, np.pi / 2), [1.0, 2.0, 1.5])
# camera 0.2 m forward, 0.05 m below the pivot, pitched +90 deg (looking down) -- as in drone1.launch.py
T_base_cam = T_from(rpy(0, np.pi / 2, 0), [0.2, 0.0, -0.05])
# realsense camera_link -> color optical (x right, y down, z forward)
T_cam_opt = T_from(rpy(-np.pi / 2, 0, -np.pi / 2), [0, 0, 0])
T_map_opt = T_map_base @ T_base_cam @ T_cam_opt

FX = FY = 600.0
CX, CY = 320.0, 240.0


def test_quat_roundtrip():
    R = rpy(0.3, -0.7, 2.0)
    assert np.allclose(quat_to_rot(*rot_to_quat(R)), R)


def test_camera_looks_down():
    z_axis_in_map = T_map_opt[:3, 2]
    assert np.allclose(z_axis_in_map, [0, 0, -1])
    # camera is 0.2 m "forward" of the drone = +y in map because of the 90 deg yaw
    assert np.allclose(T_map_opt[:3, 3], [1.0, 2.2, 1.45])


def test_center_pixel_hits_floor_below_camera():
    p = (T_map_opt @ np.append(deproject(CX, CY, 1.45, FX, FY, CX, CY), 1))[:3]
    assert np.allclose(p, [1.0, 2.2, 0.0])


def test_offset_pixel_depth_and_plane_agree():
    u, v = 500.0, 100.0
    p_plane = ray_plane(T_map_opt, u, v, FX, FY, CX, CY, 0.0)
    p_depth = (T_map_opt @ np.append(deproject(u, v, 1.45, FX, FY, CX, CY), 1))[:3]
    assert np.allclose(p_plane, p_depth)
    assert abs(p_plane[2]) < 1e-9


def test_robust_depth_ignores_holes():
    d = np.zeros((480, 640), np.float32)
    d[200:280, 300:340] = 1.2
    d[230:240, 310:320] = 0.0  # hole
    assert abs(robust_depth(d, 280, 180, 360, 300) - 1.2) < 1e-6
    assert robust_depth(np.zeros((480, 640), np.float32), 280, 180, 360, 300) is None


def test_target_in_drone2_frame():
    # drone 2 starts at (3, 0) facing +y (yaw 90). A target at map (3, 2) is 2 m straight ahead.
    T_map_odom = yaw_only(make_T([3, 0, 0.1], rot_to_quat(rpy(0.02, -0.03, np.pi / 2))))
    p = (np.linalg.inv(T_map_odom) @ np.array([3, 2, 0.1, 1]))[:3]
    assert np.allclose(p, [2, 0, 0], atol=1e-9)


def test_motive_yup_conversion():
    from fire_mapper.geometry import mocap_to_map
    # Motive Y-up: drone 1.5 m up (Y), at Motive x=1, z=-2; identity orientation (nose along Motive +X)
    p, q = mocap_to_map([1.0, 1.5, -2.0], [0, 0, 0, 1], 'y')
    assert np.allclose(p, [1.0, 2.0, 1.5])
    assert np.allclose(np.abs(q), [0, 0, 0, 1])
    # drone yawed +90 deg about Motive's up (Y) axis -> +90 deg about map Z
    s = np.sqrt(0.5)
    _, q = mocap_to_map([0, 0, 0], [0, s, 0, s], 'y')
    R = quat_to_rot(*q)
    assert np.allclose(R[:, 0], [0, 1, 0])   # nose now points +y in map
    assert np.allclose(R[:, 2], [0, 0, 1])   # still level
    # yaw offset: rigid body built with nose along its local -Z(Motive) = +Y(map) -> offset fixes it
    _, q = mocap_to_map([0, 0, 0], [0, 0, 0, 1], 'y', np.pi / 2)
    assert np.allclose(quat_to_rot(*q)[:, 0], [0, 1, 0])


def test_tcp_decode_tracked_flag():
    from fire_mapper.tcp_pose_bridge import decode_record
    f = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'tracked']
    assert decode_record((1, 2, 3, 0, 0, 0, 1.0), ['a'], f)['a'][2] is True
    assert decode_record((1, 2, 3, 0, 0, 0, 0.0), ['a'], f)['a'][2] is False
    assert decode_record((1, 2, 3, 0, 0, 0), ['a'], f[:6])['a'][2] is True   # no flag -> tracked


def test_tcp_decode_quat_and_euler():
    from fire_mapper.tcp_pose_bridge import decode_record
    out = decode_record((1, 2, 3, 0, 0, 0, 1, 9, 9, 9, 0, 0, 0.7071068, 0.7071068),
                        ['a', 'b'], ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    assert out['a'][0] == (1, 2, 3) and out['b'][1][3] > 0.7
    out = decode_record((5.0, 1, 2, 3, 0, 0, 90), ['a'], ['skip', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'], degrees=True)
    assert out['a'][0] == (1, 2, 3)
    assert np.allclose(quat_to_rot(*out['a'][1])[:, 0], [0, 1, 0], atol=1e-6)


def test_pack_for_simulink_pads_with_last_row():
    from fire_mapper.target_dispatcher import pack_for_simulink
    v = pack_for_simulink([(1, 2, 0.1, 0), (3, 4, 0.16, 0)], 4)
    assert len(v) == 1 + 4 * 4 and v[0] == 2
    M = np.array(v[1:]).reshape(4, 4, order='F')          # what Simulink's reshape(u(2:end), N, 4) gives
    assert np.allclose(M[0], [1, 2, 0.1, 0]) and np.allclose(M[1], [3, 4, 0.16, 0])
    assert np.allclose(M[2], M[1]) and np.allclose(M[3], M[1])   # padding = last real target, never (0,0,0)
    e = pack_for_simulink([], 4)
    assert e[0] == 0 and not any(e[1:])


def test_nn_tour_order():
    from fire_mapper.target_dispatcher import nn_tour
    pts = [(5, 0), (1, 0), (3, 0)]
    assert nn_tour(pts, (0, 0)) == [1, 2, 0]
    assert nn_tour(pts, None) == [0, 1, 2]


def test_matlab_matrix_format():
    from fire_mapper.target_dispatcher import pack_for_simulink, matlab_matrix
    txt = matlab_matrix(pack_for_simulink([(1, 2, 0.5, 0), (3, 4, 0.5, 0)], 3), 3)
    assert txt == 'count = 2;\nM = [1.000,2.000,0.500,0.000;3.000,4.000,0.500,0.000;3.000,4.000,0.500,0.000];'
