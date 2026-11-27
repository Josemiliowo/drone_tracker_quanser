"""Runs on the host PC next to Simulink.

Simulink -> laptop: each pose (x, y, z, roll, pitch, yaw, isTracking as doubles) from the QUARC Stream
                    Server (port 18005) goes to the laptop as ONE UDP packet (port 18010).
laptop -> Simulink: after `ros2 service call /targets/send std_srvs/srv/Trigger` on the laptop, the fire/tap
                    targets arrive here by UDP (port 18011) and are sent to Simulink as the Stream Server's
                    reply, every cycle, until the next /targets/send.

Simulink's Stream Server must RECEIVE 1 + 4*MAX_ROWS doubles (41 for 10 rows):
    [count; X(1..10); Y(1..10); Z(1..10); Yaw(1..10)]
    count = u(1);  M = reshape(u(2:41), 10, 4);    % rows = [X Y Z Yaw], world (Simulink) frame
Rows after `count` repeat the last real target (never 0,0,0). count = 0 means nothing was sent yet.

UDP never waits: if the WiFi stalls or the laptop is off, packets are simply dropped and the
Simulink loop keeps running at full speed. No reconnects, no freezes.
"""
import socket
import struct
import time

HOST = "192.168.2.11"          # Simulink's Stream Server (this PC)
PORT = 18005
LAPTOP_IP = "192.168.2.24"     # laptop running ROS (check with `ip addr` on the laptop)
LAPTOP_PORT = 18010            # tcp_pose_bridge (mode: udp)
TARGETS_PORT = 18011           # targets from the laptop's /targets/send arrive here

SIMULINK_DATA_DIMENSIONS = 7   # x, y, z, roll, pitch, yaw, isTracking
MAX_ROWS = 10                  # must match the laptop (max_rows) and Simulink's receive size
REPLY_SIZE = 1 + 4 * MAX_ROWS  # doubles Simulink's Stream Server receives (41)
SEND_REPLY = True              # False = never send anything to Simulink (diagnostic: is the reply the problem?)


def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Disconnected")
        data.extend(chunk)
    return bytes(data)


def main():
    rec = struct.Struct("<%dd" % SIMULINK_DATA_DIMENSIONS)
    reply_len = 8 * REPLY_SIZE
    reply = struct.pack("<%dd" % REPLY_SIZE, *([0.0] * REPLY_SIZE))   # count 0 until the first send

    laptop = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # UDP to the laptop
    laptop.setblocking(False)                                   # never wait on the laptop
    targets_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    targets_in.bind(("0.0.0.0", TARGETS_PORT))
    targets_in.setblocking(False)                               # checked every cycle, never waits
    sent = dropped = 0
    t_report = time.time()

    while True:                                                 # reconnect forever (model restarts etc.)
        try:
            client = socket.create_connection((HOST, PORT), timeout=3.0)
        except OSError as e:
            print("cannot connect to Simulink %s:%d (%s) - is the model running? retrying in 1 s" % (HOST, PORT, e))
            time.sleep(1.0)
            continue
        client.settimeout(2.0)                                  # no pose for 2 s -> treat as disconnected
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Connected to QUARC %s:%d, sending poses to %s:%d (UDP), targets in on UDP %d, reply to Simulink %s"
              % (HOST, PORT, LAPTOP_IP, LAPTOP_PORT, TARGETS_PORT,
                 ("ON (%d doubles)" % REPLY_SIZE) if SEND_REPLY else "OFF"))
        t_conn, n_poses, why = time.time(), 0, ""
        try:
            while True:
                data = recv_exact(client, rec.size)             # one pose from Simulink
                n_poses += 1
                try:
                    laptop.sendto(data, (LAPTOP_IP, LAPTOP_PORT))   # forward the same 56 bytes
                    sent += 1
                except OSError:                                 # buffer full / network down: drop it
                    dropped += 1

                try:                                            # new targets from the laptop?
                    while True:
                        pkt, addr = targets_in.recvfrom(4096)
                        if len(pkt) != reply_len:
                            print("ignored targets packet of %d bytes (expected %d: check MAX_ROWS)"
                                  % (len(pkt), reply_len))
                        elif pkt != reply:
                            reply = pkt
                            v = struct.unpack("<%dd" % REPLY_SIZE, reply)
                            n = int(v[0])
                            rows = [(v[1 + i], v[1 + MAX_ROWS + i], v[1 + 2 * MAX_ROWS + i]) for i in range(n)]
                            print("NEW TARGETS for Simulink: %d -> %s" % (
                                n, ", ".join("(%.2f, %.2f, %.2f)" % r for r in rows) or "none"))
                except (BlockingIOError, OSError):
                    pass

                if SEND_REPLY:
                    client.sendall(reply)                       # Simulink receives the targets every cycle

                if time.time() - t_report >= 5.0:               # one status line every 5 s
                    print("%.0f poses/s sent, %d dropped, last %s | targets for Simulink: %d" % (
                        sent / (time.time() - t_report), dropped, [round(v, 3) for v in rec.unpack(data)],
                        int(struct.unpack("<d", reply[:8])[0])))
                    sent = dropped = 0
                    t_report = time.time()
        except socket.timeout:
            why = "no data from Simulink for 2 s"
        except (ConnectionError, OSError) as e:
            why = "Simulink closed the connection (%s)" % e
        finally:
            client.close()
        print("DISCONNECTED after %.1f s and %d poses: %s. Reconnecting in 1 s..."
              % (time.time() - t_conn, n_poses, why))
        time.sleep(1.0)

if __name__ == "__main__":
    main()
