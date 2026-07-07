"""UDP input primitives for UWB acquisition."""

import socket
import time


class UdpInput:
    def __init__(self, bind_ip="0.0.0.0", port=4210):
        self.bind_ip = bind_ip
        self.port = int(port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.bind_ip, self.port))
        self.socket.setblocking(False)

    def poll_payloads(self):
        payloads = []
        while True:
            try:
                payload, addr = self.socket.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                break
            payloads.append(
                {
                    "message": payload.decode("utf-8", errors="replace").strip(),
                    "addr": addr,
                    "received_at": time.time(),
                }
            )
        return payloads

    def close(self):
        try:
            self.socket.close()
        except OSError:
            pass
