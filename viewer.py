from __future__ import annotations

import argparse
import socket
import struct

import cv2
import numpy as np


def recvall(conn: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main(port: int) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(1)
    print(f"VIEWER: Waiting for sender on port {port}...")
    conn, addr = server.accept()
    print(f"VIEWER: Connected: {addr}")

    try:
        while True:
            header = recvall(conn, 4)
            if header is None:
                print("VIEWER: Sender disconnected.")
                break
            size = struct.unpack(">I", header)[0]
            data = recvall(conn, size)
            if data is None:
                print("VIEWER: Sender disconnected.")
                break

            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            cv2.imshow("rknn", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        conn.close()
        server.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    main(args.port)
