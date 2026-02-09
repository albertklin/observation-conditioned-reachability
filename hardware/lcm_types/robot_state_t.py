"""LCM message: Robot state estimate from SLAM or other state estimator."""

import struct


class robot_state:
    __slots__ = ["timestamp", "x", "y", "theta"]

    __typenames__ = ["int64_t", "double", "double", "double"]
    __dimensions__ = [None, None, None, None]

    def __init__(self):
        self.timestamp = 0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def encode(self):
        return struct.pack(">qddd", self.timestamp, self.x, self.y, self.theta)

    def decode(data):
        msg = robot_state()
        msg.timestamp, msg.x, msg.y, msg.theta = struct.unpack_from(">qddd", data, 0)
        return msg

    decode = staticmethod(decode)

    def _get_hash_recursive(parents):
        return 0x1a2b3c4d

    _get_hash_recursive = staticmethod(_get_hash_recursive)

    def _get_packed_fingerprint():
        return struct.pack(">Q", robot_state._get_hash_recursive([]))

    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        return 0x1a2b3c4d
