"""LCM message: Velocity command to robot."""

import struct


class velocity_command:
    __slots__ = ["timestamp", "v_x", "v_yaw", "is_safe"]

    __typenames__ = ["int64_t", "double", "double", "boolean"]
    __dimensions__ = [None, None, None, None]

    def __init__(self):
        self.timestamp = 0
        self.v_x = 0.0
        self.v_yaw = 0.0
        self.is_safe = False

    def encode(self):
        return struct.pack(">qdd?", self.timestamp, self.v_x, self.v_yaw, self.is_safe)

    def decode(data):
        msg = velocity_command()
        msg.timestamp, msg.v_x, msg.v_yaw, msg.is_safe = struct.unpack_from(
            ">qdd?", data, 0
        )
        return msg

    decode = staticmethod(decode)

    def _get_hash_recursive(parents):
        return 0x2b3c4d5e

    _get_hash_recursive = staticmethod(_get_hash_recursive)

    def _get_packed_fingerprint():
        return struct.pack(">Q", velocity_command._get_hash_recursive([]))

    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        return 0x2b3c4d5e
