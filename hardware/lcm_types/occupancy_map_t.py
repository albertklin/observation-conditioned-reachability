"""LCM message: Occupancy map from SLAM."""

import struct


class occupancy_map:
    __slots__ = [
        "timestamp",
        "width",
        "height",
        "resolution",
        "origin_x",
        "origin_y",
        "map",
    ]

    __typenames__ = [
        "int64_t",
        "int32_t",
        "int32_t",
        "float",
        "float",
        "float",
        "boolean",
    ]

    __dimensions__ = [None, None, None, None, None, None, [40000]]

    def __init__(self):
        self.timestamp = 0
        self.width = 0
        self.height = 0
        self.resolution = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.map = [False] * 40000

    def encode(self):
        buf = struct.pack(">qiifff", self.timestamp, self.width, self.height,
                          self.resolution, self.origin_x, self.origin_y)
        buf += struct.pack(">%dB" % 40000, *[int(x) for x in self.map])
        return buf

    @staticmethod
    def decode(data):
        msg = occupancy_map()
        offset = 0
        (msg.timestamp, msg.width, msg.height, msg.resolution,
         msg.origin_x, msg.origin_y) = struct.unpack_from(">qiifff", data, offset)
        offset += struct.calcsize(">qiifff")
        msg.map = list(struct.unpack_from(">%dB" % 40000, data, offset))
        msg.map = [bool(x) for x in msg.map]
        return msg

    def _get_hash_recursive(parents):
        return 0x8a3d8e2c

    _packed_fingerprint = struct.pack(">Q", 0x8a3d8e2c)

    def _get_packed_fingerprint():
        return occupancy_map._packed_fingerprint
