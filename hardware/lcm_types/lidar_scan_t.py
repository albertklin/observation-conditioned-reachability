"""LCM message: Raw LiDAR scan data from sensor."""

import struct


class lidar_scan:
    __slots__ = ["timestamp", "count", "angle", "range", "quality"]

    __typenames__ = ["int64_t", "int32_t", "double", "double", "int8_t"]
    __dimensions__ = [None, None, [1400], [1400], [1400]]

    def __init__(self):
        self.timestamp = 0
        self.count = 0
        self.angle = [0.0] * 1400
        self.range = [0.0] * 1400
        self.quality = [0] * 1400

    def encode(self):
        buf = struct.pack(">qiH", self.timestamp, self.count, 1400)
        buf += struct.pack(">1400d", *self.angle)
        buf += struct.pack(">1400d", *self.range)
        buf += struct.pack(">1400b", *self.quality)
        return buf

    def decode(data):
        msg = lidar_scan()
        offset = 0
        msg.timestamp, msg.count, _ = struct.unpack_from(">qiH", data, offset)
        offset += struct.calcsize(">qiH")
        msg.angle = list(struct.unpack_from(">1400d", data, offset))
        offset += struct.calcsize(">1400d")
        msg.range = list(struct.unpack_from(">1400d", data, offset))
        offset += struct.calcsize(">1400d")
        msg.quality = list(struct.unpack_from(">1400b", data, offset))
        return msg

    decode = staticmethod(decode)

    def _get_hash_recursive(parents):
        return 0x8a3b2c1d

    _get_hash_recursive = staticmethod(_get_hash_recursive)

    def _get_packed_fingerprint():
        return struct.pack(">Q", lidar_scan._get_hash_recursive([]))

    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        return 0x8a3b2c1d
