"""LCM message: MPS planner trajectory."""

import struct


class mps_trajectory:
    __slots__ = ["timestamp", "num_points", "xs", "ys", "ths"]

    __typenames__ = ["int64_t", "int32_t", "double", "double", "double"]

    __dimensions__ = [None, None, [100], [100], [100]]

    def __init__(self):
        self.timestamp = 0
        self.num_points = 0
        self.xs = [0.0] * 100
        self.ys = [0.0] * 100
        self.ths = [0.0] * 100

    def encode(self):
        buf = struct.pack(">qi", self.timestamp, self.num_points)
        buf += struct.pack(">%dd" % 100, *self.xs)
        buf += struct.pack(">%dd" % 100, *self.ys)
        buf += struct.pack(">%dd" % 100, *self.ths)
        return buf

    @staticmethod
    def decode(data):
        msg = mps_trajectory()
        offset = 0
        msg.timestamp, msg.num_points = struct.unpack_from(">qi", data, offset)
        offset += struct.calcsize(">qi")
        msg.xs = list(struct.unpack_from(">%dd" % 100, data, offset))
        offset += struct.calcsize(">%dd" % 100)
        msg.ys = list(struct.unpack_from(">%dd" % 100, data, offset))
        offset += struct.calcsize(">%dd" % 100)
        msg.ths = list(struct.unpack_from(">%dd" % 100, data, offset))
        return msg

    def _get_hash_recursive(parents):
        return 0x7b2c9e1f

    _packed_fingerprint = struct.pack(">Q", 0x7b2c9e1f)

    def _get_packed_fingerprint():
        return mps_trajectory._packed_fingerprint
