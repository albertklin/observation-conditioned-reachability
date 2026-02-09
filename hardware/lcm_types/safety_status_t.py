"""LCM message: Safety filter status for monitoring and logging."""

import struct


class safety_status:
    __slots__ = [
        "timestamp",
        "nom_v_x",
        "nom_v_yaw",
        "safe_v_x",
        "safe_v_yaw",
        "state",
        "lidar_state",
        "value",
        "threshold",
        "is_intervening",
        "dst_dxdy",
        "dst_dth",
        "num_rays",
        "inp_lidar",
    ]

    __typenames__ = [
        "int64_t",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
        "double",
        "boolean",
        "double",
        "double",
        "int32_t",
        "double",
    ]
    __dimensions__ = [
        None,
        None,
        None,
        None,
        None,
        [3],
        [3],
        None,
        None,
        None,
        None,
        None,
        None,
        [100],
    ]

    def __init__(self):
        self.timestamp = 0
        self.nom_v_x = 0.0
        self.nom_v_yaw = 0.0
        self.safe_v_x = 0.0
        self.safe_v_yaw = 0.0
        self.state = [0.0] * 3
        self.lidar_state = [0.0] * 3
        self.value = 0.0
        self.threshold = 0.0
        self.is_intervening = False
        self.dst_dxdy = 0.0
        self.dst_dth = 0.0
        self.num_rays = 0
        self.inp_lidar = [0.0] * 100

    def encode(self):
        buf = struct.pack(
            ">qdddd",
            self.timestamp,
            self.nom_v_x,
            self.nom_v_yaw,
            self.safe_v_x,
            self.safe_v_yaw,
        )
        buf += struct.pack(">3d", *self.state)
        buf += struct.pack(">3d", *self.lidar_state)
        buf += struct.pack(
            ">dd?ddi",
            self.value,
            self.threshold,
            self.is_intervening,
            self.dst_dxdy,
            self.dst_dth,
            self.num_rays,
        )
        buf += struct.pack(">100d", *self.inp_lidar)
        return buf

    def decode(data):
        msg = safety_status()
        offset = 0

        (
            msg.timestamp,
            msg.nom_v_x,
            msg.nom_v_yaw,
            msg.safe_v_x,
            msg.safe_v_yaw,
        ) = struct.unpack_from(">qdddd", data, offset)
        offset += struct.calcsize(">qdddd")

        msg.state = list(struct.unpack_from(">3d", data, offset))
        offset += struct.calcsize(">3d")

        msg.lidar_state = list(struct.unpack_from(">3d", data, offset))
        offset += struct.calcsize(">3d")

        (
            msg.value,
            msg.threshold,
            msg.is_intervening,
            msg.dst_dxdy,
            msg.dst_dth,
            msg.num_rays,
        ) = struct.unpack_from(">dd?ddi", data, offset)
        offset += struct.calcsize(">dd?ddi")

        msg.inp_lidar = list(struct.unpack_from(">100d", data, offset))
        return msg

    decode = staticmethod(decode)

    def _get_hash_recursive(parents):
        return 0x3c4d5e6f

    _get_hash_recursive = staticmethod(_get_hash_recursive)

    def _get_packed_fingerprint():
        return struct.pack(">Q", safety_status._get_hash_recursive([]))

    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

    def get_hash(self):
        return 0x3c4d5e6f
