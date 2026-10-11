"""C.R.A.T.E.R telemetry protocol: compact binary datagrams (stdlib `struct` + `zlib.crc32`).

Datagram = header (16 B) + payload + CRC32 (4 B) over header+payload. Big-endian.

    header: magic "CR" | version u8 | type u8 | session u16 | seq u32 | t_ms u32 | len u16

  session  random id chosen when the rover's sender starts, so the ground station notices
           a restart (sequence numbers start over)
  seq      per-datagram sequence number; every datagram is ACKed by this seq
  t_ms     rover simulation time when the datagram was created [ms]

Fixed-point units: positions int16 cm (+-327 m), angles int16 0.01 deg, speeds int16 cm/s,
times u16 cs or ms, sigma u16 cm. 0xFFFF in an unsigned field = "not available".
"""
import math
import struct
import zlib

MAGIC = b'CR'
VERSION = 1
HDR = struct.Struct('!2sBBHIIH')
CRC = struct.Struct('!I')
OVERHEAD = HDR.size + CRC.size
MAX_DATAGRAM = 1200            # stays under any sane MTU

HELLO, TELEMETRY, MISSION, EVENT, CRATER, ACK = 1, 2, 3, 4, 5, 0x80
TYPE_NAMES = {HELLO: 'HELLO', TELEMETRY: 'TELEMETRY', MISSION: 'MISSION', EVENT: 'EVENT',
              CRATER: 'CRATER', ACK: 'ACK'}
STATES = ['WAIT_NAV2', 'PATROL', 'EVADE', 'SHELTER']
NA16 = 0xFFFF

_HELLO = struct.Struct('!hhBBHHH')            # station x,y | danger dm | n_wp | emu loss per mille, delay ms, jitter ms
_HELLO_TAIL = struct.Struct('!H')             # emu corrupt (per mille)
_WP = struct.Struct('!hh')
_ROVER = struct.Struct('!hhhhIHHB')           # x, y, yaw, speed | retx u32, abandoned, srtt ms | n meteors
_METEOR = struct.Struct('!HhhhhhhhhHHBHB')    # 26 B, see encode_telemetry
_MISSION = struct.Struct('!BBHHHB')           # state, wp idx, threats, evasions, wp reached, has threat
_THREAT = struct.Struct('!HhhHHhh')           # id, impact x,y, tti cs, sigma cm, escape x,y
_CRATER = struct.Struct('!HhhB')              # id, x, y, radius dm
_ACK = struct.Struct('!II')                   # ack_seq, SACK bitmap


class ProtocolError(ValueError):
    """Datagram rejected: bad magic/version/length or CRC mismatch."""


def _cm(v):
    return max(-32768, min(32767, int(round(v * 100))))


def _u16(v, scale):
    if v is None or v < 0:
        return NA16
    return min(NA16 - 1, int(round(v * scale)))


def _f16(v, scale):
    return None if v == NA16 else v / scale


# ------------------------------------------------------------------ framing
def pack(ptype, session, seq, t_ms, payload=b''):
    head = HDR.pack(MAGIC, VERSION, ptype, session & 0xFFFF, seq & 0xFFFFFFFF, t_ms & 0xFFFFFFFF, len(payload))
    body = head + payload
    return body + CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def unpack(data):
    """-> (type, session, seq, t_ms, payload) or raises ProtocolError."""
    if len(data) < OVERHEAD:
        raise ProtocolError('short datagram')
    magic, ver, ptype, session, seq, t_ms, n = HDR.unpack_from(data)
    if magic != MAGIC or ver != VERSION:
        raise ProtocolError('bad magic/version')
    if len(data) != OVERHEAD + n:
        raise ProtocolError('length mismatch')
    (crc,) = CRC.unpack_from(data, HDR.size + n)
    if zlib.crc32(data[:HDR.size + n]) & 0xFFFFFFFF != crc:
        raise ProtocolError('CRC mismatch')
    return ptype, session, seq, t_ms, data[HDR.size:HDR.size + n]


# ------------------------------------------------------------------ payloads
def encode_hello(station, danger, waypoints, emu):
    out = _HELLO.pack(_cm(station[0]), _cm(station[1]), min(255, int(round(danger * 10))),
                      len(waypoints), int(round(emu['loss'] * 1000)), int(emu['delay_ms']), int(emu['jitter_ms']))
    out += _HELLO_TAIL.pack(int(round(emu['corrupt'] * 1000)))
    return out + b''.join(_WP.pack(_cm(x), _cm(y)) for x, y in waypoints)


def decode_hello(p):
    sx, sy, dng, n, loss, delay, jitter = _HELLO.unpack_from(p)
    (corrupt,) = _HELLO_TAIL.unpack_from(p, _HELLO.size)
    off = _HELLO.size + _HELLO_TAIL.size
    wps = [[x / 100, y / 100] for x, y in (_WP.unpack_from(p, off + i * _WP.size) for i in range(n))]
    return {'station': [sx / 100, sy / 100], 'danger_radius': dng / 10, 'waypoints': wps,
            'emu': {'loss': loss / 1000, 'delay_ms': delay, 'jitter_ms': jitter, 'corrupt': corrupt / 1000}}


def encode_telemetry(rover, link, meteors):
    """rover: x, y, yaw[rad], speed | link: retx, abandoned, srtt_ms | meteors: list of dicts
    id, x, y, alt, vx, vy, vz, impact_x/impact_y (or None), tti, sigma, threat, age, n_obs."""
    meteors = meteors[:40]
    out = _ROVER.pack(_cm(rover['x']), _cm(rover['y']), _cm(math.degrees(rover['yaw'])), _cm(rover['speed']),
                      int(link['retx']) & 0xFFFFFFFF, min(NA16, int(link['abandoned'])),
                      _u16(link['srtt_ms'], 1), len(meteors))
    for m in meteors:
        has_imp = m.get('impact_x') is not None
        out += _METEOR.pack(
            m['id'] & 0xFFFF, _cm(m['x']), _cm(m['y']), _cm(m['alt']),
            _cm(m['vx']), _cm(m['vy']), _cm(m['vz']),
            _cm(m['impact_x']) if has_imp else 0, _cm(m['impact_y']) if has_imp else 0,
            _u16(m['tti'] if has_imp else None, 100), _u16(m['sigma'] if has_imp else None, 100),
            (1 if m['threat'] else 0) | (2 if has_imp else 0),
            _u16(m['age'], 1000), min(255, m['n_obs']))
    return out


def decode_telemetry(p):
    x, y, yaw, spd, retx, aband, srtt, n = _ROVER.unpack_from(p)
    rover = {'x': x / 100, 'y': y / 100, 'yaw': yaw / 100, 'speed': spd / 100}   # yaw in degrees here
    link = {'retx': retx, 'abandoned': aband, 'srtt_ms': _f16(srtt, 1)}
    meteors = []
    for i in range(n):
        (mid, mx, my, alt, vx, vy, vz, ix, iy, tti, sig, flags, age, nobs) = \
            _METEOR.unpack_from(p, _ROVER.size + i * _METEOR.size)
        m = {'id': mid, 'x': mx / 100, 'y': my / 100, 'alt': alt / 100,
             'vx': vx / 100, 'vy': vy / 100, 'vz': vz / 100, 'threat': bool(flags & 1),
             'age': _f16(age, 1000), 'n_obs': nobs}
        if flags & 2:
            m.update({'impact_x': ix / 100, 'impact_y': iy / 100, 'tti': _f16(tti, 100), 'sigma': _f16(sig, 100)})
        meteors.append(m)
    return rover, link, meteors


def encode_mission(state, wp_idx, counters, threat):
    out = _MISSION.pack(STATES.index(state) if state in STATES else 255, min(255, wp_idx),
                        min(NA16, counters.get('threats', 0)), min(NA16, counters.get('evasions_completed', 0)),
                        min(NA16, counters.get('waypoints_reached', 0)), 1 if threat else 0)
    if threat:
        out += _THREAT.pack(threat['id'] & 0xFFFF, _cm(threat['impact'][0]), _cm(threat['impact'][1]),
                            _u16(max(0.0, threat['time_to_impact']), 100), _u16(threat['sigma'], 100),
                            _cm(threat['escape'][0]), _cm(threat['escape'][1]))
    return out


def decode_mission(p):
    st, wp, thr, eva, wpr, has = _MISSION.unpack_from(p)
    m = {'state': STATES[st] if st < len(STATES) else 'UNKNOWN', 'wp_idx': wp,
         'counters': {'threats': thr, 'evasions_completed': eva, 'waypoints_reached': wpr}, 'threat': None}
    if has:
        tid, ix, iy, tti, sig, ex, ey = _THREAT.unpack_from(p, _MISSION.size)
        m['threat'] = {'id': tid, 'impact': [ix / 100, iy / 100], 'time_to_impact': _f16(tti, 100) or 0.0,
                       'sigma': _f16(sig, 100) or 0.0, 'escape': [ex / 100, ey / 100]}
    return m


def encode_event(level, text):
    raw = text.encode('utf-8')[:200]
    return bytes([1 if level == 'warn' else 0]) + raw


def decode_event(p):
    return {'level': 'warn' if p[0] == 1 else 'info', 'text': p[1:].decode('utf-8', errors='replace')}


def encode_crater(cid, x, y, r):
    return _CRATER.pack(cid & 0xFFFF, _cm(x), _cm(y), min(255, int(round(r * 10))))


def decode_crater(p):
    cid, x, y, r = _CRATER.unpack(p)
    return {'id': cid, 'x': x / 100, 'y': y / 100, 'r': r / 10}


def encode_ack(ack_seq, sack):
    return _ACK.pack(ack_seq & 0xFFFFFFFF, sack & 0xFFFFFFFF)


def decode_ack(p):
    return _ACK.unpack(p)
