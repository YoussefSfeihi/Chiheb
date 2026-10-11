#!/usr/bin/env python3
"""Stand-alone tests of the telemetry protocol and reliable UDP link (stdlib only).

    python3 src/crater_dashboard/test/test_link.py
"""
import os
import random
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from crater_dashboard import protocol as P  # noqa: E402
from crater_dashboard.link import ReliableReceiver, ReliableSender  # noqa: E402


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_roundtrip():
    emu = {'loss': 0.1, 'delay_ms': 80, 'jitter_ms': 40, 'corrupt': 0.01}
    h = P.decode_hello(P.encode_hello((-2.6, 0.0), 3.0, [(6, 6), (6, -6), (-11, -6)], emu))
    assert h['station'] == [-2.6, 0.0] and h['danger_radius'] == 3.0 and len(h['waypoints']) == 3, h
    assert abs(h['emu']['loss'] - 0.1) < 1e-9 and h['emu']['delay_ms'] == 80, h
    met = [{'id': 7, 'x': 12.345, 'y': -3.21, 'alt': 15.5, 'vx': -4.1, 'vy': 1.2, 'vz': -2.5,
            'impact_x': 3.3, 'impact_y': 1.1, 'tti': 6.42, 'sigma': 0.93, 'threat': True, 'age': 0.08, 'n_obs': 31},
           {'id': 8, 'x': 1, 'y': 2, 'alt': 3, 'vx': 0, 'vy': 0, 'vz': 0, 'impact_x': None, 'impact_y': None,
            'tti': None, 'sigma': None, 'threat': False, 'age': 1.5, 'n_obs': 4}]
    payload = P.encode_telemetry({'x': 8.62, 'y': 5.93, 'yaw': 1.0, 'speed': 0.38},
                                 {'retx': 12, 'abandoned': 0, 'srtt_ms': 161.0}, met)
    r, l, m = P.decode_telemetry(payload)
    assert abs(r['x'] - 8.62) < 0.006 and abs(r['yaw'] - 57.3) < 0.01 and l['retx'] == 12, (r, l)
    assert m[0]['threat'] and abs(m[0]['impact_x'] - 3.3) < 0.006 and abs(m[0]['tti'] - 6.42) < 0.006, m[0]
    assert 'impact_x' not in m[1] and not m[1]['threat'], m[1]
    th = {'id': 3, 'impact': [4.2, 3.3], 'time_to_impact': 6.6, 'sigma': 0.82, 'escape': [6.8, 7.0]}
    ms = P.decode_mission(P.encode_mission('EVADE', 2, {'threats': 3, 'evasions_completed': 2}, th))
    assert ms['state'] == 'EVADE' and ms['threat']['id'] == 3 and ms['counters']['threats'] == 3, ms
    ev = P.decode_event(P.encode_event('warn', '!!! INCOMING meteor #1 → EVADING'))
    assert ev == {'level': 'warn', 'text': '!!! INCOMING meteor #1 → EVADING'}, ev
    assert P.decode_crater(P.encode_crater(5, -7.5, 9.25, 1.2)) == {'id': 5, 'x': -7.5, 'y': 9.25, 'r': 1.2}
    ptype, sess, seq, t, pl = P.unpack(P.pack(P.TELEMETRY, 77, 1234, 99000, payload))
    assert (ptype, sess, seq, t, pl) == (P.TELEMETRY, 77, 1234, 99000, payload)
    print(f'[ok] round trip: telemetry frame with 2 meteors = {len(payload) + P.OVERHEAD} B on the wire')


def test_crc_rejects_every_single_bit_flip():
    dgram = P.pack(P.TELEMETRY, 1, 42, 1000, P.encode_telemetry(
        {'x': 1, 'y': 2, 'yaw': 0, 'speed': 0}, {'retx': 0, 'abandoned': 0, 'srtt_ms': None}, []))
    n = 0
    for byte in range(len(dgram)):
        for bit in range(8):
            b = bytearray(dgram)
            b[byte] ^= 1 << bit
            try:
                P.unpack(bytes(b))
            except P.ProtocolError:
                n += 1
                continue
            raise AssertionError(f'flip at byte {byte} bit {bit} not detected')
    print(f'[ok] all {n} single-bit corruptions rejected')


def test_lossy_loopback(n=600, rate=50.0):
    port = free_port()
    got, order = {}, []

    def on_packet(ptype, seq, t_ms, payload):
        (i,) = struct.unpack('!I', payload[:4])
        got[i] = got.get(i, 0) + 1
        order.append(i)

    bad = {'loss': 0.30, 'delay_ms': 40, 'jitter_ms': 30, 'corrupt': 0.10, 'seed': 1}
    rx = ReliableReceiver(port, on_packet, emulator_cfg=dict(bad, seed=2), host='127.0.0.1')
    tx = ReliableSender(('127.0.0.1', port), emulator_cfg=bad, max_retries=8)
    for i in range(n):
        tx.send(P.EVENT, i, struct.pack('!I', i) + os.urandom(random.randrange(10, 60)))
        time.sleep(1.0 / rate)
    t_end = time.time() + 40
    while tx.unacked and time.time() < t_end:
        time.sleep(0.1)
    missing = [i for i in range(n) if i not in got]
    dup = [i for i, c in got.items() if c > 1]
    reordered = sum(1 for a, b in zip(order, order[1:]) if b < a)
    print(f'[..] lossy loopback, {n} datagrams, each direction: 30% loss, 10% corruption, 40±30 ms delay')
    print(f'     sender:   sent {tx.stats["sent"]}, retransmissions {tx.stats["retransmissions"]}, '
          f'abandoned {tx.stats["abandoned"]}, corrupted ACKs ignored {tx.stats["bad_acks"]}, '
          f'SRTT {tx.srtt * 1000:.0f} ms, RTO {tx.rto * 1000:.0f} ms')
    print(f'     emulator: data dropped {tx.link.stats["dropped"]}, corrupted {tx.link.stats["corrupted"]}; '
          f'ACKs dropped {rx.link.stats["dropped"]}, corrupted {rx.link.stats["corrupted"]}')
    print(f'     receiver: CRC-rejected {rx.stats["corrupt"]}, duplicates suppressed {rx.stats["duplicates"]}, '
          f'delivered {rx.stats["delivered"]}, out-of-order deliveries {reordered}')
    assert not dup, f'delivered twice: {dup[:10]}'
    assert not missing, f'never delivered: {missing}'
    # 'abandoned' = the sender gave up waiting for an ACK. At this loss level a few packets
    # can be delivered while every ACK for them is lost; the data still arrived.
    assert tx.stats['abandoned'] <= n // 100 and not tx.unacked, (tx.stats, len(tx.unacked))
    print(f'[ok] all {n} delivered exactly once; sender gave up waiting for an ACK on '
          f'{tx.stats["abandoned"]} (ACKs lost, data was delivered)')
    tx.close()
    rx.close()


def test_sack_repairs_lost_ack():
    port = free_port()
    rx = ReliableReceiver(port, lambda *a: None, host='127.0.0.1')
    orig, state = rx.link.sendto, {'n': 0}

    def drop_first_ack(data, addr):
        state['n'] += 1
        if state['n'] > 1:
            orig(data, addr)
    rx.link.sendto = drop_first_ack
    tx = ReliableSender(('127.0.0.1', port))           # RTO starts at 0.5 s
    tx.send(P.EVENT, 0, b'\x00first')
    time.sleep(0.05)
    tx.send(P.EVENT, 0, b'\x00second')                 # its ACK carries SACK bit for seq 1
    time.sleep(0.2)
    assert not tx.unacked and tx.stats['retransmissions'] == 0, (tx.unacked, tx.stats)
    print('[ok] lost ACK repaired by the next ACK\'s SACK bitmap, no retransmission needed')
    tx.close()
    rx.close()


if __name__ == '__main__':
    test_roundtrip()
    test_crc_rejects_every_single_bit_flip()
    test_sack_repairs_lost_ack()
    test_lossy_loopback()
    print('ALL TESTS PASSED')
