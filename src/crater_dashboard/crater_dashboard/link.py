"""Reliable datagram link over UDP (stdlib `socket`), with a degraded-radio emulator.

ReliableSender   (rover)   every datagram gets a sequence number and waits in an `unacked`
                           table; anything not ACKed within the retransmission timeout (RTO)
                           is sent again with exponential backoff. RTO is adaptive
                           (Jacobson/Karels SRTT/RTTVAR, Karn's rule: only never-retransmitted
                           packets give RTT samples). After `max_retries` it is abandoned.
ReliableReceiver (ground)  datagrams with bad magic/length or a CRC mismatch are dropped
                           *without ACK*, so the sender retransmits them. Every valid datagram,
                           duplicates included, is ACKed with a SACK bitmap of the 32 previous
                           sequence numbers, so one lost ACK is repaired by the next. Duplicates
                           are never delivered twice.
LinkEmulator               wraps sendto(): random loss, delay + jitter (packets get reordered
                           naturally) and byte corruption, independently on each direction.
"""
import heapq
import itertools
import random
import socket
import threading
import time
from collections import OrderedDict, deque

from . import protocol as P


class LinkEmulator:
    def __init__(self, sock, loss=0.0, delay_ms=0.0, jitter_ms=0.0, corrupt=0.0, seed=None):
        self.sock = sock
        self.loss, self.corrupt = float(loss), float(corrupt)
        self.delay, self.jitter = float(delay_ms) / 1000.0, float(jitter_ms) / 1000.0
        self.rng = random.Random(seed)
        self.stats = {'offered': 0, 'dropped': 0, 'corrupted': 0, 'sent': 0, 'bytes': 0}
        self._heap, self._cv, self._order = [], threading.Condition(), itertools.count()
        self._closed = False
        if self.delay > 0 or self.jitter > 0:
            threading.Thread(target=self._pump, daemon=True).start()

    def settings(self):
        return {'loss': self.loss, 'delay_ms': self.delay * 1000, 'jitter_ms': self.jitter * 1000,
                'corrupt': self.corrupt}

    def sendto(self, data, addr):
        self.stats['offered'] += 1
        if self.rng.random() < self.loss:
            self.stats['dropped'] += 1
            return
        if self.rng.random() < self.corrupt:
            b = bytearray(data)
            b[self.rng.randrange(len(b))] ^= 1 << self.rng.randrange(8)
            data = bytes(b)
            self.stats['corrupted'] += 1
        if self.delay <= 0 and self.jitter <= 0:
            self._send(data, addr)
            return
        due = time.monotonic() + max(0.0, self.delay + self.rng.uniform(-self.jitter, self.jitter))
        with self._cv:
            heapq.heappush(self._heap, (due, next(self._order), data, addr))
            self._cv.notify()

    def _send(self, data, addr):
        try:
            self.sock.sendto(data, addr)
            self.stats['sent'] += 1
            self.stats['bytes'] += len(data)
        except OSError:
            pass

    def _pump(self):
        while not self._closed:
            with self._cv:
                while not self._heap:
                    self._cv.wait()
                due, _, data, addr = self._heap[0]
                wait = due - time.monotonic()
                if wait > 0:
                    self._cv.wait(wait)
                    continue
                heapq.heappop(self._heap)
            self._send(data, addr)


class ReliableSender:
    def __init__(self, peer, emulator_cfg=None, max_retries=6, window=64,
                 rto_min=0.15, rto_max=2.0, logger=None):
        self.peer = peer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', 0))
        self.sock.settimeout(0.02)
        self.link = LinkEmulator(self.sock, **(emulator_cfg or {}))
        self.session = random.randrange(1, 0xFFFF)
        self.seq = 0
        self.max_retries, self.window = max_retries, window
        self.rto_min, self.rto_max = rto_min, rto_max
        self.srtt, self.rttvar, self.rto = None, None, 0.5
        self.unacked = OrderedDict()        # seq -> [datagram, t_first, t_last, retries]
        self.lock = threading.Lock()
        self.log = logger
        self.stats = {'sent': 0, 'retransmissions': 0, 'acked': 0, 'abandoned': 0,
                      'acks_rx': 0, 'bad_acks': 0, 'bytes': 0}
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ API
    def send(self, ptype, t_ms, payload):
        with self.lock:
            self.seq = (self.seq + 1) & 0xFFFFFFFF
            dgram = P.pack(ptype, self.session, self.seq, t_ms, payload)
            now = time.monotonic()
            self.unacked[self.seq] = [dgram, now, now, 0]
            while len(self.unacked) > self.window:          # never block the rover: drop oldest
                self.unacked.popitem(last=False)
                self.stats['abandoned'] += 1
            self.stats['sent'] += 1
            self.stats['bytes'] += len(dgram)
        self.link.sendto(dgram, self.peer)
        return self.seq

    def link_stats(self):
        return {'retx': self.stats['retransmissions'], 'abandoned': self.stats['abandoned'],
                'srtt_ms': None if self.srtt is None else self.srtt * 1000}

    def close(self):
        self._run = False

    # ------------------------------------------------------------------ internals
    def _rtt_sample(self, r):
        if self.srtt is None:
            self.srtt, self.rttvar = r, r / 2
        else:
            self.rttvar = 0.75 * self.rttvar + 0.25 * abs(self.srtt - r)
            self.srtt = 0.875 * self.srtt + 0.125 * r
        self.rto = min(self.rto_max, max(self.rto_min, self.srtt + 4 * self.rttvar))

    def _on_ack(self, data):
        try:
            ptype, session, _seq, _t, payload = P.unpack(data)
        except P.ProtocolError:
            self.stats['bad_acks'] += 1                # corrupted ACK: ignore, retransmit covers it
            return
        if ptype != P.ACK or session != self.session:
            return
        ack_seq, sack = P.decode_ack(payload)
        self.stats['acks_rx'] += 1
        now = time.monotonic()
        with self.lock:
            for s in [ack_seq] + [ack_seq - 1 - i for i in range(32) if sack >> i & 1]:
                e = self.unacked.pop(s & 0xFFFFFFFF, None)
                if e is not None:
                    self.stats['acked'] += 1
                    if e[3] == 0:                      # Karn: no samples from retransmitted packets
                        self._rtt_sample(now - e[1])

    def _service(self):
        now = time.monotonic()
        resend = []
        with self.lock:
            for s, e in list(self.unacked.items()):
                if now - e[2] >= min(self.rto_max, self.rto * (2 ** e[3])):   # backoff, capped at rto_max
                    if e[3] >= self.max_retries:
                        del self.unacked[s]
                        self.stats['abandoned'] += 1
                        continue
                    e[2], e[3] = now, e[3] + 1
                    self.stats['retransmissions'] += 1
                    self.stats['bytes'] += len(e[0])
                    resend.append(e[0])
        for d in resend:
            self.link.sendto(d, self.peer)

    def _loop(self):
        while self._run:
            try:
                data, _ = self.sock.recvfrom(2048)
                self._on_ack(data)
            except socket.timeout:
                pass
            except OSError:
                time.sleep(0.02)
            self._service()


class ReliableReceiver:
    """Binds `port`; calls on_packet(ptype, seq, t_ms, payload, fresh) for each new datagram."""

    def __init__(self, port, on_packet, emulator_cfg=None, host='0.0.0.0'):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        self.link = LinkEmulator(self.sock, **(emulator_cfg or {}))
        self.on_packet = on_packet
        self.session = None
        self.seen = set()
        self.seen_order = deque()
        self.top = 0
        self.stats = {'valid': 0, 'corrupt': 0, 'malformed': 0, 'duplicates': 0, 'delivered': 0,
                      'sessions': 0, 'bytes': 0, 'acks_tx': 0}
        self.rx_log = deque(maxlen=2000)    # (wall time, bytes) of valid datagrams
        self.last_rx = None
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def close(self):
        self._run = False

    def _remember(self, seq):
        self.seen.add(seq)
        self.seen_order.append(seq)
        while len(self.seen_order) > 8192:
            self.seen.discard(self.seen_order.popleft())
        self.top = max(self.top, seq)

    def _ack(self, seq, addr):
        sack = 0
        for i in range(32):
            if (seq - 1 - i) in self.seen:
                sack |= 1 << i
        self.link.sendto(P.pack(P.ACK, self.session, 0, 0, P.encode_ack(seq, sack)), addr)
        self.stats['acks_tx'] += 1

    def _loop(self):
        while self._run:
            try:
                data, addr = self.sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                time.sleep(0.05)
                continue
            try:
                ptype, session, seq, t_ms, payload = P.unpack(data)
            except P.ProtocolError as e:
                self.stats['corrupt' if 'CRC' in str(e) else 'malformed'] += 1
                continue                                  # no ACK -> sender retransmits
            if ptype == P.ACK:
                continue
            if session != self.session:                   # rover (re)started: new sequence space
                self.session = session
                self.seen.clear()
                self.seen_order.clear()
                self.top = 0
                self.stats['sessions'] += 1
            self.stats['valid'] += 1
            self.stats['bytes'] += len(data)
            self.last_rx = time.time()
            self.rx_log.append((self.last_rx, len(data)))
            dup = seq in self.seen
            if not dup:
                self._remember(seq)
            self._ack(seq, addr)                          # ACK duplicates too (our ACK may have been lost)
            if dup:
                self.stats['duplicates'] += 1
                continue
            self.stats['delivered'] += 1
            try:
                self.on_packet(ptype, seq, t_ms, payload)
            except Exception as e:                        # a decoder bug must not kill the link
                self.stats.setdefault('handler_errors', 0)
                self.stats['handler_errors'] += 1
                self.stats['last_handler_error'] = repr(e)

    def rates(self, window=5.0):
        now = time.time()
        recent = [(t, n) for t, n in self.rx_log if now - t <= window]
        if not recent:
            return 0.0, 0.0, 0.0
        n_bytes = sum(n for _, n in recent)
        return len(recent) / window, n_bytes / window, n_bytes / len(recent)
