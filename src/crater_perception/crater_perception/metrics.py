"""Measured runtime metrics shared by the Module 2 nodes.

Every node publishes one JSON line per second on /perception/metrics:
  {"node", "fps", "proc_ms", "proc_p95_ms", "latency_ms", "cpu_pct", "rss_mb", "gpu", ...extra}
  fps          messages produced per second (wall clock)
  proc_ms      wall time spent in this node's callback (mean / p95 over the last 60)
  latency_ms   sim-time age of the source RGB frame when the output was published
               (end-to-end since the camera exposure, in simulated time)
  cpu_pct      this process, % of one core (psutil), rss_mb resident memory
Nothing is estimated: every value is measured in the running node.
"""
import json
import time
from collections import deque

import numpy as np
import psutil
from std_msgs.msg import String


class Metrics:
    def __init__(self, node, name):
        self.node, self.name = node, name
        self.pub = node.create_publisher(String, '/perception/metrics', 10)
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.proc_ms, self.lat_ms = deque(maxlen=60), deque(maxlen=60)
        self.count, self.t0 = 0, time.monotonic()
        self.extra = {}
        node.create_timer(1.0, self._publish)

    def tick(self, proc_ms=None, stamp=None):
        """Call once per produced output. stamp: the source frame's builtin_interfaces/Time."""
        self.count += 1
        if proc_ms is not None:
            self.proc_ms.append(proc_ms)
        if stamp is not None:
            now = self.node.get_clock().now().nanoseconds
            self.lat_ms.append((now - (stamp.sec * 1_000_000_000 + stamp.nanosec)) / 1e6)

    def _publish(self):
        now = time.monotonic()
        m = {'node': self.name, 'fps': round(self.count / (now - self.t0), 2)}
        self.count, self.t0 = 0, now
        if self.proc_ms:
            a = np.array(self.proc_ms)
            m['proc_ms'], m['proc_p95_ms'] = round(float(a.mean()), 2), round(float(np.percentile(a, 95)), 2)
        if self.lat_ms:
            m['latency_ms'] = round(float(np.mean(self.lat_ms)), 1)
        m['cpu_pct'] = round(self.proc.cpu_percent(None), 1)
        m['rss_mb'] = round(self.proc.memory_info().rss / 2**20, 1)
        m['gpu'] = 'n/a (CPU only)'
        m.update(self.extra)
        self.pub.publish(String(data=json.dumps(m)))
        self.last = m
