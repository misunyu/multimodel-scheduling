"""Input queue shared between one feeder process and one worker thread: capacity 2, drop-newest, lock-free.

Single producer / single consumer over shared memory. The feeder is the only writer of TAIL (requests enqueued so far),
the worker is the only writer of HEAD (requests dequeued so far) and FIN (requests completed or failed so far).
Each field is one aligned 64-bit word, so reads and writes are atomic on x86-64 and nobody ever holds a lock: a
descheduled worker thread cannot block its feeder or the boundary read (the locked version did; see
pilot/arrival/attempt3_nice_locked_queue_dev).
  queue length = TAIL - HEAD,  in-flight = HEAD - FIN (0 or 1),  pending = TAIL - FIN.
The feeder may see a slightly old HEAD; then a request arriving within microseconds of a dequeue is counted as dropped.
"""
import sys

sys.dont_write_bytecode = True

import time

_TAIL, _HEAD, _FIN, _HDR = 0, 1, 2, 3


class ShmQueue:
    def __init__(self, ctx, capacity=2):
        self.cap = capacity
        self.items = ctx.Semaphore(0)
        self.s = ctx.RawArray("q", _HDR + 2 * capacity)

    # feeder side
    def put_nowait(self, req_id):
        """Returns (ok, ts). ok = 0: the queue was full and the request is dropped (drop-newest)."""
        s = self.s
        ts = time.monotonic_ns()
        tail = s[_TAIL]
        if tail - s[_HEAD] >= self.cap:
            return 0, ts
        slot = _HDR + 2 * (tail % self.cap)
        s[slot] = req_id
        s[slot + 1] = ts
        s[_TAIL] = tail + 1
        self.items.release()
        return 1, ts

    # worker side
    def get(self, timeout):
        """Returns (request id, t_enqueue, t_dequeue) or None on timeout. The request is in-flight until finish()."""
        if not self.items.acquire(timeout=timeout):
            return None
        s = self.s
        head = s[_HEAD]
        slot = _HDR + 2 * (head % self.cap)
        idx, t_enq = s[slot], s[slot + 1]
        s[_HEAD] = head + 1
        return idx, t_enq, time.monotonic_ns()

    def finish(self):
        """Called once per dequeued request right after it completed or failed. Returns the event timestamp."""
        ts = time.monotonic_ns()
        self.s[_FIN] += 1
        return ts

    # main side
    def leftover_ids(self):
        """Request ids still queued (call only after the feeder and the worker have stopped)."""
        s = self.s
        return [int(s[_HDR + 2 * (k % self.cap)]) for k in range(s[_HEAD], s[_TAIL])]

    def snapshot(self):
        s = self.s
        fin, head, tail = s[_FIN], s[_HEAD], s[_TAIL]
        return {"ts": time.monotonic_ns(), "qsize": int(tail - head), "inflight": int(head - fin)}
