"""Read micro-batches ahead of the trainer.

Random document reads from a network file system are slow enough to dominate a training step when they run inside
it. Prefetcher wraps a stream that splits next_microbatch into plan(B) (which segments; cheap, advances the stream)
and load(segs, pool) (the reads; pure):
  * plan() always runs in the caller's thread, in order, so the batches and their order are exactly those of the
    unwrapped stream (tests/test_prefetch.py);
  * load() of the next `depth` micro-batches runs on background threads, each read (os.pread, GIL released) on a
    shared pool of `threads` workers, while the GPU computes;
  * state_dict() is the stream state right after the last batch HANDED OUT; batches planned but not yet consumed are
    planned again after a resume, so checkpoint/resume stays exact (a selecting stream also hands over the scorer
    weight versions those batches were planned with, which may have arrived after the last consumed plan).
The depth is part of a selecting run's definition (a batch is planned `depth` micro-batches before it is trained, with
the scorer weights of that time), so it stays fixed for a run. take_times(): the time next_microbatch spent planning
and waiting for reads since the last call.
"""

import collections
import time
from concurrent.futures import ThreadPoolExecutor


class Prefetcher:
    def __init__(self, stream, B, depth=4, threads=16):
        self.stream, self.B, self.depth = stream, B, depth
        self.io = ThreadPoolExecutor(threads, thread_name_prefix="daisy-io")
        self.asm = ThreadPoolExecutor(depth, thread_name_prefix="daisy-asm")
        self.q = collections.deque()          # (future of the loaded micro-batch, stream state after its plan)
        self.state = stream.state_dict()
        self.plan_s = self.wait_s = 0.0       # since the last take_times()

    def _top_up(self):
        t = time.perf_counter()
        while len(self.q) < self.depth:
            segs = self.stream.plan(self.B)
            self.q.append((self.asm.submit(self.stream.load, segs, self.io), self.stream.state_dict()))
        self.plan_s += time.perf_counter() - t

    def next_microbatch(self, B):
        if B != self.B:
            raise ValueError(f"prefetcher built for B={self.B}, asked for {B}")
        self._top_up()
        fut, st = self.q.popleft()
        t = time.perf_counter()
        out = fut.result()
        self.wait_s += time.perf_counter() - t
        self.state = st
        self._top_up()                        # the next reads start now, during this step's compute
        return out

    def take_times(self):
        """(seconds planning, seconds waiting for reads) in next_microbatch since the last call."""
        out = (self.plan_s, self.wait_s)
        self.plan_s = self.wait_s = 0.0
        return out

    def state_dict(self):
        if not hasattr(self.stream, "hist_state"):
            return self.state
        return {**self.state, "w_hist": self.stream.hist_state(self.state["n_planned"])}

    def load_state_dict(self, st):
        for fut, _ in self.q:
            fut.cancel()
        for fut, _ in self.q:
            if not fut.cancelled():
                fut.result()
        self.q.clear()
        self.stream.load_state_dict(st)
        self.state = self.stream.state_dict()

    def close(self):
        self.asm.shutdown(wait=True, cancel_futures=True)
        self.io.shutdown(wait=True, cancel_futures=True)

    def __getattr__(self, name):              # pos, epoch, skipped, ... of the wrapped stream (for logs)
        if name == "stream":
            raise AttributeError(name)
        return getattr(self.stream, name)
