"""Overlap original CPU KD-SBIR replay with GPU ranking under a bounded host-memory queue."""
from collections import deque

import torch


class PendingReplay:
    def __init__(self,replay,pool,workers):
        self.replay=replay
        self.pool=pool
        self.limit=2*workers
        self.pending=deque()
        self.maximum_error=0.

    def submit(self,rank,labels,expected,destination):
        # Different scorers can have only 32 queries in flight individually. Share the queue across scorers.
        order=rank.to(device="cpu",dtype=torch.int32)
        batches=zip(order.split(8),labels.split(8),strict=True)
        tasks=[self.pool.submit(self.replay.chunk,pair) for pair in batches]
        self.pending.append((tasks,expected.cpu(),destination))
        if len(self.pending)>=self.limit:self.pop()

    def pop(self):
        tasks,expected,destination=self.pending.popleft()
        exact=torch.cat([task.result() for task in tasks])
        error=float((exact-expected).abs().amax())
        if error>2e-6:raise ValueError("Original KD-SBIR routine differs on at least one query")
        self.maximum_error=max(self.maximum_error,error)
        destination.append(exact)

    def finish(self):
        while self.pending:self.pop()
        return self.maximum_error
