"""In-memory store of regularized samples."""
from datetime import datetime, timedelta
from typing import List, Optional, Sequence

from ..sample import Sample


class RegularizedHistoryStore:
    """
    In-memory store of regularized samples, trimmed to a configurable time window.
    """

    def __init__(self, window: timedelta):
        self._history_window = window
        self._samples: List[Sample] = []

    def extend(self, samples: Sequence[Sample]) -> None:
        """Appends samples, keeping one entry per grid timestamp (latest wins)."""
        by_ts = {s.timestamp: s for s in self._samples}
        for sample in samples:
            by_ts[sample.timestamp] = sample
        self._samples = sorted(by_ts.values(), key=lambda s: s.timestamp)

    def trim(self, now: datetime) -> None:
        """Drops samples older than now - history_window."""
        cutoff = now - self._history_window
        self._samples = [s for s in self._samples if s.timestamp >= cutoff]

    def get(self, ts: datetime) -> Optional[Sample]:
        """Returns the sample at exactly ``ts``, or None if absent."""
        for sample in self._samples:
            if sample.timestamp == ts:
                return sample
        return None

    def window(self, start: datetime, end: datetime) -> List[Sample]:
        """Returns the samples in the window."""
        return [s for s in self._samples if start <= s.timestamp < end]

    @property
    def samples(self) -> List[Sample]:
        """Returns the samples in the store."""
        return self._samples
