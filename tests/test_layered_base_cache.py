from __future__ import annotations

import numpy as np

from hac.layered_base_cache import LayeredNestedBaseCache


class _Cache:
    def __init__(self, root, existing):
        self.root = root
        self.existing = existing
        self.identity = {"same": True}
        self.labels = np.asarray([0, 1, 2, 0, 1, 2])
        self.scenarios = np.asarray(["a", "a", "a", "b", "b", "b"])
        self.sample_ids = np.asarray([f"id-{i}" for i in range(6)])

    def request(self, train):
        return {"rows": np.asarray(train).tolist()}

    def directory(self, train):
        key = "-".join(map(str, np.asarray(train).tolist()))
        path = self.root / key
        if key in self.existing:
            path.mkdir(parents=True, exist_ok=True)
            (path / "receipt.json").touch(exist_ok=True)
        return path

    def load(self, train):
        value = np.full((6, 3), 1 / 3)
        return value, {"source": self.root.name}

    def prepare(self, train):
        return {"source": self.root.name}


def test_layered_cache_prefers_historical_and_writes_current(tmp_path):
    historical = _Cache(tmp_path / "old", {"0-1-2"})
    current = _Cache(tmp_path / "new", set())
    cache = LayeredNestedBaseCache(historical, current)
    old_rows = np.asarray([0, 1, 2])
    new_rows = np.asarray([3, 4, 5])
    assert cache.prepare(old_rows)["source"] == "old"
    assert cache.prepare(new_rows)["source"] == "new"
    assert cache.directory(old_rows).parent.name == "old"
    assert cache.directory(new_rows).parent.name == "new"
