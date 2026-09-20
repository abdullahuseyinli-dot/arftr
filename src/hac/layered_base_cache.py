"""Read-through reuse of an immutable historical base cache.

Existing population caches remain in their historical directory. New
population-keyed caches are written only under the new run. Requests and
receipts must have the same feature/code/protocol identity in both layers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hac.actor_memory_base import NestedBaseCache, group_splits


class LayeredNestedBaseCache:
    def __init__(self, historical: NestedBaseCache, current: NestedBaseCache) -> None:
        if historical.identity != current.identity:
            raise ValueError("Historical and current base-cache identities differ")
        self.historical = historical
        self.current = current
        self.identity = historical.identity
        self.labels = historical.labels
        self.scenarios = historical.scenarios
        self.sample_ids = historical.sample_ids

    def request(self, train: np.ndarray) -> dict:
        historical = self.historical.request(train)
        if historical != self.current.request(train):
            raise RuntimeError("Layered base-cache requests differ")
        return historical

    def _historical_exists(self, train: np.ndarray) -> bool:
        return (self.historical.directory(train) / "receipt.json").is_file()

    def directory(self, train: np.ndarray) -> Path:
        return (
            self.historical.directory(train)
            if self._historical_exists(train)
            else self.current.directory(train)
        )

    def prepare(self, train: np.ndarray) -> dict:
        if self._historical_exists(train):
            return self.historical.load(train)[1]
        return self.current.prepare(train)

    def load(self, train: np.ndarray) -> tuple[np.ndarray, dict]:
        if self._historical_exists(train):
            return self.historical.load(train)
        return self.current.load(train)

    def held_predictions(self, train: np.ndarray, held: np.ndarray) -> np.ndarray:
        if set(self.scenarios[train]) & set(self.scenarios[held]):
            raise RuntimeError("Refusing in-scenario layered base predictions")
        return self.load(train)[0][held]

    def meta_probabilities(self, population: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        result = np.full((len(self.labels), 3), np.nan)
        ancestry = []
        for train, held in group_splits(self.labels, self.scenarios, population):
            result[held] = self.held_predictions(train, held)
            ancestry.append(
                {
                    "predicted_rows": held.tolist(),
                    "base_cache": self.directory(train).name,
                    "cache_layer": "historical"
                    if self._historical_exists(train)
                    else "current",
                }
            )
        outside = np.setdiff1d(np.arange(len(self.labels)), population)
        result[outside] = self.held_predictions(population, outside)
        ancestry.append(
            {
                "predicted_rows": outside.tolist(),
                "base_cache": self.directory(population).name,
                "cache_layer": "historical"
                if self._historical_exists(population)
                else "current",
            }
        )
        if not np.isfinite(result).all():
            raise RuntimeError("Layered meta-prediction coverage failed")
        return result, ancestry
