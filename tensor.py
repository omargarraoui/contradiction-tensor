from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from itertools import combinations

from nli import NLIScorer


@dataclass
class Belief:
    id: str
    text: str
    domain: str = "general"


class ContradictionTensor:
    """
    N × N × K tensor where:
      N = number of beliefs
      K = number of contexts (e.g. "default", "crisis", "bull_market")

    Cell [i, j, k] = contradiction score between belief i and j under context k.
    The matrix is symmetric by construction.
    """

    def __init__(self, contexts: list[str] | None = None):
        self.beliefs: list[Belief] = []
        self.contexts: list[str] = contexts or ["default"]
        self._scorer = NLIScorer()
        self._tensor: np.ndarray = np.zeros((0, 0, len(self.contexts)))

    @property
    def matrix(self) -> np.ndarray:
        return self._tensor[:, :, 0]

    def _ctx_idx(self, context: str | None) -> int:
        if context and context in self.contexts:
            return self.contexts.index(context)
        return 0

    def add(self, belief: Belief, context_weights: list[float] | None = None) -> None:
        n = len(self.beliefs)
        k = len(self.contexts)
        weights = context_weights or [1.0] * k

        if len(weights) != k:
            raise ValueError(f"expected {k} context weights, got {len(weights)}")

        self.beliefs.append(belief)
        expanded = np.zeros((n + 1, n + 1, k))

        if n > 0:
            expanded[:n, :n, :] = self._tensor
            for i, existing in enumerate(self.beliefs[:-1]):
                base = self._scorer.score(existing.text, belief.text)
                for ctx, w in enumerate(weights):
                    expanded[i, n, ctx] = base * w
                    expanded[n, i, ctx] = base * w

        self._tensor = expanded

    def max_contradiction(
        self, context: str | None = None
    ) -> tuple[Belief, Belief, float] | None:
        if len(self.beliefs) < 2:
            return None

        m = self._tensor[:, :, self._ctx_idx(context)].copy()
        np.fill_diagonal(m, -1)
        i, j = np.unravel_index(m.argmax(), m.shape)
        return self.beliefs[i], self.beliefs[j], self._tensor[i, j, self._ctx_idx(context)]

    def min_removal_set(self, threshold: float = 0.7) -> list[Belief]:
        """
        Greedy 2-approximation of Minimum Vertex Cover.
        At each step, remove the belief involved in the most conflicts above threshold.
        """
        remaining = set(range(len(self.beliefs)))
        removed: list[int] = []
        m = self.matrix.copy()

        while any(m[i, j] >= threshold for i, j in combinations(remaining, 2)):
            scores = {
                i: sum(
                    m[i, j]
                    for j in remaining
                    if j != i and m[i, j] >= threshold
                )
                for i in remaining
            }
            worst = max(scores, key=scores.get)
            removed.append(worst)
            remaining.discard(worst)

        return [self.beliefs[i] for i in removed]

    def tension_clusters(self, threshold: float = 0.6) -> list[set[str]]:
        """
        Union-Find over beliefs connected by contradiction above threshold.
        Returns clusters of mutually contradicting belief IDs.
        """
        parent = {b.id: b.id for b in self.beliefs}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (i, a), (j, b) in combinations(enumerate(self.beliefs), 2):
            if self.matrix[i, j] >= threshold:
                parent[find(a.id)] = find(b.id)

        groups: dict[str, set[str]] = {}
        for b in self.beliefs:
            root = find(b.id)
            groups.setdefault(root, set()).add(b.id)

        return [g for g in groups.values() if len(g) > 1]

    def summary(self) -> dict:
        if len(self.beliefs) < 2:
            return {"beliefs": len(self.beliefs), "mean_tension": 0.0, "max_tension": 0.0}

        upper = self.matrix[np.triu_indices(len(self.beliefs), k=1)]
        return {
            "beliefs": len(self.beliefs),
            "mean_tension": float(upper.mean()),
            "max_tension": float(upper.max()),
            "p90_tension": float(np.percentile(upper, 90)),
        }
