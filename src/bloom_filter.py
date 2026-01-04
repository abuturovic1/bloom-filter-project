import math
from dataclasses import dataclass
from typing import Iterable, Iterator

import mmh3 
from bitarray import bitarray


@dataclass(frozen=True)
class BloomParams:
    m: int  # number of bits
    k: int  # number of hash functions


def optimal_params(n: int, p: float) -> BloomParams:
    """Return (m,k) that minimize false positive rate for expected n and target p."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")

    m = int(-(n * math.log(p)) / (math.log(2) ** 2))
    k = max(1, int((m / n) * math.log(2)))
    return BloomParams(m=m, k=k)


def theoretical_fpr(n: int, m: int, k: int) -> float:
    """Classic Bloom filter false positive probability approximation."""
    # p ≈ (1 - e^{-kn/m})^k
    return (1.0 - math.exp(-(k * n) / m)) ** k


class BloomFilterClassic:
    """Classic Bloom filter using k mmh3 hashes via different seeds."""

    def __init__(self, n_expected: int, p_target: float = 0.01):
        self.params = optimal_params(n_expected, p_target)
        self.bits = bitarray(self.params.m)
        self.bits.setall(0)

    def _indexes(self, item: str) -> Iterator[int]:
        m = self.params.m
        for seed in range(self.params.k):
            yield mmh3.hash(item, seed, signed=False) % m

    def add(self, item: str) -> None:
        for idx in self._indexes(item):
            self.bits[idx] = 1

    def __contains__(self, item: str) -> bool:
        return all(self.bits[idx] for idx in self._indexes(item))


class BloomFilterDoubleHash:
    """Bloom filter variant using double hashing to generate k indices.

    Based on the idea: g_i(x) = h1(x) + i*h2(x) (mod m).
    This matches the technique described by Kirsch & Mitzenmacher.
    """

    def __init__(self, n_expected: int, p_target: float = 0.01):
        self.params = optimal_params(n_expected, p_target)
        self.bits = bitarray(self.params.m)
        self.bits.setall(0)

    def _h1_h2(self, item: str) -> tuple[int, int]:
        # Two base hashes (different seeds)
        h1 = mmh3.hash(item, 0, signed=False)
        h2 = mmh3.hash(item, 1, signed=False)
        # Ensure h2 is non-zero to avoid repeated indices (rare, but safe)
        if h2 == 0:
            h2 = 0x9E3779B1  # an arbitrary odd constant
        return h1, h2

    def _indexes(self, item: str) -> Iterator[int]:
        m = self.params.m
        k = self.params.k
        h1, h2 = self._h1_h2(item)
        for i in range(k):
            yield (h1 + i * h2) % m

    def add(self, item: str) -> None:
        for idx in self._indexes(item):
            self.bits[idx] = 1

    def __contains__(self, item: str) -> bool:
        return all(self.bits[idx] for idx in self._indexes(item))
