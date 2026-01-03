import math
import mmh3
from bitarray import bitarray


class BloomFilter:
    """
    Classic Bloom filter using bit array + k hash functions (mmh3 with different seeds).
    """

    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        if expected_items <= 0:
            raise ValueError("expected_items must be > 0")
        if not (0 < false_positive_rate < 1):
            raise ValueError("false_positive_rate must be in (0, 1)")

        # m = -(n ln p) / (ln 2)^2
        self.size = int(-(expected_items * math.log(false_positive_rate)) / (math.log(2) ** 2))

        # k = (m/n) ln 2
        self.hash_count = max(1, int((self.size / expected_items) * math.log(2)))

        self.bits = bitarray(self.size)
        self.bits.setall(0)

    def _indexes(self, item: str):
        for seed in range(self.hash_count):
            yield mmh3.hash(item, seed, signed=False) % self.size

    def add(self, item: str) -> None:
        for idx in self._indexes(item):
            self.bits[idx] = 1

    def __contains__(self, item: str) -> bool:
        return all(self.bits[idx] for idx in self._indexes(item))
