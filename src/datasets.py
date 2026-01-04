import secrets
import string
from typing import List


_ALPHABET = string.ascii_letters + string.digits


def random_strings(n: int, length: int = 16, prefix: str = "") -> List[str]:
    """Generate n unique-ish random strings (very low collision risk)."""
    out = []
    for _ in range(n):
        token = "".join(secrets.choice(_ALPHABET) for _ in range(length))
        out.append(f"{prefix}{token}")
    return out


def deterministic_strings(n: int, prefix: str) -> List[str]:
    """Deterministic dataset useful for debugging/reproducibility."""
    return [f"{prefix}_{i}" for i in range(n)]
