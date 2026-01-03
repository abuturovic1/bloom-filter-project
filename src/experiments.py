from src.bloom_filter import BloomFilter
from src.datasets import make_strings


def false_positive_rate(bf: BloomFilter, test_items: list[str]) -> float:
    fp = sum(1 for x in test_items if x in bf)
    return fp / len(test_items)


def main():
    n = 50_000
    p_target = 0.01

    bf = BloomFilter(expected_items=n, false_positive_rate=p_target)

    inserted = make_strings("in", n)
    not_inserted = make_strings("out", n)

    for x in inserted:
        bf.add(x)

    fpr = false_positive_rate(bf, not_inserted)

    print("Bloom filter params:")
    print(f"  m (bits)  = {bf.size}")
    print(f"  k (hashes)= {bf.hash_count}")
    print(f"Target p    = {p_target}")
    print(f"Measured FPR= {fpr:.6f}")


if __name__ == "__main__":
    main()
