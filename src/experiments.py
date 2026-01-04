import argparse
import os
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.bloom_filter import (
    BloomFilterClassic,
    BloomFilterDoubleHash,
    theoretical_fpr,
)
from src.datasets import deterministic_strings


def measure_fpr(bf, inserted: List[str], queries_not_inserted: List[str]) -> float:
    for x in inserted:
        bf.add(x)
    fp = sum(1 for q in queries_not_inserted if q in bf)
    return fp / len(queries_not_inserted)


def measure_speed(bf_factory: Callable[[], object], inserted: List[str], queries: List[str]) -> Tuple[float, float]:
    bf = bf_factory()

    t0 = time.perf_counter()
    for x in inserted:
        bf.add(x)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    _ = sum(1 for q in queries if q in bf)
    t3 = time.perf_counter()

    return (t1 - t0), (t3 - t2)


def ensure_results_dir(path: str = "results") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def run_suite(n_values: List[int], p_target: float, query_multiplier: int = 1) -> pd.DataFrame:
    rows: List[Dict] = []

    for n in n_values:
        inserted = deterministic_strings(n, prefix="IN")
        queries = deterministic_strings(n * query_multiplier, prefix="OUT")

        # Classic
        bf_c = BloomFilterClassic(n_expected=n, p_target=p_target)
        fpr_c = measure_fpr(bf_c, inserted, queries)
        t_ins_c, t_q_c = measure_speed(lambda: BloomFilterClassic(n_expected=n, p_target=p_target), inserted, queries)

        # Double hashing
        bf_d = BloomFilterDoubleHash(n_expected=n, p_target=p_target)
        fpr_d = measure_fpr(bf_d, inserted, queries)
        t_ins_d, t_q_d = measure_speed(lambda: BloomFilterDoubleHash(n_expected=n, p_target=p_target), inserted, queries)

        # Theory (both use same m,k computed from target p)
        m = bf_c.params.m
        k = bf_c.params.k
        p_theory = theoretical_fpr(n=n, m=m, k=k)

        rows.append({
            "n": n,
            "p_target": p_target,
            "m_bits": m,
            "k": k,
            "p_theory": p_theory,
            "p_emp_classic": fpr_c,
            "p_emp_doublehash": fpr_d,
            "insert_time_s_classic": t_ins_c,
            "query_time_s_classic": t_q_c,
            "insert_time_s_doublehash": t_ins_d,
            "query_time_s_doublehash": t_q_d,
        })

    return pd.DataFrame(rows)


def plot_fpr(df: pd.DataFrame, outdir: str) -> None:
    plt.figure()
    plt.plot(df["n"], df["p_theory"], marker="o")
    plt.plot(df["n"], df["p_emp_classic"], marker="o")
    plt.plot(df["n"], df["p_emp_doublehash"], marker="o")
    plt.xscale("log")
    plt.xlabel("n (inserted elements, log scale)")
    plt.ylabel("False positive rate")
    plt.title("False positive rate: theory vs empirical")
    plt.legend(["Theory", "Classic", "Double hashing"])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fpr_theory_vs_empirical.png"), dpi=200)
    plt.close()


def plot_speed(df: pd.DataFrame, outdir: str) -> None:
    plt.figure()
    plt.plot(df["n"], df["insert_time_s_classic"], marker="o")
    plt.plot(df["n"], df["insert_time_s_doublehash"], marker="o")
    plt.xscale("log")
    plt.xlabel("n (inserted elements, log scale)")
    plt.ylabel("Insert time (s)")
    plt.title("Insert time comparison")
    plt.legend(["Classic", "Double hashing"])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "insert_time.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["n"], df["query_time_s_classic"], marker="o")
    plt.plot(df["n"], df["query_time_s_doublehash"], marker="o")
    plt.xscale("log")
    plt.xlabel("n (queries, log scale)")
    plt.ylabel("Query time (s)")
    plt.title("Query time comparison")
    plt.legend(["Classic", "Double hashing"])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "query_time.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Bloom filter experiments (classic vs double hashing).")
    parser.add_argument("--p", type=float, default=0.01, help="target false positive rate (default: 0.01)")
    parser.add_argument("--n", type=str, default="1000,5000,10000,50000,100000",
                        help="comma-separated n values (default: 1000,5000,10000,50000,100000)")
    parser.add_argument("--qmult", type=int, default=1, help="queries_not_inserted = n * qmult (default: 1)")
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n.split(",") if x.strip()]
    df = run_suite(n_values=n_values, p_target=args.p, query_multiplier=args.qmult)

    outdir = ensure_results_dir("results")
    csv_path = os.path.join(outdir, "results.csv")
    df.to_csv(csv_path, index=False)

    plot_fpr(df, outdir)
    plot_speed(df, outdir)

    # Pretty console summary
    cols = ["n", "m_bits", "k", "p_theory", "p_emp_classic", "p_emp_doublehash",
            "insert_time_s_classic", "insert_time_s_doublehash",
            "query_time_s_classic", "query_time_s_doublehash"]
    print(df[cols].to_string(index=False))
    print(f"Saved: {csv_path} and plots in ./{outdir}/")


if __name__ == "__main__":
    main()
