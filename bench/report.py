#!/usr/bin/env python3
"""Aggregate bench/results.jsonl into a per-context-size table:
TTFT, prefill speed, decode speed, aggregate throughput, and the
concurrency (C) that maximizes stable aggregate throughput per size."""

import json
import statistics
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/root/ninfer-bonsai/bench/results.jsonl"
ORDER = {"64k": 0, "128k": 1, "256k": 2}


def median(vals):
    v = [x for x in vals if x is not None]
    return statistics.median(v) if v else None


def load():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load()
    if not rows:
        print("no results yet")
        return

    # group by (size, C) over sweep (non-soak) rounds only for the headline table
    from collections import defaultdict
    grp = defaultdict(list)
    for r in rows:
        tag = r["tag"]
        if tag.startswith("soak"):
            continue
        grp[(r["size"], r["C"])].append(r)

    sizes = sorted({s for (s, _) in grp}, key=lambda s: ORDER.get(s, 99))
    print("=" * 100)
    print("HEADLINE (sweep rounds, median across C lanes x rounds)")
    print("=" * 100)
    hdr = (f"{'size':>5} {'C':>3} | {'TTFT med(ms)':>12} {'TTFT max(ms)':>12} {'prefill(tok/s)':>14} "
           f"{'decode(tok/s)':>14} {'AGG OUT(tok/s)':>15} {'agg IN+OUT(tok/s)':>18} ok")
    print(hdr)
    print("(AGG OUT = output/completion tokens / wall-clock across the C concurrent requests — the decision metric)")
    print("-" * len(hdr))
    best = {}  # size -> (agg_out, C)
    for size in sizes:
        cs = sorted(c for (s, c) in grp if s == size)
        for c in cs:
            rs = grp[(size, c)]
            ttft_med = median([r["ttft_ms_med"] for r in rs])
            ttft_max = median([r["ttft_ms_max"] for r in rs])
            pref = median([r["prefill_tps_med"] for r in rs])
            dec = median([r["decode_tps_med"] for r in rs])
            agg_tot = median([r["agg_total_tps"] for r in rs])
            agg_dec = median([r["agg_decode_tps"] for r in rs])
            ok = median([r["ok"] / r["n"] for r in rs])
            print(f"{size:>5} {c:>3} | {ttft_med:>12.0f} {ttft_max:>12.0f} {pref:>14.0f} {dec:>14.0f} {agg_dec:>15.1f} {agg_tot:>18.0f} {ok:.0%}")
            if size not in best or (agg_dec or 0) > (best[size][0] or 0):
                best[size] = (agg_dec, c)
        print()

    print("=" * 100)
    print("OPTIMAL CONCURRENCY (max stable AGGREGATE OUTPUT throughput per size)")
    print("=" * 100)
    for size in sizes:
        agg_out, c = best[size]
        print(f"  {size:>5}: C={c}  (aggregate output ~{agg_out:.1f} tok/s)")

    # stability soak summary
    print()
    print("=" * 100)
    print("STABILITY SOAK (3 rounds each)")
    print("=" * 100)
    soaks = defaultdict(list)
    for r in rows:
        if r["tag"].startswith("soak"):
            soaks[(r["size"], r["C"])].append(r)
    for (size, c) in sorted(soaks, key=lambda k: (ORDER.get(k[0], 99), k[1])):
        rs = soaks[(size, c)]
        allerr = [e for r in rs for e in r["errors"]]
        agg_tot = median([r["agg_total_tps"] for r in rs])
        stable = (len(allerr) == 0) and all(r["ok"] == r["n"] for r in rs)
        print(f"  {size:>5} C={c}: {'STABLE' if stable else 'UNSTABLE'} "
              f"(errors={len(allerr)}, agg_total~{agg_tot:.0f} tok/s, rounds={len(rs)})")
    if allerr := [e for r in rows for e in r["errors"]]:
        print("\n  errors seen:", allerr[:10])


if __name__ == "__main__":
    main()
