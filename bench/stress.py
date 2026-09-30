#!/usr/bin/env python3
"""NInfer Ternary-Bonsai2-27B (5090, rk4v4, DFlash2-5) concurrency stress sweep.

Per context size (64k/128k/256k input tokens) sweep server max-concurrency C,
measure TTFT, prefill speed, decode speed, and aggregate throughput; run a
stability soak at the C that maximizes stable aggregate throughput.

Every lane/round uses a unique prompt window (deterministic generator), so no
cross-request or cross-round prefix reuse: every prefill is cold, matching the
model-card fresh-document methodology.

Progress lines (machine-greppable):
  RESTART C=.. / READY
  POINT size C r wall=.. agg_decode=..ttft=.. prefill=.. decode=.. ok=..
  SOAK  size C r wall=.. agg_decode=.. ok=..
  ERROR ...
  DONE
"""

import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8080/v1"
RUN_SH = "/root/ninfer-bonsai/run.sh"
RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/root/ninfer-bonsai/bench/results.jsonl"

OUT = 512          # decode budget per request
POOL = 262144      # KV pool tokens
RATIO = 1.945      # chars per token (measured 1.9434/1.9447/1.9493 @64/256/128k)
TARGET = {"64k": 64500, "128k": 130000, "256k": 261500}
# per-size candidate concurrencies, bounded by the 262,144-token KV pool:
#   256k: 261000+512 fits once only; 128k: 2x(130000+512) fits; 64k: 4x(64500+512) fits
CAND = {"256k": [1], "128k": [1, 2], "64k": [1, 2, 4]}
SWEEP_ROUNDS = 2
SOAK_ROUNDS = 3

_SENT = ("the kernel schedule moves tokens across lanes while the host pinned cache "
         "holds the frontier for the next decode round and the drafter proposes "
         "a short window that the verifier accepts or rejects before the graph "
         "replays the block on the device with the quantized state restored")


def log(msg):
    print(msg, flush=True)


def para(seed, i):
    r = random.Random((seed ^ (i * 2654435761)) & 0xFFFFFFFF)
    return (f"[doc {seed % 97} para {i:05d}] "
            + " ".join(f"{w} {r.randint(0, 999)}" for w, _ in zip(
                _SENT.split(), range(40))))


def make_prompt(target_tokens, seed):
    n_chars = int(target_tokens * RATIO * 0.995)
    parts = []
    used = 0
    i = 0
    while used < n_chars:
        p = para(seed, i)
        parts.append(p)
        used += len(p)
        i += 1
    return " ".join(parts)[:n_chars]


def post(payload, timeout=7200):
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    t1 = time.monotonic()
    return json.loads(body), t1 - t0


def build_payloads(size, C, rnd):
    target = TARGET[size]
    out = []
    for lane in range(C):
        seed = (int(size[1]) * 7919 + rnd * 104729 + lane * 15485863)
        prompt = make_prompt(target, seed)
        out.append({
            "model": "bonsai2-27b",
            "reasoning_effort": "none",
            "max_completion_tokens": OUT,
            "messages": [{"role": "user", "content": prompt}],
        })
    return out


def run_batch(size, C, rnd, tag):
    payloads = build_payloads(size, C, rnd)
    results = [None] * C
    errors = []

    def worker(lane):
        try:
            d, wall = post(payloads[lane])
            ch = d["choices"][0]
            tm = d.get("timings", {})
            results[lane] = {
                "lane": lane,
                "prompt_n": d["usage"]["prompt_tokens"],
                "pred_n": d["usage"]["completion_tokens"],
                "finish": ch.get("finish_reason"),
                "ttft_ms": tm.get("prompt_ms"),
                "prefill_tps": tm.get("prompt_per_second"),
                "decode_tps": tm.get("predicted_per_second"),
                "draft_n": tm.get("draft_n"),
                "draft_acc": tm.get("draft_n_accepted"),
                "wall_s": round(wall, 2),
            }
        except Exception as e:  # noqa: BLE001
            errors.append(f"lane{lane}: {type(e).__name__}: {e}")
            results[lane] = {"lane": lane, "error": str(e)[:200]}

    t0 = time.monotonic()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.monotonic() - t0

    ok = [r for r in results if r and "error" not in r]
    agg_decode = (sum(r["pred_n"] for r in ok) / wall) if ok and not errors else 0.0
    agg_total = (sum(r["prompt_n"] + r["pred_n"] for r in ok) / wall) if ok and not errors else 0.0
    def med(vals):
        v = sorted(x for x in vals if x is not None)
        return v[len(v) // 2] if v else None

    record = {
        "tag": tag, "size": size, "C": C, "round": rnd,
        "wall_s": round(wall, 1),
        "agg_decode_tps": round(agg_decode, 1),
        "agg_total_tps": round(agg_total, 1),
        "ttft_ms_med": med([r["ttft_ms"] for r in ok]),
        "ttft_ms_max": max((r["ttft_ms"] for r in ok if r["ttft_ms"] is not None), default=None),
        "prefill_tps_med": med([r["prefill_tps"] for r in ok]),
        "decode_tps_med": med([r["decode_tps"] for r in ok]),
        "ok": len(ok), "n": C,
        "errors": errors,
        "requests": results,
    }
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    line = (f"{'SOAK ' if tag.startswith('soak') else 'POINT'} {size} C={C} r{rnd} "
            f"wall={wall:.0f}s agg_decode={agg_decode:.1f}tok/s total={agg_total:.1f}tok/s "
            f"ttft_med={record['ttft_ms_med']}ms prefill={record['prefill_tps_med']} "
            f"decode={record['decode_tps_med']}tok/s ok={len(ok)}/{C}")
    if errors:
        line += f" ERRORS={errors}"
    log(line)
    return record


def server_restart(C):
    log(f"RESTART C={C}")
    env = dict(os.environ, EXTRA_ARGS=f"--max-concurrency {C}")
    subprocess.run(["bash", RUN_SH], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(90):
        try:
            with urllib.request.urlopen(f"{BASE}/models", timeout=3) as r:
                if r.status == 200:
                    log("READY")
                    return
        except Exception:
            time.sleep(2)
    raise SystemExit("ERROR server did not become ready")


def health():
    try:
        with urllib.request.urlopen(f"{BASE}/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def warmup():
    # Two short throwaway requests trigger one-time allocations after a
    # restart; they never touch the long unique prompts, so measured
    # prefill stays cold.
    for i in range(2):
        post({"model": "bonsai2-27b", "reasoning_effort": "none",
              "max_completion_tokens": 1,
              "messages": [{"role": "user", "content": f"warmup {i} hello"}]}, timeout=120)


def main():
    # ---- Phase A: server at C=1 (64k, 128k, 256k) ----
    server_restart(1)
    warmup()
    for size in ("64k", "128k", "256k"):
        for rnd in range(SWEEP_ROUNDS):
            run_batch(size, 1, rnd, f"sweep{size}c1")
    for rnd in range(SOAK_ROUNDS):
        run_batch("256k", 1, 100 + rnd, "soak256kc1")
    if not health():
        log("ERROR server unhealthy after 256k soak")

    # ---- Phase B: server at C=2 (64k, 128k) ----
    server_restart(2)
    warmup()
    for size in ("64k", "128k"):
        for rnd in range(SWEEP_ROUNDS):
            run_batch(size, 2, rnd, f"sweep{size}c2")
    for rnd in range(SOAK_ROUNDS):
        run_batch("128k", 2, 100 + rnd, "soak128kc2")
    if not health():
        log("ERROR server unhealthy after 128k C2 soak")

    # ---- Phase C: server at C=4 (64k) ----
    server_restart(4)
    for rnd in range(SWEEP_ROUNDS):
        run_batch("64k", 4, rnd, "sweep64kc4")
    for rnd in range(SOAK_ROUNDS):
        run_batch("64k", 4, 100 + rnd, "soak64kc4")
    if not health():
        log("ERROR server unhealthy after 64k C4 soak")

    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        log(f"ERROR {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        log(f"ERROR {type(e).__name__}: {e}")
        sys.exit(1)
