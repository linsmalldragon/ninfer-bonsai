#!/usr/bin/env python3
"""C=4 @ 128k NInfer metric capture (FIXED harness).

KV-oversubscribed config: 4 x 130k-token prompts = ~512k KV > 262,144 pool,
so at most ~2 requests fit in device KV at once. We sample /metrics + /v1/load
every 5s while the wave runs, record per-lane API timings, and dump the engine
log, to observe admission / spill / draft-acceptance behaviour.

Harness fix (2026-09-28): the original urllib/keep-alive client hung on all 4
long responses even though the engine delivered them on time (proven by
diag_c4.py: each lane's client response-arrival == the engine per-request
`total`, all 4 delivered, ~120s wave). This version uses a per-lane
http.client connection with `Connection: close` and a bounded per-socket
timeout, writes each lane as it completes, and keeps the /metrics sampler on
its own bounded connections so a stuck lane can never withhold the rest.

Outputs (bench/): c4_128k_samples.jsonl, c4_128k_results.jsonl
"""

import http.client
import json
import random
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
HOST = "127.0.0.1"
PORT = 8080
SAMPLES = "/root/ninfer-bonsai/bench/c4_128k_samples.jsonl"
RESULTS = "/root/ninfer-bonsai/bench/c4_128k_results.jsonl"

SIZE = "128k"
C = 4
OUT = 512
RATIO = 1.945
TARGET = {"64k": 64500, "128k": 130000, "256k": 261500}
LANE_TIMEOUT = 300      # per-socket, seconds; well above the ~120s wave
SAMPLE_TIMEOUT = 15

_SENT = ("the kernel schedule moves tokens across lanes while the host pinned cache "
         "holds the frontier for the next decode round and the drafter proposes "
         "a short window that the verifier accepts or rejects before the graph "
         "replays the block on the device with the quantized state restored")


def log(msg):
    print(msg, flush=True)


def para(seed, i):
    r = random.Random((seed ^ (i * 2654435761)) & 0xFFFFFFFF)
    words = [f"{w} {r.randint(0, 999)}" for w, _ in zip(_SENT.split(), range(40))]
    return f"[doc {seed % 97} para {i:05d}] " + " ".join(words)


def make_prompt(target_tokens, seed):
    n_chars = int(target_tokens * RATIO * 0.995)
    parts, used, i = [], 0, 0
    while used < n_chars:
        p = para(seed, i)
        parts.append(p)
        used += len(p)
        i += 1
    return " ".join(parts)[:n_chars]


def build_payloads(rnd):
    out = []
    for lane in range(C):
        seed = (int(SIZE[1]) * 7919 + rnd * 104729 + lane * 15485863)
        out.append({
            "model": "bonsai2-27b",
            "reasoning_effort": "none",
            "max_completion_tokens": OUT,
            "messages": [{"role": "user", "content": make_prompt(TARGET[SIZE], seed)}],
        })
    return out


def post_close(payload, t0, timeout=LANE_TIMEOUT):
    """One request on its OWN connection, Connection: close, bounded timeout.

    Returns (parsed_json, t_resp_s). Raises on timeout / connection error.
    """
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection(HOST, PORT)
    conn.connect()
    conn.sock.settimeout(timeout)
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"content-type": "application/json", "connection": "close"})
    resp = conn.getresponse()
    raw = b""
    while True:
        chunk = resp.read(1 << 16)
        if not chunk:
            break
        raw += chunk
    t_resp = time.monotonic() - t0
    if resp.status != 200:
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {raw[:300]!r}")
    conn.close()
    return json.loads(raw), t_resp


def fetch(url):
    with urllib.request.urlopen(url, timeout=SAMPLE_TIMEOUT) as r:
        return r.read()


def parse_metrics(raw):
    d = {}
    for line in raw.decode().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) >= 2:
            try:
                d[p[0]] = float(p[1])
            except ValueError:
                pass
    return d


STOP = False


def sampler(start_ts):
    while not STOP:
        try:
            m = parse_metrics(fetch(BASE + "/metrics"))
            load = json.loads(fetch(BASE + "/v1/load"))
            occ = load.get("occupancy", {})
            rq = load.get("requests", {})
            rec = {
                "t": round(time.time() - start_ts, 1),
                "kv_usage": m.get("llamacpp:kv_cache_usage_ratio"),
                "kv_tokens": m.get("llamacpp:kv_cache_tokens"),
                "req_processing": m.get("llamacpp:requests_processing"),
                "req_deferred": m.get("llamacpp:requests_deferred"),
                "busy_slots": m.get("llamacpp:n_busy_slots_per_decode"),
                "gen_tps": m.get("llamacpp:predicted_tokens_seconds"),
                "prefill_tps": m.get("llamacpp:prompt_tokens_seconds"),
                "draft_total": m.get("ninfer:draft_tokens_total"),
                "draft_acc": m.get("ninfer:draft_accepted_tokens_total"),
                "ctx_exhausted": m.get("ninfer:context_cache_exhausted_requests_total"),
                "dev_kv_tokens": occ.get("device_main_kv_tokens"),
                "host_kv_bytes": occ.get("host_kv_bytes"),
                "running": rq.get("running"),
                "prefilling": rq.get("prefilling"),
                "waiting": rq.get("waiting"),
                "decode_ready": rq.get("decode_ready"),
            }
        except Exception as e:  # noqa: BLE001
            rec = {"t": round(time.time() - start_ts, 1), "err": str(e)[:80]}
        with open(SAMPLES, "a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(5)


def main():
    open(SAMPLES, "w").close()
    open(RESULTS, "w").close()
    start_ts = time.time()
    sth = threading.Thread(target=sampler, args=(start_ts,))
    sth.start()

    payloads = build_payloads(0)
    results = [None] * C
    errors = []

    def worker(lane):
        t0 = time.monotonic()
        try:
            d, t_resp = post_close(payloads[lane], t0)
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
                "wall_s": round(t_resp, 2),
            }
            r = results[lane]
            log(f"  lane{lane} OK p={r['prompt_n']} c={r['pred_n']} fin={r['finish']} "
                f"ttft={round((r['ttft_ms'] or 0)/1000,1)}s wall={r['wall_s']}s")
        except Exception as e:  # noqa: BLE001
            errors.append(f"lane{lane}: {type(e).__name__}: {e}")
            results[lane] = {"lane": lane, "error": str(e)[:200]}
            log(f"  lane{lane} ERROR {type(e).__name__}: {e}")
        # flush this lane immediately so a stuck sibling can't hide it
        with open(RESULTS, "a") as f:
            f.write(json.dumps(results[lane]) + "\n")

    t0 = time.monotonic()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.monotonic() - t0

    global STOP
    STOP = True
    sth.join()

    ok = [r for r in results if r and "error" not in r]
    agg_decode = (sum(r["pred_n"] for r in ok) / wall) if ok and not errors else 0.0
    agg_total = (sum(r["prompt_n"] + r["pred_n"] for r in ok) / wall) if ok and not errors else 0.0
    for r in ok:
        span = r["wall_s"] - (r["ttft_ms"] or 0) / 1000
        r["phys_decode"] = round(r["pred_n"] / span, 1) if span > 0 else 0.0

    summary = {
        "tag": "c4_128k", "size": SIZE, "C": C,
        "wall_s": round(wall, 1),
        "agg_decode_tps": round(agg_decode, 1),
        "agg_total_tps": round(agg_total, 1),
        "ok": len(ok), "n": C, "errors": errors,
        "requests": results,
    }
    with open(RESULTS, "a") as f:
        f.write(json.dumps(summary) + "\n")

    log(f"wall={wall:.0f}s agg_decode={agg_decode:.1f}tok/s total={agg_total:.1f}tok/s "
        f"ok={len(ok)}/{C}")
    if errors:
        log(f"ERRORS={errors}")


if __name__ == "__main__":
    main()
