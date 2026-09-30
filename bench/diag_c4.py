#!/usr/bin/env python3
"""Diagnostic: does the NInfer HTTP layer actually deliver all 4 concurrent
128k responses, and where does a stall (if any) happen?

Each lane uses its OWN http.client connection with `Connection: close` (no
keep-alive reuse) and a per-socket timeout. Per lane we record:
  - t_hdr_s: when response status+headers arrived
  - t_body_s: when the full body was read (byte progression logged)
  - response headers (Content-Length / Transfer-Encoding / Connection)
  - completion tokens / finish_reason / wall time

A background sampler cross-checks when the engine itself goes idle
(running=0). This tells us whether a client stall is server-side
(partial send / non-flush) or was just the original harness's keep-alive.

Same deterministic 4x128k prompts as c4_128k.py (same seeds).
"""

import http.client
import json
import random
import threading
import time

HOST = "127.0.0.1"
PORT = 8080
OUT = 512
RATIO = 1.945
SIZE = "128k"
C = 4
TARGET = {"64k": 64500, "128k": 130000, "256k": 261500}
LANE_TIMEOUT = 300

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
    parts, used, i = [], 0, 0
    while used < n_chars:
        p = para(seed, i)
        parts.append(p)
        used += len(p)
        i += 1
    return " ".join(parts)[:n_chars]


def build_payloads():
    out = []
    for lane in range(C):
        seed = int(SIZE[1]) * 7919 + 0 * 104729 + lane * 15485863
        out.append({
            "model": "bonsai2-27b",
            "reasoning_effort": "none",
            "max_completion_tokens": OUT,
            "messages": [{"role": "user", "content": make_prompt(TARGET[SIZE], seed)}],
        })
    return out


def main():
    payloads = build_payloads()
    t0 = time.monotonic()
    log(f"firing {C} concurrent 128k lanes (per-socket timeout {LANE_TIMEOUT}s)")

    def run_lane(lane, payload):
        body = json.dumps(payload).encode()
        rec = {"lane": lane}
        try:
            conn = http.client.HTTPConnection(HOST, PORT)
            conn.connect()
            conn.sock.settimeout(LANE_TIMEOUT)
            conn.request("POST", "/v1/chat/completions", body=body,
                         headers={"content-type": "application/json",
                                  "connection": "close"})
            resp = conn.getresponse()
            rec["status"] = resp.status
            rec["t_hdr_s"] = round(time.monotonic() - t0, 2)
            hdr = {k.lower(): v for k, v in resp.getheaders()}
            for k in ("content-length", "transfer-encoding", "connection"):
                if k in hdr:
                    rec["hdr_" + k] = hdr[k]
            raw = b""
            last_log = 0.0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                raw += chunk
                now = time.monotonic() - t0
                if now - last_log >= 5:
                    log(f"  lane{rec['lane']}: {len(raw)} body bytes by t+{now:.0f}s")
                    last_log = now
            rec["t_body_s"] = round(time.monotonic() - t0, 2)
            rec["body_bytes"] = len(raw)
            conn.close()
            d = json.loads(raw)
            ch = d["choices"][0]
            rec["finish"] = ch.get("finish_reason")
            rec["completion_tokens"] = d["usage"]["completion_tokens"]
            rec["prompt_tokens"] = d["usage"]["prompt_tokens"]
            log(f"  lane{rec['lane']}: OK completion={rec['completion_tokens']} "
                f"finish={rec['finish']} t_hdr={rec['t_hdr_s']}s t_body={rec['t_body_s']}s")
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["t_end_s"] = round(time.monotonic() - t0, 2)
            log(f"  lane{rec['lane']}: FAIL {rec['error']} (t+{rec.get('t_end_s')}s)")
        RESULTS[lane] = rec

    RESULTS = {}
    ths = [threading.Thread(target=run_lane, args=(i, p)) for i, p in enumerate(payloads)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.monotonic() - t0
    log(f"--- all lanes settled, wall={wall:.1f}s ---")
    for i in range(C):
        r = RESULTS.get(i, {})
        log(f"  lane{i}: {json.dumps(r)}")
    with open("/root/ninfer-bonsai/bench/c4_128k_diag.jsonl", "w") as f:
        f.write(json.dumps({"wall_s": round(wall, 1), "lanes": [RESULTS.get(i) for i in range(C)]}) + "\n")


if __name__ == "__main__":
    main()
