"""
tests/tune_accumulator.py

Tunes the mid-call alarm rule (src/streaming/accumulator.py) and measures
TIME-TO-DETECTION -- "how many seconds into a call before we can flag it,
at an acceptable false-alarm rate." This is the metric the whole streaming
reformulation exists to produce.

Pure replay over the per-chunk caches written by tests/eval_streaming.py --
no model inference, so it runs in seconds and is cheap to iterate on.

DISCIPLINE: the accumulator config is swept and frozen on the DEV cache,
then validated exactly once on the EVAL cache -- the same rule applied to
detector thresholds everywhere else in this project.

Usage:
    .venv/bin/python tests/tune_accumulator.py
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detector.base import DetectionResult
from streaming.accumulator import CallAccumulator, AccumulatorConfig

WINDOW_SIZES = [5, 8, 10, 15, 20]
VOTE_FRACTIONS = [0.4, 0.5, 0.6, 0.7, 0.8]
MIN_CONFIDENCES = [0.0, 0.3, 0.5]


def load_cache(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def group_by_utterance(cache: dict, detector: str, window_ms: int):
    """-> [(utt_id, label, [DetectionResult in chunk order]), ...]"""
    m = (cache["detector"] == detector) & (cache["window_ms"] == window_ms)
    utt_ids, labels = cache["utt_id"][m], cache["label"][m]
    scores, is_spoof = cache["score"][m], cache["is_spoof"][m]
    confidence, latency = cache["confidence"][m], cache["latency_ms"][m]

    calls, current_id, current = [], None, None
    for i in range(len(utt_ids)):
        if utt_ids[i] != current_id:
            if current is not None:
                calls.append(current)
            current_id = utt_ids[i]
            current = (str(utt_ids[i]), str(labels[i]), [])
        current[2].append(DetectionResult(score=float(scores[i]),
                                           is_spoof=bool(is_spoof[i]),
                                           confidence=float(confidence[i]),
                                           latency_ms=float(latency[i])))
    if current is not None:
        calls.append(current)
    return calls


def evaluate_config(calls, config: AccumulatorConfig, window_ms: int) -> dict:
    """Replay every call through a fresh accumulator; measure alarm behaviour."""
    bona_alarms = spoof_alarms = n_bona = n_spoof = 0
    ttds, structurally_impossible = [], 0

    for _utt_id, label, results in calls:
        acc = CallAccumulator(config, chunk_duration_ms=window_ms)
        alarmed = False
        for r in results:
            if acc.update(r):
                alarmed = True
                break
        if label == "bonafide":
            n_bona += 1
            bona_alarms += alarmed
        else:
            n_spoof += 1
            spoof_alarms += alarmed
            # A call with fewer chunks than window_size can never reach quorum,
            # regardless of how spoof-like it sounds -- a real limitation of
            # N-of-M voting on short calls, worth reporting separately.
            if len(results) < config.window_size:
                structurally_impossible += 1
            if alarmed:
                ttds.append(acc.time_to_detection_ms)

    return {
        "false_alarm_rate": bona_alarms / n_bona if n_bona else float("nan"),
        "detection_rate": spoof_alarms / n_spoof if n_spoof else float("nan"),
        "median_ttd_ms": float(np.median(ttds)) if ttds else float("nan"),
        "mean_ttd_ms": float(np.mean(ttds)) if ttds else float("nan"),
        "n_bonafide": n_bona,
        "n_spoof": n_spoof,
        "structurally_undetectable_spoof_rate":
            structurally_impossible / n_spoof if n_spoof else float("nan"),
    }


def sweep(calls, window_ms: int) -> list[dict]:
    rows = []
    for ws in WINDOW_SIZES:
        for frac in VOTE_FRACTIONS:
            votes = max(1, int(round(ws * frac)))
            for min_conf in MIN_CONFIDENCES:
                cfg = AccumulatorConfig(window_size=ws, required_votes=votes,
                                         min_confidence=min_conf)
                r = evaluate_config(calls, cfg, window_ms)
                r.update({"window_size": ws, "required_votes": votes,
                           "vote_fraction": frac, "min_confidence": min_conf,
                           "window_ms": window_ms})
                rows.append(r)
    return rows


def pick_best(rows: list[dict], max_far: float) -> dict | None:
    """Lowest median time-to-detection subject to a false-alarm-rate ceiling.
    Ties/no-detections are excluded -- a config that never alarms trivially
    has a perfect false-alarm rate and is useless."""
    viable = [r for r in rows
              if r["false_alarm_rate"] <= max_far and not np.isnan(r["median_ttd_ms"])
              and r["detection_rate"] > 0]
    if not viable:
        return None
    return min(viable, key=lambda r: (r["median_ttd_ms"], -r["detection_rate"]))


def plot_tradeoff(rows: list[dict], detector: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for window_ms in sorted({r["window_ms"] for r in rows}):
        pts = [(r["false_alarm_rate"] * 100, r["median_ttd_ms"] / 1000)
               for r in rows if r["window_ms"] == window_ms
               and not np.isnan(r["median_ttd_ms"]) and r["detection_rate"] > 0]
        if pts:
            pts.sort()
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                        label=f"{window_ms}ms chunks", alpha=0.7)
    ax.set_xlabel("False-alarm rate on bonafide calls (%)")
    ax.set_ylabel("Median time-to-detection (s)")
    ax.set_title(f"Alarm tradeoff — {detector}")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-cache", type=Path,
                         default=Path("results/cache/streaming_chunks_dev_clean.npz"))
    parser.add_argument("--eval-cache", type=Path,
                         default=Path("results/cache/streaming_chunks_eval_clean.npz"))
    parser.add_argument("--degraded-cache", type=Path, default=None,
                         help="Optional third cache to also validate the frozen config on.")
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.05)
    parser.add_argument("--detector", default="aasist_l")
    parser.add_argument("--sweep-out", type=Path, default=Path("results/accumulator_sweep.csv"))
    parser.add_argument("--final-out", type=Path, default=Path("results/accumulator_final.json"))
    args = parser.parse_args()

    dev = load_cache(args.dev_cache)
    windows = sorted({int(w) for w in dev["window_ms"]})
    print(f"DEV sweep on {args.dev_cache.name}, windows={windows}, "
          f"detector={args.detector}, FAR ceiling={args.max_false_alarm_rate:.0%}")

    all_rows = []
    for window_ms in windows:
        calls = group_by_utterance(dev, args.detector, window_ms)
        rows = sweep(calls, window_ms)
        all_rows.extend(rows)
        print(f"  {window_ms}ms: swept {len(rows)} configs over {len(calls)} calls")

    args.sweep_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sweep_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"Full sweep -> {args.sweep_out}")
    plot_tradeoff(all_rows, args.detector, Path("results/accumulator_tradeoff.png"))
    print("Tradeoff plot -> results/accumulator_tradeoff.png")

    best = pick_best(all_rows, args.max_false_alarm_rate)
    if best is None:
        raise SystemExit(f"No config met FAR <= {args.max_false_alarm_rate:.0%} while "
                          f"still detecting anything. Loosen the ceiling or improve "
                          f"per-chunk accuracy first.")

    print(f"\n=== FROZEN CONFIG (chosen on DEV only) ===")
    print(f"  chunk window: {best['window_ms']}ms")
    print(f"  window_size (N): {best['window_size']}, required_votes (M): {best['required_votes']}"
          f", min_confidence: {best['min_confidence']}")
    print(f"  dev false-alarm rate: {best['false_alarm_rate'] * 100:.2f}%")
    print(f"  dev detection rate:   {best['detection_rate'] * 100:.2f}%")
    print(f"  dev median time-to-detection: {best['median_ttd_ms'] / 1000:.2f}s")

    frozen = AccumulatorConfig(window_size=best["window_size"],
                                required_votes=best["required_votes"],
                                min_confidence=best["min_confidence"])
    result = {"frozen_config": {"window_size": frozen.window_size,
                                 "required_votes": frozen.required_votes,
                                 "min_confidence": frozen.min_confidence,
                                 "chunk_window_ms": best["window_ms"]},
              "detector": args.detector,
              "max_false_alarm_rate": args.max_false_alarm_rate,
              "dev": {k: best[k] for k in ("false_alarm_rate", "detection_rate",
                                             "median_ttd_ms", "n_bonafide", "n_spoof",
                                             "structurally_undetectable_spoof_rate")}}

    print("\nConfig FROZEN. Validating once on held-out caches.")
    for name, cache_path in [("eval_clean", args.eval_cache),
                              ("eval_degraded", args.degraded_cache)]:
        if cache_path is None or not Path(cache_path).exists():
            continue
        cache = load_cache(Path(cache_path))
        calls = group_by_utterance(cache, args.detector, best["window_ms"])
        r = evaluate_config(calls, frozen, best["window_ms"])
        result[name] = r
        print(f"  [{name}] FAR={r['false_alarm_rate'] * 100:.2f}%  "
              f"detection={r['detection_rate'] * 100:.2f}%  "
              f"median TTD={r['median_ttd_ms'] / 1000:.2f}s  "
              f"(structurally undetectable: "
              f"{r['structurally_undetectable_spoof_rate'] * 100:.1f}% of spoof calls)")

    with open(args.final_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFinal -> {args.final_out}")


if __name__ == "__main__":
    main()
