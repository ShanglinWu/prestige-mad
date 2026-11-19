"""Recover round-1 reasoning for a prestige_mad_v2 store written before it was saved, so the
run can be resumed instead of restarted. Reads the per-call log (calls.jsonl), where every model
reply is recorded in full. No API calls.

    python3 scripts/backfill_rationales.py runs/prestige_gemini_gpqa_extended_n546.jsonl \
        --problems gpqa_extended.jsonl --n 546 --log calls.jsonl

The original store is kept as <store>.bak. A round-1 answer is matched to a logged call by agent,
question text, first round, and the same answer and confidence; if several match, the latest wins.
Answers with no logged match keep an empty reasoning (reported), so prestige and the debates
already finished stay exactly as they were.
"""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from mad_dual import load_jsonl


def clip(c):
    return min(0.99, max(0.01, float(c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--problems", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--log", default="calls.jsonl", help="comma-separated call logs to search")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.store) if l.strip()]
    probs = load_jsonl(a.problems, a.n)
    want = {}                                   # (agent, question text) -> list of (pid, answer, conf)
    for r in rows:
        if r["kind"] == "r0":
            q = probs[r["pid"]]["question"]
            for aid, t in r["turns"].items():
                want.setdefault((aid, q), []).append((r["pid"], t["answer"], t["confidence"]))

    found = {}                                  # (pid, agent) -> (ts, rationale, key_uncertainty)
    for path in a.log.split(","):
        if not os.path.exists(path):
            print(f"(log {path} not found, skipped)"); continue
        with open(path) as f:
            for line in f:
                if '"round": 0' not in line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if c.get("round") != 0:
                    continue
                key = (c.get("agent"), c.get("q"))
                if key not in want:
                    continue
                out = c.get("out") or {}
                for pid, ans, conf in want[key]:
                    if str(out.get("answer", "")).strip() == str(ans).strip() and \
                            abs(clip(out.get("confidence", -1)) - conf) < 1e-6:
                        prev = found.get((pid, key[0]))
                        if prev is None or c.get("ts", 0) >= prev[0]:
                            found[(pid, key[0])] = (c.get("ts", 0), out.get("rationale", ""),
                                                    out.get("key_uncertainty", ""))

    total = filled = 0
    for r in rows:
        if r["kind"] == "r0":
            for aid, t in r["turns"].items():
                total += 1
                hit = found.get((r["pid"], aid))
                t["rationale"] = hit[1] if hit else t.get("rationale", "")
                if hit:
                    t["key_uncertainty"] = hit[2]
                    filled += 1
    shutil.copy(a.store, a.store + ".bak")
    with open(a.store, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"recovered reasoning for {filled}/{total} round-1 answers; original kept as {a.store}.bak")
    if filled < total:
        print(f"{total - filled} answers had no logged match and keep empty reasoning "
              f"(their answers and confidence are unchanged).")


if __name__ == "__main__":
    main()
