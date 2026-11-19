"""Prestige x confidence plot for a prestige_mad_v2 store. No API calls.

    python3 analysis/pc_plot.py runs/prestige_gemini_gpqa_extended_n546.jsonl
    python3 analysis/pc_plot.py STORE.jsonl --problems gpqa_extended.jsonl --block 50

Left panel: x = stated confidence, y = accuracy (prestige). The diagonal is perfect calibration;
below it = overconfident, above it = underconfident. Dashed lines at the panel's mean c and mean p
split the plot into four quadrants (reliable expert, humble expert, bluffer, aware novice).
  big dot     one per agent, whole run (warm-up included)
  trail       one point per block of questions (block accuracy vs block mean confidence), in order
  hollow      one per agent x domain, if --problems is given and its rows carry a domain
Right panel: resolution. Mean confidence on questions the agent got right vs wrong, and AUROC
(chance that a right answer got higher confidence than a wrong one; 0.5 = confidence carries no signal).
Writes <store>_pc.png and prints the numbers behind it.
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
from mad_dual import answers_match

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stem_key(question):
    return hashlib.md5(question.split("\n\n(A)")[0].strip().encode()).hexdigest()[:12]


def auroc(right, wrong):
    if not right or not wrong:
        return float("nan")
    win = sum((r > w) + 0.5 * (r == w) for r in right for w in wrong)
    return win / (len(right) * len(wrong))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--problems", help="problem file, to split points by domain (matched by question stem)")
    ap.add_argument("--block", type=int, default=50, help="questions per trail point")
    ap.add_argument("--out", help="output png (default <store>_pc.png)")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.store) if l.strip()]
    r0 = sorted((r for r in rows if r["kind"] == "r0"), key=lambda r: r["pid"])
    if not r0:
        sys.exit("no round-1 rows in store")
    ids = sorted(r0[0]["turns"])
    c = {x: [r["turns"][x]["confidence"] for r in r0] for x in ids}
    ok = {x: [int(answers_match(r["turns"][x]["answer"], r["truth"])) for r in r0] for x in ids}
    n = len(r0)

    dom = None
    if a.problems:
        m = {}
        for l in open(a.problems):
            if l.strip():
                p = json.loads(l)
                m[stem_key(p["question"])] = p.get("domain") or p.get("level") or ""
        dom = [m.get(r.get("stem", ""), "") for r in r0]
        if not any(dom):
            print("(no domain field matched in --problems; domain points skipped)")
            dom = None

    mean = lambda v: sum(v) / len(v) if v else float("nan")
    P = {x: mean(ok[x]) for x in ids}
    C = {x: mean(c[x]) for x in ids}
    expert = max(ids, key=P.get)
    label = {x: (f"{x} (top)" if x == expert else x) for x in ids}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.35, 1]})
    colors = plt.cm.tab10.colors

    print(f"{a.store}: {n} questions, {len(ids)} agents\n")
    print(f"{'agent':22s} {'p (acc)':>8s} {'c (conf)':>9s} {'c - p':>7s} {'c|right':>8s} {'c|wrong':>8s} {'AUROC':>6s}")
    res = {}
    for k, x in enumerate(ids):
        col = colors[k % 10]
        # trail
        bp, bc = [], []
        for s in range(0, n, a.block):
            if n - s < a.block // 2 and bp:
                break
            bp.append(mean(ok[x][s:s + a.block])); bc.append(mean(c[x][s:s + a.block]))
        ax.plot(bc, bp, "-", color=col, alpha=0.35, lw=1)
        ax.scatter(bc, bp, color=col, s=14, alpha=0.5)
        if len(bc) > 1:
            ax.annotate("", xy=(bc[-1], bp[-1]), xytext=(bc[-2], bp[-2]),
                        arrowprops=dict(arrowstyle="->", color=col, alpha=0.6))
        # domains
        if dom:
            for d in sorted(set(dom) - {""}):
                idx = [i for i in range(n) if dom[i] == d]
                ax.scatter(mean([c[x][i] for i in idx]), mean([ok[x][i] for i in idx]),
                           facecolors="none", edgecolors=col, s=60, lw=1.2)
                if x == expert or len(ids) <= 2:     # weak agents cluster; label one agent only
                    ax.annotate(d[:4], (mean([c[x][i] for i in idx]), mean([ok[x][i] for i in idx])),
                                fontsize=7, color=col, xytext=(4, -8), textcoords="offset points")
        ax.scatter(C[x], P[x], color=col, s=140, edgecolors="black", zorder=5, label=label[x])
        # resolution
        right = [c[x][i] for i in range(n) if ok[x][i]]
        wrong = [c[x][i] for i in range(n) if not ok[x][i]]
        res[x] = (mean(right), mean(wrong), auroc(right, wrong))
        print(f"{label[x]:22s} {P[x]:8.3f} {C[x]:9.3f} {C[x] - P[x]:+7.3f} "
              f"{res[x][0]:8.3f} {res[x][1]:8.3f} {res[x][2]:6.3f}")

    if dom:
        print("\nby domain (p / c):")
        for d in sorted(set(dom) - {""}):
            idx = [i for i in range(n) if dom[i] == d]
            print(f"  {d:20s} n={len(idx):3d}  " + "  ".join(
                f"{x}: {mean([ok[x][i] for i in idx]):.2f}/{mean([c[x][i] for i in idx]):.2f}" for x in ids))

    allv = [v for x in ids for v in (P[x], C[x])]
    lo = max(0.0, min(allv + [0.4]) - 0.08)
    ax.plot([0, 1], [0, 1], color="gray", lw=1)
    ax.fill_between([0, 1], [0, 1], [0, 0], color="red", alpha=0.04)
    ax.fill_between([0, 1], [0, 1], [1, 1], color="blue", alpha=0.04)
    mp, mc = mean(list(P.values())), mean(list(C.values()))
    ax.axhline(mp, ls="--", color="gray", lw=0.8); ax.axvline(mc, ls="--", color="gray", lw=0.8)
    for (tx, ty, t) in ((lo + 0.01, 0.99, "humble expert"), (0.99, 0.99, "reliable expert"),
                        (lo + 0.01, lo + 0.01, "aware novice"), (0.99, lo + 0.01, "bluffer")):
        ax.text(tx, ty, t, fontsize=9, color="gray", ha="left" if tx < 0.5 + lo / 2 else "right",
                va="top" if ty > 0.5 + lo / 2 else "bottom")
    ax.text(0.5 * (lo + 1) + 0.03, 0.5 * (lo + 1) - 0.03, "calibrated (p = c)", rotation=45,
            fontsize=8, color="gray", ha="center", va="center")
    ax.set_xlim(lo, 1); ax.set_ylim(lo, 1); ax.set_aspect("equal")
    ax.set_xlabel("stated confidence c"); ax.set_ylabel("accuracy / prestige p")
    ax.set_title(f"Prestige x confidence (trail = blocks of {a.block})")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=8)

    xs = range(len(ids))
    bx.bar([i - 0.2 for i in xs], [res[x][0] for x in ids], 0.4, label="when right", color="#4a7")
    bx.bar([i + 0.2 for i in xs], [res[x][1] for x in ids], 0.4, label="when wrong", color="#c55")
    for i, x in enumerate(ids):
        bx.text(i, max(res[x][0], res[x][1] if res[x][1] == res[x][1] else 0) + 0.01,
                f"AUROC {res[x][2]:.2f}", ha="center", fontsize=8)
    bx.set_xticks(list(xs)); bx.set_xticklabels([label[x] for x in ids], rotation=20, fontsize=8)
    bx.set_ylim(lo, 1.05); bx.set_ylabel("mean stated confidence")
    bx.set_title("Resolution: does confidence separate right from wrong?")
    bx.legend(fontsize=8)

    fig.tight_layout()
    out = a.out or os.path.splitext(a.store)[0] + "_pc.png"
    fig.savefig(out, dpi=160)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
