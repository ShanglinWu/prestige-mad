"""prestige_mad_v2.py — clean redesign of prestige-weighted debate, v2: every debate runs in two
random seatings (flip rates), the report adds paired significance tests, and --fresh-against
reports the questions earlier runs never used.

Two signals, each one thing:
  * prestige p_i(t)  = agent i's running ACCURACY on round-1 answers before question t
                       (uniform 0.5 during a warm-up of the first W questions)
  * confidence c_i   = the agent's STATED confidence this question (no self-consistency term)

We track D_i = c_i - p_i, the overconfidence gap, per agent per question, and plot it over time.

Debate (policy 'prestige'):
  1. Every agent answers once (round 1).
  2. If they disagree: each distinct answer gets ONE advocate (its highest-prestige holder),
     and the advocates are laid out most-trusted LAST, since agents weight the last argument
     they read most. Everyone then revises. Repeat until agreement or --rounds is reached.
  3. Final answer: a vote weighted by log-odds of prestige (a chronically-wrong agent's vote
     shrinks toward zero). During the warm-up the vote is one-agent-one-vote.

Baselines: 'all_speak' (standard simultaneous debate) and 'none' (majority vote, no debate).

    python3 prestige_mad_v2.py --mock --n 60 --policies prestige,all_speak,none
    python3 prestige_mad_v2.py --design expert --n 250 --problems gpqa_main_extra.jsonl --policies prestige,all_speak
    python3 prestige_mad_v2.py report --store runs/foo.jsonl      # accuracy + D-over-time plot

No API calls in --mock or in report.
"""
import argparse, hashlib, json, math, os, random, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mad_dual import (AgentTurn, GeminiAgent, OpenAIAgent, MockAgent, answers_match,
                      load_jsonl, synthetic, panel_spec, _spec_agent, _qid, _levels)

PRIOR = 0.5


# ---------- signals ----------
def prestige_path(round1, ids, warmup):
    """round1: list per question (in order) of {agent: correct(bool)}.
    Returns p[i] before each question = running accuracy, or 0.5 during warm-up."""
    path, hit, seen = [], {a: 0 for a in ids}, 0
    for r in round1:
        if seen < warmup:
            path.append({a: PRIOR for a in ids})
        else:
            path.append({a: hit[a] / seen for a in ids})
        for a in ids:
            hit[a] += int(r[a])
        seen += 1
    return path


def stem_key(question):
    """Hash of the question text without its answer options, so the same GPQA question
    matches across files whose options were shuffled differently."""
    stem = question.split("\n\n(A)")[0].strip()
    return hashlib.md5(stem.encode()).hexdigest()[:12]


def logodds(p):
    p = min(max(p, 0.02), 0.98)
    return math.log(p / (1 - p))


# ---------- one debate ----------
def advocates(turns, prestige, conf):
    """One advocate per distinct answer; most-trusted (prestige, then confidence) LAST."""
    best = {}
    for a, ans in turns.items():
        for g in best:
            if answers_match(ans, g):
                if (prestige[a], conf[a]) > (prestige[best[g]], conf[best[g]]):
                    best[g] = a
                break
        else:
            best[ans] = a
    return sorted(best.values(), key=lambda a: (prestige[a], conf[a]))   # ascending -> top last


def vote(turns, prestige, conf, weighted):
    tally = defaultdict(float)
    for a, ans in turns.items():
        w = logodds(prestige[a]) if weighted else 1.0
        for g in list(tally):
            if answers_match(ans, g):
                tally[g] += w
                break
        else:
            tally[ans] += w
    return max(tally, key=tally.get)


def run_debate(agents, problem, prestige, policy, rounds, first, warm):
    ids = [a.agent_id for a in agents]
    ans = {a.agent_id: first[a.agent_id].answer for a in agents}
    conf = {a.agent_id: first[a.agent_id].confidence for a in agents}
    rat = {a.agent_id: first[a.agent_id].rationale for a in agents}
    transcript, rnds = [], {aid: [(ans[aid], round(conf[aid], 3))] for aid in ids}
    for rnd in range(rounds - 1):
        if len(set(ans.values())) == 1:
            break
        if policy == "all_speak":
            speakers = ids
        else:
            speakers = advocates(ans, prestige, conf)
        board = [{"round": rnd, "speaker": s, "speaker_prestige": prestige[s],
                  "confidence": conf[s], "answer": ans[s], "rationale": rat[s]} for s in speakers]
        transcript += board
        outs = {}
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            res = list(ex.map(lambda a: (a.agent_id, a.propose(problem, transcript, prestige)), agents))
        for aid, t in res:
            t = t.clipped()
            ans[aid], conf[aid], rat[aid] = t.answer, t.confidence, t.rationale
            rnds[aid].append((t.answer, round(t.confidence, 3)))
    final = vote(ans, prestige, conf, weighted=(policy == "prestige" and not warm))
    return {"final": final, "speakers": [b["speaker"] for b in transcript], "rounds": rnds}


# ---------- trial loop ----------
def build_panel(a):
    if a.mock:
        accs = [0.85, 0.62, 0.60, 0.58, 0.58]
        return [MockAgent(f"{'expert' if i == 0 else 'weak%d' % i}", accuracy=accs[i], conf_bias=0.25 if i else 0.1)
                for i in range(5)]
    spec = panel_spec(a)
    if not spec:
        sys.exit("give --design or --panel-file (or --mock)")
    return [_spec_agent(d, a) for d in spec]


def cmd_run(a):
    """Resumable: rerun the same command with the same --store and it continues where it stopped.
    A call that still fails after --max-retries only skips that question/debate for this pass;
    the run makes up to three passes and tells you if anything is still missing."""
    os.makedirs(os.path.dirname(a.store) or ".", exist_ok=True)
    probs = (load_jsonl(a.problems, a.n, levels=_levels(a)) if a.problems else synthetic(a.n, 0))[:a.n]
    agents = build_panel(a)
    for ag in agents:                       # --max-retries applies to every backend (Gemini included)
        ag.max_retries = a.max_retries
    ids = [ag.agent_id for ag in agents]
    cfg = {"n": a.n, "problems": a.problems, "warmup": a.warmup, "rounds": a.rounds,
           "policies": a.policies, "agents": ids}

    old = [json.loads(l) for l in open(a.store) if l.strip()] if os.path.exists(a.store) else []
    if old:
        ocfg = next((r["cfg"] for r in old if r["kind"] == "config"), {})
        same = {k: ocfg.get(k) for k in ("n", "problems", "warmup", "rounds", "agents")} == \
               {k: cfg[k] for k in ("n", "problems", "warmup", "rounds", "agents")}
        if not same:
            sys.exit(f"{a.store} was started with different settings {ocfg}; use a new --store")
        if any(r["kind"] == "r0" and any("rationale" not in t for t in r["turns"].values()) for r in old):
            sys.exit(f"{a.store} comes from an older version (no round-1 reasoning saved) and cannot be "
                     f"resumed. Move it away and start again: mv {a.store} {a.store}.old")
        print(f"resuming {a.store}")
    store = open(a.store, "a")
    def emit(r): store.write(json.dumps(r) + "\n"); store.flush()
    if not old:
        emit({"kind": "config", "cfg": cfg})

    # phase 1: one independent answer per agent per question
    r1 = [None] * len(probs)
    for r in old:
        if r["kind"] == "r0" and r["pid"] < len(probs):
            r1[r["pid"]] = {aid: AgentTurn(rationale=t["rationale"], answer=t["answer"], confidence=t["confidence"],
                                           key_uncertainty=t.get("key_uncertainty", ""))
                            for aid, t in r["turns"].items()}
    def ask(i):
        p = probs[i]
        try:
            with ThreadPoolExecutor(max_workers=len(agents)) as ex:
                outs = list(ex.map(lambda ag: (ag.agent_id, ag.propose(p, [], None).clipped()), agents))
            return i, {aid: t for aid, t in outs}
        except Exception as e:
            print(f"  round 1, question {i}: {str(e)[:120]} (will retry)", flush=True)
            return i, None
    for attempt in range(3):
        todo = [i for i in range(len(probs)) if r1[i] is None]
        if not todo:
            break
        print(f"round 1: {len(todo)} questions x {len(ids)} agents" + (f" (pass {attempt + 1})" if attempt else ""))
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, turns in ex.map(ask, todo):
                if turns is None:
                    continue
                r1[i] = turns
                emit({"kind": "r0", "pid": i, "qid": _qid(probs[i]), "stem": stem_key(probs[i]["question"]),
                      "truth": probs[i]["answer"],
                      "turns": {aid: {"answer": t.answer, "confidence": t.confidence, "rationale": t.rationale}
                                for aid, t in turns.items()}})
    missing = [i for i in range(len(probs)) if r1[i] is None]
    if missing:
        store.close()
        sys.exit(f"{len(missing)} questions still lack round-1 answers (API trouble). "
                 f"Rerun the same command later; it resumes and nothing is lost.")

    correct1 = [{aid: answers_match(t.answer, probs[i]["answer"]) for aid, t in r1[i].items()} for i in range(len(probs))]
    path = prestige_path(correct1, ids, a.warmup)

    # D = c - p per agent per question (round-1 stated confidence vs prestige-before)
    if not any(r["kind"] == "D" for r in old):
        for i in range(len(probs)):
            emit({"kind": "D", "pid": i, "D": {aid: r1[i][aid].confidence - path[i][aid] for aid in ids},
                  "p": path[i], "c": {aid: r1[i][aid].confidence for aid in ids}})

    # phase 2: debates (only contested questions change anything), each in two seatings
    done = {(r["policy"], r["pid"]) for r in old if r["kind"] == "debate"}
    def do(i, pol):
        q = _qid(probs[i])
        seats = agents[:]; random.Random(f"seat:{q}").shuffle(seats)     # random seating per question
        perm = agents[:]; random.Random(f"order:{q}").shuffle(perm)      # a second seating, for flips
        try:
            with ThreadPoolExecutor(max_workers=2) as ex2:
                f1 = ex2.submit(run_debate, seats, probs[i], path[i], pol, a.rounds, r1[i], i < a.warmup)
                f2 = ex2.submit(run_debate, perm, probs[i], path[i], pol, a.rounds, r1[i], i < a.warmup)
                return i, pol, f1.result(), f2.result(), [x.agent_id for x in seats], [x.agent_id for x in perm]
        except Exception as e:
            print(f"  {pol}, question {i}: {str(e)[:120]} (will retry)", flush=True)
            return i, pol, None, None, None, None
    contested = [i for i in range(len(probs)) if len({r1[i][aid].answer for aid in ids}) > 1]
    pols = [p for p in a.policies.split(",") if p != "none"]
    for attempt in range(3):
        todo = [(i, pol) for pol in pols for i in contested if (pol, i) not in done]
        if not todo:
            break
        print(f"debates: {len(todo)} to run ({len(contested)} contested questions x {len(pols)} policies)"
              + (f" (pass {attempt + 1})" if attempt else ""))
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, pol, res, res2, seats, perm in ex.map(lambda x: do(*x), todo):
                if res is None:
                    continue
                done.add((pol, i))
                emit({"kind": "debate", "policy": pol, "pid": i, "qid": _qid(probs[i]),
                      "truth": probs[i]["answer"], "prestige": path[i],
                      "canon": {**res, "seats": seats}, "order": {**res2, "perm": perm}})
    store.close()
    left = [(i, pol) for pol in pols for i in contested if (pol, i) not in done]
    if left:
        sys.exit(f"{len(left)} debates still missing (API trouble). Rerun the same command later to finish them.")
    print(f"-> {a.store}")
    report(a.store, a.fresh_against)


# ---------- report ----------
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def boot_ci(diff, B=5000, seed=0):
    rng = random.Random(seed); n = len(diff)
    if n == 0:
        return float("nan"), float("nan")
    xs = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return xs[int(.025 * B)], xs[int(.975 * B)]


def seen_stems(spec):
    """spec: 'file[:n],file[:n]' -> stems of the problems those runs used (same loader and shuffle)."""
    out = set()
    for part in filter(None, (spec or "").split(",")):
        path, _, n = part.partition(":")
        if not os.path.exists(path):
            print(f"(fresh subset: {path} not found, skipped)"); continue
        for p in load_jsonl(path, int(n) if n else None):
            out.add(stem_key(p["question"]))
    return out


def report(path, fresh_against=None):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    r1 = {r["pid"]: r for r in rows if r["kind"] == "r0"}
    cfg = next(r["cfg"] for r in rows if r["kind"] == "config")
    ids, pols = cfg["agents"], [p for p in cfg["policies"].split(",") if p != "none"]
    deb = defaultdict(dict)
    for r in rows:
        if r["kind"] == "debate":
            deb[r["policy"]][r["pid"]] = r
    n = len(r1)
    print(f"\n{path}: {n} questions, agents {ids}")
    acc1 = {a: sum(answers_match(r1[i]["turns"][a]["answer"], r1[i]["truth"]) for i in r1) / n for a in ids}
    print("round-1 accuracy (= final prestige): " + ", ".join(f"{a} {acc1[a]:.3f}" for a in ids))
    top = max(ids, key=lambda a: acc1[a])

    def final_ok(pol, i, which):
        if i in deb[pol]:
            return answers_match(deb[pol][i][which]["final"], deb[pol][i]["truth"])
        return _maj(r1[i], ids) == 1                                   # unanimous: no debate ran
    def flipped(pol, i):
        d = deb[pol].get(i)
        return bool(d) and "order" in d and not answers_match(d["canon"]["final"], d["order"]["final"])

    def table(keys, label):
        if not keys:
            return
        cont = [i for i in keys if len({r1[i]["turns"][a]["answer"] for a in ids}) > 1]
        maj = sum(_maj(r1[i], ids) == 1 for i in keys) / len(keys)
        exp = sum(answers_match(r1[i]["turns"][top]["answer"], r1[i]["truth"]) for i in keys) / len(keys)
        print(f"\n== {label}: {len(keys)} questions, {len(cont)} contested")
        print(f"   majority vote (no debate) {maj:.1%}   top-prestige agent alone ({top}) {exp:.1%}")
        print(f"   {'policy':12s} {'all (2 seatings avg)':>21s} {'contested':>10s} {'flips':>7s}")
        mean = {}
        for pol in pols:
            m = {i: (final_ok(pol, i, "canon") + final_ok(pol, i, "order" if i in deb[pol] and "order" in deb[pol][i] else "canon")) / 2 for i in keys}
            mean[pol] = m
            fl = sum(flipped(pol, i) for i in keys) / len(keys)
            print(f"   {pol:12s} {sum(m.values())/len(keys):21.1%} {sum(m[i] for i in cont)/max(1,len(cont)):10.1%} {fl:7.1%}")
        if "prestige" in mean and "all_speak" in mean and cont:
            b = sum(final_ok("prestige", i, "canon") and not final_ok("all_speak", i, "canon") for i in cont)
            c = sum(final_ok("all_speak", i, "canon") and not final_ok("prestige", i, "canon") for i in cont)
            diff = [mean["prestige"][i] - mean["all_speak"][i] for i in cont]
            lo, hi = boot_ci(diff)
            print(f"   prestige vs all_speak, contested: {sum(diff)/len(diff):+.1%} (both seatings) 95% CI [{lo:+.1%}, {hi:+.1%}]")
            print(f"     exact McNemar, first seating: prestige right & all_speak wrong {b}, reverse {c}, p = {mcnemar(b, c):.3f}")
            e = {i: answers_match(r1[i]["turns"][top]["answer"], r1[i]["truth"]) for i in keys}
            for pol in ("prestige", "all_speak"):
                b2 = sum(final_ok(pol, i, "canon") and not e[i] for i in keys)
                c2 = sum(e[i] and not final_ok(pol, i, "canon") for i in keys)
                print(f"     {pol} vs top agent alone: {b2} vs {c2}, p = {mcnemar(b2, c2):.3f}")

    table(list(r1), "all questions")
    if fresh_against:
        seen = seen_stems(fresh_against)
        fresh = [i for i in r1 if r1[i].get("stem") and r1[i]["stem"] not in seen]
        table(fresh, f"untouched questions (not in {fresh_against})")

    # D-over-time plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Ds = sorted([r for r in rows if r["kind"] == "D"], key=lambda r: r["pid"])
        if Ds:
            run = {a: [] for a in ids}; tot = {a: 0.0 for a in ids}
            for t, r in enumerate(Ds, 1):
                for a in ids:
                    tot[a] += r["D"][a]; run[a].append(tot[a] / t)
            plt.figure(figsize=(7, 4.3))
            for a in ids:
                plt.plot(range(1, len(Ds) + 1), run[a], lw=2.2 if a == top else 1,
                         color="#185FA5" if a == top else "#993C1D", alpha=1 if a == top else .6)
            plt.axhline(0, color="#999", lw=.6); plt.xlabel("questions seen (t)")
            plt.ylabel("running mean D = c - p")
            plt.title("Overconfidence gap over time (blue = top-prestige agent)")
            out = os.path.splitext(path)[0] + "_D.png"; plt.tight_layout(); plt.savefig(out, dpi=130)
            print(f"\nwrote {out}")
    except ImportError:
        print("(matplotlib not available; skipped D plot)")


def _maj(r, ids):
    from collections import Counter
    c = Counter()
    for a in ids:
        placed = False
        for g in list(c):
            if answers_match(r["turns"][a]["answer"], g):
                c[g] += 1; placed = True; break
        if not placed:
            c[r["turns"][a]["answer"]] += 1
    win = max(c, key=c.get)
    return int(answers_match(win, r["truth"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "report"])
    ap.add_argument("store_arg", nargs="?", help="report: path to a store")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--design", default=None); ap.add_argument("--panel-file", default=None)
    ap.add_argument("--expert-model", default=None); ap.add_argument("--weak-model", default=None)
    ap.add_argument("--expert-thinking", default="low"); ap.add_argument("--weak-thinking", default="none")
    ap.add_argument("--problems", default=None); ap.add_argument("--level", default=None)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-retries", type=int, default=8); ap.add_argument("--schema-mode", default="json_schema")
    ap.add_argument("--openai-extra", default=None); ap.add_argument("--provider", default=None); ap.add_argument("--api", default=None)
    ap.add_argument("--log", default="calls.jsonl"); ap.add_argument("--stagger", type=float, default=0.0)
    ap.add_argument("--policies", default="prestige,all_speak,none")
    ap.add_argument("--store", default="runs/prestige_run.jsonl")
    ap.add_argument("--fresh-against", default=None,
                    help="report: 'file[:n],...' problem files earlier runs used; also report the questions not in them")
    a = ap.parse_args()
    if a.cmd == "report":
        report(a.store_arg or a.store, a.fresh_against)
    else:
        from mad_dual import _RATE_LIMITER
        _RATE_LIMITER.min_interval = a.stagger
        cmd_run(a)


if __name__ == "__main__":
    main()