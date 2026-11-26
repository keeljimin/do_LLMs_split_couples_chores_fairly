"""Step 3. Parse and check the outputs, build the dataset, print a quick summary.

    python validate.py outputs_gemma.jsonl outputs_qwen.jsonl

Output:
  responses.csv  one row per response: status and problems
  dataset.csv    one row per response x task (valid parses only), hours seen from the WFH partner's side
  printed summary: status counts, failure rate by couple type, first look at WFH shares
"""
import argparse
import csv
import difflib
import json
import os
import re
from collections import Counter, defaultdict

from make_prompts import TASKS

TASK_TYPE = {t: ty for t, ty, _ in TASKS}
TASK_HOURS = {t: h for t, _, h in TASKS}
TOL = 0.05  # hours


def norm(s):
    s = str(s).lower().replace("’", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", s).strip()


NORM_TASKS = {norm(t): t for t in TASK_TYPE}


def match_task(name):
    n = norm(name)
    if n in NORM_TASKS:
        return NORM_TASKS[n]
    hit = difflib.get_close_matches(n, NORM_TASKS, n=1, cutoff=0.8)
    return NORM_TASKS[hit[0]] if hit else None


def split_thinking(text):
    """Return (thinking, answer). Qwen-style <think>...</think>; anything before the last tag is thinking."""
    if "</think>" in text:
        th, ans = text.rsplit("</think>", 1)
        return th.replace("<think>", "").strip(), ans
    return "", text


def _find_json(text):
    dec = json.JSONDecoder()
    found = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("tasks"), list):
            found = obj
    return found


REASON = re.compile(r'("reason"\s*:\s*")(.*?)("\s*,\s*"hours")', re.S)


def repair_reason_quotes(text):
    """Escape double quotes the model left unescaped inside "reason" strings,
    e.g. "handles the final "reset" on weekends". Hours are untouched."""
    return REASON.sub(lambda m: m.group(1) + re.sub(r'(?<!\\)"', r'\\"', m.group(2)) + m.group(3), text)


def extract_json(text):
    """Last JSON object in the text that has a 'tasks' list.
    Returns (obj, repaired) where repaired=True if quotes in reasons had to be escaped."""
    obj = _find_json(text)
    if obj is not None:
        return obj, False
    obj = _find_json(repair_reason_quotes(text))
    return obj, obj is not None


def match_person(key, name_1, name_2):
    k = norm(key)
    for n in (name_1, name_2):
        if k == norm(n):
            return n
    for n in (name_1, name_2):  # first name only, e.g. "Deepa" (exact, so "Deepa" never matches "Deepak")
        if k and k == norm(n).split()[0]:
            return n
    return None


def to_num(x):
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace("hours", "").replace("hrs", "").strip())
    except ValueError:
        return None


def check(prompt, out):
    problems = []
    if out["finish_reason"] == "length":
        problems.append("truncated")
    thinking, answer = split_thinking(out["text"])
    obj, repaired = extract_json(answer)
    if obj is None:
        return "no_json", problems + ["no_json"], [], thinking, False
    rows, seen = [], set()
    n1, n2 = prompt["name_1"], prompt["name_2"]
    for item in obj["tasks"]:
        if not isinstance(item, dict):
            problems.append("bad_item")
            continue
        task = match_task(item.get("task", ""))
        if task is None:
            problems.append(f"unknown_task:{item.get('task')}")
            continue
        if task in seen:
            problems.append(f"duplicate_task:{task}")
            continue
        seen.add(task)
        hours = item.get("hours") or {}
        if not isinstance(hours, dict):
            problems.append(f"bad_hours:{task}")
            continue
        h = {}
        for k, v in hours.items():
            who = match_person(k, n1, n2)
            val = to_num(v)
            if who is None:
                problems.append(f"unknown_person:{k}")
            elif val is None or val < 0:
                problems.append(f"bad_number:{task}")
            else:
                h[who] = h.get(who, 0) + val
        h1, h2 = h.get(n1, 0.0), h.get(n2, 0.0)
        total = TASK_HOURS[task]
        if abs(h1 + h2 - total) > TOL:
            problems.append(f"sum_mismatch:{task}")
        h_wfh = h1 if n1 == prompt["wfh_name"] else h2
        h_off = h2 if n1 == prompt["wfh_name"] else h1
        rows.append(dict(task=task, task_type=TASK_TYPE[task], total_hours=total,
                         hours_wfh=h_wfh, hours_office=h_off,
                         share_wfh=(h_wfh / (h_wfh + h_off)) if (h_wfh + h_off) > 0 else None,
                         reason=item.get("reason", "")))
    missing = set(TASK_TYPE) - seen
    problems += [f"missing_task:{t}" for t in sorted(missing)]
    status = "ok" if not problems else "problem"
    return status, problems, rows, thinking, repaired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--outdir", default=".", help="folder for responses.csv and dataset.csv")
    args = ap.parse_args()

    prompts = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(args.prompts)}
    meta_cols = ["condition", "couple_type", "wfh_name", "wfh_role", "wfh_race", "office_name", "office_role",
                 "office_race", "intro_order", "pair_id", "repeat", "seed"]

    resp_rows, data_rows = [], []
    for path in args.outputs:
        for line in open(path):
            try:
                out = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = prompts[out["prompt_id"]]
            status, problems, rows, thinking, repaired = check(p, out)
            base = dict(model=out["model"], prompt_id=out["prompt_id"], **{c: p[c] for c in meta_cols})
            resp_rows.append(dict(base, status=status, n_problems=len(problems),
                                  problems="; ".join(problems), json_repaired=repaired,
                                  finish_reason=out["finish_reason"],
                                  n_output_tokens=out["n_output_tokens"], thinking_chars=len(thinking)))
            for r in rows:
                data_rows.append(dict(base, response_status=status, **r))

    for name, rows in (("responses.csv", resp_rows), ("dataset.csv", data_rows)):
        if rows:
            with open(os.path.join(args.outdir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"wrote responses.csv ({len(resp_rows)} rows) and dataset.csv ({len(data_rows)} rows)\n")

    # ---- summary ----
    for model in sorted({r["model"] for r in resp_rows}):
        R = [r for r in resp_rows if r["model"] == model]
        print(f"===== {model}: {len(R)} responses =====")
        print("status:", dict(Counter(r["status"] for r in R)))
        kinds = Counter(pr.split(":")[0] for r in R for pr in r["problems"].split("; ") if pr)
        print("problem types:", dict(kinds))
        print("valid after repairing unescaped quotes in reasons:",
              sum(r["json_repaired"] and r["status"] == "ok" for r in R))
        print(f"avg output tokens: {sum(r['n_output_tokens'] for r in R) / len(R):.0f} | "
              f"with thinking: {sum(r['thinking_chars'] > 0 for r in R)}")

        fail = defaultdict(lambda: [0, 0])
        for r in R:
            fail[r["couple_type"]][0] += r["status"] != "ok"
            fail[r["couple_type"]][1] += 1
        worst = sorted(fail.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:5]
        print("highest failure rates by couple type:")
        for ct, (bad, n) in worst:
            print(f"   {bad}/{n}  {ct}")

        # first look: WFH partner's share of total hours, valid responses only
        tot = defaultdict(lambda: [0.0, 0.0])
        for d in data_rows:
            if d["model"] == model and d["response_status"] == "ok":
                tot[d["prompt_id"]][0] += d["hours_wfh"]
                tot[d["prompt_id"]][1] += d["hours_wfh"] + d["hours_office"]
        groups = defaultdict(list)
        for pid, (w, t) in tot.items():
            p = prompts[pid]
            key = "baseline" if p["condition"] == "baseline" else f"{p['wfh_role']} WFH + {p['office_role']}"
            groups[key].append(w / t)
        print("WFH partner's share of total hours (valid responses):")
        for k, v in sorted(groups.items()):
            print(f"   {k:28s} {sum(v) / len(v):.3f}  (n={len(v)})")
        print()


if __name__ == "__main__":
    main()
