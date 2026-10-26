#!/usr/bin/env python3
"""
Build performance.csv (language, model, task, score) from the per-sample CSVs written by your evaluation
scripts, so you do not have to type scores by hand.

  MT            eval_opus_*.py      columns Source, Reference, Prediction, BLEU, chrF++   -> mean chrF++
  Summarization eval_xlsum_*.py     columns Source Text, Reference Summary, Generated Summary
                                                                                         -> mean ROUGE-1 F1 (x100)
  QA            eval_belebele_*.py  columns Context, Question, Gold Answer, Predicted Answer
                                                                                         -> accuracy (%)

The task comes from the file's columns; the language and the model are inferred from the file path
(e.g. results/xlsum/gemma_xlsum_am.csv -> Gemma-3, Amharic). Check the printed table; fix any mistake with
--name_map (CSV with columns pattern,language,model[,task]; `pattern` is a substring of the path).

ROUGE-1 uses a script-agnostic tokenizer (letters, marks and digits of any script). The default rouge_score
tokenizer deletes every non-Latin character, which would turn the script gap into a metric artifact.

  python collect_scores.py --results_dir results --out performance.csv
"""
import argparse
import collections
import csv
import datetime
import os
import re
import sys
import unicodedata

from make_tables import ALIAS

csv.field_size_limit(sys.maxsize)

# (model key as used in the tokenizer CSVs, substrings of the path with non-alphanumerics removed); order matters
MODEL_PATTERNS = [
    ("mt5-multi", ["mt5multi", "multilingualxlsum"]), ("mt5-base", ["mt5base", "mt5"]),
    ("mbart", ["mbart"]), ("nllb", ["nllb"]),
    ("llama3", ["llama3", "llama"]), ("mistral", ["mistral"]), ("bloom", ["bloom"]),
    ("gemma3", ["gemma3", "gemma"]), ("aya", ["aya"]), ("xglm", ["xglm"]), ("qwen3", ["qwen3", "qwen"]),
]


def tokenize(text):
    text = unicodedata.normalize("NFC", text.lower())
    tokens, cur = [], []
    for ch in text:
        if unicodedata.category(ch)[0] in ("L", "M", "N"):
            cur.append(ch)
        elif cur:
            tokens.append("".join(cur))
            cur = []
    if cur:
        tokens.append("".join(cur))
    return tokens


def rouge1_f(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    overlap = sum((collections.Counter(p) & collections.Counter(r)).values())
    if not p or not r or overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def infer_model(path):
    flat = re.sub(r"[^a-z0-9]", "", path.lower())
    for key, subs in MODEL_PATTERNS:
        if any(s in flat for s in subs):
            return key
    return None


def infer_language(path):
    tokens = [t for t in re.split(r"[^a-z0-9]+", path.lower()) if t]
    cands = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]            # yor + latn -> yor_latn
    found = []
    for c in cands:
        if c in ALIAS and ALIAS[c][0] != "English" and ALIAS[c][0] not in found:
            found.append(ALIAS[c][0])
    return found


def score_file(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return None
    if {"Prediction", "chrF++"} <= cols:
        return "MT", sum(float(r["chrF++"]) for r in rows) / len(rows), len(rows)
    if "Generated Summary" in cols:
        v = [rouge1_f(r["Generated Summary"], r["Reference Summary"]) for r in rows]
        return "Summarization", 100 * sum(v) / len(v), len(rows)
    if {"Gold Answer", "Predicted Answer"} <= cols:
        ok = sum(r["Gold Answer"].strip() == r["Predicted Answer"].strip() for r in rows)
        return "QA", 100 * ok / len(rows), len(rows)
    return None


def main(args):
    overrides = []
    if args.name_map:
        with open(args.name_map, encoding="utf-8") as f:
            overrides = list(csv.DictReader(f))

    results, problems, skipped = {}, [], []
    candidates = collections.defaultdict(list)          # (task, lang, model) -> [(path, score, n, mtime)]
    ignored = []
    print(f"{'file':60s} {'task':13s} {'language':9s} {'model':10s} {'n':>5s} {'score':>7s}")
    for root, _, files in os.walk(args.results_dir):
        for fn in sorted(files):
            if not fn.endswith(".csv"):
                continue
            path = os.path.join(root, fn)
            if any(pat in path for pat in args.ignore) or (args.ignore_regex and re.search(args.ignore_regex, path)):
                ignored.append(path)
                continue
            scored = score_file(path)
            if scored is None:                             # not recognised: report it, do not hide it
                with open(path, encoding="utf-8", newline="") as fh:
                    header = next(csv.reader(fh), [])
                skipped.append((path, header))
                continue
            task, score, n = scored
            lang, model = None, infer_model(path)
            langs = infer_language(path)
            if len(langs) == 1:
                lang = langs[0]
            for o in overrides:
                if o["pattern"] in path:
                    lang = o["language"] or lang
                    model = o["model"] or model
                    task = o.get("task") or task
            if lang is None or model is None:
                problems.append((path, langs, model))
                print(f"{path[-60:]:60s} {task:13s} {'?':9s} {str(model):10s} {n:5d} {score:7.2f}   <-- needs --name_map")
                continue
            candidates[(task, lang, model)].append((path, score, n, os.path.getmtime(path)))
            print(f"{path[-60:]:60s} {task:13s} {lang:9s} {model:10s} {n:5d} {score:7.2f}")

    # Resolve duplicates: keep the most recently modified file (NOT the alphabetically last one), and say so.
    dup_rows, max_gap = [], 0.0
    for key, cands in sorted(candidates.items()):
        cands.sort(key=lambda c: c[3])
        keep = cands[-1]
        results[key] = (keep[1], keep[2])
        if len(cands) > 1:
            max_gap = max(max_gap, max(c[1] for c in cands) - min(c[1] for c in cands))
            for c in cands:
                dup_rows.append([key[0], key[1], key[2], "KEPT" if c is keep else "dropped", c[0],
                                 f"{c[1]:.2f}", c[2], datetime.datetime.fromtimestamp(c[3]).strftime("%Y-%m-%d %H:%M")])
    if dup_rows:
        n_dup = len({tuple(r[:3]) for r in dup_rows})
        print(f"\n⚠️ {n_dup} (task, language, model) cells have more than one result file; the most recently "
              f"modified file is kept (largest score difference between versions: {max_gap:.1f} points).")
        if args.duplicates_csv:
            with open(args.duplicates_csv, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["task", "language", "model", "decision", "file", "score", "n", "modified"])
                w.writerows(dup_rows)
            print(f"   all candidates written to {args.duplicates_csv}; exclude a version with --ignore PATTERN")
        else:
            print("   rerun with --duplicates_csv dup.csv to see every candidate, and --ignore PATTERN to choose.")
    if ignored:
        print(f"ℹ️ ignored {len(ignored)} file(s) matching --ignore / --ignore_regex:")
        for p in ignored[:60]:
            print(f"     {p}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["language", "model", "task", "score", "n_samples"])
        for (task, lang, model), (score, n) in sorted(results.items()):
            w.writerow([lang, model, task, f"{score:.4f}", n])
    print(f"\n✅ {len(results)} cells written to {args.out}")

    tasks = ["MT", "Summarization", "QA"]
    models = sorted({m for (_, _, m) in results})
    if models:
        print(f"\nCells found per model and task (number of languages):")
        print(f"  {'model':11s}" + "".join(f"{t:>15s}" for t in tasks))
        for m in models:
            counts = [sum(1 for (t, _, mm) in results if mm == m and t == task) for task in tasks]
            print(f"  {m:11s}" + "".join(f"{c:15d}" for c in counts))
    if skipped:
        print(f"\n⚠️ {len(skipped)} CSV file(s) were NOT recognised as evaluation outputs (unexpected columns):")
        for path, header in skipped[:25]:
            print(f"   {path}\n      columns: {header[:8]}")
        print("   -> if these are your missing runs, their columns differ from the current eval scripts "
              "(MT: Prediction + chrF++; Summarization: Generated Summary; QA: Gold Answer + Predicted Answer).")
    if problems:
        print(f"⚠️ {len(problems)} file(s) could not be attributed (language or model unclear); add them to a "
              f"--name_map CSV (pattern,language,model[,task]) and rerun.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", default="performance.csv")
    ap.add_argument("--name_map", default=None)
    ap.add_argument("--ignore", nargs="+", default=[], metavar="PATTERN",
                    help="Skip files whose path contains any of these substrings (e.g. an old version of a run)")
    ap.add_argument("--ignore_regex", default=None,
                    help="Skip files whose path matches this regular expression, e.g. '(xlsum_llama_|_llama_eval)'")
    ap.add_argument("--duplicates_csv", default=None,
                    help="Write every candidate file of cells that have more than one (score, date, decision)")
    main(ap.parse_args())