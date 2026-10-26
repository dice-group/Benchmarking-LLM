#!/usr/bin/env python3
"""
Why does the regression pool contain so few models?

For every (model, task) this prints how many cells survive each stage that mediation_analysis.py applies:

  tokenizer   languages with a tokenizer row for that model and task (what a complete model would have)
  in perf     rows of the model in your performance CSV
  scored      ... with a non-blank score
  tok match   ... that find a tokenizer row (name and language match)
  Joshi       ... whose language has a Joshi class
  KEPT        ... that enter the regression

and gives a verdict per row: model absent from the performance file, model name that does not match the
tokenizer key, blank scores, languages without a Joshi class, or partly missing languages.

  python pool_coverage.py --perf_csv performance.csv --joshi_csv joshi.csv \\
      --token_csv token_analysis_reports/*.csv
"""
import argparse
import csv
import difflib
from collections import defaultdict

import mediation_analysis as ma
from make_tables import lang_info

TASKS = ["MT", "Summarization", "QA"]


def main(args):
    for spec in args.model_alias:
        ma.set_alias(spec)
    tokens = ma.read_tokens(args.token_csv, dict(kv.split("=") for kv in args.task_dataset))
    joshi = ma.read_simple(args.joshi_csv, ["language"], "joshi_class") if args.joshi_csv else {}
    excluded = {ma.norm_model(m) for m in args.exclude_models}

    tok_langs = defaultdict(set)                  # (model, task) -> languages with tokenizer rows
    for (task, lang, model) in tokens:
        tok_langs[(model, task)].add(lang)
    tok_models = {m for m, _ in tok_langs}

    perf = defaultdict(dict)                      # (model, task) -> {language: has_score}
    raw_names = defaultdict(set)
    with open(args.perf_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = ma.norm_model(r["model"])
            raw_names[m].add(r["model"])
            has = any((r.get(c) or "").strip() != "" for c in ("score", "score_z"))
            lang = lang_info(r["language"])[0]
            if lang == "English" and not args.include_english:
                continue
            perf[(m, ma.canon_task(r["task"]))][lang] = has

    models = sorted(tok_models | {m for m, _ in perf})
    print(f"{'model':11s} {'task':14s} {'tokenizer':>9s} {'in perf':>8s} {'scored':>7s} {'tok match':>9s} "
          f"{'Joshi':>6s} {'KEPT':>5s}   verdict")
    kept_by_model = defaultdict(int)
    missing_rows = []
    for m in models:
        for task in TASKS:
            exp = {l for l in tok_langs.get((m, task), set()) if args.include_english or l != "English"}
            cells = perf.get((m, task), {})
            scored = {l for l, h in cells.items() if h}
            tok_ok = {l for l in scored if l in exp}
            joshi_ok = {l for l in tok_ok if l in joshi}
            kept = joshi_ok
            kept_by_model[m] += len(kept)
            if m not in excluded:
                for l in sorted(exp - kept):
                    missing_rows.append((m, task, l))
            if not exp and not cells:
                continue
            if m in excluded:
                verdict = "excluded by design (--exclude_models)"
            elif not cells and exp:
                verdict = "ABSENT from the performance file"
            elif cells and not scored:
                verdict = "all scores blank"
            elif cells and exp and len(tok_ok) < len(scored):
                miss = sorted(scored - exp)
                verdict = f"no tokenizer row for {len(scored) - len(tok_ok)} scored cells, e.g. {miss[:3]}"
            elif not exp and cells:
                verdict = "no tokenizer rows for this model/task (name mismatch?)"
            elif len(joshi_ok) < len(tok_ok):
                verdict = f"no Joshi class for {sorted(tok_ok - joshi_ok)}"
            elif len(kept) < len(exp):
                verdict = f"partial: missing languages {sorted(exp - kept)}"
            else:
                verdict = "complete"
            print(f"{m:11s} {task:14s} {len(exp):9d} {len(cells):8d} {len(scored):7d} {len(tok_ok):9d} "
                  f"{len(joshi_ok):6d} {len(kept):5d}   {verdict}")

    print("\nObservations per model that enter the regression:",
          {m: n for m, n in sorted(kept_by_model.items()) if m not in excluded})

    if args.write_missing:
        with open(args.write_missing, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "task", "language"])
            w.writerows(missing_rows)
        print(f"\n{len(missing_rows)} missing (model, task, language) cells written to {args.write_missing}")

    absent = sorted(m for m in tok_models if not any(k[0] == m for k in perf) and m not in excluded)
    if absent:
        print(f"\nModels with tokenizer results but NO rows in the performance file: {absent}")
        print("  -> their scores were never collected, or collect_scores.py could not attribute their files "
              "(look for '<-- needs --name_map' lines).")
    unmatched = sorted(m for m in {k[0] for k in perf} if m not in tok_models and m not in excluded)
    for m in unmatched:
        close = difflib.get_close_matches(m, sorted(tok_models), n=2, cutoff=0.4)
        prefix = [t for t in tok_models if t in m or m in t]
        guess = prefix or close
        print(f"\nName mismatch: performance file has {sorted(raw_names[m])} (normalised '{m}'), no tokenizer key "
              f"by that name. Closest keys: {guess}")
        if guess:
            print(f"  -> rerun with  --model_alias {sorted(raw_names[m])[0]}={guess[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf_csv", required=True)
    ap.add_argument("--token_csv", nargs="+", required=True)
    ap.add_argument("--joshi_csv", default=None)
    ap.add_argument("--task_dataset", nargs="+", default=["MT=opus", "Summarization=xlsum", "QA=belebele"])
    ap.add_argument("--exclude_models", nargs="+", default=["nllb", "mbart", "mt5-multi"])
    ap.add_argument("--model_alias", nargs="+", default=[], metavar="NAME=KEY")
    ap.add_argument("--include_english", action="store_true")
    ap.add_argument("--write_missing", default=None, help="CSV listing every missing (model, task, language) cell")
    main(ap.parse_args())