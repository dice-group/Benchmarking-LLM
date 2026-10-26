#!/usr/bin/env python3
"""
Fill every \\tbd{} of the "Does Tokenization Explain the Gap?" section.

Inputs
  --perf_csv   language, model, task, score            (+ optional score_z, joshi_class)
               task is MT | Summarization | QA (case-insensitive; "translation", "summ", "belebele" also work)
  --joshi_csv  language, joshi_class                    (only if joshi_class is not in --perf_csv)
  --token_csv  one or more CSVs written by token_analysis_v2.py (OPUS-100, XL-Sum, Belebele);
               the task's own dataset supplies each observation's tokenizer metrics:
               MT -> opus, Summarization -> xlsum, QA -> belebele   (change with --task_dataset)
  optional     --coverage_csv  language, model, documented (0/1)   documented pre-training coverage
               --corpus_csv    language, size                      monolingual corpus size (log10 is taken)

Put ONLY the pooled general-purpose models in the performance CSV (the script also drops nllb, mbart and
mt5-multi, and English, by default). `language` and `model` must match the tokenizer CSVs
(yo / yor_Latn / Yoruba and llama3 / Llama-3 are all fine).

Outputs (in --out_dir): fill_sheet.txt, tab_mediation_rows.tex, tab_mediation_all_rows.tex,
                        fertility_vs_score.pdf / .png

Example
  python mediation_analysis.py --perf_csv performance.csv --joshi_csv joshi.csv \\
      --token_csv token_analysis_reports/opus100_token_analysis_results.csv \\
                  token_analysis_reports/xlsum_token_analysis_results.csv \\
                  token_analysis_reports/token_analysis_results.csv
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np

from make_tables import lang_info   # only the language/script table is shared with make_tables.py


# ----------------------------------------------------------------------------- bootstrap (self-contained)
# Copied here so this script works with any version of make_tables.py.
def _fit(X, y, starts, XtX_inv, c):
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    S = np.add.reduceat(X * e[:, None], starts, axis=0)          # per-cluster score sums (rows sorted by cluster)
    V = c * XtX_inv @ (S.T @ S) @ XtX_inv                         # CR1 sandwich
    return beta, np.sqrt(np.maximum(np.diag(V), 1e-300)), e


def wild_cluster_inference(X, y, starts, sizes, B, seed, test_cols=None):
    """Rows of X/y must be sorted by cluster. Returns beta, se, p (null imposed), ci_lo, ci_hi (percentile-t)."""
    n, k = X.shape
    G = len(starts)
    XtX_inv = np.linalg.inv(X.T @ X)
    c = G / (G - 1) * (n - 1) / (n - k)
    beta, se, e = _fit(X, y, starts, XtX_inv, c)
    t_hat = beta / se
    rng = np.random.default_rng(seed)
    W = rng.choice([-1.0, 1.0], size=(B, G))                      # Rademacher weights, one per cluster
    row_cluster = np.repeat(np.arange(G), sizes)

    fitted = X @ beta                                             # 95% CI: unrestricted residuals, percentile-t
    t_star = np.empty((B, k))
    for b in range(B):
        bs, ses, _ = _fit(X, fitted + W[b][row_cluster] * e, starts, XtX_inv, c)
        t_star[b] = (bs - beta) / ses
    lo = beta - np.quantile(t_star, 0.975, axis=0) * se
    hi = beta - np.quantile(t_star, 0.025, axis=0) * se

    pvals = np.full(k, np.nan)                                    # p-values: null-imposed (restricted residuals)
    for j in (range(1, k) if test_cols is None else test_cols):
        Xr = np.delete(X, j, axis=1)
        fr = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
        er = y - fr
        hits = 0
        for b in range(B):
            bs, ses, _ = _fit(X, fr + W[b][row_cluster] * er, starts, XtX_inv, c)
            hits += abs(bs[j] / ses[j]) >= abs(t_hat[j])
        pvals[j] = hits / B
    return beta, se, pvals, lo, hi

METRIC_COLS = {"fert": "avg_tokens_per_word", "oov": "avg_oov_rate",
               "cov": "avg_vocab_coverage", "frag": "avg_subword_fragmentation_rate"}
METRIC_LABEL = {"fert": "Fertility (tok/word)", "oov": "OOV rate", "cov": r"Vocab.\ coverage",
                "frag": "Subword fragmentation"}


MODEL_ALIASES = {}


def _squash(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def norm_model(s):
    """Normalise a model name (case and punctuation removed) and apply --model_alias renames."""
    n = _squash(s)
    return MODEL_ALIASES.get(n, n)


def set_alias(spec):
    """'Mistral-7B-v0.3=mistral' -> performance rows named like the left side match the tokenizer key on the right."""
    k, sep, v = spec.partition("=")
    if not sep or not v:
        raise SystemExit(f"--model_alias expects NAME=KEY, got '{spec}'")
    MODEL_ALIASES[_squash(k)] = _squash(v)


def canon_task(s):
    s = s.strip().lower()
    if s.startswith(("mt", "trans", "opus")):
        return "MT"
    if s.startswith(("summ", "xl")):
        return "Summarization"
    if s.startswith(("qa", "belebele", "question")):
        return "QA"
    raise SystemExit(f"Unknown task '{s}' (use MT, Summarization or QA).")


# ----------------------------------------------------------------------------- data loading
def read_tokens(paths, task_dataset):
    data = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ds = r["dataset"].lower()
                for task, key in task_dataset.items():
                    if key in ds:                                  # later rows overwrite earlier reruns
                        data[(task, lang_info(r["language"])[0], norm_model(r["model"]))] = {
                            k: float(r[c]) for k, c in METRIC_COLS.items()}
    if not data:
        raise SystemExit("No tokenizer rows matched the task datasets; check --token_csv and --task_dataset.")
    return data


def read_simple(path, key_cols, value_col):
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get(value_col) or "").strip() == "":
                continue
            k = tuple(lang_info(r[c])[0] if c == "language" else norm_model(r[c]) for c in key_cols)
            out[k if len(k) > 1 else k[0]] = float(r[value_col])
    return out


def build_observations(args):
    joshi = read_simple(args.joshi_csv, ["language"], "joshi_class") if args.joshi_csv else {}
    tokens = read_tokens(args.token_csv, dict(kv.split("=") for kv in args.task_dataset))
    excluded = {norm_model(m) for m in args.exclude_models}

    grouped = defaultdict(list)
    with open(args.perf_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lang, grp = lang_info(r["language"])
            model = norm_model(r["model"])
            if model in excluded or (lang == "English" and not args.include_english):
                continue
            task_ = canon_task(r["task"])
            if args.only_task and task_ != args.only_task:
                continue
            grouped[(task_, lang, model)].append(r)

    rows, dropped, no_joshi = [], 0, set()
    reasons, examples = defaultdict(int), {}
    for (task, lang, model), rs in grouped.items():
        tok = tokens.get((task, lang, model))
        jc = float(rs[0]["joshi_class"]) if rs[0].get("joshi_class") not in (None, "") else joshi.get(lang)
        score = np.mean([float(r["score"]) for r in rs]) if rs[0].get("score") not in (None, "") else None
        z = np.mean([float(r["score_z"]) for r in rs]) if rs[0].get("score_z") not in (None, "") else None
        if jc is None:
            no_joshi.add(lang)
        missing = []
        if tok is None:
            missing.append("no tokenizer row for this (task, language, model)")
        if jc is None:
            missing.append("no Joshi class for the language")
        if score is None and z is None:
            missing.append("empty score (the score column is blank)")
        for m in missing:
            reasons[m] += 1
            examples.setdefault(m, (task, lang, model))
        if missing:
            dropped += 1
            continue
        rows.append({"task": task, "lang": lang, "model": model, "nonlatin": 1.0 if lang_info(lang)[1] == "NL" else 0.0,
                     "joshi": jc, "score": score, "y": z, **tok})
    if dropped:
        print(f"⚠️ dropped {dropped} of {len(grouped)} (task, language, model) cells. Reasons (a cell can have several):",
              file=sys.stderr)
        for m, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"     {c:4d} x {m}   e.g. {examples[m]}", file=sys.stderr)
        if reasons.get("empty score (the score column is blank)", 0) == len(grouped):
            print("   -> every score is blank. Fill the `score` column of performance_template.csv, or build "
                  "performance.csv from your result files with collect_scores.py.", file=sys.stderr)
        if reasons.get("no tokenizer row for this (task, language, model)", 0):
            print(f"   tokenizer rows available -> tasks {sorted({k[0] for k in tokens})}, "
                  f"models {sorted({k[2] for k in tokens})}", file=sys.stderr)
    if no_joshi:
        print(f"⚠️ no Joshi class for: {sorted(no_joshi)} -> their observations are excluded from every model",
              file=sys.stderr)

    by_task = defaultdict(list)                                   # z-standardise within task
    for o in rows:
        if o["y"] is None:
            by_task[o["task"]].append(o)
    for task, os_ in by_task.items():
        s = np.array([o["score"] for o in os_])
        z = (s - s.mean()) / (s.std(ddof=1) or 1.0)
        for o, zz in zip(os_, z):
            o["y"] = float(zz)
    rows.sort(key=lambda o: (o["lang"], o["task"], o["model"]))
    return rows


# ----------------------------------------------------------------------------- estimation
def design(obs, cols):
    parts = [np.ones(len(obs))]
    for c in cols:
        parts.append(np.array([o[c] for o in obs], dtype=float))
    return np.column_stack(parts)


def point_fit(obs, cols):
    X, y = design(obs, cols), np.array([o["y"] for o in obs])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    return beta, 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))


def boot_fit(obs, cols, B, seed, test=None):
    """OLS + Rademacher wild cluster bootstrap (clusters = languages). Returns dict per column name."""
    X, y = design(obs, cols), np.array([o["y"] for o in obs])
    uniq, starts, sizes = np.unique([o["lang"] for o in obs], return_index=True, return_counts=True)
    names = ["const"] + list(cols)
    test_idx = None if test is None else [names.index(t) for t in test]
    beta, se, p, lo, hi = wild_cluster_inference(X, y, starts, sizes, B, seed, test_cols=test_idx)
    e = y - X @ beta
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return {"coef": dict(zip(names, beta)), "p": dict(zip(names, p)), "lo": dict(zip(names, lo)),
            "hi": dict(zip(names, hi)), "r2": r2, "n": len(obs), "clusters": len(uniq)}


def star(p):
    return "$^{*}$" if p == p and p < 0.05 else ""


def sgn(v):
    return f"{v:+.2f}"


# ----------------------------------------------------------------------------- main analysis
def main(args):
    for spec in args.model_alias:
        set_alias(spec)
    obs = build_observations(args)
    if len(obs) < 20:
        raise SystemExit(f"Only {len(obs)} observations after merging; check language/model/task names.")
    os.makedirs(args.out_dir, exist_ok=True)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    n = len(obs)
    mt = [o for o in obs if o["task"] == "MT"]
    out(f"N_total = {n}   languages = {len({o['lang'] for o in obs})}   models = {sorted({o['model'] for o in obs})}")
    for t in ("MT", "Summarization", "QA"):
        sub = [o for o in obs if o["task"] == t]
        out(f"  {t:13s} N = {len(sub):3d}  Latin = {sum(1 - o['nonlatin'] for o in sub):.0f}  "
            f"non-Latin = {sum(o['nonlatin'] for o in sub):.0f}")
    out(f"N_MT_Latin = {sum(1 - o['nonlatin'] for o in mt):.0f}   N_MT_nonLatin = {sum(o['nonlatin'] for o in mt):.0f}")

    # --- Table: M1-M4 -----------------------------------------------------------------------
    specs = [("M1", ["nonlatin"]), ("M2", ["nonlatin", "joshi"]), ("M3", ["fert", "joshi"]),
             ("M4", ["nonlatin", "fert", "joshi"])]
    res = {}
    out("\n== tab:mediation  (coef, wild-cluster-bootstrap p, 95% CI)")
    for name, cols in specs:
        res[name] = boot_fit(obs, cols, args.B, args.seed)
        for c in cols:
            out(f"  {name} {c:9s} {res[name]['coef'][c]:+.3f}  p={res[name]['p'][c]:.3f}  "
                f"CI [{res[name]['lo'][c]:+.2f}, {res[name]['hi'][c]:+.2f}]")
        out(f"  {name} R2 = {res[name]['r2']:.3f}   N = {res[name]['n']}   clusters = {res[name]['clusters']}")

    b2, b4 = res["M2"]["coef"]["nonlatin"], res["M4"]["coef"]["nonlatin"]
    same = b2 * b4 > 0
    verdict = ("magnitude GROWS (suppression, not mediation)" if same and abs(b4) > abs(b2) else
               "magnitude SHRINKS (pattern expected under mediation)" if same else "sign FLIPS")
    out(f"\n  M2 -> M4 non-Latin: {b2:+.2f} -> {b4:+.2f}: {verdict}")
    f4 = res["M4"]
    inside = -args.equiv_margin <= f4["lo"]["fert"] and f4["hi"]["fert"] <= args.equiv_margin
    out(f"  M4 fertility {f4['coef']['fert']:+.2f}, CI [{f4['lo']['fert']:+.2f}, {f4['hi']['fert']:+.2f}] is "
        f"{'INSIDE' if inside else 'NOT inside'} the +/-{args.equiv_margin:.2f} SD equivalence margin")
    # The margin above is per tokens/word. Per ONE SD of fertility it is a much stricter (and more
    # interpretable) test, so report that too.
    sd_f = float(np.std([o["fert"] for o in obs], ddof=1))
    b_s, lo_s, hi_s = f4["coef"]["fert"] * sd_f, f4["lo"]["fert"] * sd_f, f4["hi"]["fert"] * sd_f
    inside_s = -args.equiv_margin <= lo_s and hi_s <= args.equiv_margin
    out(f"  M4 fertility per 1 SD of fertility ({sd_f:.2f} tok/word): {b_s:+.3f} SD of score, CI [{lo_s:+.2f}, {hi_s:+.2f}] is "
        f"{'INSIDE' if inside_s else 'NOT inside'} the +/-{args.equiv_margin:.2f} SD margin")

    rows = []
    for var, label in [("nonlatin", "non-Latin (script)"), ("fert", "Fertility (tok/word)"), ("joshi", "Joshi class")]:
        cells = [(f"{sgn(res[m]['coef'][var])}{star(res[m]['p'][var])}" if var in res[m]["coef"] else "--")
                 for m, _ in specs]
        rows.append(f"    {label:20s} & " + " & ".join(f"{c:11s}" for c in cells) + r" \\")
    rows.append(r"    \midrule")
    rows.append(f"    {'$R^2$':20s} & " + " & ".join(f"{res[m]['r2']:.2f}".ljust(11) for m, _ in specs) + r" \\")
    rows.append(f"    {'$N$':20s} & " + " & ".join(str(res[m]['n']).ljust(11) for m, _ in specs) + r" \\")
    open(os.path.join(args.out_dir, "tab_mediation_rows.tex"), "w", encoding="utf-8").write("\n".join(rows) + "\n")

    # --- Table: each tokenizer metric added to the base model ---------------------------------
    out("\n== tab:mediation-all  (base = M2; metric added one at a time; gap absorbed = % shrink of |script coef|)")
    base = point_fit(obs, ["nonlatin", "joshi"])[0][1]
    out(f"  base non-Latin coefficient (M2) = {base:+.3f}")
    rows_all = []
    for m in ("fert", "oov", "cov", "frag"):
        b = point_fit(obs, ["nonlatin", "joshi", m])[0][1]
        sd = float(np.std([o[m] for o in obs], ddof=1))
        absorbed = 100 * (abs(base) - abs(b)) / abs(base)
        out(f"  {METRIC_LABEL[m]:22s} SD={sd:.2f}  script coef={b:+.2f}  gap absorbed={absorbed:+.0f}%")
        rows_all.append(f"    {METRIC_LABEL[m]:30s} & {sd:.2f} & {sgn(b)} & {absorbed:+.0f}\\% \\\\")
    open(os.path.join(args.out_dir, "tab_mediation_all_rows.tex"), "w", encoding="utf-8").write("\n".join(rows_all) + "\n")

    # --- Check 1: matched fertility (common support) --------------------------------------------
    lat = [o["fert"] for o in obs if o["nonlatin"] == 0]
    lo_f, hi_f = min(lat), max(lat)
    sub = [o for o in obs if lo_f <= o["fert"] <= hi_f]
    out(f"\n== matched-fertility check: fertility in the Latin range [{lo_f:.2f}, {hi_f:.2f}] tokens/word")
    if len({o['nonlatin'] for o in sub}) == 2 and len({o['lang'] for o in sub}) > 3:
        r = boot_fit(sub, ["nonlatin", "joshi"], args.B, args.seed, test=["nonlatin"])
        out(f"  non-Latin coef = {r['coef']['nonlatin']:+.3f}  p={r['p']['nonlatin']:.3f}  "
            f"CI [{r['lo']['nonlatin']:+.2f}, {r['hi']['nonlatin']:+.2f}]  N = {r['n']}  clusters = {r['clusters']}")
    else:
        out("  not enough observations of both scripts in the common-support range")

    # --- Check 2: language fixed effects ------------------------------------------------------------
    out("\n== language fixed effects: only differences between models WITHIN a language identify fertility")
    langs = sorted({o["lang"] for o in obs})
    for o in obs:
        for l in langs[1:]:
            o[f"fe_{l}"] = 1.0 if o["lang"] == l else 0.0
    fe_cols = ["fert"] + [f"fe_{l}" for l in langs[1:]]
    r = boot_fit(obs, fe_cols, args.B, args.seed, test=["fert"])
    out(f"  fertility coef = {r['coef']['fert']:+.3f}  p={r['p']['fert']:.3f}  "
        f"CI [{r['lo']['fert']:+.2f}, {r['hi']['fert']:+.2f}]  N = {r['n']}")

    # --- Check 2b: language AND model fixed effects ---------------------------------------------------
    # With few models, within-language differences in fertility are mostly differences BETWEEN models
    # (model capability). Adding model dummies leaves only the language x model interaction in fertility.
    models_ = sorted({o["model"] for o in obs})
    for o in obs:
        for m in models_[1:]:
            o[f"mfe_{m}"] = 1.0 if o["model"] == m else 0.0
    out("\n== language + model fixed effects: fertility identified only by language x model variation")
    cols2 = ["fert"] + [f"fe_{l}" for l in langs[1:]] + [f"mfe_{m}" for m in models_[1:]]
    r = boot_fit(obs, cols2, args.B, args.seed, test=["fert"])
    out(f"  fertility coef = {r['coef']['fert']:+.3f}  p={r['p']['fert']:.3f}  "
        f"CI [{r['lo']['fert']:+.2f}, {r['hi']['fert']:+.2f}]  N = {r['n']}")

    # --- Descriptive: where is there a script gap at all? ----------------------------------------------
    out("\n== non-Latin coefficient (no controls) by task and by model; wild-cluster-bootstrap p and 95% CI")
    for key, vals in [("task", ["MT", "Summarization", "QA"]), ("model", models_)]:
        for v in vals:
            sub = [o for o in obs if o[key] == v]
            if len({o["nonlatin"] for o in sub}) < 2 or len({o["lang"] for o in sub}) < 4:
                continue
            r = boot_fit(sub, ["nonlatin"], args.B, args.seed, test=["nonlatin"])
            out(f"  {key}={v:14s} coef = {r['coef']['nonlatin']:+.3f}  p={r['p']['nonlatin']:.3f}  "
                f"CI [{r['lo']['nonlatin']:+.2f}, {r['hi']['nonlatin']:+.2f}]  N = {r['n']}")

    # --- Optional: pre-training data checks (same method; independent of fertility) ---------------
    if args.coverage_csv:
        cov = read_simple(args.coverage_csv, ["language", "model"], "documented")
        sub = [dict(o, doc=cov[(o["lang"], o["model"])]) for o in obs if (o["lang"], o["model"]) in cov]
        b0 = point_fit(sub, ["nonlatin", "joshi"])[0][1]
        r = boot_fit(sub, ["nonlatin", "joshi", "doc"], args.B, args.seed, test=["doc"])
        out(f"\n== documented pre-training coverage: coef = {r['coef']['doc']:+.3f}  p={r['p']['doc']:.3f}  N = {r['n']}  "
            f"gap absorbed = {100 * (abs(b0) - abs(r['coef']['nonlatin'])) / abs(b0):+.1f}%")
    if args.corpus_csv:
        size = read_simple(args.corpus_csv, ["language"], "size")
        sub = [dict(o, logdata=float(np.log10(size[o["lang"]]))) for o in obs if size.get(o["lang"], 0) > 0]
        b0 = point_fit(sub, ["nonlatin", "joshi"])[0][1]
        r = boot_fit(sub, ["nonlatin", "joshi", "logdata"], args.B, args.seed, test=["logdata"])
        out(f"== log10 corpus size: coef = {r['coef']['logdata']:+.3f}  p={r['p']['logdata']:.3f}  N = {r['n']}  "
            f"gap absorbed = {100 * (abs(b0) - abs(r['coef']['nonlatin'])) / abs(b0):+.1f}%")

    # --- Figure: fertility vs score ----------------------------------------------------------------
    x, y = np.array([o["fert"] for o in obs]), np.array([o["y"] for o in obs])
    rr = float(np.corrcoef(x, y)[0, 1])
    out(f"\n== fertility vs score: Pearson r = {rr:+.3f} (N = {n});  Latin fertility {lo_f:.2f}-{hi_f:.2f}, "
        f"non-Latin {min(o['fert'] for o in obs if o['nonlatin']):.2f}-{max(o['fert'] for o in obs if o['nonlatin']):.2f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for flag, color, marker, label in [(0, "#3f9b5f", "o", "Latin"), (1, "#7b3f98", "^", "nonLatin")]:
        pts = [(o["fert"], o["y"]) for o in obs if o["nonlatin"] == flag]
        ax.scatter(*zip(*pts), s=22, color=color, marker=marker, alpha=0.75, linewidths=0, label=label)
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + intercept, "k--", linewidth=1)
    ax.set_xlabel("Tokenizer fertility (tokens / word)")
    ax.set_ylabel("Task score (z, within task)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fertility_vs_score.pdf"))
    fig.savefig(os.path.join(args.out_dir, "fertility_vs_score.png"), dpi=250)
    plt.close(fig)

    open(os.path.join(args.out_dir, "fill_sheet.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\n✅ wrote fill_sheet.txt, tab_mediation_rows.tex, tab_mediation_all_rows.tex and "
          f"fertility_vs_score.pdf/.png to {args.out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf_csv", required=True)
    ap.add_argument("--token_csv", nargs="+", required=True)
    ap.add_argument("--joshi_csv", default=None)
    ap.add_argument("--coverage_csv", default=None)
    ap.add_argument("--corpus_csv", default=None)
    ap.add_argument("--task_dataset", nargs="+", default=["MT=opus", "Summarization=xlsum", "QA=belebele"],
                    help="TASK=substring of the dataset name in the token CSV")
    ap.add_argument("--exclude_models", nargs="+", default=["nllb", "mbart", "mt5-multi"])
    ap.add_argument("--include_english", action="store_true")
    ap.add_argument("--model_alias", nargs="+", default=[], metavar="NAME=KEY",
                    help="Rename models of the performance CSV to the tokenizer-CSV keys, e.g. Mistral-7B-v0.3=mistral")
    ap.add_argument("--only_task", choices=["MT", "Summarization", "QA"], default=None,
                    help="Restrict every analysis to one task (e.g. where a script gap exists)")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--equiv_margin", type=float, default=0.20)
    ap.add_argument("--out_dir", default="mediation_out")
    main(ap.parse_args())