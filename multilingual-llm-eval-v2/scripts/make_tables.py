#!/usr/bin/env python3
"""
Rebuild the two paper tables from your own results.

  table16  Tokenizer analysis: 4 panels (tokens/word, OOV rate, vocab coverage, subword
           fragmentation) x languages (grouped L / NL by script) x tokenizers; best value per
           language in bold; arrows give the better direction.
           Input : CSV written by token_analysis_v2.py
  table6   Predicting task performance with M1-M4 (script, +Joshi class, fertility+class, all);
           cluster-robust OLS with a Rademacher wild cluster bootstrap (clusters = languages).
           Input : a performance CSV (+ the token-analysis CSV for fertility)

Outputs (LaTeX, needs booktabs + graphicx; Markdown for quick viewing): <out_prefix>.tex / .md
(table6 also writes <out_prefix>_diagnostics.txt with CIs and p-values for writing the caption/text).

Examples
  python make_tables.py table16 --token_csv token_analysis_reports/token_analysis_results.csv
  python make_tables.py table6  --perf_csv performance.csv --joshi_csv joshi.csv \
      --token_csv token_analysis_reports/token_analysis_results.csv

performance.csv columns : language, model, task, score   (optional: score_z, joshi_class)
joshi.csv columns       : language, joshi_class           (only if joshi_class isn't in performance.csv)
`language` / `model` must match the token CSV (ISO codes like yo or FLORES codes like yor_Latn both work;
`model` is the key used in token_analysis_v2.py, e.g. llama3, mistral, bloom, xglm, qwen3, gemma3, aya).
"""
import argparse
import csv
import re
import sys
from collections import defaultdict

import numpy as np

# --- Languages: canonical name, script group (L = Latin, NL = non-Latin), aliases ------------------
LANGS = [
    ("Somali", "L", ["so", "som_Latn"]),
    ("Swahili", "L", ["sw", "swh_Latn", "swa_Latn"]),
    ("Yoruba", "L", ["yo", "yor_Latn"]),
    ("Hausa", "L", ["ha", "hau_Latn"]),
    ("Igbo", "L", ["ig", "ibo_Latn"]),
    ("English", "L", ["en", "eng_Latn"]),
    ("French", "L", ["fr", "fra_Latn"]),
    ("Pashto", "NL", ["ps", "pbt_Arab", "pus_Arab"]),
    ("Kannada", "NL", ["kn", "kan_Knda"]),
    ("Sinhala", "NL", ["si", "sin_Sinh"]),
    ("Marathi", "NL", ["mr", "mar_Deva"]),
    ("Punjabi", "NL", ["pa", "pan_Guru"]),
    ("Tajik", "NL", ["tg", "tgk_Cyrl"]),
    ("Kyrgyz", "NL", ["ky", "kir_Cyrl"]),
    ("Telugu", "NL", ["te", "tel_Telu"]),
    ("Amharic", "NL", ["am", "amh_Ethi"]),
    ("Burmese", "NL", ["my", "mya_Mymr"]),
    ("Nepali", "NL", ["ne", "npi_Deva"]),
]
ALIAS = {a.lower(): (name, grp) for name, grp, aliases in LANGS for a in [name, *aliases]}
LANG_ORDER = {name: i for i, (name, _, _) in enumerate(LANGS)}
_warned = set()


def lang_info(code):
    """code (yo / yor_Latn / Yoruba) -> (canonical name, 'L' or 'NL')."""
    key = code.strip().lower()
    if key in ALIAS:
        return ALIAS[key]
    m = re.match(r".*_([A-Z][a-z]{3})$", code.strip())
    if m:
        return code.strip(), ("L" if m.group(1) == "Latn" else "NL")
    if code not in _warned:
        print(f"⚠️ Unknown language '{code}': add it to LANGS (assuming Latin script).", file=sys.stderr)
        _warned.add(code)
    return code.strip(), "L"


#MODEL_DISPLAY = {"nllb": "NLLB", "mbart": "mBART","llama3": "Llama-3", "mistral": "Mistral", "bloom": "BLOOM", "xglm": "XGLM",
#                 "qwen3": "Qwen3", "gemma3": "Gemma-3", "aya": "Aya"}
MODEL_DISPLAY = {"nllb": "NLLB", "mbart": "mBART","llama3": "Llama-3", "mistral": "Mistral", "bloom": "BLOOM", "gemma3": "Gemma-3", "aya": "Aya"}
#MODEL_ORDER = ["llama3", "mistral", "bloom", "xglm", "qwen3", "gemma3", "aya"]

MODEL_ORDER = ["nllb","mbart","llama3", "mistral", "bloom", "gemma3", "aya"]
METRICS = [  # column in the token CSV, panel title, better direction
    ("avg_tokens_per_word", "Avg. tokens / word", "down"),
    ("avg_oov_rate", "Avg. OOV rate", "down"),
    ("avg_vocab_coverage", "Avg. vocab. coverage", "up"),
    ("avg_subword_fragmentation_rate", "Avg. subword fragmentation", "down"),
]


def fmt(v):
    return f"{v:.3f}" if 0 < abs(v) < 0.01 else f"{v:.2f}"


def join_and(items):
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


# =============================================================================
# Table 16
# =============================================================================
def read_token_csv(path):
    data, meta = {}, {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang, _ = lang_info(row["language"])
            key = (lang, row["model"])
            data[key] = {col: float(row[col]) for col, _, _ in METRICS}     # later rows overwrite reruns
            meta[key] = {"n_texts": int(row["n_texts"]), "vocab_size": int(row["vocab_size"]),
                         "dataset": row["dataset"]}
    return data, meta


def panel_rows(metric, direction, langs, models, data):
    """Yield (language, group, [(text, is_best) or None per model])."""
    for lang, grp in langs:
        vals = [data.get((lang, m), {}).get(metric) for m in models]
        present = [v for v in vals if v is not None]
        best = (min(present) if direction == "down" else max(present)) if present else None
        cells = [None if v is None else (fmt(v), fmt(v) == fmt(best)) for v in vals]
        yield lang, grp, cells


def latex_panel(letter, title, metric, direction, langs, models, data):
    arrow = r"$\downarrow$" if direction == "down" else r"$\uparrow$"
    out = [rf"\textbf{{({letter}) {title} ({arrow})}}\\[3pt]",
           r"\resizebox{\linewidth}{!}{%",
           rf"\begin{{tabular}}{{l{'c' * len(models)}}}", r"\toprule",
           "Language & " + " & ".join(MODEL_DISPLAY.get(m, m) for m in models) + r" \\", r"\midrule"]
    prev = None
    for lang, grp, cells in panel_rows(metric, direction, langs, models, data):
        if prev is not None and grp != prev:
            out.append(r"\addlinespace")
        prev = grp
        cell_txt = ["--" if c is None else (rf"\textbf{{{c[0]}}}" if c[1] else c[0]) for c in cells]
        out.append(rf"{lang}$_{{{grp}}}$ & " + " & ".join(cell_txt) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(out)


def md_panel(letter, title, metric, direction, langs, models, data):
    arrow = "↓" if direction == "down" else "↑"
    out = [f"**({letter}) {title} ({arrow})**", "",
           "| Language | " + " | ".join(MODEL_DISPLAY.get(m, m) for m in models) + " |",
           "|---|" + "---:|" * len(models)]
    for lang, grp, cells in panel_rows(metric, direction, langs, models, data):
        txt = ["–" if c is None else (f"**{c[0]}**" if c[1] else c[0]) for c in cells]
        out.append(f"| {lang} ({grp}) | " + " | ".join(txt) + " |")
    return "\n".join(out)


def build_caption(dataset_label, langs, models, data, meta):
    sizes = defaultdict(list)
    for m in models:
        vs = [meta[k]["vocab_size"] for k in meta if k[1] == m]
        if vs:
            sizes[f"{round(vs[0] / 1000)}K"].append(MODEL_DISPLAY.get(m, m))
    size_txt = join_and(f"{'/'.join(names)} {k}" for k, names in sizes.items())
    counts = {lang: max(meta[(lang, m)]["n_texts"] for m in models if (lang, m) in meta)
              for lang, _ in langs if any((lang, m) in meta for m in models)}
    top = max(counts.values())
    few = [(lang, n) for lang, n in counts.items() if n < 0.9 * top]
    few_txt = ""
    if few:
        few_txt = (f" {join_and(l for l, _ in few)} have fewer analysis texts available "
                   f"({join_and(str(n) for _, n in few)}, respectively).")
    return (f"Tokenizer analysis on {dataset_label} text for the {len(langs)} evaluation languages, by tokenizer. "
            f"We report four metrics: (a) average tokens per word, (b) average out-of-vocabulary (OOV) rate, "
            f"(c) average vocabulary coverage, and (d) average subword fragmentation. Arrows indicate the better "
            f"direction; per language, the best value across tokenizers is in \\textbf{{bold}}. Languages are "
            f"grouped by script (L--Latin, NL--non-Latin). All runs use each model's own pretrained tokenizer on "
            f"the same texts (up to {top:,} analysis texts per language); tokenizer vocabulary sizes are "
            f"{size_txt}.{few_txt}")


def render_png(path, langs, models, data):
    """Picture of the four panels (like the paper figure): best value per language in bold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(2 * (2.2 + 0.85 * len(models)), 0.23 * (len(langs) + 1) * 2 + 2.2))
    for ax, letter, (col, title, direction) in zip(axes.flat, "abcd", METRICS):
        ax.axis("off")
        arrow = "↓" if direction == "down" else "↑"
        ax.set_title(f"({letter}) {title} ({arrow})", fontsize=11, fontweight="bold")
        rows = list(panel_rows(col, direction, langs, models, data))
        text = [[f"{lang} ({grp})"] + ["–" if c is None else c[0] for c in cells] for lang, grp, cells in rows]
        tbl = ax.table(cellText=text, colLabels=["Language"] + [MODEL_DISPLAY.get(m, m) for m in models],
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.3)
        for j in range(len(models) + 1):
            tbl[0, j].set_text_props(fontweight="bold")
            tbl[0, j].set_facecolor("#e9e9e9")
        prev = None
        for i, (_, grp, cells) in enumerate(rows, start=1):
            tbl[i, 0].set_text_props(ha="left")
            for j, c in enumerate(cells, start=1):
                if c and c[1]:
                    tbl[i, j].set_text_props(fontweight="bold")
            if prev is not None and grp != prev:                 # visual gap between Latin / non-Latin groups
                for j in range(len(models) + 1):
                    tbl[i, j].visible_edges = "TLR" if j in (0, len(models)) else "T"
            prev = grp
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cmd_table16(args):
    data, meta = read_token_csv(args.token_csv)
    present_models = {m for _, m in data}
    models = [m for m in (args.models or MODEL_ORDER) if m in present_models]
    models += sorted(present_models - set(models)) if not args.models else []
    langs = sorted({(l, lang_info(l)[1]) for l, _ in data}, key=lambda x: LANG_ORDER.get(x[0], 999))
    langs = sorted(langs, key=lambda x: (x[1] != "L", LANG_ORDER.get(x[0], 999)))       # Latin first
    dataset_label = args.dataset_label or next(iter(meta.values()))["dataset"].split("/")[-1]
    caption = build_caption(dataset_label, langs, models, data, meta)

    letters = "abcd"
    panels_tex = [latex_panel(letters[i], t, c, d, langs, models, data) for i, (c, t, d) in enumerate(METRICS)]
    tex = "\n".join([
        "% Requires \\usepackage{booktabs,graphicx}",
        r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}", r"\label{tab:tokenizer-analysis}",
        r"\begin{minipage}[t]{0.49\textwidth}\centering", panels_tex[0], r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.49\textwidth}\centering", panels_tex[1], r"\end{minipage}",
        r"\vspace{1.5em}",
        r"\begin{minipage}[t]{0.49\textwidth}\centering", panels_tex[2], r"\end{minipage}\hfill",
        r"\begin{minipage}[t]{0.49\textwidth}\centering", panels_tex[3], r"\end{minipage}",
        r"\end{table*}", ""])
    md = ["**Table 16.** " + caption.replace("\\textbf{bold}", "**bold**").replace("--", "–"), ""]
    for i, (c, t, d) in enumerate(METRICS):
        md += [md_panel(letters[i], t, c, d, langs, models, data), ""]

    with open(args.out_prefix + ".tex", "w", encoding="utf-8") as f:
        f.write(tex)
    with open(args.out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    if args.png:
        render_png(args.out_prefix + ".png", langs, models, data)
    print(f"✅ Wrote {args.out_prefix}.tex, {args.out_prefix}.md"
          f"{' and ' + args.out_prefix + '.png' if args.png else ''} "
          f"({len(langs)} languages x {len(models)} tokenizers)")


# =============================================================================
# Table 6: OLS with cluster-robust SEs + Rademacher wild cluster bootstrap
# =============================================================================
def _fit(X, y, starts, XtX_inv, c):
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    S = np.add.reduceat(X * e[:, None], starts, axis=0)          # per-cluster score sums (rows sorted by cluster)
    V = c * XtX_inv @ (S.T @ S) @ XtX_inv                         # CR1 sandwich
    return beta, np.sqrt(np.maximum(np.diag(V), 1e-300)), e


def wild_cluster_inference(X, y, starts, sizes, B, seed):
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
    for j in range(1, k):
        Xr = np.delete(X, j, axis=1)
        fr = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
        er = y - fr
        hits = 0
        for b in range(B):
            bs, ses, _ = _fit(X, fr + W[b][row_cluster] * er, starts, XtX_inv, c)
            hits += abs(bs[j] / ses[j]) >= abs(t_hat[j])
        pvals[j] = hits / B
    return beta, se, pvals, lo, hi


def load_perf(args):
    joshi = {}
    if args.joshi_csv:
        with open(args.joshi_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                joshi[lang_info(r["language"])[0]] = float(r["joshi_class"])
    tok, _ = read_token_csv(args.token_csv)

    rows = []
    with open(args.perf_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lang, grp = lang_info(r["language"])
            rows.append({"lang": lang, "nonlatin": 1.0 if grp == "NL" else 0.0, "model": r["model"], "task": r["task"],
                         "score": float(r["score"]) if r.get("score") not in (None, "") else None,
                         "score_z": float(r["score_z"]) if r.get("score_z") not in (None, "") else None,
                         "joshi": float(r["joshi_class"]) if r.get("joshi_class") not in (None, "") else joshi.get(lang)})
    by_task = defaultdict(list)
    for r in rows:
        if r["score_z"] is None:                                  # z-standardize within task
            by_task[r["task"]].append(r)
    for task, rs in by_task.items():
        s = np.array([r["score"] for r in rs])
        sd = s.std(ddof=1)
        for r, z in zip(rs, (s - s.mean()) / (sd if sd > 0 else 1.0)):
            r["score_z"] = float(z)

    kept, dropped = [], 0
    for r in rows:
        t = tok.get((r["lang"], r["model"]))
        if t is None or r["joshi"] is None or r["score_z"] is None:
            dropped += 1
            continue
        r["fert"] = t["avg_tokens_per_word"]
        kept.append(r)
    if dropped:
        print(f"⚠️ Dropped {dropped} rows without fertility / Joshi class / score.", file=sys.stderr)
    kept.sort(key=lambda r: r["lang"])                            # sorted by cluster for reduceat
    return kept


SPECS = [("M1", "script", ["nonlatin"]), ("M2", "+class", ["nonlatin", "joshi"]),
         ("M3", "fert.+class", ["fert", "joshi"]), ("M4", "all", ["nonlatin", "fert", "joshi"])]
ROW_LABELS = {"nonlatin": "non-Latin (script)", "fert": "Fertility (tok/word)", "joshi": "Joshi class"}


def cmd_table6(args):
    rows = load_perf(args)
    if len(rows) < 10:
        raise SystemExit("Too few observations after merging; check language/model codes match the token CSV.")
    y = np.array([r["score_z"] for r in rows])
    clusters = [r["lang"] for r in rows]
    uniq, starts, sizes = np.unique(clusters, return_index=True, return_counts=True)
    n, G = len(rows), len(uniq)

    results = {}
    for name, _, vars_ in SPECS:
        X = np.column_stack([np.ones(n)] + [[r[v] for r in rows] for v in vars_])
        beta, se, p, lo, hi = wild_cluster_inference(X, y, starts, sizes, args.B, args.seed)
        e = y - X @ beta
        r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
        results[name] = {"vars": {v: dict(b=beta[i + 1], se=se[i + 1], p=p[i + 1], lo=lo[i + 1], hi=hi[i + 1])
                                  for i, v in enumerate(vars_)}, "r2": r2}
        print(f"  fitted {name} ({', '.join(vars_)})")

    def cell(model, var, tex):
        d = results[model]["vars"].get(var)
        if d is None:
            return "--"
        s = f"{d['b']:+.2f}"
        star = "^{*}" if d["p"] < 0.05 else ""
        return f"${s}{star}$" if tex else f"{s}{'*' if star else ''}"

    tasks = join_and(sorted({r["task"] for r in rows}))
    caption = (f"Predicting task performance (scores $z$-standardized within task; pooled over {tasks}; "
               f"$N={n}$ language--model observations, {G} language clusters). Significance is assessed with a "
               f"Rademacher wild cluster bootstrap (Cameron et al. 2008; $B={args.B}$). "
               f"${{}}^{{*}}p<0.05$ (bootstrap).")
    head = " & ".join(rf"\textbf{{{n_}}}" for n_, _, _ in SPECS)
    sub = " & ".join(s for _, s, _ in SPECS)
    body = [f"{ROW_LABELS[v]} & " + " & ".join(cell(m, v, True) for m, _, _ in SPECS) + r" \\"
            for v in ["nonlatin", "fert", "joshi"]]
    tex = "\n".join([
        "% Requires \\usepackage{booktabs}", r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
        r"\label{tab:mediation}", r"\begin{tabular}{lcccc}", r"\toprule",
        rf" & {head} \\", rf" & {sub} \\", r"\midrule", *body, r"\midrule",
        r"$R^2$ & " + " & ".join(f"{results[m]['r2']:.2f}" for m, _, _ in SPECS) + r" \\",
        r"$N$ & " + " & ".join(str(n) for _ in SPECS) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

    md = ["**Table 6.** " + caption.replace("$", "").replace("{}^{*}", "*"), "",
          "| | " + " | ".join(f"{n_} ({s})" for n_, s, _ in SPECS) + " |", "|---|---:|---:|---:|---:|"]
    for v in ["nonlatin", "fert", "joshi"]:
        md.append(f"| {ROW_LABELS[v]} | " + " | ".join(cell(m, v, False).replace("--", "–") for m, _, _ in SPECS) + " |")
    md.append("| R² | " + " | ".join(f"{results[m]['r2']:.2f}" for m, _, _ in SPECS) + " |")
    md.append(f"| N | " + " | ".join(str(n) for _ in SPECS) + " |")

    diag = [f"N={n}, clusters={G}, B={args.B}, seed={args.seed}", ""]
    for name, _, _ in SPECS:
        for v, d in results[name]["vars"].items():
            diag.append(f"{name} {ROW_LABELS[v]:22s} coef={d['b']:+.3f}  boot 95% CI [{d['lo']:+.2f}, {d['hi']:+.2f}]  "
                        f"p={d['p']:.3f}")
    b2, b4 = results["M2"]["vars"]["nonlatin"]["b"], results["M4"]["vars"]["nonlatin"]["b"]
    same_sign = b2 * b4 > 0
    verdict = ("magnitude grows (suppression pattern, not mediation)" if same_sign and abs(b4) > abs(b2)
               else "magnitude shrinks (the pattern expected under mediation)" if same_sign
               else "sign flips")
    diag += ["", f"Non-Latin coefficient: M2 {b2:+.2f} -> M4 {b4:+.2f} when fertility is added: {verdict}"]
    f4 = results["M4"]["vars"]["fert"]
    inside = -args.equiv_margin <= f4["lo"] and f4["hi"] <= args.equiv_margin
    diag.append(f"Fertility in M4: {f4['b']:+.2f}, CI [{f4['lo']:+.2f}, {f4['hi']:+.2f}] is "
                f"{'within' if inside else 'NOT within'} the +/-{args.equiv_margin:.2f} SD equivalence margin")

    with open(args.out_prefix + ".tex", "w", encoding="utf-8") as f:
        f.write(tex)
    with open(args.out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(args.out_prefix + "_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(diag))
    print("\n".join(diag))
    print(f"\n✅ Wrote {args.out_prefix}.tex, .md and _diagnostics.txt")


# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p16 = sub.add_parser("table16", help="Tokenizer-analysis table (4 panels)")
    p16.add_argument("--token_csv", required=True)
    p16.add_argument("--models", nargs="+", default=None, help="Column order (registry keys)")
    p16.add_argument("--dataset_label", default=None, help="Name used in the caption (default: from the CSV)")
    p16.add_argument("--png", action="store_true", help="Also render the four panels as a PNG image")
    p16.add_argument("--out_prefix", default="table16_tokenizer_analysis")
    p16.set_defaults(fn=cmd_table16)

    p6 = sub.add_parser("table6", help="Regression table M1-M4 with wild cluster bootstrap")
    p6.add_argument("--perf_csv", required=True)
    p6.add_argument("--token_csv", required=True)
    p6.add_argument("--joshi_csv", default=None)
    p6.add_argument("--B", type=int, default=2000, help="Bootstrap replications")
    p6.add_argument("--seed", type=int, default=0)
    p6.add_argument("--equiv_margin", type=float, default=0.20, help="Equivalence margin in SD units")
    p6.add_argument("--out_prefix", default="table6_mediation")
    p6.set_defaults(fn=cmd_table6)

    a = ap.parse_args()
    a.fn(a)
