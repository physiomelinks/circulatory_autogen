"""Build the CUFLynx paper's LaTeX benchmark tables from the benchmark harness output.

Two tables are produced into a single ``.tex`` file:

1. **Calibration summary** -- one row per benchmark model: wall-clock calibration time and
   maximum absolute parameter error. By default each model is represented by its best optimiser
   (lowest max param err, ties broken by wall-clock); ``--all-methods`` lists every optimiser
   grouped under its model instead.
2. **Parallel scaling** -- for the slowest model (auto-detected as the one with the largest total
   benchmark wall-clock, override with ``--scaling-benchmark``): one wall-clock column per core
   count plus the speedup over the smallest core count measured.

Inputs (both understood, JSON preferred because it is the harness's canonical serialisation):

* ``--emit-json`` payloads written by ``run_benchmarks.py`` -- ``{"num_ranks": N, "result": {...}}``
  where ``result`` is :func:`benchmarks.docs_results.benchmark_result_to_dict`.
* the Markdown tables written by ``run_benchmarks.py --results-out`` -- parsed back into the same
  structure, so tables can be rebuilt from a run that only saved Markdown.

Scaling data is read from the per-core cache the scaling orchestrator leaves behind,
``benchmarks/_results/<name>/scaling_<C>core.json``.

Examples
--------
    # from the Markdown a normal run wrote out; writes to
    # benchmarks/results/figs_tables/CUFLynx_paper_tables.tex by default
    python benchmarks/create_CUFLynx_paper_tables.py --results result_*.md

    # from JSON, expanding every optimiser, and a compilable preview document
    python benchmarks/create_CUFLynx_paper_tables.py \
        --results-dir benchmarks/_results --all-methods --standalone

The emitted tables need ``booktabs``. ``--standalone`` wraps them in a minimal document so the
file compiles on its own for previewing.
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "benchmarks", "_results")
# Generated tables land here, next to the (gitignored) raw run output but tracked themselves, so
# the exact tables a paper cites are versioned alongside the code that produced them.
FIGS_TABLES_DIR = os.path.join(ROOT, "benchmarks", "results", "figs_tables")
DEFAULT_OUT = os.path.join(FIGS_TABLES_DIR, "CUFLynx_paper_tables.tex")

# Acronyms used in the table body, kept short so the Model column stays narrow. The captions
# expand whichever ones actually appear, so each table stays self-contained.
DISPLAY_NAMES = {
    "fitzhugh_nagumo": "FHN",
    "three_compartment": "3CVS",
    "goodwin": "GO",
    "teusink": "TG",
}

FULL_NAMES = {
    "fitzhugh_nagumo": "FitzHugh--Nagumo",
    "three_compartment": "3-compartment cardiovascular",
    "goodwin": "Goodwin oscillator",
    "teusink": "Teusink 2000 yeast glycolysis",
}


def acronym_legend(names):
    """'GO: Goodwin oscillator; TG: ...' for the models present, in the given order."""
    parts = [f"{DISPLAY_NAMES[n]}: {FULL_NAMES[n]}" for n in names
             if n in DISPLAY_NAMES and n in FULL_NAMES]
    return "; ".join(parts)


# ------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------

def _rows_from_dict(payload):
    """Normalise a ``benchmark_result_to_dict`` payload's rows to plain dicts."""
    rows = []
    for r in payload.get("rows", []):
        rows.append({
            "method": r.get("method"),
            "cost": r.get("cost"),
            "time_s": r.get("time_s"),
            "param_err": r.get("param_err"),
            "skipped_reason": r.get("skipped_reason"),
        })
    return rows


def load_json_result(path):
    """Load one ``--emit-json`` payload (or a bare result dict) into our internal form."""
    with open(path) as f:
        data = json.load(f)
    result = data.get("result", data)
    return {
        "name": result.get("name") or _name_from_path(path),
        "title": result.get("title", ""),
        "description": result.get("description", ""),
        "env_note": result.get("env_note", ""),
        "true_params": result.get("true_params"),
        "param_labels": result.get("param_labels"),
        "rows": _rows_from_dict(result),
        "source": path,
    }


_MD_SKIPPED = re.compile(r"_skipped\s*[—-]\s*(.*?)_")


def _md_cells(line):
    """Split a Markdown table row into its cells."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _num(text):
    """Parse a numeric table cell, returning None for blanks/placeholders."""
    text = text.strip()
    if not text or text in {"-", "--", "—"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_markdown_results(path):
    """Parse ``run_benchmarks.py --results-out`` Markdown back into result dicts.

    A file may hold several ``### Title`` sections (``--set all`` writes one per benchmark), so
    this returns a list.
    """
    with open(path) as f:
        text = f.read()

    results = []
    # Split on the '### ' section headers, keeping each header with its body.
    sections = re.split(r"(?m)^###\s+", text)[1:]
    for sec in sections:
        lines = sec.splitlines()
        title = lines[0].strip()
        rows, header, description, true_params, param_labels = [], None, "", None, None
        env_note = ""
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("|"):
                cells = _md_cells(stripped)
                if header is None:
                    header = [c.lower() for c in cells]
                    continue
                if all(set(c) <= set("-: ") for c in cells):
                    continue  # the |---|---| separator
                method = cells[0].strip().strip("`")
                skipped = _MD_SKIPPED.search(stripped)
                if skipped:
                    rows.append({"method": method, "cost": None, "time_s": None,
                                 "param_err": None,
                                 "skipped_reason": skipped.group(1).strip()})
                    continue
                by_name = dict(zip(header, cells))
                rows.append({
                    "method": method,
                    "cost": _num(by_name.get("best cost", "")),
                    "time_s": _num(by_name.get("time (s)", "")),
                    "param_err": _num(by_name.get("max param err", "")),
                    "skipped_reason": None,
                })
            elif stripped.startswith("True parameters:"):
                true_params, param_labels = _parse_true_params(stripped)
            elif stripped.startswith("*") and stripped.endswith(".*"):
                env_note = stripped.strip("*").rstrip(".")
            elif stripped and not description and header is None:
                description = stripped
        if rows:
            results.append({
                "name": _name_from_title(title) or _name_from_path(path),
                "title": title, "description": description, "env_note": env_note,
                "true_params": true_params, "param_labels": param_labels,
                "rows": rows, "source": path,
            })
    return results


def _parse_true_params(line):
    """'True parameters: a_i=72, b_i=2.' -> (['72','2'] as floats, ['a_i','b_i'])."""
    body = line.split(":", 1)[1].strip().rstrip(".")
    labels, values = [], []
    for part in body.split(","):
        if "=" not in part:
            continue
        lab, val = part.split("=", 1)
        try:
            values.append(float(val.strip()))
        except ValueError:
            continue
        labels.append(lab.strip())
    return (values or None), (labels or None)


def _name_from_title(title):
    """Map a harness title back to its benchmark id, so display names/lookups still work."""
    low = title.lower()
    for name in DISPLAY_NAMES:
        stem = name.replace("_", "")
        if stem in low.replace("-", "").replace(" ", "").replace("_", ""):
            return name
    return None


def _name_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    for prefix in ("result_", "results_", "benchmark_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def load_results(paths):
    """Load every given path (JSON or Markdown), de-duplicating by benchmark name.

    When the same benchmark appears more than once the later path wins, so a caller can layer a
    fresh result over an older one just by ordering the arguments.
    """
    by_name = {}
    for path in paths:
        try:
            if path.endswith(".json"):
                loaded = [load_json_result(path)]
            else:
                loaded = parse_markdown_results(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[tables] could not read {path}: {exc}", file=sys.stderr)
            continue
        for result in loaded:
            if not result["rows"]:
                continue
            by_name[result["name"]] = result
    return by_name


def load_scaling(name, results_dir=RESULTS_DIR):
    """Read ``scaling_<C>core.json`` for one benchmark into {cores: result-dict}."""
    jdir = os.path.join(results_dir, name)
    per_core = {}
    if not os.path.isdir(jdir):
        return per_core
    for fname in sorted(os.listdir(jdir)):
        m = re.fullmatch(r"scaling_(\d+)core\.json", fname)
        if not m:
            continue
        try:
            with open(os.path.join(jdir, fname)) as f:
                per_core[int(m.group(1))] = json.load(f)["result"]
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            continue
    return per_core


# ------------------------------------------------------------------------------------------
# Selection helpers
# ------------------------------------------------------------------------------------------

def live_rows(result):
    """Rows that actually ran (skipped optimisers carry no numbers)."""
    return [r for r in result["rows"]
            if not r["skipped_reason"] and r["time_s"] is not None]


def best_row(result):
    """The optimiser that represents this model: lowest max param err, then lowest time.

    Falls back to lowest cost when the benchmark reports no parameter error.
    """
    rows = live_rows(result)
    if not rows:
        return None
    if any(r["param_err"] is not None for r in rows):
        return min((r for r in rows if r["param_err"] is not None),
                   key=lambda r: (r["param_err"], r["time_s"]))
    return min(rows, key=lambda r: (r["cost"] if r["cost"] is not None else float("inf"),
                                    r["time_s"]))


def slowest_benchmark(by_name):
    """The benchmark with the largest total wall-clock across its optimisers."""
    totals = {name: sum(r["time_s"] for r in live_rows(res))
              for name, res in by_name.items() if live_rows(res)}
    if not totals:
        return None
    return max(totals, key=totals.get)


def display_name(result):
    name = DISPLAY_NAMES.get(result["name"])
    if name:
        return name
    title = result.get("title") or result["name"]
    return tex_escape(title.split(" (")[0])


# ------------------------------------------------------------------------------------------
# LaTeX formatting
# ------------------------------------------------------------------------------------------

_TEX_SPECIALS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                 "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                 "^": r"\textasciicircum{}"}


def tex_escape(text):
    return "".join(_TEX_SPECIALS.get(ch, ch) for ch in str(text))


def fmt_method(method):
    return r"\texttt{" + tex_escape(method) + "}"


def fmt_time(seconds):
    """Wall-clock in seconds, with enough precision to stay useful across 5 orders of magnitude."""
    if seconds is None:
        return "---"
    if seconds >= 100:
        return f"{seconds:.0f}"
    if seconds >= 10:
        return f"{seconds:.1f}"
    return f"{seconds:.2f}"


def fmt_err(err):
    if err is None:
        return "---"
    if err == 0:
        return "0"
    if err < 1e-4:
        return fmt_sci(err)
    return f"{err:.4f}"


def fmt_sci(value, digits=2):
    """Scientific notation as LaTeX math, e.g. 1.38\\times 10^{-6}."""
    if value is None:
        return "---"
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def fmt_cost(cost):
    return fmt_sci(cost)


def table_calibration(by_name, order, all_methods=False, label="tab:ca-calibration"):
    """Table 1 -- calibration time and max parameter error per model."""
    cols = "llrrr" if all_methods else "lrlrr"
    lines = [r"% ---------------------------------------------------------------------------",
             r"% Table 1: calibration cost, wall-clock and parameter accuracy per model.",
             r"% ---------------------------------------------------------------------------",
             r"\begin{table}[htbp]",
             r"  \centering",
             r"  \caption{Calibration performance of CUFLynx across the benchmark models. "
             r"Wall-clock time is the full parameter-identification run; the maximum parameter "
             r"error is the largest deviation of a recovered parameter from its known true "
             r"value (absolute, except 3CVS whose parameters span five orders of magnitude and "
             r"which reports relative error). "
             + (rf"Models --- {acronym_legend(order)}." if acronym_legend(order) else "")
             + r"}",
             rf"  \label{{{label}}}",
             rf"  \begin{{tabular}}{{{cols}}}",
             r"    \toprule"]

    if all_methods:
        lines.append(r"    Model & Optimiser & Best cost & Time (s) & Max param.\ err. \\")
        lines.append(r"    \midrule")
        for i, name in enumerate(order):
            result = by_name[name]
            rows = live_rows(result)
            if not rows:
                continue
            if i:
                lines.append(r"    \midrule")
            for j, row in enumerate(rows):
                model_cell = display_name(result) if j == 0 else ""
                lines.append(
                    f"    {model_cell} & {fmt_method(row['method'])} & {fmt_cost(row['cost'])} "
                    f"& {fmt_time(row['time_s'])} & {fmt_err(row['param_err'])} \\\\")
    else:
        lines.append(r"    Model & Params & Best optimiser & Time (s) & Max param.\ err. \\")
        lines.append(r"    \midrule")
        for name in order:
            result = by_name[name]
            row = best_row(result)
            if row is None:
                continue
            n_params = len(result["true_params"] or []) or "---"
            lines.append(
                f"    {display_name(result)} & {n_params} & {fmt_method(row['method'])} "
                f"& {fmt_time(row['time_s'])} & {fmt_err(row['param_err'])} \\\\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_scaling(result, per_core, label="tab:ca-scaling"):
    """Table 2 -- wall-clock per core count, with speedup, for one model."""
    cores = sorted(per_core)
    ref = min(cores)
    methods = [r["method"] for r in per_core[ref].get("rows", [])]

    # A speedup column needs at least two core counts to compare; with one (e.g. a scaling sweep
    # still in progress) it would read "speedup at 8 cores relative to 8" over a column of dashes.
    show_speedup = len(cores) > 1
    col_spec = "l" + "r" * len(cores) + ("r" if show_speedup else "")
    core_headers = " & ".join(rf"{c}" for c in cores)
    lines = [r"% ---------------------------------------------------------------------------",
             r"% Table 2: parallel scaling of the slowest benchmark model.",
             r"% ---------------------------------------------------------------------------",
             r"\begin{table}[htbp]",
             r"  \centering",
             r"  \caption{Parallel scaling of CUFLynx calibration on the "
             f"{display_name(result)} ({FULL_NAMES.get(result.get('name'), '')}) model, "
             r"the most expensive benchmark. "
             r"Entries are wall-clock seconds"
             + (rf"; the final column is the speedup at {max(cores)} cores relative to {ref}.}}"
                if show_speedup else ".}"),
             rf"  \label{{{label}}}",
             rf"  \begin{{tabular}}{{{col_spec}}}",
             r"    \toprule",
             rf"    & \multicolumn{{{len(cores)}}}{{c}}{{Wall-clock time (s) by core count}}"
             + (r" & \\" if show_speedup else r" \\"),
             rf"    \cmidrule(lr){{2-{len(cores) + 1}}}",
             rf"    Optimiser & {core_headers}" + (r" & Speedup \\" if show_speedup else r" \\"),
             r"    \midrule"]

    for method in methods:
        times = {}
        for c in cores:
            row = next((r for r in per_core[c].get("rows", []) if r["method"] == method), None)
            if row and not row.get("skipped_reason") and row.get("time_s") is not None:
                times[c] = row["time_s"]
        if not times:
            continue
        cells = [fmt_time(times.get(c)) for c in cores]
        row = f"    {fmt_method(method)} & " + " & ".join(cells)
        if show_speedup:
            lo, hi = min(times), max(times)
            speedup = f"{times[lo] / times[hi]:.2f}$\\times$" if hi != lo and times[hi] else "---"
            row += f" & {speedup}"
        lines.append(row + " \\\\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


PREAMBLE = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage[margin=2cm]{geometry}
\begin{document}
"""


def build_document(tables, standalone=False, provenance=()):
    parts = []
    if standalone:
        parts.append(PREAMBLE)
    parts.append("% Generated by benchmarks/create_CUFLynx_paper_tables.py -- do not hand-edit;\n"
                 "% re-run the script after regenerating the benchmark results.")
    if provenance:
        parts.append("% Sources:\n" + "\n".join(f"%   {p}" for p in provenance))
    parts.extend(tables)
    if standalone:
        parts.append(r"\end{document}")
    return "\n\n".join(parts) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="*", default=None,
                        help="benchmark result files (.json from --emit-json, or .md from "
                             "--results-out); globs are expanded")
    parser.add_argument("--results-dir", default=None,
                        help=f"directory to scan for result files (default: {RESULTS_DIR} plus "
                             f"the current directory) when --results is not given")
    parser.add_argument("--scaling-benchmark", default=None,
                        help="benchmark id for the scaling table (default: the slowest model)")
    parser.add_argument("--scaling-dir", default=RESULTS_DIR,
                        help="directory holding <benchmark>/scaling_<C>core.json")
    parser.add_argument("--all-methods", action="store_true",
                        help="list every optimiser per model instead of only the best one")
    parser.add_argument("--standalone", action="store_true",
                        help="wrap the tables in a minimal compilable LaTeX document")
    parser.add_argument("--order", default=None,
                        help="comma-separated benchmark ids fixing the row order of table 1")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output .tex path (default: benchmarks/results/figs_tables/"
                             "CUFLynx_paper_tables.tex)")
    args = parser.parse_args(argv)

    paths = []
    if args.results:
        for pattern in args.results:
            paths.extend(sorted(glob.glob(pattern)) or [pattern])
    else:
        search_dirs = [args.results_dir] if args.results_dir else [RESULTS_DIR, os.getcwd()]
        for d in search_dirs:
            paths.extend(sorted(glob.glob(os.path.join(d, "result*.md"))))
            paths.extend(sorted(glob.glob(os.path.join(d, "result*.json"))))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("[tables] no benchmark result files found; pass --results explicitly",
              file=sys.stderr)
        return 1

    by_name = load_results(paths)
    if not by_name:
        print("[tables] no usable results parsed from: " + ", ".join(paths), file=sys.stderr)
        return 1

    if args.order:
        wanted = [n.strip() for n in args.order.split(",") if n.strip()]
        order = [n for n in wanted if n in by_name]
        order += [n for n in by_name if n not in order]
    else:
        # Cheapest to most expensive: the natural narrative order for the paper.
        order = sorted(by_name, key=lambda n: sum(r["time_s"] for r in live_rows(by_name[n])))

    print(f"[tables] loaded {len(by_name)} benchmark(s): {', '.join(order)}")

    tables = [table_calibration(by_name, order, all_methods=args.all_methods)]

    scaling_name = args.scaling_benchmark or slowest_benchmark(by_name)
    per_core = load_scaling(scaling_name, args.scaling_dir) if scaling_name else {}
    if per_core:
        result = by_name.get(scaling_name) or {"name": scaling_name,
                                               "title": per_core[min(per_core)].get("title", "")}
        tables.append(table_scaling(result, per_core))
        print(f"[tables] scaling table for '{scaling_name}' from cores {sorted(per_core)}")
    else:
        where = os.path.join(args.scaling_dir, str(scaling_name), "scaling_<C>core.json")
        print(f"[tables] NO scaling data for '{scaling_name}' (looked for {where}); table 2 "
              f"omitted. Generate it with:\n"
              f"    ./benchmarks/run_benchmarks.sh --scaling --benchmark {scaling_name}",
              file=sys.stderr)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(build_document(tables, standalone=args.standalone, provenance=paths))
    print(f"[tables] wrote {args.out} ({len(tables)} table(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
