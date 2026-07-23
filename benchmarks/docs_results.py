"""Benchmark result data types, Markdown formatting, and docs splicing.

Kept free of the heavy runtime imports (solver backends, MPI) so it can be used and tested on
its own. The parameter-identification docs contain a marker-delimited region;
``update_docs_section`` replaces everything between the markers with freshly generated tables
so the benchmark runner (and CI) can keep the published results current without hand-editing.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkRow:
    method: str
    cost: Optional[float] = None
    time_s: Optional[float] = None
    param_err: Optional[float] = None
    params: Optional[list] = None
    skipped_reason: Optional[str] = None


@dataclass
class BenchmarkResult:
    name: str                       # short id, e.g. 'fitzhugh_nagumo'
    title: str                      # human title for the docs
    description: str                # one/two line summary
    env_note: str = ""              # e.g. '1 MPI rank; 2000 cost evals; 8 starts'
    rows: list = field(default_factory=list)
    true_params: Optional[list] = None
    param_labels: Optional[list] = None

    def cost(self, method):
        for r in self.rows:
            if r.method == method:
                return r.cost
        return None


@dataclass
class ScalingRow:
    """One optimiser's result in a core-scaling table: a single best cost / param error (the same
    work is run at every core count, so these are core-independent) plus a wall-clock time per
    core count."""
    method: str
    cost: Optional[float] = None
    param_err: Optional[float] = None
    times_by_core: dict = field(default_factory=dict)   # {n_cores: seconds}
    skipped_reason: Optional[str] = None


@dataclass
class ScalingBenchmarkResult:
    name: str
    title: str
    description: str
    cores: list                     # e.g. [1, 2, 4, 8, 16]
    env_note: str = ""
    rows: list = field(default_factory=list)
    true_params: Optional[list] = None
    param_labels: Optional[list] = None


def benchmark_result_to_dict(result):
    """Serialise a ``BenchmarkResult`` to a plain dict (JSON-safe), so a per-core child process
    can hand its results back to the scaling orchestrator."""
    return {
        'name': result.name,
        'title': result.title,
        'description': result.description,
        'env_note': result.env_note,
        'true_params': result.true_params,
        'param_labels': result.param_labels,
        'rows': [{'method': r.method, 'cost': r.cost, 'time_s': r.time_s,
                  'param_err': r.param_err, 'skipped_reason': r.skipped_reason}
                 for r in result.rows],
    }


DOCS_START_MARKER = "<!-- BENCHMARK_RESULTS_START -->"
DOCS_END_MARKER = "<!-- BENCHMARK_RESULTS_END -->"


def _fmt(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) else str(x)


def result_to_markdown(result):
    """Render one BenchmarkResult as a Markdown table with surrounding prose."""
    lines = [f"### {result.title}", "", result.description, ""]
    if result.env_note:
        lines.append(f"*{result.env_note}.*")
        lines.append("")

    show_err = any(r.param_err is not None for r in result.rows)
    header = ["method", "best cost", "time (s)"]
    if show_err:
        header.append("max param err")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in result.rows:
        if r.skipped_reason is not None:
            note = f"_skipped — {r.skipped_reason}_"
            span = ["`" + r.method + "`", note, ""]
            if show_err:
                span.append("")
            lines.append("| " + " | ".join(span) + " |")
            continue
        cells = ["`" + r.method + "`", _fmt(r.cost, ".4e"), _fmt(r.time_s, ".1f")]
        if show_err:
            cells.append(_fmt(r.param_err, ".4f"))
        lines.append("| " + " | ".join(cells) + " |")

    if result.true_params is not None:
        labels = result.param_labels or [f"p{i}" for i in range(len(result.true_params))]
        true_str = ", ".join(f"{lab}={val:g}" for lab, val in zip(labels, result.true_params))
        lines += ["", f"True parameters: {true_str}."]
    lines.append("")
    return "\n".join(lines)


def results_to_markdown(results, generated_note=None):
    """Render all results into a single Markdown block for the docs region."""
    blocks = []
    if generated_note:
        blocks.append(f"*{generated_note}*\n")
    blocks += [result_to_markdown(r) for r in results]
    return "\n".join(blocks).rstrip() + "\n"


def _core_header(n):
    return f"{n} core{'s' if n != 1 else ''} (s)"


def scaling_result_to_markdown(result):
    """Render one ``ScalingBenchmarkResult`` as a Markdown table with a wall-clock column per
    core count (the parallel-scaling view)."""
    lines = [f"### {result.title}", "", result.description, ""]
    if result.env_note:
        lines += [f"*{result.env_note}.*", ""]

    show_err = any(r.param_err is not None for r in result.rows)
    core_headers = [_core_header(c) for c in result.cores]
    header = ["method", "best cost"] + (["max param err"] if show_err else []) + core_headers
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in result.rows:
        if r.skipped_reason is not None:
            span = ["`" + r.method + "`", f"_skipped — {r.skipped_reason}_"]
            span += [""] if show_err else []
            span += [""] * len(result.cores)
            lines.append("| " + " | ".join(span) + " |")
            continue
        cells = ["`" + r.method + "`", _fmt(r.cost, ".4e")]
        if show_err:
            cells.append(_fmt(r.param_err, ".4f"))
        for c in result.cores:
            t = r.times_by_core.get(c)
            cells.append(_fmt(t, ".1f") if t is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    if result.true_params is not None:
        labels = result.param_labels or [f"p{i}" for i in range(len(result.true_params))]
        true_str = ", ".join(f"{lab}={val:g}" for lab, val in zip(labels, result.true_params))
        lines += ["", f"True parameters: {true_str}."]
    lines.append("")
    return "\n".join(lines)


def scaling_results_to_markdown(results, generated_note=None):
    """Render all scaling results into a single Markdown block for the docs region."""
    blocks = []
    if generated_note:
        blocks.append(f"*{generated_note}*\n")
    blocks += [scaling_result_to_markdown(r) for r in results]
    return "\n".join(blocks).rstrip() + "\n"


def update_docs_section(markdown, docs_path,
                        start_marker=DOCS_START_MARKER, end_marker=DOCS_END_MARKER):
    """Replace the marker-delimited region of ``docs_path`` with ``markdown``.

    Returns True if the file changed. Raises if the markers are missing.
    """
    with open(docs_path, "r") as f:
        content = f.read()
    if start_marker not in content or end_marker not in content:
        raise ValueError(
            f"Benchmark markers not found in {docs_path}; expected {start_marker!r} and "
            f"{end_marker!r}.")
    pre, rest = content.split(start_marker, 1)
    _, post = rest.split(end_marker, 1)
    new_content = f"{pre}{start_marker}\n{markdown}{end_marker}{post}"
    if new_content == content:
        return False
    with open(docs_path, "w") as f:
        f.write(new_content)
    return True
