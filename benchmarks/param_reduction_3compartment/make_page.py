"""The result page of the parameter-reduction study, from the files run_study.py and compare.py
wrote: python make_page.py <out> <commit> <page.html>. It opens with the provenance diagram of
how the numbers were made, then the modes, the reductions, the fits and the traces."""
import csv
import html
import json
import math
import os
import sys

OUT = os.path.abspath(sys.argv[1])
COMMIT = sys.argv[2]
PAGE = sys.argv[3]
S = json.load(open(f'{OUT}/compare/summary.json'))
T = json.load(open(f'{OUT}/compare/traces.json'))
modes_rows = list(csv.DictReader(open(f'{OUT}/run/reduce_pca/pca_mode_sobol_indices.csv')))
simple_scores = {}
for row in csv.DictReader(open(S['screen']['sobol_indices'])):
    simple_scores[row['Parameter']] = max(float(v) for k, v in row.items() if k.startswith('ST_'))

P = S['parameters']
LABEL = {'global/q_lv_init': 'q_lv_init', 'aortic_root/C': 'aortic_root/C',
         'global/E_lv_A': 'E_lv_A', 'global/E_lv_B': 'E_lv_B', 'venous_svc/C': 'venous_svc/C',
         'aortic_root/R': 'aortic_root/R', 'systemic_T/R_T': 'systemic_T/R_T'}
WHAT = {'global/q_lv_init': 'initial LV volume (sets blood volume)', 'aortic_root/C': 'aortic compliance',
        'global/E_lv_A': 'LV active elastance', 'global/E_lv_B': 'LV passive elastance',
        'venous_svc/C': 'venous compliance', 'aortic_root/R': 'aortic root resistance',
        'systemic_T/R_T': 'systemic resistance'}
pca_sel = {r['parameter']: r for r in S['reductions']['pca']['selected']}
simple_sel = [r['parameter'] for r in S['reductions']['simple']['selected']]
modes = S['pca_report']['modes']
fits = S['fits']
e = html.escape


def fmt(x, d=3):
    if x == 0:
        return '0'
    if abs(x) >= 1e4 or abs(x) < 1e-2:
        m, ex = f'{x:.{d-1}e}'.split('e')
        return f'{m}×10<sup>{int(ex)}</sup>'
    return f'{x:.{d}g}'


# ---------- provenance diagram (HTML flow)
def box(title, body, mark, cls=''):
    return (f'<div class="pbox {cls}"><div class="ptop"><span class="pname">{title}</span>'
            f'<span class="mark mark-{mark.split()[0].replace(" ", "-")}">{mark}</span></div>'
            f'<div class="pbody">{body}</div></div>')


truth_list = ', '.join(f'<code>{e(LABEL[p])}</code> {fmt(S["truth"][i])}' for i, p in enumerate(P))
kept_pca = ', '.join(f'<code>{e(LABEL[p])}</code>' for p in pca_sel)
prov = f'''
<div class="flow">
 {box('Model source', '<code>resources/3compartment_module_array.csv</code> and <code>3compartment_parameters.json</code> at '
      f'commit <code>{COMMIT}</code>. Every parameter not listed below keeps its value from this file.', 'inherited')}
 <div class="arrow" aria-hidden="true"></div>
 {box('Truth (make_synthetic_obs.py)', f'Seven values chosen by us inside the calibration bounds: {truth_list}. '
      'Each differs from its parameters-file default, so a reduction that fixes a parameter at its default does not fix it at the truth.', 'set by hand')}
 <div class="arrow" aria-hidden="true"></div>
 {box('Synthetic obs_data', 'The six features of <code>3compartment_obs_data.json</code> (mean and max aortic flow, stroke volume, mean and pulse aortic pressure, mean LA pressure) simulated at the truth. '
      'Each std is 10% of its value (chosen by us); the weights (1, 1, 2, 5, 5, 5) are the shipped file’s.', 'set by hand')}
 <div class="arrow" aria-hidden="true"></div>
 {box('screen (sensitivity)', 'Sobol analysis of all seven parameters over their bounds: Saltelli sampling, N = 256, '
      f'256 × (2·7 + 2) = 4096 runs. Writes the indices, samples and outputs.', 'computed')}
 <div class="arrow" aria-hidden="true"></div>
 <div class="row2">
  {box('reduce_pca', f'<code>method: pca</code>, defaults (min_mode_std 1, variance_fraction 0.99, min_index 0.05). Found 4 informative modes and kept {kept_pca}.', 'computed')}
  {box('reduce_simple', '<code>method: sobol</code>, top n by max ST over the six outputs. '
       '<b>n = 4 was set by the driver</b> to the number reduce_pca kept, so both reductions keep the same number.', 'set by hand')}
 </div>
 <div class="arrow" aria-hidden="true"></div>
 <div class="row3">
  {box('fit_full_s0–s2', 'CMA-ES, 2000 calls, seeds 0, 1, 2. Calibrates all seven against the synthetic obs_data.', 'calibrated')}
  {box('fit_pca_s0–s2', 'Same settings. Calibrates the four reduce_pca kept; the other three stay at their parameters-file defaults.', 'calibrated')}
  {box('fit_simple_s0–s2', 'Same settings. Calibrates the four reduce_simple kept; the other three stay at their defaults.', 'calibrated')}
 </div>
 <div class="arrow" aria-hidden="true"></div>
 {box('compare.py', 'Simulates each fit (its calibrated values, defaults for the rest) to get the features, central-difference Fisher information '
      'of the calibrated parameters (log scale, step 0.1%), and four traces over the logged 2 s, compared with the truth’s.', 'computed')}
</div>'''


# ---------- modes chart: mode std (log) vs noise floor
def modes_chart():
    w, h, left, right, top, bottom = 560, 220, 56, 16, 14, 40
    lo, hi = -1.5, 1.5  # log10 std
    def y(v):
        return top + (hi - math.log10(max(v, 10 ** lo))) / (hi - lo) * (h - top - bottom)
    n = len(modes)
    bw = (w - left - right) / n
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Standard deviation of each PCA mode, in noise stds, on a log scale">']
    for t in (-1, 0, 1):
        yy = y(10 ** t)
        parts.append(f'<line x1="{left}" x2="{w-right}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>'
                     f'<text x="{left-8}" y="{yy+4:.1f}" class="tick" text-anchor="end">{10**t:g}</text>')
    y1 = y(1)
    parts.append(f'<line x1="{left}" x2="{w-right}" y1="{y1:.1f}" y2="{y1:.1f}" class="floor"/>'
                 f'<text x="{w-right}" y="{y1-6:.1f}" class="floorlab" text-anchor="end">noise floor (min_mode_std = 1)</text>')
    for i, m in enumerate(modes):
        x = left + i * bw + bw * 0.22
        yy = y(m['std'])
        cls = 'bar-on' if m['informative'] else 'bar-off'
        parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw*0.56:.1f}" height="{h-bottom-yy:.1f}" class="{cls}"/>'
                     f'<text x="{x+bw*0.28:.1f}" y="{yy-5:.1f}" class="val" text-anchor="middle">{m["std"]:.2g}</text>'
                     f'<text x="{x+bw*0.28:.1f}" y="{h-bottom+16:.1f}" class="tick" text-anchor="middle">mode {m["mode"]}</text>'
                     f'<text x="{x+bw*0.28:.1f}" y="{h-bottom+30:.1f}" class="tick dim" text-anchor="middle">{m["variance_share"]:.1%}</text>')
    parts.append(f'<text x="14" y="{(h-bottom+top)/2:.0f}" class="tick" transform="rotate(-90 14 {(h-bottom+top)/2:.0f})" text-anchor="middle">std (noise stds)</text>')
    parts.append('</svg>')
    return ''.join(parts)


# ---------- ST per mode heat table
def heat():
    head = ''.join(f'<th>mode {m["mode"]}{"" if m["informative"] else "<br><span class=dim>below noise</span>"}</th>' for m in modes)
    rows = []
    by = {r['Parameter']: r for r in modes_rows}
    for p in P:
        cells = []
        for m in modes:
            v = max(0.0, float(by[p][f'ST_mode_{m["mode"]}']))
            chosen = p in pca_sel and pca_sel[p]['mode'] == m['mode']
            cells.append(f'<td class="heat{" chosen" if chosen else ""}{" off" if not m["informative"] else ""}" style="--v:{min(v,1):.3f}">{v:.2f}</td>')
        rows.append(f'<tr><th scope="row"><code>{e(LABEL[p])}</code></th>{"".join(cells)}</tr>')
    return f'<div class="scroll"><table class="heatt"><thead><tr><th>ST of the mode score</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


# ---------- reductions table
def red_table():
    rows = []
    ranked = sorted(P, key=lambda p: -simple_scores[p])
    for p in ranked:
        i = P.index(p)
        ps = pca_sel.get(p)
        rows.append(
            f'<tr><th scope="row"><code>{e(LABEL[p])}</code><div class="dim small">{WHAT[p]}</div></th>'
            f'<td class="num">{fmt(S["truth"][i])}</td><td class="num">{fmt(S["defaults"][i])}</td>'
            f'<td class="num">{simple_scores[p]:.3f}</td>'
            f'<td>{"<span class=kept>kept</span>" if p in simple_sel else "<span class=drop>fixed</span>"}</td>'
            f'<td>{("<span class=kept>kept</span> mode " + str(ps["mode"])) if ps else "<span class=drop>fixed</span>"}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Parameter</th><th class="num">Truth</th><th class="num">Default</th>'
            '<th class="num">Max ST over outputs</th><th>Simple (n = 4)</th><th>PCA</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>')


# ---------- parameter error dot plot
COLORS = {'full': 'var(--full)', 'pca': 'var(--pca)', 'simple': 'var(--simple)'}


def err_chart():
    w, row_h, left, right, top = 620, 34, 120, 20, 30
    h = top + row_h * len(P) + 30
    lo, hi = -0.5, 0.5
    def x(v):
        v = max(lo, min(hi, v))
        return left + (v - lo) / (hi - lo) * (w - left - right)
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="log10 of fitted over true value for each parameter, each fit">']
    for t in (-0.4, -0.2, 0, 0.2, 0.4):
        parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top-6}" y2="{h-26}" class="{"zero" if t == 0 else "grid"}"/>'
                     f'<text x="{x(t):.1f}" y="{h-10}" class="tick" text-anchor="middle">{t:+.1f}</text>')
    parts.append(f'<text x="{x(0):.1f}" y="{top-14}" class="tick" text-anchor="middle">log10(value / truth)</text>')
    offs = {'full': -9, 'pca': 0, 'simple': 9}
    for r, p in enumerate(P):
        yc = top + r * row_h + row_h / 2
        parts.append(f'<text x="{left-10}" y="{yc+4:.1f}" class="tick" text-anchor="end">{e(LABEL[p])}</text>')
        if r % 2 == 0:
            parts.append(f'<rect x="{left}" y="{yc-row_h/2:.1f}" width="{w-left-right}" height="{row_h}" class="band"/>')
        for f in fits:
            v = f['log10_error'][p]
            cal = p in f['calibrated']
            yy = yc + offs[f['method']] + (f['seed'] - 1) * 2.2
            col = COLORS[f['method']]
            parts.append(f'<circle cx="{x(v):.1f}" cy="{yy:.1f}" r="3.6" style="fill:{col if cal else "var(--bg)"};stroke:{col}" stroke-width="1.6">'
                         f'<title>{f["step"]}: {LABEL[p]} {v:+.3f} ({"calibrated" if cal else "fixed at default"})</title></circle>')
    parts.append('</svg>')
    return ''.join(parts)


# ---------- fits table
def fits_table():
    rows = []
    for f in fits:
        fi = f['fisher']
        cond = fi['condition_number']
        cond_s = '∞ (singular)' if not math.isfinite(cond) else fmt(cond)
        rows.append(
            f'<tr><th scope="row"><span class="dot" style="background:{COLORS[f["method"]]}"></span>{f["step"]}</th>'
            f'<td class="num">{len(f["calibrated"])}</td><td class="num">{f["chi2"]:.3f}</td>'
            f'<td class="num">{cond_s}</td>'
            f'<td class="num">{fi["max_abs_correlation"]:.3f}</td><td class="small">{" / ".join(LABEL[n] for n in fi["most_correlated"])}</td>'
            f'<td class="num">{f["mean_trace_error"]:.3f}</td><td class="num">{f["trace_error"]["venous_svc/u"]:.3f}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Fit</th><th class="num">Calibrated</th><th class="num">χ² = Σ w z²</th>'
            '<th class="num">FIM condition</th><th class="num">Max |corr|</th><th>Most correlated pair</th>'
            '<th class="num">Mean trace error</th><th class="num">Venous pressure error</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>')


# ---------- traces small multiples
TR = [('heart/u_lv', 'LV pressure', 'mmHg', 1 / 133.322), ('heart/q_lv', 'LV volume', 'mL', 1e6),
      ('aortic_root/v', 'Aortic root flow', 'mL/s', 1e6), ('venous_svc/u', 'Venous pressure (not observed)', 'mmHg', 1 / 133.322)]


def trace_chart(name, label, unit, k):
    w, h, left, right, top, bottom = 300, 170, 44, 8, 22, 26
    t = T['time']
    series = [('truth', 'var(--fg)', 2.2, ''), ('full', 'var(--full)', 1.6, ''), ('pca', 'var(--pca)', 1.6, '5 3')]
    vals = [v * k for s, *_ in series for v in T[s][name]]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    def X(v):
        return left + (v - t[0]) / (t[-1] - t[0]) * (w - left - right)
    def Y(v):
        return top + (hi - v) / (hi - lo) * (h - top - bottom)
    parts = [f'<figure class="mini"><svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{e(label)}: truth, full fit seed 0 and PCA fit seed 0">']
    for tv in (lo + pad, (lo + hi) / 2, hi - pad):
        parts.append(f'<line x1="{left}" x2="{w-right}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" class="grid"/>'
                     f'<text x="{left-5}" y="{Y(tv)+4:.1f}" class="tick" text-anchor="end">{tv:.3g}</text>')
    for tt in (0, 1, 2):
        parts.append(f'<text x="{X(tt):.1f}" y="{h-8}" class="tick" text-anchor="middle">{tt} s</text>')
    for s, col, sw, dash in series:
        pts = ' '.join(f'{X(a):.1f},{Y(b*k):.1f}' for a, b in zip(t, T[s][name]))
        parts.append(f'<polyline points="{pts}" fill="none" style="stroke:{col}" stroke-width="{sw}" stroke-dasharray="{dash}"/>')
    parts.append(f'<text x="{left}" y="13" class="minititle">{e(label)} <tspan class="dim">({unit})</tspan></text></svg></figure>')
    return ''.join(parts)


ft = S['fisher_at_truth_all_seven']
eig = ft['eigenvalues']
full_err_q = [f['log10_error']['global/q_lv_init'] for f in fits if f['method'] == 'full']
pca_fits = [f for f in fits if f['method'] == 'pca']
full_fits = [f for f in fits if f['method'] == 'full']
pca_conds = [f['fisher']['condition_number'] for f in pca_fits]
venC = [f['log10_error']['venous_svc/C'] for f in pca_fits]

page = f'''<title>PCA Parameter Reduction Study</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,560&display=swap">
<style>
/* Layout: one reading column; the provenance flow and wide tables break out to a wider measure. */
:root {{
  --bg: #f6f7f8; --surface: #ffffff; --fg: #1a2026; --muted: #5d6873; --rule: #d7dce1;
  --accent: #0b6e75; --full: #b4532a; --pca: #0b7d84; --simple: #6b4fb3; --warn: #a3620a;
  --heat: 11 110 117;
  --display: "Fraunces", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12171b; --surface: #1a2127; --fg: #e3e8ec; --muted: #94a1ad; --rule: #2e3840;
  --accent: #5cc2c8; --full: #e08a5f; --pca: #4fc4cb; --simple: #a68cf0; --warn: #e0a24a;
  --heat: 79 196 203; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #12171b; --surface: #1a2127; --fg: #e3e8ec; --muted: #94a1ad; --rule: #2e3840;
  --accent: #5cc2c8; --full: #e08a5f; --pca: #4fc4cb; --simple: #a68cf0; --warn: #e0a24a;
  --heat: 79 196 203; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }}
main {{ max-width: 1000px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 34px; }}
section {{ display: grid; gap: 14px; min-width: 0; }}
.prose {{ max-width: 68ch; }}
h1 {{ font: 560 clamp(28px, 4.5vw, 40px)/1.1 var(--display); margin: 0; text-wrap: balance; letter-spacing: -0.01em; }}
h2 {{ font: 560 23px/1.2 var(--display); margin: 0; text-wrap: balance; }}
h3 {{ font: 600 15px/1.3 var(--body); margin: 0; }}
p {{ margin: 0; }}
.eyebrow {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
.meta {{ font: 13px/1.5 var(--mono); color: var(--muted); display: flex; flex-wrap: wrap; gap: 4px 18px; }}
code {{ font: 0.88em var(--mono); }}
.dim {{ color: var(--muted); }} .small {{ font-size: 13px; }}
.flow {{ display: grid; gap: 0; }}
.row2, .row3 {{ display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
.pbox {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 13px; display: grid; gap: 4px; min-width: 0; }}
.ptop {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
.pname {{ font: 600 14px var(--mono); }}
.pbody {{ font-size: 13.5px; color: var(--fg); }}
.arrow {{ height: 22px; position: relative; }}
.arrow::before {{ content: ""; position: absolute; left: 24px; top: 2px; bottom: 6px; border-left: 2px solid var(--rule); }}
.arrow::after {{ content: ""; position: absolute; left: 19px; bottom: 1px; border: 6px solid transparent; border-top: 7px solid var(--rule); border-bottom: 0; }}
.mark {{ font: 500 11px/1 var(--mono); text-transform: uppercase; letter-spacing: .06em; padding: 4px 7px; border-radius: 99px; border: 1px solid currentColor; white-space: nowrap; }}
.mark-inherited {{ color: var(--muted); }} .mark-set {{ color: var(--warn); }}
.mark-calibrated {{ color: var(--pca); }} .mark-computed {{ color: var(--accent); }}
.findings {{ display: grid; gap: 10px; margin: 0; padding: 0; list-style: none; }}
.findings li {{ padding-left: 18px; position: relative; max-width: 72ch; }}
.findings li::before {{ content: ""; position: absolute; left: 0; top: .62em; width: 8px; height: 8px; border-radius: 2px; background: var(--accent); }}
.scroll {{ overflow-x: auto; min-width: 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
thead th {{ font: 500 12px var(--mono); color: var(--muted); letter-spacing: .03em; }}
.num {{ text-align: right; white-space: nowrap; }}
.kept {{ font: 500 12px var(--mono); color: var(--pca); }} .drop {{ font: 12px var(--mono); color: var(--muted); }}
.heatt td {{ text-align: center; background: rgb(var(--heat) / calc(var(--v) * .55)); font-family: var(--mono); font-size: 12.5px; }}
.heatt td.off {{ opacity: .55; }}
.heatt td.chosen {{ outline: 2px solid var(--fg); outline-offset: -2px; font-weight: 600; }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }}
.chart {{ width: 100%; height: auto; display: block; overflow: visible; }}
.chart text {{ fill: var(--muted); font: 11px var(--mono); }}
.chart .val {{ fill: var(--fg); }} .chart .minititle {{ fill: var(--fg); font: 600 12px var(--body); }}
.chart .floorlab {{ fill: var(--warn); }}
.chart .grid {{ stroke: var(--rule); stroke-width: 1; }} .chart .zero {{ stroke: var(--muted); stroke-width: 1.2; }}
.chart .floor {{ stroke: var(--warn); stroke-width: 1.4; stroke-dasharray: 5 4; }}
.chart .bar-on {{ fill: var(--pca); }} .chart .bar-off {{ fill: var(--rule); }}
.chart .band {{ fill: var(--surface); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.swatch {{ width: 18px; height: 0; border-top: 2.4px solid; display: inline-block; }}
.minis {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
.mini {{ margin: 0; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; min-width: 0; }}
.callout {{ border-left: 3px solid var(--warn); padding: 4px 0 4px 14px; max-width: 72ch; }}
</style>
<main>
<header style="display:grid;gap:10px">
  <div class="eyebrow">libcuflynx · param_reduction_options · PR #639</div>
  <h1>PCA parameter reduction on 3compartment</h1>
  <p class="prose">Seven parameters, six synthetic features. We compare calibrating all seven with calibrating the four kept by the new <code>method: pca</code> reduction and the four kept by top-n Sobol ranking.</p>
  <div class="meta"><span>run: {e(OUT)}/run</span><span>libcuflynx commit {COMMIT} (feat/pca-param-reduction)</span><span>2026-11-13</span></div>
</header>

<section>
  <h2>How these numbers were made</h2>
  {prov}
</section>

<section>
  <h2>What the study shows</h2>
  <ul class="findings">
   <li><b>The data resolve four directions.</b> PCA of the 4096 Sobol outputs, in units of the obs_data noise, gives four modes whose std exceeds one noise std (22, 15, 8.2 and 4.1) and two below it (0.85 and 0.13). The reduction keeps one parameter per mode: {kept_pca}.</li>
   <li><b>Calibrating all seven is not identifiable.</b> At the truth the 7×7 Fisher information has a zero eigenvalue and one of {eig[1]:.2f} (both below 1). The three full fits fit the data (χ² 0.01–0.93) but end in different places: <code>q_lv_init</code> ranges {min(full_err_q):+.2f} to {max(full_err_q):+.2f} decades. Their FIM condition numbers are 10<sup>16</sup> or singular, with <code>q_lv_init</code>, <code>venous_svc/C</code>, <code>aortic_root/R</code> and <code>E_lv_A</code> correlated near ±1.</li>
   <li><b>The reduced fits are identifiable and repeatable.</b> All three seeds reach the same values, with condition numbers {min(pca_conds):.0f}–{max(pca_conds):.0f}. <code>aortic_root/C</code>, <code>E_lv_B</code> and <code>systemic_T/R_T</code> are within 0.03 decades of the truth.</li>
   <li><b>Fixing a parameter at a wrong default moves its partner.</b> <code>q_lv_init</code> is fixed at its default, 2× the truth. <code>venous_svc/C</code>, which shares its mode, absorbs that: it fits {min(venC):+.2f} to {max(venC):+.2f} decades off. The features still fit (χ² ≈ 0.4) because blood volume over venous compliance is what the data see.</li>
   <li><b>Here the top-n ranking keeps the same four parameters,</b> so the pca and simple fits are identical. What the PCA supplies on this model is the count. Without it, top-n has no natural stopping point: every one of the seven scores at least {min(simple_scores.values()):.2f} max-ST, so a 0.05 threshold keeps all seven, the unidentifiable set.</li>
  </ul>
</section>

<section>
  <h2>Modes of the Sobol outputs</h2>
  <p class="prose">Each output is scaled by its std over the square root of its weight, so 1 on this axis is one noise std. The share under each mode is its fraction of the scaled variance.</p>
  {modes_chart()}
  <p class="prose">Sobol total index of each parameter for each mode’s score. Outlined cells are the parameters kept, one per informative mode, largest mode first.</p>
  {heat()}
</section>

<section>
  <h2>The two reductions</h2>
  {red_table()}
</section>

<section>
  <h2>Where each fit ends</h2>
  <div class="legend"><span><span class="dot" style="background:var(--full)"></span>full (7 calibrated)</span><span><span class="dot" style="background:var(--pca)"></span>pca</span><span><span class="dot" style="background:var(--simple)"></span>simple</span><span>filled: calibrated · hollow: fixed at its default · three dots per method are seeds 0–2</span></div>
  <div class="scroll">{err_chart()}</div>
  {fits_table()}
  <p class="prose small dim">χ² sums w·z² over the six features (z in obs_data stds). FIM is the Gauss–Newton Fisher information of the calibrated parameters in log scale. Trace error is the RMSE over the 2 s window divided by the truth trace’s range, for the four traces below; the venous pressure is not in the data.</p>
</section>

<section>
  <h2>Traces at seed 0</h2>
  <div class="legend"><span><span class="swatch" style="border-color:var(--fg)"></span>truth</span><span><span class="swatch" style="border-color:var(--full)"></span>full, seed 0</span><span><span class="swatch" style="border-color:var(--pca);border-top-style:dashed"></span>pca, seed 0 (simple is identical)</span></div>
  <div class="minis">{"".join(trace_chart(*a) for a in TR)}</div>
</section>

<section>
  <h2>Limits of this study</h2>
  <div class="callout prose"><p>On this model both reductions keep the same parameters, so the study shows that the PCA count and choice are sound, not that PCA picks better parameters than top-n here. The unit test <code>test_pca_keeps_one_of_two_parameters_that_move_the_data_together</code> covers the case where they differ: an effect below the noise, which top-n keeps and PCA does not.</p></div>
  <ul class="findings">
   <li>The fits use CMA-ES rather than the GA in the plan: the GA’s population of 744 would leave a 2000-call budget about 3 generations.</li>
   <li>The Fisher information is local (at each fit); the Sobol modes are global over the prior box. They agree here on the count of well-resolved directions.</li>
   <li>The fixed parameters keep the model’s defaults, which are wrong by design. A reduced calibration is only as good as the values it fixes.</li>
  </ul>
</section>
</main>
'''
open(PAGE, 'w').write(page)
print('wrote', PAGE, len(page))
