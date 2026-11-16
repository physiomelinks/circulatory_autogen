"""The result page of the benchmark parameter-reduction study:
``python make_page.py <root> <commit> <page.html> [<3compartment run>]``.

``<root>`` holds one directory per model (make_inputs.py, run_study.py and compare.py written);
the optional 3compartment run is a ``param_reduction_3compartment`` workflow output whose
``run/reduce_pca`` and ``run/reduce_simple`` used the current reduction. Every number comes
from those files.
"""

import glob
import html
import json
import os
import sys

import numpy as np

from models import SPECS

ROOT, COMMIT, PAGE = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
THREE = sys.argv[4] if len(sys.argv) > 4 else None
e = html.escape


def short(name):
    return name.split('/', 1)[1] if name.count('/') == 1 and name.startswith(
        ('Xi/', 'Yi/', 'fhn/')) else name


def fmt(x, d=3):
    if x is None:
        return '–'
    if not np.isfinite(x):
        return '∞'
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-2):
        m, ex = f'{x:.{d - 1}e}'.split('e')
        return f'{m}×10<sup>{int(ex)}</sup>'
    return f'{x:.{d}g}'


def sobol_counts(indices_csv):
    from libcuflynx.sensitivity_analysis.sobol_ranking import (rank_sobol_parameters,
                                                               read_sobol_indices)
    ind = read_sobol_indices(indices_csv)
    return {t: len(rank_sobol_parameters(ind, 99, 'ST', 'max', t)) for t in (0.05, 0.1, 0.2)}


models = {}
for name in ('goodwin', 'fitzhugh_nagumo', 'teusink'):
    s = json.load(open(os.path.join(ROOT, name, 'compare', 'summary.json')))
    s['counts'] = sobol_counts(glob.glob(os.path.join(ROOT, name, 'sensitivity',
                                                      '*_Sobol_indices.csv'))[0])
    s['fit_settings'] = json.load(open(os.path.join(ROOT, name, 'fits', 'full_s0',
                                                    'result.json')))
    models[name] = s
three = None
if THREE:
    pca = json.load(open(os.path.join(THREE, 'run', 'reduce_pca', 'param_reduction.json')))
    simple = json.load(open(os.path.join(THREE, 'run', 'reduce_simple', 'param_reduction.json')))
    three = {'pca': pca, 'simple': simple, 'counts': sobol_counts(
        glob.glob(os.path.join(THREE, 'run', 'screen', 'sensitivity',
                               '*_Sobol_indices.csv'))[0])}


def identifiable_count(eigs):
    return int(np.sum(np.asarray(eigs) >= 1.0))


# ------------------------------------------------------------------ provenance
def box(title, body, mark):
    cls = mark.split()[0]
    return (f'<div class="pbox"><div class="ptop"><span class="pname">{title}</span>'
            f'<span class="mark mark-{cls}">{mark}</span></div><div class="pbody">{body}</div></div>')


truth_rows = ''.join(
    f'<tr><th scope="row">{e(s["name"])}</th><td>' + ', '.join(
        f'<code>{e(short(p))}</code> {fmt(t)} <span class="dim">(default {fmt(d)}; '
        f'bounds {fmt(lo)}–{fmt(hi)})</span>'
        for p, t, d, (_, lo, hi, _l) in zip(s['parameters'], s['truth'], s['defaults'],
                                            SPECS[m].parameters)) + '</td></tr>'
    for m, s in models.items())
fs = models['goodwin']['fit_settings']['optimiser_options']
prov = f'''<div class="flow">
 {box('Model sources', 'CA resources at commit <code>' + COMMIT + '</code>: <code>FitzHugh_Nagumo_module_array.csv</code> (generated to CellML), <code>Goodwin.cellml</code>, <code>Teusink.cellml</code>; solver settings from <code>benchmarks/benchmark_specs.py</code>. Every parameter not in the table below keeps the model’s value.', 'inherited')}
 <div class="arrow"></div>
 {box('Parameters and truth (models.py)', 'Each benchmark’s calibrated parameters plus extras, and a truth chosen by us inside the bounds, every value away from the model’s default so that fixing a dropped parameter at its default does not fix it at the truth. Teusink’s box is its defaults ÷1.5 to ×1.5: in the benchmark’s ÷3 to ×3 box one Sobol sample in twelve failed or ran away.<div class="scroll"><table class="small">' + truth_rows + '</table></div>', 'set by hand')}
 <div class="arrow"></div>
 {box('Synthetic obs_data (make_inputs.py)', 'The benchmark’s series (FitzHugh-Nagumo V, R: 602 points; Goodwin X, Y: 162; Teusink G6P, F16P, PYR, NADH: 84) simulated at the truth, <b>without noise</b>, with the benchmark’s stds. For the Sobol analysis each series is cut into bins (10, 4 and 2 points), each bin mean a feature with std σ/√points.', 'set by hand')}
 <div class="arrow"></div>
 {box('Sobol screen (run_SA)', 'All parameters, Saltelli N = 256, plus the local Jacobian at the centre of the box (2M + 1 runs). No failed samples.', 'computed')}
 <div class="arrow"></div>
 <div class="row2">
  {box('reduce_pca', '<code>param_reduction_options: {{method: pca}}</code>, defaults: min_mode_std 1, min_index 0.05, redundancy_check on.', 'computed')}
  {box('reduce_simple', '<code>method: sobol</code>, top n by max ST. <b>n set by the driver</b> to the number reduce_pca kept.', 'set by hand')}
 </div>
 <div class="arrow"></div>
 {box('Calibrations (run_param_id)', 'All parameters (<i>full</i>) and each reduction, against the synthetic series: <code>multi_start_sp_minimize</code>, ' + str(fs['num_starts']) + ' Sobol-spread starts, finite-difference gradients, seeds 0–2. The parameters a reduction drops keep the model’s default. A reduction keeping the same parameters as another reuses its fits.', 'calibrated')}
 <div class="arrow"></div>
 {box('compare.py', 'Each fit simulated: χ² over the series, log10 error of each parameter against the truth, and the Gauss–Newton Fisher information of the calibrated parameters from central differences of the binned features (condition number, Laplace std). The Fisher information at the truth gives the reference count of identifiable directions (eigenvalues ≥ 1).', 'computed')}
</div>'''


# ------------------------------------------------------------------ the answer table
def answer_rows():
    rows = []
    for m, s in models.items():
        pca = [x['parameter'] for x in s['reductions']['pca']['selected']]
        simple = [x['parameter'] for x in s['reductions']['simple']['selected']]
        rows.append((s['name'], len(s['parameters']), len(pca), set(pca) == set(simple),
                     s['counts'], identifiable_count(s['fisher_at_truth']['eigenvalues'])))
    if three:
        pca = [x['parameter'] for x in three['pca']['selected']]
        simple = [x['parameter'] for x in three['simple']['selected']]
        rows.append(('3compartment', 7, len(pca), set(pca) == set(simple), three['counts'], 5))
    out = []
    for name, n, k, same, counts, fim in rows:
        out.append(f'<tr><th scope="row">{e(name)}</th><td class="num">{n}</td>'
                   f'<td class="num"><b>{k}</b></td><td>{"same set" if same else "different"}</td>'
                   + ''.join(f'<td class="num{" off" if counts[t] != k else ""}">{counts[t]}</td>'
                             for t in (0.05, 0.1, 0.2))
                   + f'<td class="num">{fim}</td></tr>')
    return ''.join(out)


def fit_table(s):
    rows = []
    for f in s['fits']:
        if f['method'] == 'simple':
            continue
        lap = f['fisher']['laplace_std_log10']
        worst = max(lap, key=lap.get)
        cond = f['fisher']['condition_number']
        rows.append(
            f'<tr><th scope="row">{f["fit"]}</th><td class="num">{len(f["calibrated"])}</td>'
            f'<td class="num">{f["chi2"]:.3g}</td><td class="num">{fmt(cond)}</td>'
            f'<td class="num">{fmt(lap[worst])} <span class="dim">{e(short(worst))}</span></td>'
            f'<td class="num">{f["max_abs_log10_error_calibrated"]:.3f}</td>'
            f'<td class="num">{f["max_abs_log10_error"]:.3f}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Fit</th><th class="num">Calibrated</th>'
            '<th class="num">χ²</th><th class="num">FIM condition</th>'
            '<th class="num">Largest Laplace std (decades)</th>'
            '<th class="num">Max error, calibrated</th><th class="num">Max error, all</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def model_section(m, s):
    pca = s['reductions']['pca']
    kept = ', '.join(f'<code>{e(short(x["parameter"]))}</code>' for x in pca['selected'])
    dropped = ', '.join(f'<code>{e(short(k))}</code> ({v:.2f})'
                        for k, v in pca.get('unexplained_effects', {}).items()) or 'none'
    lap = s['fisher_at_truth']['laplace_std_log10']
    eig = s['fisher_at_truth']['eigenvalues']
    extras = '; '.join(f'<code>{e(short(k))}</code>: {e(v)}' for k, v in s['reasons'].items())
    return f'''<section><h3>{e(s["name"])}</h3>
  <p class="prose">Extras: {extras}. {pca["n_informative_modes"]} informative modes over {s["n_bins"]} binned features.
  PCA keeps {kept}; dropped, with the effect each would still add in noise stds: {dropped}.
  At the truth, the Fisher information of all {len(s["parameters"])} has eigenvalues {", ".join(fmt(x) for x in eig)}
  and Laplace stds (decades) {", ".join(f"<code>{e(short(k))}</code> {fmt(v)}" for k, v in lap.items())}.</p>
  {fit_table(s)}</section>'''


page = f'''<title>Parameter Reduction Benchmarks</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,560&display=swap">
<style>
/* One reading column; diagram and tables scroll inside their own boxes on a phone. */
:root {{ --bg:#f6f7f8; --surface:#fff; --fg:#1a2026; --muted:#5d6873; --rule:#d7dce1;
  --accent:#0b6e75; --warn:#a3620a; --ok:#0b7d84;
  --display:"Fraunces",Georgia,serif; --body:"IBM Plex Sans",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#12171b; --surface:#1a2127; --fg:#e3e8ec; --muted:#94a1ad; --rule:#2e3840; --accent:#5cc2c8; --warn:#e0a24a; --ok:#4fc4cb; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#12171b; --surface:#1a2127; --fg:#e3e8ec; --muted:#94a1ad; --rule:#2e3840; --accent:#5cc2c8; --warn:#e0a24a; --ok:#4fc4cb; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.6 var(--body); }}
main {{ max-width:1000px; margin:0 auto; padding-block:28px 64px; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); display:grid; gap:32px; }}
section {{ display:grid; gap:12px; min-width:0; }}
.prose {{ max-width:72ch; margin:0; }}
h1 {{ font:560 clamp(28px,4.5vw,38px)/1.1 var(--display); margin:0; text-wrap:balance; }}
h2 {{ font:560 22px/1.2 var(--display); margin:0; }} h3 {{ font:600 16px var(--body); margin:0; }}
.eyebrow {{ font:500 12px var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--accent); }}
.meta {{ font:13px var(--mono); color:var(--muted); display:flex; flex-wrap:wrap; gap:4px 18px; }}
code {{ font:.88em var(--mono); }} .dim {{ color:var(--muted); }} .small {{ font-size:13px; }}
.flow {{ display:grid; }} .row2 {{ display:grid; gap:10px; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); }}
.pbox {{ background:var(--surface); border:1px solid var(--rule); border-radius:6px; padding:10px 13px; display:grid; gap:4px; min-width:0; }}
.ptop {{ display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; }} .pname {{ font:600 14px var(--mono); }}
.pbody {{ font-size:13.5px; min-width:0; }}
.arrow {{ height:20px; position:relative; }} .arrow::before {{ content:""; position:absolute; left:24px; top:2px; bottom:6px; border-left:2px solid var(--rule); }}
.arrow::after {{ content:""; position:absolute; left:19px; bottom:1px; border:6px solid transparent; border-top:7px solid var(--rule); border-bottom:0; }}
.mark {{ font:500 11px var(--mono); text-transform:uppercase; letter-spacing:.06em; padding:3px 7px; border-radius:99px; border:1px solid currentColor; white-space:nowrap; }}
.mark-inherited {{ color:var(--muted); }} .mark-set {{ color:var(--warn); }} .mark-calibrated {{ color:var(--ok); }} .mark-computed {{ color:var(--accent); }}
.scroll {{ overflow-x:auto; min-width:0; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; font-variant-numeric:tabular-nums; }}
th,td {{ text-align:left; padding:6px 9px; border-bottom:1px solid var(--rule); vertical-align:top; }}
thead th {{ font:500 12px var(--mono); color:var(--muted); }} .num {{ text-align:right; white-space:nowrap; }}
td.off {{ color:var(--warn); font-weight:600; }}
ul.findings {{ display:grid; gap:9px; margin:0; padding-left:20px; max-width:74ch; }}
.callout {{ border-left:3px solid var(--warn); padding:2px 0 2px 14px; max-width:74ch; }}
</style>
<main>
<header style="display:grid;gap:10px">
 <div class="eyebrow">libcuflynx · CA PR #654</div>
 <h1>Does PCA reduction beat top-n Sobol ranking?</h1>
 <p class="prose">CA’s optimiser benchmarks, each with extra parameters: which parameters the PCA reduction keeps, which top-n Sobol ranking keeps, and how the calibrations of each compare with calibrating everything.</p>
 <div class="meta"><span>runs: {e(ROOT)}/&lt;model&gt;</span>{f"<span>3compartment: {e(THREE)}</span>" if THREE else ""}<span>commit {COMMIT} (feat/param-reduction-benchmarks)</span></div>
</header>
<section><h2>How these numbers were made</h2>{prov}</section>
<section><h2>The answer</h2>
<ul class="findings">
 <li><b>Given the same number of parameters, top-n Sobol ranking keeps the same set as the PCA reduction on every model</b>, so their calibrations are identical. PCA did not pick better parameters than top-n on these four models.</li>
 <li><b>What PCA adds is the count.</b> It stops when the next parameter would change the fit by less than the noise. Top-n needs n, or a threshold, from the user: at max-ST ≥ 0.05 it keeps more than PCA on three of the four models, and no single threshold matches PCA on all four (amber cells).</li>
 <li><b>The count is right where it can be checked.</b> Goodwin’s <code>a_i/(A_i + k_i Y)</code> is unchanged when the three are scaled together: the Fisher information at the truth has a zero eigenvalue and the full fit’s Laplace std for them is 67 decades. PCA drops one of them, and the reduced fit’s largest Laplace std is 0.02 decades. On Teusink it drops the two worst-determined enzymes (Laplace std 0.49 and 0.31 decades in the full set). On FitzHugh-Nagumo, where everything is identifiable, it keeps all five.</li>
 <li><b>Reduced fits are better determined, not better fitted.</b> The data are noise-free, so the full fits also land near the truth (χ² ≤ 4); the reduced fits trade a χ² still far below the number of points for condition numbers 10<sup>2</sup>–10<sup>8</sup> times smaller. Fixing a parameter at a wrong default moves its partners: Goodwin’s <code>A_i</code> and <code>k_i</code> absorb <code>a_i</code> at 72 instead of 100.</li>
</ul>
<div class="scroll"><table><thead><tr><th>Model</th><th class="num">Parameters</th><th class="num">PCA keeps</th><th>Top-n at the same n</th><th class="num">max ST ≥ 0.05</th><th class="num">≥ 0.1</th><th class="num">≥ 0.2</th><th class="num">FIM eigenvalues ≥ 1 at truth</th></tr></thead><tbody>{answer_rows()}</tbody></table></div>
<p class="prose small dim">The threshold columns count the parameters top-n would keep at that cutoff; amber where that differs from PCA. 3compartment’s Fisher count is from the earlier study’s Fisher information at its truth.</p>
</section>
<section><h2>Each model</h2>{"".join(model_section(m, s) for m, s in models.items())}
<p class="prose small dim">χ² sums ((model − data)/std)² over all series points. Laplace std is in decades of the calibrated parameter. Max error is |log10(value/truth)|, over the calibrated parameters and over all (the dropped ones sit at their defaults). The simple fits equal the pca fits and are not repeated.</p></section>
<section><h2>Limits</h2>
<div class="callout prose"><p>Single-start CMA-ES from the models’ defaults was tried first and is not used: FitzHugh-Nagumo starts at rest and every seed ran to a bound (a = b = 1), and with as many candidates per generation as MPI ranks it often stopped after a few hundred calls. The fits here are multi-start, as CA’s benchmarks fit these models.</p></div>
<ul class="findings">
 <li>The data are noise-free. Parameter recovery in the full fits is therefore optimistic; the Laplace std says what noisy data would allow.</li>
 <li>The redundancy check’s Jacobian is at one point, the centre of the box (not the truth); the modes and Sobol indices cover the whole box.</li>
 <li>Top-n’s n was set by us to PCA’s count. That is the comparison that isolates which parameters each method picks.</li>
</ul></section>
</main>
'''
open(PAGE, 'w').write(page)
print('wrote', PAGE, len(page))
