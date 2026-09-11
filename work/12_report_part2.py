import json, html
from pathlib import Path
Q = json.load(open("/home/user/forgedge/work/out/payload2.json"))
SRC = Path("/tmp/claude-0/-home-user-forgedge/fb49d182-1300-5313-98ab-950890ac9071/scratchpad/holdout_verdict.html")
doc = SRC.read_text()

EXTRA_CSS = """
/* ---- part two ---- */
.partmark{margin-top:76px;padding-top:30px;border-top:2px solid var(--ink)}
.partmark .eyebrow{display:block;margin-bottom:12px}
.gt{width:100%;border-collapse:collapse;font-size:.86rem;margin-top:16px;min-width:520px}
.gt th,.gt td{padding:10px 14px;border-bottom:1px solid var(--rule);text-align:left}
.gt thead th{font-family:var(--mono);font-size:.66rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);font-weight:600;border-bottom:1px solid var(--rule-2)}
.gt td.n,.gt th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.gt tbody tr:last-child{background:var(--neg-soft)}
.gt tbody tr:last-child td{font-weight:600;border-bottom:0}
.gt .lead{font-family:var(--mono);font-size:.8rem}
.quad{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin:18px 0}
.quad .win{border-top-width:3px}
.win.ho1{border-top-color:var(--warn)}
.win.ho2{border-top-color:var(--neg)}
.win dl.tight{font-size:.79rem}
.legrow{display:flex;flex-wrap:wrap;gap:7px;margin:12px 0 2px}
.leg{font-family:var(--mono);font-size:.7rem;padding:3px 8px;border:1px solid var(--rule);
  background:var(--surface);border-radius:2px;color:var(--muted);white-space:nowrap}
.leg b{color:var(--ink);font-weight:600}
.leg.dead{border-style:dashed;opacity:.62}
.leg.win{border-color:var(--pos);color:var(--pos)}
.leg.lose{border-color:var(--neg);color:var(--neg)}
.setnote{background:var(--surface-2);border-left:3px solid var(--accent);padding:15px 18px;
  margin:20px 0 6px;font-size:.9rem;color:var(--ink-2)}
.setnote b{color:var(--ink)}
.setnote.warn{border-left-color:var(--warn)}
@media (max-width:820px){.quad{grid-template-columns:repeat(2,1fr)}}
@media (max-width:520px){.quad{grid-template-columns:1fr}}
"""

def f(v,d=2):
    if v is None: return "—"
    try: x=float(v)
    except: return html.escape(str(v))
    return "—" if x!=x else f"{x:,.{d}f}"
def s(v,d=2,suf=""):
    if v is None: return "—"
    x=float(v); c="pos" if x>0 else ("neg" if x<0 else "")
    return f'<span class="{c}">{x:+,.{d}f}{suf}</span>'

WL={"IS":("In-sample","is"),"OOS":("Out-of-sample","oos"),
    "HO1":("Hold-out 1st half","ho1"),"HO2":("Hold-out 2nd half","ho2")}

def quad(wins, sealed):
    o=['<div class="quad">']
    for w in wins:
        lab,cls=WL[w["window"]]
        mark=' <span class="seal">sealed</span>' if w["window"]==sealed else ""
        o.append(f'''<div class="win {cls}">
<h4><span>{lab}{mark}</span></h4>
<dl class="tight">
<dt>Aperture</dt><dd>{w["n_trades"]:,}</dd>
<dt>Win rate</dt><dd>{f(w["win_rate"]*100,1)}%</dd>
<dt>Profit factor</dt><dd>{f(w["profit_factor"],2)}</dd>
<dt>Net gain</dt><dd>{s(w["net_gain_eur"],0," €")}</dd>
<div class="sep"></div>
<dt>Max DD</dt><dd>{f(w["max_dd_eur"],0)} €</dd>
<dt>Max DD %</dt><dd>{f(w["max_dd_pct"],2)}%</dd>
<dt>Peak conc.</dt><dd>{w["max_concurrent"]}</dd>
<dt>Peak exp.</dt><dd>{f(w["max_exposure_eur"],0)} €</dd>
<div class="sep"></div>
<dt>Expectancy</dt><dd>{s(w["expectancy_eur"],2," €")}</dd>
<dt>Sharpe</dt><dd>{s(w["sharpe_ann"],2)}</dd>
<dt>Months +</dt><dd>{f(w["pct_months_positive"],0)}%</dd>
</dl></div>''')
    o.append('</div>'); return "".join(o)

def legs(rules):
    o=['<div class="legrow">']
    for r in rules:
        c = "dead" if r["ho2_n"]==0 else ("win" if r["ho2_net"]>0 else "lose")
        val = "no trades" if r["ho2_n"]==0 else f'{r["ho2_n"]}t {r["ho2_net"]:+.0f}€'
        o.append(f'<span class="leg {c}"><b>{r["ticker"]}</b> h={r["target_h"]} · {val}</span>')
    o.append('</div>'); return "".join(o)

def book(name, b, sealed):
    return f'''<div class="book">
<div class="bookhead"><h3>{name}</h3>
<span class="tag {'fail' if b["windows"][-1]["net_gain_eur"]<0 else 'ok'}">HO2 {b["windows"][-1]["net_gain_eur"]:+,.0f} € · PF {b["windows"][-1]["profit_factor"]:.2f}</span></div>
{legs(b["rules"])}
{quad(b["windows"], sealed)}
</div>'''

def gtable(exp):
    rows=[]
    for r in exp["rows"]:
        rows.append(f'<tr><td class="lead">{r["gate"]}</td><td class="n">{r["n"]:,}</td>'
                    f'<td class="n">{r["pct"]:.1f}%</td><td class="n">{r["med_pf"]:.2f}</td>'
                    f'<td class="n">{r["med_net"]:+,.0f} €</td></tr>')
    return f'''<div class="tablewrap"><table class="gt">
<thead><tr><th>Rules selected by</th><th class="n">Rules</th><th class="n">Profitable in HO2</th>
<th class="n">Median HO2 PF</th><th class="n">Median HO2 P&amp;L</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>'''

bc = Q["by_class"]
def classtable():
    keys = sorted(set(bc["IS + OOS"]) | set(bc["IS + OOS + HO1"]))
    rows=[]
    for k in keys:
        a=bc["IS + OOS"].get(k); b=bc["IS + OOS + HO1"].get(k)
        if not a or not b: continue
        rows.append(f'<tr><td class="lead">{k}</td>'
                    f'<td class="n">{a["n"]}</td><td class="n">{a["pct"]:.1f}%</td>'
                    f'<td class="n">{b["n"]}</td><td class="n">{b["pct"]:.1f}%</td>'
                    f'<td class="n">{b["pct"]-a["pct"]:+.1f} pt</td></tr>')
    return f'''<div class="tablewrap"><table class="gt">
<thead><tr><th>Asset class</th><th class="n">n (IS+OOS)</th><th class="n">HO2 survival</th>
<th class="n">n (+HO1)</th><th class="n">HO2 survival</th><th class="n">Change</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>'''

B=Q["books"]; C=Q["counts"]
ea, ec = Q["exp_all"], Q["exp_cap"]

PART2 = f'''
<section class="partmark">
<span class="eyebrow">Part two · does a three-window check fix it?</span>
<h2>Adding windows to the gate makes the next one worse</h2>
<p class="sub">The natural response to an overfit selection is to demand more: require a rule to work in-sample, out-of-sample <i>and</i> in the hold-out. That is testable — and it fails a controlled test.</p>

<div class="col">
<p>Selecting on a window spends it. To measure the idea rather than assume it, the hold-out was split in two: <b>HO1</b> (2026-01-01 → 2026-05-01) joins the gate, <b>HO2</b> (2026-05-01 → 2026-09-02) stays sealed and is the only thing any of these numbers are measured against. Every rule was re-scored on all four windows in one pass — the entry-bar filter makes a single full-range replay exactly equivalent to four windowed ones, verified against the earlier run to the last decimal.</p>
<p>The gates thin the pool hard: {C["IS+OOS"]:,} of {Q["n_total"]:,} rules clear IS + OOS, {C["IS+OOS+HO"]:,} clear IS + OOS + the whole hold-out, and just {C["IS+OOS+HO1+HO2"]:,} clear all four windows. A stricter filter feels like stronger evidence. It is not.</p>
</div>

{gtable(ea)}
<p style="font-size:.78rem;color:var(--muted);margin-top:8px">All {ea["universe"]:,} rules with at least 3 hold-out-2 trades. "Profitable in HO2" is the share with positive net P&amp;L in the sealed window.</p>

<div class="col">
<p>Requiring a rule to also work in HO1 <b>drops</b> its chance of working in HO2, from {ea["rows"][1]["pct"]:.1f}% to {ea["rows"][2]["pct"]:.1f}% — below even the {ea["rows"][0]["pct"]:.1f}% of applying no gate at all. Each window you add to the selection buys a tighter fit to the past and pays for it in the future.</p>
<p>One measurement flaw had to be ruled out first: several selected rules carry horizons of 96–100 daily bars, longer than the four-month HO2 window, so their trades cannot close inside it and are dropped. Capping horizons at 20 bars, so every rule can genuinely trade in HO2, sharpens the same result rather than softening it:</p>
</div>

{gtable(ec)}
<p style="font-size:.78rem;color:var(--muted);margin-top:8px">{ec["universe"]:,} rules with horizon ≤ 20 bars and at least 3 HO2 trades. The gap widens from {ea["rows"][1]["pct"]-ea["rows"][2]["pct"]:.1f} to {ec["rows"][1]["pct"]-ec["rows"][2]["pct"]:.1f} points.</p>

<div class="col" style="margin-top:34px">
<h3 style="margin-bottom:8px">What the gate does and does not move</h3>
<p class="sub" style="margin-bottom:0">Broken out by asset class, under the horizon cap. The gate barely touches the one group that works and prunes the two that don't — in the wrong direction.</p>
</div>
{classtable()}
<div class="col" style="margin-top:14px">
<p>Index rules survive at about two in three whichever gate is applied — {bc["IS + OOS"]["index"]["pct"]:.1f}% under IS + OOS, {bc["IS + OOS + HO1"]["index"]["pct"]:.1f}% with HO1 added, a difference of well under a point. Commodity rules fall from {bc["IS + OOS"]["commodity"]["pct"]:.1f}% to {bc["IS + OOS + HO1"]["commodity"]["pct"]:.1f}% and crypto from {bc["IS + OOS"]["crypto"]["pct"]:.1f}% to {bc["IS + OOS + HO1"]["crypto"]["pct"]:.1f}%. <b>The instrument predicts survival; the number of windows in the gate does not.</b> Gating harder simply selects more aggressively inside the asset classes that were not going to work.</p>
</div>
</section>

<section>
<h2>The same portfolios, built three ways</h2>
<p class="sub">Three books per method, five rules each, distinct tickers, at most two per asset class — the construction is held fixed so only the selection gate varies.</p>

<div class="setnote warn"><b>Set A — gated on IS + OOS + the entire hold-out.</b> This is the literal three-window check. All {C["IS+OOS+HO"]:,} qualifying rules were profitable in every window by construction, HO2 included, so the HO2 column below is <i>not</i> a measurement — it is the filter reading itself back. Shown because the contrast with Set B is the whole point.</div>
{"".join(book(n, B[n], None) for n in ("A1","A2","A3"))}

<div class="setnote"><b>Set B — gated on IS + OOS + HO1, with HO2 sealed.</b> Identical logic, one window held back. Note what happens to HO1: profit factors of {B["B1"]["windows"][2]["profit_factor"]:.1f}–{B["B3"]["windows"][2]["profit_factor"]:.1f} and win rates above 80%, <i>better</i> than IS or OOS. That is not the rules improving; it is a four-month window being selected on. Then HO2 arrives.</div>
{"".join(book(n, B[n], "HO2") for n in ("B1","B2","B3"))}

<div class="setnote"><b>Set C — Set B plus a 20-bar horizon cap,</b> so no leg is structurally unable to trade inside HO2. In Set B the S&amp;P 500 legs — the only asset class that survives — contributed zero HO2 trades in all three books, which would have been a fair objection. It is not the explanation.</div>
{"".join(book(n, B[n], "HO2") for n in ("C1","C2","C3"))}

<div class="col" style="margin-top:30px">
<p>Set A shows profit factors of {B["A3"]["windows"][3]["profit_factor"]:.2f}–{B["A1"]["windows"][3]["profit_factor"]:.2f} in HO2. Sets B and C, same construction with that window withheld, show {B["B2"]["windows"][3]["profit_factor"]:.2f}–{B["C3"]["windows"][3]["profit_factor"]:.2f}. The difference between the two is not a better portfolio — it is whether the last window was allowed to vote on its own result.</p>
</div>
</section>

<section>
<h2>Answering the original doubt</h2>
<div class="col">
<ul class="notes">
<li><b>The overfitting was real, and verifying more windows does not remove it.</b> It relocates it. Each window added to the gate looks like corroboration and behaves like another parameter fitted to the past — measured here as a {ec["rows"][1]["pct"]-ec["rows"][2]["pct"]:.1f}-point drop in survival on the next untouched window.</li>
<li><b>A window is evidence exactly once.</b> Once HO1 entered the gate, its profit factors jumped to 3.9–10.7 while the same books scored 0.4–0.8 four months later. Those HO1 figures were never a forecast; they were the selection looking at its own reflection.</li>
<li><b>Something durable did turn up.</b> Asset class survives the test that window-stacking fails: index rules held around 67% across both gates, on a diagnostic that had every opportunity to erode them. That is worth a proper run — discovery restricted to S&amp;P 500 and DAX, with a genuinely untouched window reserved before it starts.</li>
<li><b>The budget is spent.</b> HO1 and HO2 have both now been read. Any further selection on this data is fitting, not testing; the next honest measurement needs bars that none of this analysis has seen.</li>
</ul>
</div>
</section>
'''

doc = doc.replace("</style>", EXTRA_CSS + "</style>", 1)
doc = doc.replace("<footer>", PART2 + "\n<footer>", 1)
doc = doc.replace("<title>Hold-Out Verdict on Three FORGE Portfolios</title>",
                  "<title>Hold-Out Verdict on Three FORGE Portfolios</title>", 1)
doc = doc.replace('<p class="standfirst">Three five-rule portfolios',
                  '<p class="standfirst">Three five-rule portfolios', 1)
SRC.write_text(doc)
print("patched:", len(doc), "bytes")
