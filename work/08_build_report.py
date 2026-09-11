import json, html
from pathlib import Path
P = json.load(open("/home/user/forgedge/work/out/payload.json"))
OUT = Path("/tmp/claude-0/-home-user-forgedge/fb49d182-1300-5313-98ab-950890ac9071/scratchpad/holdout_verdict.html")

CSS = """
<title>Hold-Out Verdict on Three FORGE Portfolios</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Spectral:ital,wght@0,400;0,600;1,400&display=swap">
<style>
:root{
  --ground:#F4F6F8; --surface:#FFFFFF; --surface-2:#EDF1F4;
  --ink:#131A22; --ink-2:#35404B; --muted:#5C6874; --faint:#8B97A3;
  --rule:#DCE2E8; --rule-2:#C6D0D9;
  --accent:#1F5F8B; --accent-soft:#E3EDF5;
  --pos:#12736A; --pos-soft:#E0F0ED;
  --neg:#A8322D; --neg-soft:#F7E5E3;
  --warn:#9A6B12;
  --serif:Spectral,"Iowan Old Style",Georgia,serif;
  --sans:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;
  --mono:"IBM Plex Mono","SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0E141A; --surface:#151D26; --surface-2:#1C2630;
  --ink:#E6EDF3; --ink-2:#C3CEDA; --muted:#95A3B1; --faint:#6C7B89;
  --rule:#253039; --rule-2:#33414D;
  --accent:#6BA8D6; --accent-soft:#17293A;
  --pos:#3DA893; --pos-soft:#0F2B29;
  --neg:#E0736B; --neg-soft:#321A1A;
  --warn:#C89A3C;
}}
:root[data-theme="dark"]{
  --ground:#0E141A; --surface:#151D26; --surface-2:#1C2630;
  --ink:#E6EDF3; --ink-2:#C3CEDA; --muted:#95A3B1; --faint:#6C7B89;
  --rule:#253039; --rule-2:#33414D;
  --accent:#6BA8D6; --accent-soft:#17293A;
  --pos:#3DA893; --pos-soft:#0F2B29;
  --neg:#E0736B; --neg-soft:#321A1A;
  --warn:#C89A3C;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding-inline:20px;padding-block:0 72px}
.col{max-width:68ch}
h1,h2,h3,h4{font-family:var(--serif);text-wrap:balance;margin:0;font-weight:600;letter-spacing:-.01em}
h1{font-size:clamp(2.1rem,5.2vw,3.1rem);line-height:1.08}
h2{font-size:clamp(1.45rem,3.2vw,1.9rem);line-height:1.18}
h3{font-size:1.12rem;line-height:1.3}
p{margin:0 0 1em}
a{color:var(--accent)}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);font-weight:500}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.pos{color:var(--pos)} .neg{color:var(--neg)}

/* ---- masthead ---- */
header.mast{border-bottom:2px solid var(--ink);padding-block:40px 20px;margin-bottom:8px}
.mast .eyebrow{display:block;margin-bottom:14px}
.mast .standfirst{font-family:var(--serif);font-size:1.2rem;line-height:1.5;color:var(--ink-2);
  max-width:62ch;margin-top:18px}
.runbar{display:flex;flex-wrap:wrap;gap:0;border-top:1px solid var(--rule);margin-top:26px;
  font-family:var(--mono);font-size:.74rem}
.runbar div{padding:9px 18px 9px 0;margin-right:18px;border-right:1px solid var(--rule);color:var(--muted)}
.runbar div:last-child{border-right:0}
.runbar b{color:var(--ink);font-weight:500}

/* ---- verdict ---- */
.verdict{background:var(--surface);border:1px solid var(--rule);border-left:4px solid var(--neg);
  padding:26px 28px;margin:34px 0 12px}
.verdict h2{font-size:1.35rem;margin-bottom:12px}
.verdict p:last-child{margin-bottom:0}
.bignum{display:flex;flex-wrap:wrap;gap:34px;margin:22px 0 4px;padding-top:20px;border-top:1px solid var(--rule)}
.bignum div{min-width:130px}
.bignum .v{font-family:var(--mono);font-size:1.9rem;font-weight:600;line-height:1;
  font-variant-numeric:tabular-nums}
.bignum .k{font-size:.76rem;color:var(--muted);margin-top:7px;line-height:1.35}

section{margin-top:60px}
section > h2{margin-bottom:6px}
.sub{color:var(--muted);font-size:.95rem;margin-bottom:26px;max-width:66ch}

/* ---- window triad ---- */
.triad{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}
.win{background:var(--surface);border:1px solid var(--rule);padding:16px 16px 14px;border-top:3px solid var(--rule-2)}
.win.is{border-top-color:var(--faint)}
.win.oos{border-top-color:var(--accent)}
.win.ho{border-top-color:var(--neg);background:var(--surface)}
.win h4{font-family:var(--mono);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;
  font-weight:600;color:var(--ink);display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.win .span{font-family:var(--mono);font-size:.66rem;color:var(--faint);letter-spacing:0;text-transform:none;font-weight:400}
.win dl{margin:12px 0 0;display:grid;grid-template-columns:1fr auto;gap:5px 10px;font-size:.82rem}
.win dt{color:var(--muted)}
.win dd{margin:0;font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;font-weight:500}
.win dl .sep{grid-column:1/-1;border-top:1px solid var(--rule);margin:5px 0 1px}
.seal{display:inline-block;font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;
  background:var(--neg-soft);color:var(--neg);padding:2px 6px;border-radius:2px;text-transform:uppercase}

/* ---- rules table ---- */
.tablewrap{overflow-x:auto;border:1px solid var(--rule);background:var(--surface);margin:8px 0 4px}
table{border-collapse:collapse;width:100%;font-size:.82rem;min-width:680px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--rule);vertical-align:top}
thead th{font-family:var(--mono);font-size:.66rem;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);font-weight:600;border-bottom:1px solid var(--rule-2);white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
th.n{text-align:right}
code.f{font-family:var(--mono);font-size:.74rem;color:var(--ink-2);line-height:1.45;display:block;
  max-width:46ch;overflow-wrap:anywhere}
.tk{font-family:var(--mono);font-weight:600;font-size:.78rem;white-space:nowrap}
.dir{font-family:var(--mono);font-size:.64rem;letter-spacing:.06em;text-transform:uppercase;
  padding:1px 5px;border-radius:2px;background:var(--surface-2);color:var(--muted)}
.dir.short{background:var(--neg-soft);color:var(--neg)}

/* ---- chart ---- */
figure{margin:22px 0 0}
figcaption{font-size:.78rem;color:var(--muted);margin-top:9px;font-family:var(--sans)}
.chart{width:100%;height:auto;display:block;background:var(--surface);border:1px solid var(--rule)}

/* ---- book header ---- */
.book{margin-top:46px;padding-top:26px;border-top:1px solid var(--rule-2)}
.book:first-of-type{border-top:0;padding-top:0}
.bookhead{display:flex;flex-wrap:wrap;align-items:baseline;gap:14px;margin-bottom:4px}
.bookhead h3{font-size:1.3rem}
.tag{font-family:var(--mono);font-size:.68rem;padding:2px 8px;border:1px solid var(--rule-2);
  color:var(--muted);border-radius:2px}
.tag.fail{border-color:var(--neg);color:var(--neg);background:var(--neg-soft)}
.tag.ok{border-color:var(--pos);color:var(--pos);background:var(--pos-soft)}

/* ---- survival bars ---- */
.surv{display:grid;gap:9px;margin-top:18px}
.survrow{display:grid;grid-template-columns:88px 1fr 116px;gap:12px;align-items:center;font-size:.82rem}
.survrow .tk{font-size:.76rem}
.bar{height:16px;background:var(--surface-2);position:relative;border:1px solid var(--rule)}
.bar i{position:absolute;inset:0 auto 0 0;display:block}
.survrow .v{font-family:var(--mono);font-size:.74rem;color:var(--muted);text-align:right;
  font-variant-numeric:tabular-nums}

/* ---- steps ---- */
ol.steps{list-style:none;padding:0;margin:20px 0 0;counter-reset:s;display:grid;gap:2px}
ol.steps li{counter-increment:s;display:grid;grid-template-columns:32px 1fr;gap:14px;
  padding:14px 0;border-top:1px solid var(--rule)}
ol.steps li::before{content:"0" counter(s);font-family:var(--mono);font-size:.72rem;color:var(--accent);
  font-weight:600;padding-top:2px}
ol.steps b{display:block;margin-bottom:3px}
ol.steps span{font-size:.9rem;color:var(--muted)}

ul.notes{padding-left:1.1em;margin:14px 0 0}
ul.notes li{margin-bottom:.6em;font-size:.94rem;color:var(--ink-2)}
footer{margin-top:70px;padding-top:22px;border-top:1px solid var(--rule);
  font-family:var(--mono);font-size:.72rem;color:var(--faint);line-height:1.7}
@media (max-width:720px){
  .triad{grid-template-columns:1fr}
  .survrow{grid-template-columns:74px 1fr 92px}
  .bignum{gap:22px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
"""

def fmt(v, d=2, dash="—"):
    if v is None: return dash
    try:
        f=float(v)
    except (TypeError,ValueError): return html.escape(str(v))
    if f != f: return dash
    return f"{f:,.{d}f}"

def sgn(v, d=2, suffix=""):
    if v is None: return "—"
    cls = "pos" if float(v) > 0 else ("neg" if float(v) < 0 else "")
    s = f"{float(v):+,.{d}f}{suffix}"
    return f'<span class="{cls}">{s}</span>'

WLABEL = {"IS":("In-sample","is"),"OOS":("Out-of-sample","oos"),"HO":("Hold-out","ho")}

def triad(windows):
    out=['<div class="triad">']
    for w in windows:
        lab,cls = WLABEL[w["window"]]
        span = f'{w["start"]} → {w["end"] or "2026-09-02"}'
        seal = ' <span class="seal">sealed</span>' if cls=="ho" else ""
        out.append(f'''<div class="win {cls}">
<h4><span>{lab}{seal}</span><span class="span">{span}</span></h4>
<dl>
<dt>Aperture (trades)</dt><dd>{w["n_trades"]:,}</dd>
<dt>Signals fired</dt><dd>{w["n_signals"]:,}</dd>
<dt>Fill rate</dt><dd>{fmt(w["fill_rate"]*100,1)}%</dd>
<div class="sep"></div>
<dt>Win rate</dt><dd>{fmt(w["win_rate"]*100,1)}%</dd>
<dt>Profit factor</dt><dd>{fmt(w["profit_factor"],2)}</dd>
<dt>Net gain</dt><dd>{sgn(w["net_gain_eur"],2," €")}</dd>
<dt>Expectancy / trade</dt><dd>{sgn(w["expectancy_eur"],3," €")}</dd>
<div class="sep"></div>
<dt>Max drawdown</dt><dd>{fmt(w["max_dd_eur"],2)} €</dd>
<dt>Max DD (% capital)</dt><dd>{fmt(w["max_dd_pct"],2)}%</dd>
<div class="sep"></div>
<dt>Peak concurrency</dt><dd>{w["max_concurrent"]}</dd>
<dt>Peak exposure</dt><dd>{fmt(w["max_exposure_eur"],0)} €</dd>
<dt>Mean concurrency</dt><dd>{fmt(w["mean_concurrent"],2)}</dd>
<dt>Return on peak exp.</dt><dd>{sgn(w["return_on_max_exposure_pct"],2,"%")}</dd>
<div class="sep"></div>
<dt>Sharpe (annualised)</dt><dd>{sgn(w["sharpe_ann"],2)}</dd>
<dt>Months positive</dt><dd>{fmt(w["pct_months_positive"],1)}%</dd>
<dt>Avg holding</dt><dd>{fmt(w["avg_hold_days"],1)} d</dd>
<dt>Trades / month</dt><dd>{fmt(w["trades_per_month"],2)}</dd>
</dl></div>''')
    out.append('</div>')
    return "\n".join(out)

def rules_table(rules, per_rule):
    rows=[]
    for r in rules:
        aid=r["alpha_id"]
        pr={w:per_rule[w].get(aid,{}) for w in ("IS","OOS","HO")}
        d = "short" if r["direction"]=="short" else "long"
        rows.append(f'''<tr>
<td><span class="tk">{html.escape(r["ticker"])}</span><br><span class="dir {d}">{d}</span></td>
<td><code class="f">{html.escape(r["formula"])}</code></td>
<td class="n">{r["target_h"]}d<br><span style="color:var(--faint)">{fmt(r["sell_pct"]*100,2)}% TP</span></td>
<td class="n">{html.escape(r["buy_type"])}</td>
<td class="n">{pr["IS"].get("n",0)}<br>{sgn(pr["IS"].get("net"),1)}</td>
<td class="n">{pr["OOS"].get("n",0)}<br>{sgn(pr["OOS"].get("net"),1)}</td>
<td class="n">{pr["HO"].get("n",0)}<br>{sgn(pr["HO"].get("net"),1)}</td>
</tr>''')
    return f'''<div class="tablewrap"><table>
<thead><tr><th>Ticker</th><th>Event rule (thresholds are distributional)</th><th class="n">Horizon</th>
<th class="n">Entry</th><th class="n">IS n / €</th><th class="n">OOS n / €</th><th class="n">HO n / €</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p style="font-size:.78rem;color:var(--muted);margin-top:9px">Per-leg cells show trade count and net P&amp;L in euro at 100 € per opened position.</p>'''

# ---------- equity chart ----------
def equity_svg(book, wkey_order=("IS","OOS","HO")):
    import datetime as dt
    series=[]
    for w in wkey_order:
        e=book["equity"][w]
        if not e["x"]: continue
        series.append((w,e["x"],e["y"]))
    if not series: return ""
    W,H = 900,250
    ml,mr,mt,mb = 56,14,16,30
    iw,ih = W-ml-mr, H-mt-mb
    def dnum(s): 
        y,m,d = s.split("-"); return dt.date(int(y),int(m),int(d)).toordinal()
    allx=[dnum(x) for _,xs,_ in series for x in xs]
    ally=[v for _,_,ys in series for v in ys]
    x0,x1 = min(allx), max(allx)
    y0,y1 = min(min(ally),0), max(max(ally),0)
    pad=(y1-y0)*0.10 or 1
    y0-=pad; y1+=pad
    sx=lambda v:(ml+(v-x0)/(x1-x0)*iw) if x1>x0 else ml
    sy=lambda v:(mt+ih-(v-y0)/(y1-y0)*ih)
    COL={"IS":"var(--faint)","OOS":"var(--accent)","HO":"var(--neg)"}
    parts=[f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Cumulative realised P and L in euro">']
    # zero line + y ticks
    ticks=[]
    step=(y1-y0)/4
    for i in range(5):
        v=y0+step*i; ticks.append(v)
    for v in ticks:
        yy=sy(v)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{W-mr}" y2="{yy:.1f}" stroke="var(--rule)" stroke-width="1" fill="none"/>')
        parts.append(f'<text x="{ml-8}" y="{yy+3.5:.1f}" text-anchor="end" fill="var(--faint)" font-family="IBM Plex Mono, monospace" font-size="10">{v:,.0f}</text>')
    parts.append(f'<line x1="{ml}" y1="{sy(0):.1f}" x2="{W-mr}" y2="{sy(0):.1f}" stroke="var(--rule-2)" stroke-width="1.2" fill="none"/>')
    for w,xs,ys in series:
        pts=" ".join(f"{sx(dnum(x)):.1f},{sy(y):.1f}" for x,y in zip(xs,ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{COL[w]}" stroke-width="1.8" stroke-linejoin="round"/>')
        # segment boundary + label
        bx=sx(dnum(xs[0]))
        parts.append(f'<line x1="{bx:.1f}" y1="{mt}" x2="{bx:.1f}" y2="{mt+ih}" stroke="{COL[w]}" stroke-width="1" stroke-dasharray="3 3" opacity="0.55" fill="none"/>')
        parts.append(f'<text x="{bx+5:.1f}" y="{mt+11}" fill="{COL[w]}" font-family="IBM Plex Mono, monospace" font-size="10" font-weight="600">{w}</text>')
        ex,ey=sx(dnum(xs[-1])),sy(ys[-1])
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" fill="{COL[w]}"/>')
    # x labels
    for v,anch in ((x0,"start"),(x1,"end")):
        lbl=dt.date.fromordinal(v).strftime("%b %Y")
        parts.append(f'<text x="{sx(v):.1f}" y="{H-10}" text-anchor="{anch}" fill="var(--faint)" font-family="IBM Plex Mono, monospace" font-size="10">{lbl}</text>')
    parts.append('</svg>')
    return "".join(parts)

def book_block(name, book, note):
    w = {x["window"]:x for x in book["windows"]}
    ho = w["HO"]
    tag = 'fail' if ho["net_gain_eur"] < 0 else 'ok'
    tagtxt = f'hold-out {ho["net_gain_eur"]:+,.0f} € · PF {ho["profit_factor"]:.2f}'
    return f'''<div class="book">
<div class="bookhead"><h3>{html.escape(name)}</h3><span class="tag {tag}">{tagtxt}</span></div>
<p class="sub">{note}</p>
{rules_table(book["rules"], book["per_rule"])}
{triad(book["windows"])}
<figure>{equity_svg(book)}
<figcaption>Cumulative realised P&amp;L in euro, each window measured on its own from zero — the three lines are not a continuous equity curve. Entry bars are restricted to the window; fill and exit windows may reach past it.</figcaption></figure>
</div>'''

# ---------- survival bars ----------
def survival():
    bt = P["pool"]["by_ticker"]
    bh = P["bh"]
    order = sorted(bt, key=lambda k:-bt[k]["pct_pos"])
    rows=[]
    for tk in order:
        d=bt[tk]; pct=d["pct_pos"]*100
        col = "var(--pos)" if pct>=50 else ("var(--warn)" if pct>=33 else "var(--neg)")
        rows.append(f'''<div class="survrow">
<span class="tk">{tk}</span>
<span class="bar"><i style="width:{pct:.1f}%;background:{col}"></i></span>
<span class="v">{pct:.0f}% of {d["n"]} · B&amp;H {bh.get(tk,0):+.0f}%</span></div>''')
    return f'<div class="surv">{"".join(rows)}</div>'

# ---------- assemble ----------
pool=P["pool"]
v1=P["v1"]; v2=P["v2"]
NOTES = {
 "P1 — Core":"Highest-ranked rule from five distinct tickers. Copper volume-dispersion, Ether range-position reversal, DAX volatility contraction, Brent SMA-vs-lagged-high, Bitcoin z-score dip.",
 "P2 — Diversified":"Second draft round — same construction, disjoint rules. Ether and Brent again carry the largest legs.",
 "P3 — Satellite":"Third draft round, the only book holding an S&P 500 leg and the only one whose hold-out loss stays inside 5% of peak exposure.",
}
V2NOTES = {
 "V2-A":"Brent lower-wick reversal short drafted first, then long legs capped at two per asset class.",
 "V2-B":"Brent close-position z-score short with S&P 500, DAX and both crypto names.",
 "V2-C":"Brent EMA-spread short — the only leg in any book whose rotation p-value cleared 1.00 (0.864).",
}

v1_html = "\n".join(book_block(k, v1[k], NOTES[k]) for k in v1)
v2_html = "\n".join(book_block(k, v2[k], V2NOTES[k]) for k in v2)

runs = P["runs"]
tot_cand = sum(v[0] for v in runs.values()); tot_prom=sum(v[1] for v in runs.values()); tot_trade=sum(v[2] for v in runs.values())

HTML = CSS + f'''
<div class="wrap">
<header class="mast">
<span class="eyebrow">FORGE · daily bars · nine-month sealed hold-out</span>
<h1>Every rule passed. The hold-out said no.</h1>
<p class="standfirst">Three five-rule portfolios, built from {tot_trade:,} tradeable rules mined across eight instruments, sized at 100 € per opened position. All three clear in-sample and out-of-sample with profit factors between 3.8 and 6.2. All three lose money on the eight months of 2026 that selection never saw.</p>
<div class="runbar">
<div>Discovery <b>2021-01-03 → 2025-12-31</b></div>
<div>Hold-out <b>2026-01-01 → 2026-09-02</b></div>
<div>Preset <b>balanced · 1D</b></div>
<div>Fee <b>0.10%/side</b></div>
<div>Size <b>100 € / trade</b></div>
</div>
</header>

<div class="col">
<div class="verdict">
<h2>The gates selected for luck, not edge</h2>
<p>Replaying every rule in the pool on the sealed window settles what the three portfolios alone could not. Of the 815 rules that passed both the in-sample and out-of-sample gates, <b>{pool["elig_pos"]*100:.1f}%</b> made money in the hold-out. Of all {pool["n_all"]:,} tradeable rules — gates ignored — <b>{pool["all_pos"]*100:.1f}%</b> did.</p>
<p>Passing the IS and OOS filters made a rule <i>less</i> likely to survive. That is the signature of selection overfitting, not of a market that simply turned: the filters were fitted to the same 815-rule surface they were ranking.</p>
<div class="bignum">
<div><div class="v neg">{pool["elig_pos"]*100:.1f}%</div><div class="k">of gated rules profitable<br>in hold-out</div></div>
<div><div class="v" style="color:var(--muted)">{pool["all_pos"]*100:.1f}%</div><div class="k">of ungated rules<br>profitable</div></div>
<div><div class="v neg">{pool["elig_med_pf"]:.2f}</div><div class="k">median hold-out<br>profit factor</div></div>
<div><div class="v neg">{pool["elig_med_net"]:+,.0f} €</div><div class="k">median hold-out P&amp;L<br>per rule</div></div>
</div>
</div>
</div>

<section>
<h2>How the three books were built</h2>
<p class="sub">Nothing after step 3 was allowed to read the hold-out. The pipeline ran once per instrument on the discovery span only.</p>
<div class="col">
<ol class="steps">
<li><b>Seal the hold-out first</b><span>Everything from 2026-01-01 onward — 8.1 months, ~170 daily bars — was cut before the KPI tables reached <code>forge()</code>. The discovery frame ends 2025-12-31.</span></li>
<li><b>Mine each instrument independently</b><span>Eight <code>forge()</code> runs at the <code>balanced</code> preset on 1D bars: {tot_cand:,} event candidates → {tot_prom:,} promoted alpha contracts → {tot_trade:,} rules reaching a tradeable verdict.</span></li>
<li><b>Replay every rule on IS and OOS</b><span>Each rule re-evaluated with <code>EventCandidate.apply()</code> over the full table, then backtested with its own validated <code>BacktestParams</code>, restricted by entry date to each window. The IS/OOS boundary is the session <code>TimeBudget</code> split — mid-2024 for the five-year instruments, March 2025 for crypto.</span></li>
<li><b>Gate, rank, draft</b><span>815 rules cleared the gates (≥10 IS trades, ≥4 OOS trades, IS PF ≥ 1.20, OOS PF ≥ 1.10, both windows net-positive, horizon ≤ 20 bars). Ranked on a blend of OOS and IS profit factor, IS/OOS agreement, walk-forward consistency and deflated Sharpe, then snake-drafted into three disjoint books of five, one ticker per book per leg, rejecting any rule firing on &gt;60% the same days as one already drafted.</span></li>
<li><b>Break the seal once</b><span>The hold-out was replayed after the three books were fixed. Every hold-out number on this page was produced in that single pass.</span></li>
</ol>
</div>
</section>

<section>
<h2>The three portfolios</h2>
<p class="sub">Each book holds five rules on five distinct instruments. Every opened position is 100 € of notional; positions overlap freely, so peak concurrency is what decides the capital the book actually needs.</p>
{v1_html}
</section>

<section>
<h2>Where the hold-out broke</h2>
<p class="sub">Survival was not uniform. The share of gated rules that made money in the hold-out, per instrument, against that instrument's own buy-and-hold return over the same window.</p>
{survival()}
<div class="col" style="margin-top:26px">
<p>Equity indices held: S&amp;P 500 rules survived at {pool["by_ticker"]["SP500"]["pct_pos"]*100:.0f}% with a median hold-out profit factor of {pool["by_ticker"]["SP500"]["med_pf"]:.2f}, DAX at {pool["by_ticker"]["DAX"]["pct_pos"]*100:.0f}%. Crypto and Brent did not: Bitcoin {pool["by_ticker"]["BTCEUR"]["pct_pos"]*100:.0f}%, Ether {pool["by_ticker"]["ETHEUR"]["pct_pos"]*100:.0f}%, Brent {pool["by_ticker"]["BRENT"]["pct_pos"]*100:.0f}%.</p>
<p>Two things compounded. First, the eligible pool was itself concentrated: <b>483 of the 815 gated rules were Brent</b>, an instrument that then delivered a 37.5% peak-to-trough fall inside the hold-out even while finishing the window up 51.9%. Second, {P["pool"]["by_dir"]["long"]["n"]} of the 815 were long and only {P["pool"]["by_dir"]["short"]["n"]} short, and nearly all of the long rules are dip-buying events — volatility contraction, z-score extremes, range-position lows — closed by a fixed take-profit or a horizon exit. Ether fell 19.2% over the hold-out with a 52.8% maximum drawdown along the way. Buying that dip repeatedly, with no stop and a take-profit that never prints, is the single largest loss in two of the three books.</p>
<p>The monthly P&amp;L locates it precisely: February 2026 and June 2026 account for almost the entire loss in all three books, and the Ether leg alone contributes −313 € in P1 and −442 € in P2.</p>
</div>
</section>

<section>
<h2>Testing the obvious fix</h2>
<p class="sub">If concentration and long-only bias caused the failure, forcing diversification should repair it. It does not — which is what tells us the problem is upstream of portfolio construction.</p>
<div class="col">
<p>A second selection was run with three structural changes, all decided on portfolio-construction principle rather than tuned against the hold-out: rank on the <i>weaker</i> of the IS and OOS profit factors instead of the OOS figure alone; cap any asset class at two of five legs; and require at least one short leg, drafted first so the quota is reachable.</p>
<p><b>This set was built after the hold-out had already been observed.</b> Its hold-out figures are therefore no longer an unbiased estimate, and are reported only as a diagnostic on the pool — not as a result.</p>
</div>
{v2_html}
<div class="col" style="margin-top:26px">
<p>All three still lose, the best of them (V2-C, −110 €) landing roughly where the best of the original three did (P3, −63 €). Structural diversification did not rescue a pool in which only 31.5% of gated rules were profitable — you cannot diversify your way out of an edge that is not there.</p>
</div>
</section>

<section>
<h2>What this run actually establishes</h2>
<div class="col">
<ul class="notes">
<li><b>The hold-out earned its keep.</b> On IS and OOS evidence alone these books look like a finished product: PF 3.8–6.2, win rates 71–82%, max drawdown under 3% of peak exposure, 68–90% of months positive. Every one of those numbers is an artefact of having chosen the rules that produced them.</li>
<li><b>No rule ever cleared the rotation null.</b> All {tot_trade:,} tradeable rules came back <code>PARTIAL-EDGE</code>, and every selected leg but one carries <code>rotation_p = 1.00</code>. FORGE was saying, in its own vocabulary, that nothing in this search survived the multiple-testing surface it was mined from — a search of {tot_cand:,} candidates across roughly 87,000 return tests per instrument. The hold-out confirmed the warning rather than contradicting it.</li>
<li><b>The OOS window was not out-of-sample for the portfolio.</b> It was out-of-sample for each individual rule, but ranking 815 rules on it makes it a selection window. Only the sealed hold-out was ever genuinely clean, which is why it is the only window that disagreed.</li>
<li><b>Long-only mean reversion with no stop is a directional bet.</b> Peak concurrency of 18–25 positions means 1,800–2,500 € of simultaneous exposure — these books carry far more beta than their in-sample Sharpe of 4–8 suggests.</li>
<li><b>What would be worth trying next.</b> Restrict discovery to the instruments whose rules actually survived (S&amp;P 500, DAX, Copper); pair the <code>sweep</code> preset with <code>RotationConfig(k≥100)</code> and a <code>min_lift</code> filter so the rotation null does real work instead of saturating at 1.00; and carve a second sealed window so a revised selection rule can be tested without spending the only clean data that remains.</li>
</ul>
</div>
</section>

<footer>
forgedge 0.2.0 · preset <b>balanced</b> · timeframe 1D · fee 0.10% per side · PYTHONHASHSEED=0<br>
Instruments: BTCEUR, ETHEUR (2023-03-14 →), EURUSD, GBPUSD, COPPER.CMDUSD, E_Brent, E_DAAX, E_SandP-500 (2021-01-03 →) · all series end 2026-09-02<br>
Drawdown is measured on the closed-trade equity curve with starting capital set to peak concurrent exposure. Economics are nominal; overlapping positions share a price path and are not independent observations.
</footer>
</div>
'''
OUT.write_text(HTML)
print("wrote", OUT, len(HTML), "bytes")
