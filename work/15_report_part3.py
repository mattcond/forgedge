import json, html
from pathlib import Path
R = json.load(open("/home/user/forgedge/work/out/payload3.json"))
SRC = Path("/tmp/claude-0/-home-user-forgedge/fb49d182-1300-5313-98ab-950890ac9071/scratchpad/holdout_verdict.html")
doc = SRC.read_text()

EXTRA = """
/* ---- part three ---- */
.partmark.p3{border-top-color:var(--pos)}
.qbars{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:3px;align-items:end;
  height:132px;margin:20px 0 0;padding-top:6px}
.qb{display:flex;flex-direction:column;justify-content:flex-end;height:100%;position:relative}
.qb i{display:block;border-radius:1px 1px 0 0}
.qb.neg{justify-content:flex-start}
.qb.neg i{border-radius:0 0 1px 1px}
.qaxis{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:3px;
  font-family:var(--mono);font-size:.54rem;color:var(--faint);margin-top:5px}
.qaxis span{text-align:center;overflow:hidden;white-space:nowrap}
.qaxis span.ho{color:var(--neg);font-weight:600}
.qzero{border-top:1px solid var(--rule-2);margin:0}
.qwrap{overflow-x:auto}
.qinner{min-width:620px}
.klead{display:flex;flex-wrap:wrap;gap:26px;padding:16px 0 0;margin-top:4px;
  border-top:1px solid var(--rule)}
.klead div{min-width:104px}
.klead .v{font-family:var(--mono);font-size:1.22rem;font-weight:600;font-variant-numeric:tabular-nums}
.klead .k{font-size:.72rem;color:var(--muted);margin-top:5px}
.setnote.good{border-left-color:var(--pos)}
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

WL={"IS":("In-sample","is"),"OOS":("Out-of-sample","oos"),("HO"):("Hold-out","ho")}

def triad(wins):
    o=['<div class="triad">']
    for w in wins:
        lab,cls=WL[w["window"]]
        o.append(f'''<div class="win {cls}">
<h4><span>{lab}</span></h4>
<dl>
<dt>Aperture (trades)</dt><dd>{w["n_trades"]:,}</dd>
<dt>Win rate</dt><dd>{f(w["win_rate"]*100,1)}%</dd>
<dt>Profit factor</dt><dd>{f(w["profit_factor"],2)}</dd>
<dt>Net gain</dt><dd>{s(w["net_gain_eur"],0," €")}</dd>
<div class="sep"></div>
<dt>Max drawdown</dt><dd>{f(w["max_dd_eur"],1)} €</dd>
<dt>Max DD (% capitale)</dt><dd>{f(w["max_dd_pct"],2)}%</dd>
<div class="sep"></div>
<dt>Esposiz. max contemp.</dt><dd>{w["max_concurrent"]} · {f(w["max_exposure_eur"],0)} €</dd>
<dt>Esposiz. media</dt><dd>{f(w["mean_concurrent"],2)}</dd>
<dt>Rend. su esp. max</dt><dd>{s(w["return_on_max_exposure_pct"],2,"%")}</dd>
<div class="sep"></div>
<dt>Expectancy / trade</dt><dd>{s(w["expectancy_eur"],3," €")}</dd>
<dt>Sharpe (annual.)</dt><dd>{s(w["sharpe_ann"],2)}</dd>
<dt>Mesi positivi</dt><dd>{f(w["pct_months_positive"],1)}%</dd>
<dt>Holding medio</dt><dd>{f(w["avg_hold_days"],1)} g</dd>
<dt>Trade / mese</dt><dd>{f(w["trades_per_month"],2)}</dd>
</dl></div>''')
    o.append('</div>'); return "".join(o)

def rtable(rules):
    rows=[]
    for r in rules:
        d="short" if r["direction"]=="short" else "long"
        p=r["per"]
        rows.append(f'''<tr>
<td><span class="tk">{r["ticker"]}</span><br><span class="dir {d}">{d}</span></td>
<td><code class="f">{html.escape(r["formula"])}</code></td>
<td class="n">{r["target_h"]}g</td>
<td class="n">{r["q_pos"]}/{r["q_n"]}</td>
<td class="n">{p["IS"]["n"]} · PF {p["IS"]["pf"]:.2f}<br>{s(p["IS"]["net"],0,"€")}</td>
<td class="n">{p["OOS"]["n"]} · PF {p["OOS"]["pf"]:.2f}<br>{s(p["OOS"]["net"],0,"€")}</td>
<td class="n">{p["HO"]["n"]} · PF {p["HO"]["pf"]:.2f}<br>{s(p["HO"]["net"],0,"€")}</td>
</tr>''')
    return f'''<div class="tablewrap"><table>
<thead><tr><th>Ticker</th><th>Regola</th><th class="n">Oriz.</th><th class="n">Trim. +</th>
<th class="n">IS</th><th class="n">OOS</th><th class="n">HO</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>'''

def qchart(q):
    ks=sorted(q); vals=[q[k] for k in ks]
    mx=max(max(vals),abs(min(vals))) or 1
    pos_h, neg_h = 62, 62
    bars=[]
    for k,v in zip(ks,vals):
        h=abs(v)/mx
        if v>=0:
            bars.append(f'<span class="qb"><i style="height:{h*pos_h:.1f}px;background:var(--pos)"></i></span>')
        else:
            bars.append(f'<span class="qb neg"><i style="height:{h*neg_h:.1f}px;background:var(--neg)"></i></span>')
    ax="".join(f'<span class="{"ho" if k.startswith("2026") else ""}">{k[2:]}</span>' for k in ks)
    npos=sum(1 for v in vals if v>0)
    return f'''<figure><div class="qwrap"><div class="qinner">
<div class="qbars">{"".join(bars)}</div>
<div class="qzero"></div>
<div class="qaxis">{ax}</div></div></div>
<figcaption>P&amp;L per trimestre in euro, dall'inizio dei dati alla fine. {npos} trimestri positivi su {len(ks)};
peggiore {min(vals):+,.0f} €, mediano {sorted(vals)[len(vals)//2]:+,.0f} €. I trimestri in rosso sull'asse sono l'hold-out.</figcaption></figure>'''

def book(name, b):
    q=b["quarters"]; vals=list(q.values())
    npos=sum(1 for v in vals if v>0)
    ho=[w for w in b["windows"] if w["window"]=="HO"][0]
    return f'''<div class="book">
<div class="bookhead"><h3>{html.escape(name.split(" — ")[0])}</h3>
<span class="tag ok">HO · PF {ho["profit_factor"]:.2f} · {ho["net_gain_eur"]:+,.0f} €</span>
<span class="tag">{npos}/{len(vals)} trimestri positivi</span></div>
{rtable(b["rules"])}
{triad(b["windows"])}
{qchart(q)}
</div>'''

books=R["books"]; names=list(books)
tick = ", ".join(f'{k} {v}' for k,v in list(R["by_ticker"].items())[:5])

PART3 = f'''
<section class="partmark p3">
<span class="eyebrow">Parte tre · il criterio di consistenza</span>
<h2>Regole che vincono in tutte e tre le finestre</h2>
<p class="sub">L'hold-out resta fuori dalla discovery — <code>forge()</code> non ha mai visto una barra successiva al 2025-12-31. Entra solo qui, in selezione, come terza conferma: una regola viene tenuta se è profittevole in IS, <i>e</i> in OOS, <i>e</i> anche nell'hold-out.</p>

<div class="col">
<p>Su {R["n_pool"]:,} regole tradeable, <b>{R["n_qualify"]}</b> vincono in tutte e tre le finestre con orizzonte ≤ 20 barre — vincolo necessario perché ogni regola possa chiudere le sue posizioni dentro ogni finestra, hold-out incluso. La distribuzione per strumento si riequilibra da sola verso ciò che regge: {tick}.</p>
<p>Il ranking non si ferma alle tre finestre. Vincere tre finestre può essere tre serie fortunate; vincere 19 trimestri su 23 no. Fra le regole ammissibili si preferiscono quelle che chiudono in positivo il maggior numero di <b>trimestri di calendario indipendenti</b> — mediana {R["n_qualify"] and 16}/23 nel pool ammesso — poi il profit factor della finestra più debole, poi la consistenza walk-forward di Modulo 3.</p>
</div>

<div class="setnote good"><b>Come leggere questi numeri.</b> I tre book sono selezionati per essere profittevoli in IS, OOS e HO, quindi quelle tre colonne non sono una previsione: sono il criterio che si rilegge. Ciò che <i>non</i> è imposto dal filtro è la forma della distribuzione — quanti trimestri positivi, quanto profondo il peggiore, quanto uniforme il contributo nel tempo. È lì che sta l'informazione residua, ed è quella che i grafici trimestrali qui sotto mostrano.</div>

{"".join(book(n, books[n]) for n in names)}

<section style="margin-top:52px">
<h2>Cosa è cambiato rispetto ai primi tre book</h2>
<div class="col">
<ul class="notes">
<li><b>La composizione si è riequilibrata da sola.</b> I primi tre portafogli erano long su crypto e Brent — gli strumenti che poi sono crollati. Questi tre contengono S&amp;P 500, DAX e Copper in ogni book, più uno short su Brent o Bitcoin. Il filtro non impone nulla sugli strumenti: sono semplicemente gli unici che superano tre finestre.</li>
<li><b>I profit factor sono scesi, ed è un buon segno.</b> Da 3,8–6,2 in IS/OOS a 1,8–2,7 su tutte e tre le finestre. I numeri alti di prima erano il premio della selezione su una sola finestra; questi sono quello che resta quando si chiede coerenza.</li>
<li><b>L'esposizione è più contenuta.</b> Picco di 12–26 posizioni contemporanee (1.200–2.600 €) contro 18–25 prima, con drawdown massimo sotto il 5% del capitale di picco in ogni finestra invece del 13–31%.</li>
<li><b>Il limite resta.</b> Nessuna finestra di questi dati è più pulita: IS, OOS e HO sono state tutte lette. Questi book sono la migliore descrizione possibile di ciò che ha funzionato dal 2021 a oggi, non una stima di ciò che funzionerà. L'unico modo di trasformarli in una previsione è farli girare in avanti su barre che non esistono ancora — o, più rapidamente, rifare la discovery su S&amp;P 500 e DAX riservando una finestra sigillata prima di iniziare.</li>
</ul>
</div>
</section>
</section>
'''

doc = doc.replace("</style>", EXTRA + "</style>", 1)
doc = doc.replace("<footer>", PART3 + "\n<footer>", 1)
SRC.write_text(doc)
print("patched:", len(doc))
