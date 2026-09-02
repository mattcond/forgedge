"""
ForgeEdge — Post-M2 Grade-Based Event Pairing (prototype for issue #254)
==========================================================================
Dataset: 1D OHLCV CSV (default: examples/data/AMZN_1D.csv)

`ANDComposer` normally runs inside Module 1 (Event Discovery), so the only
criteria available to decide which 1D events get paired/tripled are
structural (co-activation rate, episodic dispersion, distinct
``transform_key``) — M1 never sees the forward return.

This script prototypes an alternative: run the AND-composition *after* a
first pass of Module 2 (Alpha Discovery) instead of inside M1, and use the
A–D grade Alpha Discovery assigns each 1D event as the pairing criterion —

  1. M1 with ``max_and_components=1`` → pool of 1D events (returns-blind,
     M1's no-look-ahead invariant untouched).
  2. First AlphaDiscovery pass on the whole 1D pool → every promoted event
     gets a grade. Grade D and non-promoted events are discarded.
  3. Pair/triple events by grade: same grade (A-A, B-B, C-C), or a "root +
     lower-grade helper" scheme — grade A pairs with {A, B}, B with {B, C},
     C with {C, D} — every composite must contain at least one *root*-grade
     component. tpm and ``transform_key`` are still checked exactly (via a
     matmul co-activation matrix + the real ``ConsistencyGate``), but the
     grade — not co-activation — is what decides *which* events get tried
     together.
  4. Second AlphaDiscovery pass on the composed candidates: target,
     holding period and direction are re-derived from scratch for each
     composite (nothing is inherited from the 1D components), so there is
     no leakage between the single-event and the composite-event discovery.
  5. FastRotationNull + RuleDiscovery (M3) on the promoted composites, same
     as the rest of the pipeline — M3 remains the sole economic judge.
  6. Rank the resulting PARTIAL-EDGE/EDGE rules by a fold-variance-penalized
     score (``mean(fold_pf) - std(fold_pf)``, each fold PF capped to blunt
     forgedge's "zero losing trades" 9999.0 sentinel) instead of pooled WF
     PF, and show the best-ranked rule's full fold table plus its
     performance on a genuinely fresh, held-out window.

Why the stratified sampling
----------------------------
Root B's cross-tier population (B roots paired with C partners) vastly
outnumbers its same-tier population (B-B pairs). A single shared sampling
cap per root lets the abundant cross-tier candidates crowd out same-tier
ones, silently under-testing same-tier pairs relative to a same-grade-only
run — which then looks like widening the search *reduces* the number of
PARTIAL-EDGE/EDGE rules found, a sampling artifact rather than a real
regression. Sampling ``SAME_TIER_CAP``/``CROSS_TIER_CAP`` independently per
(root, same|cross) stratum avoids this.

Reference numbers
------------------
The docstrings below cite specific counts (5 / 76 / 122 PARTIAL-EDGE/EDGE,
p=0.0497, ...) from one documented run of this exact design on
``AMZN_1D.csv`` with the ``balanced`` preset — see issue #254. They are
context, not an assertion this script will reproduce them: ``and_composer``
has open bugs (#124, #226) whose fixes will shift these numbers, and a
single asset/timeframe is not proof the approach generalizes (also flagged
in #254 as a follow-up step).

This script is a prototype / research aid for #254, not part of the
forgedge public API — nothing here is imported by the library itself.

Esecuzione
----------
    python examples/post_m2_grade_pairing.py [path/to/data_1D.csv] \\
        [--preset balanced] [--ticker AMZN] [--timeframe 1D] [--months 6]
"""
from __future__ import annotations

import os
import sys

# Deve girare prima di ogni altro import — si veda la stessa nota in
# wf_period_reduction_test.py: PYTHONHASHSEED va fissato prima che
# l'interprete parta, quindi ci si ri-esegue con la variabile già in
# ambiente se non lo è già.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import dataclasses
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "src")
sys.path.insert(0, "examples")

import numpy as np
import pandas as pd

from wf_period_reduction_test import build_kpi, DEFAULT_CSV, replay_rule_on_window
from forgedge import MarketContext, EventDiscovery, AlphaDiscovery, RuleDiscovery, forge_preset
from forgedge.calibration import FastRotationNull
from forgedge.event_discovery.and_composer import _make_composed_event
from forgedge.event_discovery.consistency_gate import ConsistencyGate, _build_month_index, _count_by_month
from forgedge.event_discovery.models import RawEvent

REPO_ROOT = Path(__file__).resolve().parents[1]

TIERS = ["A", "B", "C", "D"]
ROOTS = ["A", "B", "C"]  # D non fa mai da radice: non esiste un grado sotto D

# Cap di campionamento INDIPENDENTI per strato (radice, same|cross) — non un
# cap unico condiviso. Si veda la nota "Why the stratified sampling" sopra.
SAME_TIER_CAP = 1000
CROSS_TIER_CAP = 400
TRIPLE_SAMPLE_PER_ROOT = 4000
TRIPLE_KEEP_PER_ROOT = 200

# PF per-fold cappato prima del ranking penalizzato per varianza, per non
# lasciare che un fold con pochissimi trade e nessuna perdita (sentinella
# 9999.0 di forgedge quando neg==0) distorca media e deviazione standard.
FOLD_PF_CAP = 10.0


def build_1d_pool(kpi_full: pd.DataFrame, disc_cfg):
    """M1 con ``max_and_components=1``: pool di eventi 1D, returns-blind."""
    disc_cfg_1d = dataclasses.replace(disc_cfg, max_and_components=1)
    enriched = MarketContext(kpi_full.copy()).run()
    ed = EventDiscovery(enriched.copy(), config=disc_cfg_1d)
    candidates = [c for c in ed.run() if " AND " not in c.expression]
    print(f"M1 (max_and_components=1): {len(candidates)} eventi 1D gated")
    return candidates, ed


def transform_key_of(candidate) -> str:
    """Replica ``RawEvent.transform_key`` su un ``EventCandidate`` a 1
    componente: stessa feature sorgente + stesso transform + stessi
    parametri temporali (window/lag) — soglia esclusa apposta, per
    escludere AND degeneri "stesso slot, soglia diversa"."""
    comp = candidate.components[0]
    temporal = {"window", "lag"}
    params_str = "_".join(
        f"{k}{v}" for k, v in sorted(comp.transform_params.items()) if k in temporal
    )
    return f"{comp.source_feature}__{comp.transform}__{params_str}"


def evaluate_and3(gate: ConsistencyGate, s1, s2, s3, month_index, n_total_months):
    active = (s1 & s2 & s3).astype(bool)
    period_counts = _count_by_month(active, month_index, n_total_months)
    return gate.evaluate(active, period_counts, n_total_months, month_index)


def lower_tier(t: str) -> str:
    return TIERS[TIERS.index(t) + 1]


def sample_pairs_and_triples(candidates_1d, ed, disc_cfg, grade_by_event_id, rng):
    """Costruisce coppie e triple radice+aiutante-di-grado-adiacente, con
    campionamento stratificato indipendente per (radice, same|cross)."""
    by_tier = {t: [i for i, c in enumerate(candidates_1d) if grade_by_event_id.get(c.event_id) == t] for t in TIERS}
    for t in TIERS:
        print(f"  grado {t}: {len(by_tier[t])} eventi")

    idx = ed.df.index
    n_bars = len(idx)
    timestamp_col = "open_dt" if "open_dt" in ed.df.columns else None
    timestamps = ed.df[timestamp_col] if timestamp_col else pd.Series(ed.df.index, index=ed.df.index)
    month_index, n_total_months = _build_month_index(timestamps)
    gate = ConsistencyGate(disc_cfg.gate_params)

    def series_of(i):
        return candidates_1d[i].event_series.reindex(idx).fillna(0).to_numpy() > 0

    full_series_mat = np.zeros((n_bars, len(candidates_1d)), dtype=np.int32)
    for j in range(len(candidates_1d)):
        full_series_mat[:, j] = series_of(j).astype(np.int32)

    keys = np.array([transform_key_of(c) for c in candidates_1d])

    # ── Coppie ──────────────────────────────────────────────────────────
    pair_kept = []  # (i, j, tpm, gate_result, root_tier, "same"|"cross")
    for R in ROOTS:
        Rp = lower_tier(R)
        root_idx, partner_idx = by_tier[R], by_tier[Rp]
        combined = root_idx + partner_idx
        if len(combined) < 2:
            continue
        # matmul esatto (float32: numpy non ha un percorso BLAS per matmul
        # intero — ~275-350x più lento — mentre float32 rappresenta ogni
        # intero fino a 2^24 senza perdita, ben oltre i conteggi possibili
        # qui) sulla sola sottomatrice radice+partner di questo strato.
        Mf = full_series_mat[:, combined].astype(np.float32)
        C = (Mf.T @ Mf).round().astype(np.int64)
        tpm = C.astype(np.float64) / n_total_months

        n_root = len(root_idx)
        same_pairs = [(a, b) for a in range(n_root) for b in range(a + 1, n_root)]
        cross_pairs = [(a, b) for a in range(n_root) for b in range(n_root, len(combined))]

        for stratum_name, pairs_list, cap in (("same", same_pairs, SAME_TIER_CAP),
                                                ("cross", cross_pairs, CROSS_TIER_CAP)):
            valid = []
            for a, b in pairs_list:
                gi, gj = combined[a], combined[b]
                if keys[gi] == keys[gj]:
                    continue
                t = tpm[a, b]
                if t < disc_cfg.gate_params.min_tpm:
                    continue
                valid.append((a, b, t))

            sample_n = min(cap, len(valid))
            if sample_n == 0:
                print(f"radice {R} strato {stratum_name}: popolazione valida=0, salto")
                continue
            sample_pos = rng.choice(len(valid), size=sample_n, replace=False)

            kept_this_stratum = 0
            for si in sample_pos:
                a, b, t = valid[si]
                gi, gj = combined[a], combined[b]
                result = gate.evaluate_and(series_of(gi), series_of(gj), month_index, n_total_months)
                if result.passed:
                    pair_kept.append((gi, gj, float(t), result, R, stratum_name))
                    kept_this_stratum += 1
            print(f"radice {R} strato {stratum_name}: popolazione valida={len(valid)}  "
                  f"campionate={sample_n}  passano il gate={kept_this_stratum}")

    print(f"\nTotale coppie che passano il gate esatto: {len(pair_kept)}")
    print(f"  per radice/tipo: {Counter((p[4], p[5]) for p in pair_kept)}")

    # ── Triple: almeno un membro deve appartenere alla radice ──────────
    triple_kept = []  # (i, j, k, gate_result, root_tier, "same"|"cross")
    for R in ROOTS:
        Rp = lower_tier(R)
        root_idx, partner_idx = by_tier[R], by_tier[Rp]
        if not root_idx:
            continue
        found_this_root, seen_trios, attempts = [], set(), 0
        while attempts < TRIPLE_SAMPLE_PER_ROOT and len(found_this_root) < TRIPLE_KEEP_PER_ROOT:
            attempts += 1
            max_root = min(3, len(root_idx))
            n_root_pick = rng.integers(1, max_root + 1)
            n_partner_pick = 3 - n_root_pick
            if n_partner_pick > len(partner_idx):
                n_partner_pick = len(partner_idx)
                n_root_pick = 3 - n_partner_pick
                if n_root_pick > len(root_idx):
                    continue
            picks_root = rng.choice(root_idx, size=n_root_pick, replace=False)
            picks_partner = rng.choice(partner_idx, size=n_partner_pick, replace=False) if n_partner_pick > 0 else []
            trio = list(picks_root) + list(picks_partner)
            if len(set(trio)) != 3:
                continue
            gi, gj, gk = sorted(trio)
            trio_key = (gi, gj, gk)
            if trio_key in seen_trios:
                continue
            seen_trios.add(trio_key)
            if len({keys[gi], keys[gj], keys[gk]}) != 3:
                continue
            result = evaluate_and3(gate, series_of(gi), series_of(gj), series_of(gk), month_index, n_total_months)
            subkind = "same" if n_partner_pick == 0 else "cross"
            if result.passed:
                found_this_root.append((gi, gj, gk, result, R, subkind))
        triple_kept.extend(found_this_root)
        print(f"radice {R} (triple): tentativi={attempts}, tenute={len(found_this_root)}")

    print(f"\nTotale triple che passano il gate esatto: {len(triple_kept)}")
    return pair_kept, triple_kept, timestamps


def build_composed_candidates(pair_kept, triple_kept, candidates_1d, ed, timestamps):
    """Veri ``EventCandidate`` (via ``_make_composed_event`` +
    ``EventDiscovery._to_candidate``), non ``CustomEvent``: solo così
    ``EventCandidate.apply()`` può ricostruire il segnale dai componenti
    sorgente invece di cercare una colonna derivata che non esiste in
    nessun frame persistito."""
    def raw_of(i):
        return RawEvent(series=candidates_1d[i].event_series, component=candidates_1d[i].components[0])

    composed, origin, root_event_idx = [], [], []

    for gi, gj, t, gr, R, kind in pair_kept:
        and_series = (
            candidates_1d[gi].event_series.fillna(0).astype(bool)
            & candidates_1d[gj].event_series.fillna(0).astype(bool)
        ).astype(float)
        raw = _make_composed_event(raw_of(gi), raw_of(gj), and_series, gr)
        composed.append(ed._to_candidate(raw, len(composed), timestamps))
        origin.append(("coppia", R, kind))
        root_event_idx.append(gi)

    for gi, gj, gk, gr, R, subkind in triple_kept:
        and_series = (
            candidates_1d[gi].event_series.fillna(0).astype(bool)
            & candidates_1d[gj].event_series.fillna(0).astype(bool)
            & candidates_1d[gk].event_series.fillna(0).astype(bool)
        ).astype(float)
        raw = _make_composed_event(raw_of(gi), raw_of(gj), and_series, gr, third=raw_of(gk))
        composed.append(ed._to_candidate(raw, len(composed), timestamps))
        origin.append(("tripla", R, subkind))
        root_event_idx.append(gi)

    return composed, origin, root_event_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=str(DEFAULT_CSV), help="CSV OHLCV 1D (default: examples/data/AMZN_1D.csv)")
    ap.add_argument("--preset", default="balanced", choices=["sniper", "balanced", "sweep", "burst"])
    ap.add_argument("--ticker", default="AMZN")
    ap.add_argument("--timeframe", default="1D")
    ap.add_argument("--months", type=int, default=6, help="mesi finali da usare come finestra fresh per la regola migliore (default 6)")
    ap.add_argument("--seed", type=int, default=0, help="seed per il campionamento stratificato (default 0)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    kpi_full = build_kpi(csv_path)
    end_full = kpi_full["open_dt"].max()
    cutoff = end_full - pd.DateOffset(months=args.months)
    disc_cfg, alpha_cfg, rd_cfg = forge_preset(args.preset, timeframe=args.timeframe, asset=args.ticker)
    rng = np.random.default_rng(args.seed)

    print(f"Dataset: {csv_path}")
    print(f"Storico: {kpi_full['open_dt'].min().date()} -> {end_full.date()}  ({len(kpi_full)} barre)")
    print(f"Finestra fresh per la regola migliore: ultimi {args.months} mesi (dal {cutoff.date()})")
    print()

    candidates_1d, ed = build_1d_pool(kpi_full, disc_cfg)

    print("\nPrima passata AlphaDiscovery sul pool 1D (per ottenere i gradi)…")
    ad1 = AlphaDiscovery(ed.df, candidates_1d, alpha_cfg)
    contracts1 = ad1.run()
    promoted1 = ad1.promoted_contracts()
    grade_by_event_id = {c.event_candidate_id: c.alpha_score.grade for c in promoted1 if c.alpha_score is not None}
    contract1_by_event_id = {c.event_candidate_id: c for c in promoted1}
    print(f"1a passata: {len(contracts1)} valutati, {len(promoted1)} promossi, "
          f"gradi: {dict(Counter(grade_by_event_id.values()))}")

    print("\nCampionamento stratificato di coppie/triple per grado…")
    pair_kept, triple_kept, timestamps = sample_pairs_and_triples(candidates_1d, ed, disc_cfg, grade_by_event_id, rng)
    if not pair_kept and not triple_kept:
        print("Nessuna coppia/tripla sopravvive al gate. Interrompo.")
        return

    composed_all, origin, root_event_idx = build_composed_candidates(pair_kept, triple_kept, candidates_1d, ed, timestamps)
    print(f"\n{len(composed_all)} EventCandidate composti totali (coppie+triple)")

    print("\nSeconda passata AlphaDiscovery sui composti — target/h/lato riderivati da zero…")
    ad2 = AlphaDiscovery(ed.df, composed_all, alpha_cfg)
    contracts2 = ad2.run()
    promoted2 = ad2.promoted_contracts()
    print(f"2a passata: {len(contracts2)} valutati, {len(promoted2)} promossi")

    if promoted2:
        print("\nFastRotationNull…")
        cal_report = FastRotationNull(ed.df, composed_all, alpha_cfg).run(promoted2, alpha=0.05)
        print(f"Fast rotation null — search p={cal_report.tippett_p:.4f}, "
              f"{len(cal_report.survivors)}/{len(promoted2)} sopra la null bar")

    cand_by_id = {c.event_id: c for c in composed_all}
    origin_by_id = {composed_all[k].event_id: origin[k] for k in range(len(composed_all))}
    root_by_id = {composed_all[k].event_id: root_event_idx[k] for k in range(len(composed_all))}

    rule_responses = []
    for contract in promoted2:
        cand = cand_by_id.get(contract.event_candidate_id)
        if cand is None:
            continue
        rule_responses.append((contract, RuleDiscovery(ed.df, contract, cand, config=rd_cfg).run()))

    edges = [(c, r) for c, r in rule_responses if r.verdict in ("PARTIAL-EDGE", "EDGE")]
    print()
    print("=" * 100)
    print(f"RISULTATO — {len(edges)}/{len(contracts2)} PARTIAL-EDGE/EDGE (pairing/tripling per grado adiacente)")
    print("=" * 100)
    for contract, resp in edges:
        wf = resp.walk_forward
        kind, R, subkind = origin_by_id[contract.event_candidate_id]
        wf_pf = wf.oos_summary.profit_factor if wf else None
        print(f"\nalpha_id={contract.alpha_id}  {kind} ({subkind}, radice={R})  verdetto={resp.verdict}")
        print(f"  {contract.event_expression}")
        print(f"  IS pf={resp.in_sample_summary.profit_factor:.4f}  WF pf={wf_pf}")

    if not edges:
        return

    # ── Radice standalone vs composto, solo per gli abbinamenti incrociati
    # con esito PARTIAL-EDGE/EDGE: un evento di grado più basso abbinato a
    # una radice migliore ne migliora davvero le performance? ──────────────
    root_ids_used = sorted({root_by_id[c.event_candidate_id] for c, r in edges})
    print(f"\nBacktest standalone (M3) di {len(root_ids_used)} eventi radice usati, per confronto…")
    root_wf_pf = {}
    for gi in root_ids_used:
        contract_root = contract1_by_event_id.get(candidates_1d[gi].event_id)
        if contract_root is None:
            continue
        resp_root = RuleDiscovery(ed.df, contract_root, candidates_1d[gi], config=rd_cfg).run()
        wf = resp_root.walk_forward
        root_wf_pf[gi] = wf.oos_summary.profit_factor if wf else None

    cross_edges = [(c, r) for c, r in edges if origin_by_id[c.event_candidate_id][2] == "cross"]
    deltas = []
    for contract, resp in cross_edges:
        gi = root_by_id[contract.event_candidate_id]
        root_pf = root_wf_pf.get(gi)
        wf = resp.walk_forward
        wf_pf = wf.oos_summary.profit_factor if wf else None
        if wf_pf is not None and root_pf is not None:
            deltas.append(wf_pf - root_pf)
    print()
    print("=" * 78)
    print("RADICE vs COMPOSTO — solo abbinamenti incrociati (radice + grado inferiore)")
    print("=" * 78)
    if deltas:
        n_improved = sum(1 for d in deltas if d > 0)
        print(f"abbinamenti incrociati PARTIAL-EDGE/EDGE con delta calcolabile: {len(deltas)}  "
              f"migliorati rispetto alla radice: {n_improved}/{len(deltas)}")
        print(f"delta medio (WF pf composto - WF pf radice): {np.mean(deltas):+.4f}  mediano: {np.median(deltas):+.4f}")
    else:
        print("Nessun abbinamento incrociato con esito PARTIAL-EDGE/EDGE e delta calcolabile.")

    # ── Ranking penalizzato per varianza tra fold (vedi issue #253) ────────
    print()
    print("=" * 100)
    print("RANKING PENALIZZATO PER VARIANZA TRA FOLD (score = media_fold_pf − std_fold_pf)")
    print("=" * 100)
    scored = []
    for contract, resp in edges:
        wf = resp.walk_forward
        if wf is None or not wf.splits:
            continue
        fold_pfs = [min(sp.test_summary.profit_factor, FOLD_PF_CAP) for sp in wf.splits]
        mean_pf = float(np.mean(fold_pfs))
        std_pf = float(np.std(fold_pfs, ddof=1)) if len(fold_pfs) > 1 else 0.0
        scored.append((contract, resp, fold_pfs, mean_pf, std_pf, mean_pf - std_pf))

    if not scored:
        print("Nessun edge con walk_forward calcolato.")
        return

    by_pooled = sorted(scored, key=lambda t: -t[1].walk_forward.oos_summary.profit_factor)
    by_score = sorted(scored, key=lambda t: -t[5])

    print("\nTop 5 per PF WF pooled (ranking senza penalità di varianza):")
    for contract, resp, fold_pfs, mean_pf, std_pf, score in by_pooled[:5]:
        pooled = resp.walk_forward.oos_summary.profit_factor
        print(f"  {contract.alpha_id}  pooled={pooled:.4f}  fold={['%.2f' % p for p in fold_pfs]}  "
              f"media={mean_pf:.3f}  std={std_pf:.3f}  score={score:.3f}")

    print("\nTop 5 per score penalizzato (media_fold_pf − std_fold_pf):")
    for contract, resp, fold_pfs, mean_pf, std_pf, score in by_score[:5]:
        pooled = resp.walk_forward.oos_summary.profit_factor
        print(f"  {contract.alpha_id}  pooled={pooled:.4f}  fold={['%.2f' % p for p in fold_pfs]}  "
              f"media={mean_pf:.3f}  std={std_pf:.3f}  score={score:.3f}")

    best_contract, best_resp, best_folds, best_mean, best_std, best_score = by_score[0]
    print()
    print("=" * 100)
    print("MIGLIORE PER SCORE PENALIZZATO — tabella WF + finestra fresh")
    print("=" * 100)
    kind, R, subkind = origin_by_id[best_contract.event_candidate_id]
    wf = best_resp.walk_forward
    best_cand = cand_by_id[best_contract.event_candidate_id]
    params = best_resp.validated_rule.params

    print(f"alpha_id={best_contract.alpha_id}  {kind} ({subkind}, radice={R})  verdetto={best_resp.verdict}")
    print(f"espressione: {best_contract.event_expression}")
    print(f"score penalizzato: {best_score:.4f}  (media_fold={best_mean:.4f}, std_fold={best_std:.4f})")
    print()
    print(f"PF IS: {best_resp.in_sample_summary.profit_factor:.4f}  (n={best_resp.in_sample_summary.total_trades})")
    for i, sp in enumerate(wf.splits, start=1):
        print(f"PF WF {i}: {sp.test_summary.profit_factor:.4f}  (n={sp.test_summary.total_trades})")
    print(f"PF WF pooled: {wf.oos_summary.profit_factor:.4f}  (n={wf.oos_summary.total_trades}, consistency={wf.consistency})")

    replay_summary, _ = replay_rule_on_window(ed.df, best_cand, params, cutoff)
    print(f"PF Fresh (ultimi {args.months} mesi, {cutoff.date()} -> {end_full.date()}): "
          f"{replay_summary.profit_factor:.4f}  (n={replay_summary.total_trades})")


if __name__ == "__main__":
    main()
