# FORGE — Codebase Specifications (v0.1.0)

> **Purpose of this directory**
> Files in `specs/` describe the behaviour that is actually implemented in the
> codebase, derived directly from source-code inspection.
> They serve as the reference for verifying how closely the functional analysis
> (`docs/README.md` and `docs/modules/`) matches the current state of the code.

---

## Structure

| File | Contents |
|---|---|
| `modulo_0_it.md` / `modulo_0_en.md` | Market Context — regime classifier (Module 0) |
| `modulo_1_it.md` / `modulo_1_en.md` | Event Discovery — 5-step pipeline (Module 1) |
| `modulo_2_it.md` / `modulo_2_en.md` | Alpha Discovery — predictivity measurement (Module 2) |

---

## Implementation status (v0.1.0)

| Module | Implemented | Notes |
|---|---|---|
| Module 0 — Market Context | ✅ Complete | `EMAProxyClassifier`, Hurst/OU auto-window |
| Module 1 — Event Discovery | ✅ Complete | Steps 0–5, walk-forward, `sql_expression` |
| Module 2 — Alpha Discovery | ✅ Complete | 8 steps, FDR, scoring, pattern family |
| Module 3 — Rule Discovery | ❌ Not implemented | Documented in functional analysis only |
| Module 4 — Rule Registry | ❌ Not implemented | Documented in functional analysis only |

---

## Alignment conventions

In each module spec, highlighted sections indicate:

- ✅ **Aligned** — implementation matches the functional analysis
- ➕ **Added in code** — present in the code, not documented in the functional analysis
- ⚠️ **Divergence** — behaviour differs from what the functional analysis describes
- ❌ **Missing in code** — documented in the functional analysis, not yet implemented

---

## Runtime dependencies

The package depends solely on `numpy` and `pandas`.
All statistical primitives (Spearman correlation, t-test, OU regression,
Benjamini-Hochberg FDR) are implemented in pure numpy —
no dependency on `scipy` or `statsmodels`.
