#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
🌿 QAOA BIOMASSA — MEGA-VARREDURA GLOBAL JUSTA (k=4..12) — v2.0 EMS-READY
================================================================================
Objetivo: Comparar QAOA (simulado, AerSimulator) contra SA, GA, Exact Enumeration
e baselines de filter (RF_topK, ET_topK) na seleção de features para estimativa
de biomassa no Cerrado brasileiro, usando uma formulação QUBO única por k.

NOTA IMPORTANTE — CV interno vs CV final:
  O cv_oof_grouped() aqui usa RF+ET+meta-RF (ensemble leve, n_estimators=90)
  exclusivamente para RANQUEAR subsets candidatos no QUBO. A validação de
  desempenho preditivo final é feita no Script 2 (ensemble completo RF+ET+XGB
  +LGBM+SVR+Ridge, nested CV com GRID blocks, LO-MRO). Essa distinção é
  intencional: usar o ensemble completo internamente aqui tornaria o Script 1
  computacionalmente inviável (k=4..12 × 9 métodos × 12 blocos × n_estimators=300+).

Adições v2.0 para Environmental Modelling & Software (EMS):
  [G] QUBO landscape analysis pós-varredura:
      - hash (frozenset) do subset de cada método por k
      - Jaccard entre pares de métodos
      - energy gap: custo QUBO do 1º mínimo vs 2º mínimo de mesma cardinalidade
      - diagnóstico automático: "unique_minimum" vs "degenerate" vs "converged_heuristics"
  [H] Todos os 3 warm-starts do QAOA logados individualmente (variância entre runs)
  [I] Stats do circuito QAOA (depth, n_gates) após transpilação — salvo no JSON
  [J] Feature diagnostics (RF/ET importances + correlações + marginal R²) salvo em CSV
  [K] Ranking final completo (todos k × métodos) salvo em CSV

Fixes definitivos v1.3 (mantidos):
  [A] JSON crash-proof: converte recursivamente np.int64/float64/ndarray
  [B] CHECKPOINT/RESUME por k
  [C] penalty_lambda robusto com busca adaptativa
  [D] seeds reprodutíveis por k e por método
  [E] fail-safe QAOA
  [F] logs de tempo por k
================================================================================
"""

import os
import re
import json
import time
import hashlib
import itertools
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from scipy.optimize import differential_evolution

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
FEATURES_CSV = "/mnt/e/PROJETOS/biomassa_quantum/results/goias_df_features_buffer50m_v3_2018.csv"
BIOMASSA_CSV = "/mnt/e/PROJETOS/biomassa_quantum/results/biomassa_por_UA_corrigido.csv"

OUTPUT_DIR   = "/mnt/e/PROJETOS/biomassa_quantum/results/qaoa_ibm_real/"
OUT_JSON     = "qaoa_mega_varredura_justa_resultados.json"
OUT_JSON_TMP = "qaoa_mega_varredura_justa_resultados.PARTIAL.json"  # checkpoint
OUT_CSV_RANKING    = "script1_ranking_completo.csv"        # [J] ranking k × método
OUT_CSV_FEATURES   = "script1_feature_diagnostics.csv"     # [K] importâncias + correlações
OUT_CSV_LANDSCAPE  = "script1_qubo_landscape.csv"          # [G] energy gap + Jaccard

SEED         = 42
TARGET_COL   = "Biomassa_Mg_ha"
MAPBIOMAS_COL= "mapbiomas_2018"
ID_UA_COL    = "UA"

CLASSES_VEGETACAO = {3,4,5,6,11,12,29,32,49,50}

# Allowlist (mesma filosofia do workflow)
BAND_NAMES = [
    'B2_seca','B3_seca','B4_seca','B8_seca','B11_seca','B12_seca',
    'NDVI_seca','NDWI_seca','NBR_seca','NDVI_RE_seca','MSI_seca','EVI_seca',
    'elevation','slope','VV_dB','VH_dB','HV_dB',
    'mapbiomas_2018','clay_pct','canopy_height','canopy_height_sd'
]

# FIREWALL anti leakage
LEAKY_PATTERNS = re.compile(
    r"(?:^|_)(?:bio|biom|agb|carbon|carb|estoque|stock|mg_ha|mg_h|biomassa)(?:_|$)|"
    r"(?:^|_)(?:n_arvor|narvor|arvore|arvores|dap|dbh|ht|altura|height_field|volume|basal)(?:_|$)|"
    r"(?:^|_)(?:Biomassa|BIO|AGB)(?:_|$)",
    flags=re.IGNORECASE
)

# QUBO pool size
N_FEATURES = 15
K_RANGE    = list(range(4, 13))

# CV espacial (grid blocks)
GRID_BLOCKS_NX, GRID_BLOCKS_NY = 4, 3  # 12 blocks

# QAOA
QAOA_LAYERS = 2
SHOTS_FINAL = 8192
SHOTS_OBJ   = 1024

# Differential Evolution (tempo!)
DE_MAXITER  = 12
DE_POPSIZE  = 10
DE_POLISH   = False

# Penalidade: busca robusta
PENALTY_INITIAL_GRID = (0.005, 0.40, 80)   # (min, max, n_grid)
PENALTY_REFINE_STEPS = 30                  # refinamento local
PENALTY_MAX_TRIES    = 6                   # expandir range se não bater cardinalidade

# Pesos do QUBO
QUBO_WEIGHTS = {
    'w_rf': 0.5,
    'w_marg': 0.4,
    'alpha': 0.4,
    'lambda_vif': 0.0,
    'lambda_corr': 0.1,
    'w_synergy': 0.0
}

# =============================================================================
# PRINT HEADER
# =============================================================================
print("=" * 92)
print(" 🌿 QAOA BIOMASSA — MEGA-VARREDURA GLOBAL JUSTA (k=4..12) — v2.0 EMS-READY")
print("=" * 92)

# =============================================================================
# JSON SAFE (definitivo)
# =============================================================================
def to_py(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_py(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kk = k
            if not isinstance(kk, (str, int, float, bool)):
                kk = str(kk)
            out[str(kk)] = to_py(v)
        return out
    return str(obj)

def json_dump_safe(payload: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_py(payload), f, indent=2, ensure_ascii=False)

def try_load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def subset_hash(feature_list: List[str]) -> str:
    """Hash determinístico de um subset de features (frozenset → SHA256[:16])."""
    key = "|".join(sorted(feature_list))
    return hashlib.sha256(key.encode()).hexdigest()[:16]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0

# =============================================================================
# HELPERS
# =============================================================================
def robust_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=";")
    df.columns = df.columns.astype(str).str.strip()
    return df

def grid_partition(lon: np.ndarray, lat: np.ndarray, nx: int, ny: int) -> np.ndarray:
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    eps = 1e-12
    lon_bins = np.linspace(lon_min, lon_max + eps, nx + 1)
    lat_bins = np.linspace(lat_min, lat_max + eps, ny + 1)
    ix = np.clip(np.digitize(lon, lon_bins) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(lat, lat_bins) - 1, 0, ny - 1)
    return (iy * nx + ix).astype(int)

def qubo_cost_vectorized(X_matrix: np.ndarray, linear: np.ndarray, quad: np.ndarray) -> np.ndarray:
    X = np.asarray(X_matrix, dtype=float)
    linear = np.asarray(linear, dtype=float)
    quad = np.asarray(quad, dtype=float)
    n = len(linear)
    cost = X @ linear
    for i in range(n):
        for j in range(i+1, n):
            cost += quad[i, j] * X[:, i] * X[:, j]
    return cost

def compute_metrics_from_oof(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias": float(np.mean(y_pred - y_true))
    }

def cv_oof_grouped(feature_indices: List[int], X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   seed: int, n_estimators: int = 90) -> Tuple[float, np.ndarray]:
    if feature_indices is None or len(feature_indices) == 0:
        return float("-inf"), np.full(len(y), np.nan)

    X_sub = X[:, feature_indices]
    y = np.asarray(y, float)
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    if len(uniq) < 3:
        raise RuntimeError(f"Poucos blocos para GroupKFold: {len(uniq)}")

    gkf = GroupKFold(n_splits=len(uniq))
    pred_sum = np.zeros(len(y), dtype=float)
    pred_cnt = np.zeros(len(y), dtype=int)

    for tr, te in gkf.split(X_sub, y, groups=groups):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_sub[tr])
        Xte = scaler.transform(X_sub[te])

        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(Xtr, y[tr])
        et = ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed+1, n_jobs=-1).fit(Xtr, y[tr])

        meta_tr = np.column_stack([rf.predict(Xtr), et.predict(Xtr)])
        meta_te = np.column_stack([rf.predict(Xte), et.predict(Xte)])

        stk = RandomForestRegressor(n_estimators=60, random_state=seed+2, n_jobs=-1).fit(meta_tr, y[tr])

        pred_sum[te] += stk.predict(meta_te)
        pred_cnt[te] += 1

    y_oof = pred_sum / np.maximum(pred_cnt, 1)
    r2 = r2_score(y, y_oof)
    return float(r2), y_oof

def topk_from_scores(scores: np.ndarray, k: int) -> List[int]:
    s = np.asarray(scores, float)
    return list(np.argsort(s)[::-1][:k])

# =============================================================================
# QUBO BUILDER
# =============================================================================
def build_qubo_for_k(k: int,
                     rf_imp: np.ndarray,
                     marg_r2: np.ndarray,
                     corr_tgt: np.ndarray,
                     vif: np.ndarray,
                     spearman: np.ndarray,
                     syn: np.ndarray,
                     weights: Dict[str, float],
                     penalty_lambda: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    n = len(rf_imp)
    linear = np.zeros(n, dtype=float)
    for i in range(n):
        q = weights['w_rf']*rf_imp[i] + weights['w_marg']*marg_r2[i] + weights['alpha']*corr_tgt[i]
        linear[i] = -q + penalty_lambda * (1.0 - 2.0 * k)

    quad = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            quad_ij = 0.0
            quad_ij += weights['lambda_vif'] * vif[i, j]
            quad_ij += weights['lambda_corr'] * spearman[i, j]
            quad_ij -= weights['w_synergy'] * syn[i, j]
            quad_ij += 2.0 * penalty_lambda
            quad[i, j] = quad[j, i] = quad_ij

    scale = max(float(np.max(np.abs(linear))), float(np.max(np.abs(quad))), 1.0)
    return linear, quad, linear/scale, quad/scale, scale

def best_solution_size_for_lambda(all_x: np.ndarray, lin: np.ndarray, quad: np.ndarray) -> Tuple[int, int, float]:
    costs = qubo_cost_vectorized(all_x, lin, quad)
    idx = int(np.argmin(costs))
    size = int(np.sum(all_x[idx]))
    return size, idx, float(costs[idx])

def choose_penalty_lambda_robust(k: int,
                                all_x: np.ndarray,
                                rf_imp: np.ndarray,
                                marg_r2: np.ndarray,
                                corr_tgt: np.ndarray,
                                vif: np.ndarray,
                                spearman: np.ndarray,
                                syn: np.ndarray,
                                weights: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    """
    Busca robusta:
    - grid amplo
    - se não bater size==k, expande range e refina em torno do melhor
    - retorna lambda e diagnóstico
    """
    lo, hi, n_grid = PENALTY_INITIAL_GRID
    best = {"diff": 10**9, "pl": None, "size": None, "idx": None, "cost": None}

    diag = {"tries": []}

    for t in range(PENALTY_MAX_TRIES):
        grid = np.linspace(lo, hi, n_grid)
        trial = []

        for pl in grid:
            lin, quad, _, _, _ = build_qubo_for_k(k, rf_imp, marg_r2, corr_tgt, vif, spearman, syn, weights, float(pl))
            size, idx, cost = best_solution_size_for_lambda(all_x, lin, quad)
            diff = abs(size - k)

            trial.append((float(pl), int(size), int(idx), float(cost), int(diff)))

            if diff < best["diff"]:
                best.update({"diff": diff, "pl": float(pl), "size": int(size), "idx": int(idx), "cost": float(cost)})

            if diff == 0:
                diag["tries"].append({
                    "try_id": t, "lo": float(lo), "hi": float(hi), "n_grid": int(n_grid),
                    "found_exact": True, "pl": float(pl), "size": int(size), "idx": int(idx), "cost": float(cost)
                })
                return float(pl), diag

        # não achou exato: refina entorno do melhor pl
        pl_center = float(best["pl"])
        span = (hi - lo) * 0.20  # 20% do range atual
        lo2 = max(1e-6, pl_center - span)
        hi2 = pl_center + span

        # refinamento
        refine = np.linspace(lo2, hi2, PENALTY_REFINE_STEPS)
        for pl in refine:
            lin, quad, _, _, _ = build_qubo_for_k(k, rf_imp, marg_r2, corr_tgt, vif, spearman, syn, weights, float(pl))
            size, idx, cost = best_solution_size_for_lambda(all_x, lin, quad)
            diff = abs(size - k)
            if diff < best["diff"]:
                best.update({"diff": diff, "pl": float(pl), "size": int(size), "idx": int(idx), "cost": float(cost)})
            if diff == 0:
                diag["tries"].append({
                    "try_id": t, "lo": float(lo), "hi": float(hi), "n_grid": int(n_grid),
                    "found_exact": True, "pl": float(pl), "size": int(size), "idx": int(idx), "cost": float(cost),
                    "refined": True
                })
                return float(pl), diag

        # expande range e tenta de novo
        diag["tries"].append({
            "try_id": t,
            "lo": float(lo), "hi": float(hi), "n_grid": int(n_grid),
            "found_exact": False,
            "best_so_far": best.copy()
        })
        # expande e desloca: se size<k, aumentar lambda tende a aumentar cardinalidade;
        # se size>k, diminuir lambda tende a reduzir cardinalidade. Ajuste leve.
        if best["size"] is not None:
            if best["size"] < k:
                lo, hi = lo, hi * 1.8
            elif best["size"] > k:
                lo, hi = lo * 0.6, hi
            else:
                lo, hi = lo * 0.8, hi * 1.2
        else:
            lo, hi = lo * 0.8, hi * 1.5

        lo = max(1e-6, float(lo))
        hi = max(lo * 1.1, float(hi))

    # fallback: retorna o melhor aproximado
    diag["fallback_used"] = True
    diag["best_final"] = best.copy()
    return float(best["pl"]), diag

# =============================================================================
# CLASSICAL SOLVERS
# =============================================================================
def sa_qubo(linear: np.ndarray, quad: np.ndarray, seed: int, steps: int = 6500) -> Tuple[List[int], float]:
    rng = np.random.default_rng(seed)
    n = len(linear)
    current = rng.integers(2, size=n)
    current_cost = qubo_cost_vectorized(current.reshape(1, -1), linear, quad)[0]
    best, best_cost = current.copy(), float(current_cost)

    T, cooling, T_min = 1.0, 0.992, 1e-4
    for _ in range(steps):
        if T < T_min:
            break
        neighbor = current.copy()
        flip_idx = rng.integers(n)
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        neighbor_cost = qubo_cost_vectorized(neighbor.reshape(1, -1), linear, quad)[0]
        delta = neighbor_cost - current_cost
        if (delta < 0) or (rng.random() < np.exp(-delta / max(T, 1e-12))):
            current, current_cost = neighbor, neighbor_cost
            if current_cost < best_cost:
                best, best_cost = current.copy(), float(current_cost)
        T *= cooling

    idx = np.where(best == 1)[0].tolist()
    return idx, float(best_cost)

def ga_qubo(linear: np.ndarray, quad: np.ndarray, seed: int,
            pop_size: int = 60, gens: int = 140, mut_rate: float = 0.18) -> Tuple[List[int], float]:
    rng = np.random.default_rng(seed)
    n = len(linear)
    pop = rng.integers(2, size=(pop_size, n))
    best_ever = pop[0].copy()
    best_cost = float("inf")

    for _ in range(gens):
        costs = qubo_cost_vectorized(pop, linear, quad)
        order = np.argsort(costs)
        if float(costs[order[0]]) < best_cost:
            best_cost = float(costs[order[0]])
            best_ever = pop[order[0]].copy()

        elite = pop[order[:6]].copy()
        new_pop = [e.copy() for e in elite]

        while len(new_pop) < pop_size:
            t1 = pop[rng.choice(pop_size, 3, replace=False)]
            p1 = t1[np.argmin(qubo_cost_vectorized(t1, linear, quad))]
            t2 = pop[rng.choice(pop_size, 3, replace=False)]
            p2 = t2[np.argmin(qubo_cost_vectorized(t2, linear, quad))]

            mask = rng.integers(2, size=n).astype(bool)
            child = np.where(mask, p1, p2).copy()

            if rng.random() < mut_rate:
                flip = rng.integers(n)
                child[flip] = 1 - child[flip]

            new_pop.append(child)

        pop = np.asarray(new_pop, dtype=int)

    idx = np.where(best_ever == 1)[0].tolist()
    return idx, float(best_cost)

# =============================================================================
# QAOA
# =============================================================================
def build_qaoa_ws(n: int, h_lin: np.ndarray, J_quad: np.ndarray,
                 gammas: List[float], betas: List[float], p: int, ws_idx: List[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n)

    for i in range(n):
        if i in ws_idx:
            qc.x(i)
            qc.ry(-np.pi/4, i)
        else:
            qc.ry(np.pi/4, i)

    for layer in range(p):
        gamma, beta = gammas[layer], betas[layer]

        for i in range(n):
            if abs(h_lin[i]) > 1e-8:
                qc.rz(2 * gamma * h_lin[i], i)

        quad_terms = [(i, j, J_quad[i, j]) for i in range(n) for j in range(i+1, n) if abs(J_quad[i, j]) > 1e-8]
        quad_terms.sort(key=lambda x: abs(x[2]), reverse=True)

        for i, j, q in quad_terms[:8]:
            qc.cx(i, j)
            qc.rz(2 * gamma * q, j)
            qc.cx(i, j)

        for i in range(n):
            qc.rx(2 * beta, i)

    return qc

def sample_best_bitstring_by_qubo(counts: Dict[str, int], linear: np.ndarray, quad: np.ndarray) -> Tuple[List[int], float]:
    best_idx = []
    best_cost = float("inf")
    for bs, cnt in counts.items():
        x_vec = np.array([int(b) for b in bs[::-1]], dtype=int)
        if int(x_vec.sum()) == 0:
            continue
        c = qubo_cost_vectorized(x_vec.reshape(1, -1), linear, quad)[0]
        if float(c) < best_cost:
            best_cost = float(c)
            best_idx = np.where(x_vec == 1)[0].tolist()
    return best_idx, float(best_cost)

# =============================================================================
# LOAD DATA + FIREWALL
# =============================================================================
print("\n📊 Carregando e processando dados (veg-only + coords + firewall)...")

df_feat = robust_read_csv(FEATURES_CSV)
df_bio  = robust_read_csv(BIOMASSA_CSV)

df = df_feat.merge(df_bio[[ID_UA_COL, TARGET_COL]], on=ID_UA_COL, how="inner")

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df["mb_int"]   = pd.to_numeric(df[MAPBIOMAS_COL], errors="coerce").round().fillna(-1).astype(int)
df["lon"]      = pd.to_numeric(df.get("lon_pc", np.nan), errors="coerce")
df["lat"]      = pd.to_numeric(df.get("lat_pc", np.nan), errors="coerce")

df = df[
    (df[TARGET_COL] > 0) &
    (df[TARGET_COL] < 300) &
    df["lon"].notna() & df["lat"].notna() &
    df["mb_int"].isin(CLASSES_VEGETACAO)
].copy().reset_index(drop=True)

allowed = [c for c in BAND_NAMES if c in df.columns]
feature_pool = [c for c in allowed if not LEAKY_PATTERNS.search(c)]

suspicious = [c for c in df.columns if LEAKY_PATTERNS.search(c)]
if suspicious:
    print("\n⚠️ Colunas suspeitas presentes no CSV (NÃO usadas aqui):")
    for c in suspicious[:60]:
        print(f"   - {c}")
    if len(suspicious) > 60:
        print(f"   ... (+{len(suspicious)-60} outras)")

print(f"\n   -> UAs válidas  : {len(df)} | y̅={df[TARGET_COL].mean():.1f} ± {df[TARGET_COL].std():.1f} Mg/ha")
print(f"   -> Allowlist existentes no CSV: {len(allowed)}")
print(f"   -> Feature pool final (exógeno + firewall): {len(feature_pool)}")

if len(feature_pool) < N_FEATURES:
    raise RuntimeError(f"Feature pool ({len(feature_pool)}) < N_FEATURES={N_FEATURES}. Ajuste N_FEATURES/allowlist.")

preferred = [
    'canopy_height', 'canopy_height_sd', 'clay_pct', 'elevation', 'slope',
    'NDVI_seca', 'NDWI_seca', 'NBR_seca', 'EVI_seca', 'NDVI_RE_seca',
    'HV_dB', 'VV_dB', 'VH_dB', 'B8_seca', 'mapbiomas_2018'
]
FEATURE_NAMES = [c for c in preferred if c in feature_pool]
if len(FEATURE_NAMES) < N_FEATURES:
    remaining = [c for c in feature_pool if c not in FEATURE_NAMES]
    FEATURE_NAMES = FEATURE_NAMES + remaining[:(N_FEATURES - len(FEATURE_NAMES))]
FEATURE_NAMES = FEATURE_NAMES[:N_FEATURES]

df = df.dropna(subset=FEATURE_NAMES + [TARGET_COL]).reset_index(drop=True)

print(f"\n✅ FEATURE_NAMES (n={len(FEATURE_NAMES)}): {FEATURE_NAMES}")

X_all = df[FEATURE_NAMES].values.astype(float)
y     = df[TARGET_COL].values.astype(float)

groups = grid_partition(df["lon"].values, df["lat"].values, GRID_BLOCKS_NX, GRID_BLOCKS_NY)
df["spatial_block_cv"] = groups

counts_blocks = pd.Series(groups).value_counts().sort_index()
counts_blocks = {int(k): int(v) for k, v in counts_blocks.items()}

print("\n🧱 GRID blocks (CV) contíguos (GroupKFold):")
print(f"   -> nx={GRID_BLOCKS_NX}, ny={GRID_BLOCKS_NY}, total={GRID_BLOCKS_NX*GRID_BLOCKS_NY}")
print(f"   -> Blocos: {counts_blocks}")

# Pré-cálculos
scaler_all = StandardScaler()
X_scaled = scaler_all.fit_transform(X_all)

rf_imp = RandomForestRegressor(n_estimators=600, random_state=SEED, n_jobs=-1).fit(X_scaled, y).feature_importances_
rf_imp = rf_imp / (rf_imp.sum() + 1e-12)

et_imp = ExtraTreesRegressor(n_estimators=600, random_state=SEED+1, n_jobs=-1).fit(X_scaled, y).feature_importances_
et_imp = et_imp / (et_imp.sum() + 1e-12)

corr_target = np.nan_to_num([
    abs(np.corrcoef(X_all[:, i], y)[0, 1]) if np.std(X_all[:, i]) > 0 else 0.0
    for i in range(N_FEATURES)
])

spearman_matrix = np.zeros((N_FEATURES, N_FEATURES), dtype=float)
for i in range(N_FEATURES):
    for j in range(i+1, N_FEATURES):
        s = spearmanr(X_all[:, i], X_all[:, j]).correlation
        spearman_matrix[i, j] = spearman_matrix[j, i] = float(abs(s) if np.isfinite(s) else 0.0)

vif_spearman = spearman_matrix**2 / (1.0 - spearman_matrix**2 + 1e-8)

marginal_r2 = np.zeros(N_FEATURES, dtype=float)
for i in range(N_FEATURES):
    r2_i, _ = cv_oof_grouped([i], X_all, y, groups, seed=SEED + 1000 + i, n_estimators=80)
    marginal_r2[i] = max(float(r2_i), 0.0)
marginal_r2_norm = marginal_r2 / (marginal_r2.sum() + 1e-12)

synergy_norm = np.zeros((N_FEATURES, N_FEATURES), dtype=float)

# [K] Salvar diagnósticos de features imediatamente após pré-cálculo
ensure_dir(OUTPUT_DIR)
_feat_diag_early = pd.DataFrame([{
    "feature": FEATURE_NAMES[i],
    "RF_importance": float(rf_imp[i]),
    "ET_importance": float(et_imp[i]),
    "corr_target_abs": float(corr_target[i]),
    "marginal_r2": float(marginal_r2[i]),
    "marginal_r2_norm": float(marginal_r2_norm[i]),
} for i in range(N_FEATURES)]).sort_values("RF_importance", ascending=False)

_feat_diag_path_early = os.path.join(OUTPUT_DIR, OUT_CSV_FEATURES)
_feat_diag_early.to_csv(_feat_diag_path_early, index=False)
print(f"\n📋 Feature diagnostics (pré-varredura) salvo: {_feat_diag_path_early}")
print(_feat_diag_early[["feature","RF_importance","ET_importance","corr_target_abs","marginal_r2"]].to_string(index=False))

# Enumerável: 2^15
all_32768_x = np.array(list(itertools.product([0, 1], repeat=N_FEATURES)), dtype=int)

# =============================================================================
# CHECKPOINT / RESUME
# =============================================================================
ensure_dir(OUTPUT_DIR)
final_path = os.path.join(OUTPUT_DIR, OUT_JSON)
partial_path = os.path.join(OUTPUT_DIR, OUT_JSON_TMP)

state = try_load_json(partial_path)
if state is None:
    state = {
        "results_by_k": {},
        "run_config": {
            "seed": int(SEED),
            "features_csv": FEATURES_CSV,
            "biomassa_csv": BIOMASSA_CSV,
            "target_col": TARGET_COL,
            "allowlist": BAND_NAMES,
            "firewall_regex": LEAKY_PATTERNS.pattern,
            "N_FEATURES": int(N_FEATURES),
            "K_RANGE": K_RANGE,
            "grid_blocks": {"nx": int(GRID_BLOCKS_NX), "ny": int(GRID_BLOCKS_NY), "counts": counts_blocks},
            "qaoa": {
                "layers": int(QAOA_LAYERS),
                "shots_obj": int(SHOTS_OBJ),
                "shots_final": int(SHOTS_FINAL),
                "de_maxiter": int(DE_MAXITER),
                "de_popsize": int(DE_POPSIZE),
                "de_polish": bool(DE_POLISH)
            },
            "qubo_weights": QUBO_WEIGHTS
        },
        "meta": {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "feature_names": FEATURE_NAMES
        }
    }
else:
    print("\n🧩 CHECKPOINT detectado — retomando do PARTIAL.json (K já concluídos serão pulados).")

done_ks = set(int(k) for k in state.get("results_by_k", {}).keys())

# =============================================================================
# RUN
# =============================================================================
sim = AerSimulator()

print("\n" + "="*92)
print("🚀 Iniciando/Retomando varredura k=4..12 (todos competem na MESMA QUBO)")
print("="*92)

global_t0 = time.time()

for k in K_RANGE:
    if int(k) in done_ks:
        print(f"\n⏭️  K={k} já concluído no checkpoint — pulando.")
        continue

    t0 = time.time()

    print(f"\n{'='*92}")
    print(f" 🎯 ALVO DE CARDINALIDADE (soft): K = {k}")
    print(f"{'='*92}")

    # --- penalty robusto
    pl, pl_diag = choose_penalty_lambda_robust(
        k=k,
        all_x=all_32768_x,
        rf_imp=rf_imp,
        marg_r2=marginal_r2_norm,
        corr_tgt=corr_target,
        vif=vif_spearman,
        spearman=spearman_matrix,
        syn=synergy_norm,
        weights=QUBO_WEIGHTS
    )

    lin_f, quad_f, lin_n, quad_n, scale_f = build_qubo_for_k(
        k, rf_imp, marginal_r2_norm, corr_target, vif_spearman, spearman_matrix, synergy_norm, QUBO_WEIGHTS, pl
    )

    best_size, idx_best, cost_best = best_solution_size_for_lambda(all_32768_x, lin_f, quad_f)
    exact_hit = (best_size == k)

    print(f" 📐 QUBO gerada (penalty_lambda={pl:.6f}) | scale={scale_f:.3f} | best_size={best_size} | exact_hit={exact_hit}")

    # --- Ising
    h_ising = np.zeros(N_FEATURES, dtype=float)
    J_ising = np.zeros((N_FEATURES, N_FEATURES), dtype=float)
    for i in range(N_FEATURES):
        h_ising[i] = -lin_n[i] / 2.0
        for j in range(i+1, N_FEATURES):
            J_ising[i, j] = quad_n[i, j] / 4.0
            h_ising[i] -= quad_n[i, j] / 4.0
            h_ising[j] -= quad_n[i, j] / 4.0

    methods_k: Dict[str, Any] = {}

    # --- Exact enumeration (verdade do QUBO)
    sol_exact = all_32768_x[int(idx_best)].copy()
    feat_exact = np.where(sol_exact == 1)[0].tolist()
    r2_ex, ypred_ex = cv_oof_grouped(feat_exact, X_all, y, groups, seed=SEED + 10_000 + k, n_estimators=90)
    methods_k["Exact_Enumerated_QUBO"] = {
        **compute_metrics_from_oof(y, ypred_ex),
        "features": [FEATURE_NAMES[i] for i in feat_exact],
        "indices": feat_exact,
        "selected_size": int(len(feat_exact)),
        "qubo_cost": float(cost_best),
        "subset_hash": subset_hash([FEATURE_NAMES[i] for i in feat_exact])
    }
    print(f" 🏆 Exact_Enumerated: R²={r2_ex:.4f} | size={len(feat_exact)} | hash={methods_k['Exact_Enumerated_QUBO']['subset_hash']}")

    # --- SA
    sa_idx, sa_cost = sa_qubo(lin_f, quad_f, seed=SEED + 20_000 + k, steps=6500)
    r2_sa, ypred_sa = cv_oof_grouped(sa_idx, X_all, y, groups, seed=SEED + 21_000 + k, n_estimators=90)
    methods_k["Simulated_Annealing"] = {
        **compute_metrics_from_oof(y, ypred_sa),
        "features": [FEATURE_NAMES[i] for i in sa_idx],
        "indices": sa_idx,
        "selected_size": int(len(sa_idx)),
        "qubo_cost": float(sa_cost),
        "subset_hash": subset_hash([FEATURE_NAMES[i] for i in sa_idx])
    }
    print(f" 🔥 SA: R²={r2_sa:.4f} | size={len(sa_idx)} | hash={methods_k['Simulated_Annealing']['subset_hash']}")

    # --- GA
    ga_idx, ga_cost = ga_qubo(lin_f, quad_f, seed=SEED + 30_000 + k)
    r2_ga, ypred_ga = cv_oof_grouped(ga_idx, X_all, y, groups, seed=SEED + 31_000 + k, n_estimators=90)
    methods_k["Genetic_Algorithm"] = {
        **compute_metrics_from_oof(y, ypred_ga),
        "features": [FEATURE_NAMES[i] for i in ga_idx],
        "indices": ga_idx,
        "selected_size": int(len(ga_idx)),
        "qubo_cost": float(ga_cost),
        "subset_hash": subset_hash([FEATURE_NAMES[i] for i in ga_idx])
    }
    print(f" 🧬 GA: R²={r2_ga:.4f} | size={len(ga_idx)} | hash={methods_k['Genetic_Algorithm']['subset_hash']}")

    # --- Baselines topK
    rf_top = topk_from_scores(rf_imp, k)
    et_top = topk_from_scores(et_imp, k)

    r2_rf, ypred_rf = cv_oof_grouped(rf_top, X_all, y, groups, seed=SEED + 40_000 + k, n_estimators=90)
    methods_k["RF_topK"] = {
        **compute_metrics_from_oof(y, ypred_rf),
        "features": [FEATURE_NAMES[i] for i in rf_top],
        "indices": rf_top,
        "selected_size": int(len(rf_top)),
        "subset_hash": subset_hash([FEATURE_NAMES[i] for i in rf_top])
    }

    r2_et, ypred_et = cv_oof_grouped(et_top, X_all, y, groups, seed=SEED + 41_000 + k, n_estimators=90)
    methods_k["ET_topK"] = {
        **compute_metrics_from_oof(y, ypred_et),
        "features": [FEATURE_NAMES[i] for i in et_top],
        "indices": et_top,
        "selected_size": int(len(et_top)),
        "subset_hash": subset_hash([FEATURE_NAMES[i] for i in et_top])
    }

    # --- QAOA
    print(f" ⚛️ QAOA (DE, warm-starts: RF/ET/Exact) ...")
    ws_dict = {"RF": rf_top, "ET": et_top, "Exact": feat_exact}

    best_qaoa = None
    best_qaoa_cost = float("inf")

    # [H] Logar TODOS os warm-starts individualmente para quantificar variância
    qaoa_all_warmstarts = {}

    for ws_name, ws_idx in ws_dict.items():
        ws_seed = SEED + 50_000 + (k * 10) + (0 if ws_name == "RF" else 1 if ws_name == "ET" else 2)

        def objective(params):
            gammas = params[:QAOA_LAYERS]
            betas  = params[QAOA_LAYERS:]
            qc = build_qaoa_ws(N_FEATURES, h_ising, J_ising, gammas, betas, QAOA_LAYERS, ws_idx)
            qc.measure_all()
            counts = sim.run(transpile(qc, sim, optimization_level=1), shots=SHOTS_OBJ).result().get_counts()

            exp_val = 0.0
            total = sum(counts.values())
            for bs, cnt in counts.items():
                x_vec = np.array([int(b) for b in bs[::-1]], dtype=int)
                c = qubo_cost_vectorized(x_vec.reshape(1, -1), lin_n, quad_n)[0]
                exp_val += float(c) * (cnt / max(total, 1))
            return float(exp_val)

        bnds = [(0.01, 1.0)] * (2 * QAOA_LAYERS)
        try:
            res = differential_evolution(
                objective,
                bounds=bnds,
                maxiter=DE_MAXITER,
                popsize=DE_POPSIZE,
                seed=ws_seed,
                polish=DE_POLISH,
                updating="deferred"
            )
        except Exception as e:
            qaoa_all_warmstarts[ws_name] = {"status": f"de_failed: {str(e)}"}
            continue

        gammas_opt = res.x[:QAOA_LAYERS].tolist()
        betas_opt  = res.x[QAOA_LAYERS:].tolist()

        qc_final = build_qaoa_ws(N_FEATURES, h_ising, J_ising, gammas_opt, betas_opt, QAOA_LAYERS, ws_idx)
        qc_final.measure_all()

        try:
            qc_transpiled = transpile(qc_final, sim, optimization_level=1)
            counts = sim.run(qc_transpiled, shots=SHOTS_FINAL).result().get_counts()

            # [I] Stats do circuito QAOA após transpilação
            circuit_stats = {
                "depth": int(qc_transpiled.depth()),
                "n_gates": int(sum(qc_transpiled.count_ops().values())),
                "gate_counts": {str(g): int(c) for g, c in qc_transpiled.count_ops().items()},
                "n_qubits": int(qc_transpiled.num_qubits),
                "n_parameters": int(2 * QAOA_LAYERS),
                "de_nfev": int(res.nfev),
                "de_success": bool(res.success)
            }
        except Exception:
            qaoa_all_warmstarts[ws_name] = {"status": "circuit_run_failed"}
            continue

        qaoa_idx, qaoa_cost = sample_best_bitstring_by_qubo(counts, lin_f, quad_f)
        if len(qaoa_idx) == 0:
            qaoa_all_warmstarts[ws_name] = {"status": "empty_bitstring"}
            continue

        r2_q_ws, ypred_q_ws = cv_oof_grouped(qaoa_idx, X_all, y, groups, seed=SEED + 60_000 + k, n_estimators=90)
        ws_result = {
            "status": "ok",
            "r2": float(r2_q_ws),
            "rmse": float(np.sqrt(mean_squared_error(y, ypred_q_ws))),
            "bias": float(np.mean(ypred_q_ws - y)),
            "qubo_cost": float(qaoa_cost),
            "features": [FEATURE_NAMES[i] for i in qaoa_idx],
            "selected_size": int(len(qaoa_idx)),
            "subset_hash": subset_hash([FEATURE_NAMES[i] for i in qaoa_idx]),
            "qaoa_params": {"gammas": gammas_opt, "betas": betas_opt},
            "circuit_stats": circuit_stats
        }
        qaoa_all_warmstarts[ws_name] = ws_result

        if float(qaoa_cost) < best_qaoa_cost:
            best_qaoa_cost = float(qaoa_cost)
            best_qaoa = {**ws_result, "warm_start": ws_name}

        print(f"   ws={ws_name}: R²={r2_q_ws:.4f} | size={len(qaoa_idx)} | cost={qaoa_cost:.4f} | depth={circuit_stats['depth']}")

    if best_qaoa is None:
        methods_k["QAOA_Simulado"] = {
            "r2": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "features": [],
            "indices": [],
            "selected_size": 0,
            "qubo_cost": float("nan"),
            "status": "no_solution",
            "all_warmstarts": qaoa_all_warmstarts
        }
        print(" ⚛️ QAOA: sem solução válida (registrado, sem crash).")
    else:
        # Quantificar variância entre warm-starts válidos
        valid_ws = [v for v in qaoa_all_warmstarts.values() if isinstance(v, dict) and v.get("status") == "ok"]
        r2_vals = [v["r2"] for v in valid_ws]
        cost_vals = [v["qubo_cost"] for v in valid_ws]
        hashes = [v["subset_hash"] for v in valid_ws]

        best_qaoa["all_warmstarts"] = qaoa_all_warmstarts
        best_qaoa["warmstart_variance"] = {
            "n_valid": len(valid_ws),
            "r2_mean": float(np.mean(r2_vals)) if r2_vals else float("nan"),
            "r2_std": float(np.std(r2_vals, ddof=1)) if len(r2_vals) > 1 else 0.0,
            "cost_mean": float(np.mean(cost_vals)) if cost_vals else float("nan"),
            "cost_std": float(np.std(cost_vals, ddof=1)) if len(cost_vals) > 1 else 0.0,
            "n_unique_subsets": len(set(hashes)),
            "all_same_subset": len(set(hashes)) == 1
        }

        methods_k["QAOA_Simulado"] = best_qaoa
        print(f" ⚛️ QAOA best: R²={best_qaoa['r2']:.4f} | size={best_qaoa['selected_size']} | "
              f"cost={best_qaoa['qubo_cost']:.4f} | R²_std={best_qaoa['warmstart_variance']['r2_std']:.4f}")

    # --- registra K
    state["results_by_k"][str(int(k))] = {
        "penalty_lambda": float(pl),
        "penalty_search_diag": pl_diag,
        "qubo_scale": float(scale_f),
        "best_size_at_lambda": int(best_size),
        "exact_hit_size_eq_k": bool(exact_hit),
        "methods": methods_k,
        "qubo_arrays": {
            "linear_final": lin_f.tolist(),
            "quad_final": quad_f.tolist()
        }
    }

    # -----------------------------------------------------------------------
    # [G] QUBO LANDSCAPE ANALYSIS per k — energy gap + convergence diagnosis
    # -----------------------------------------------------------------------
    # Energy gap: diferença de custo entre o 1º e 2º melhor subset de mesma
    # cardinalidade exata (size == k). Um gap grande indica mínimo bem definido.
    costs_all = qubo_cost_vectorized(all_32768_x, lin_f, quad_f)
    mask_exact_k = (all_32768_x.sum(axis=1) == k)
    costs_k = costs_all[mask_exact_k]

    if len(costs_k) >= 2:
        sorted_costs = np.sort(costs_k)
        energy_gap_abs  = float(sorted_costs[1] - sorted_costs[0])
        energy_gap_rel  = float(energy_gap_abs / (abs(sorted_costs[0]) + 1e-12))
        landscape_type  = "unique_minimum" if energy_gap_abs > 0.01 else "near_degenerate"
    else:
        energy_gap_abs = float("nan")
        energy_gap_rel = float("nan")
        landscape_type = "insufficient_solutions"

    # Convergência entre heurísticas: SA, GA, Exact
    heuristic_methods = ["Exact_Enumerated_QUBO", "Simulated_Annealing", "Genetic_Algorithm"]
    heuristic_hashes  = {m: methods_k[m]["subset_hash"] for m in heuristic_methods if m in methods_k}
    unique_h_hashes   = set(heuristic_hashes.values())

    if len(unique_h_hashes) == 1:
        convergence_diag = "converged_all_heuristics"  # SA=GA=Exact → mesmo subset
    elif len(unique_h_hashes) == 2:
        convergence_diag = "partial_convergence"
    else:
        convergence_diag = "diverged"

    # Jaccard entre todos os pares de métodos que têm subset_hash
    all_methods_with_hash = {m: d["features"] for m, d in methods_k.items()
                             if isinstance(d, dict) and "features" in d and len(d["features"]) > 0}
    pairwise_jaccard = {}
    for (m1, f1), (m2, f2) in itertools.combinations(all_methods_with_hash.items(), 2):
        pairwise_jaccard[f"{m1}_vs_{m2}"] = round(jaccard(f1, f2), 4)

    landscape_k = {
        "k_target": int(k),
        "energy_gap_abs": energy_gap_abs,
        "energy_gap_rel": energy_gap_rel,
        "landscape_type": landscape_type,
        "convergence_heuristics": convergence_diag,
        "heuristic_hashes": heuristic_hashes,
        "n_unique_heuristic_subsets": len(unique_h_hashes),
        "pairwise_jaccard": pairwise_jaccard
    }

    print(f" 🔍 LANDSCAPE k={k}: gap_abs={energy_gap_abs:.4f} | {landscape_type} | heuristics: {convergence_diag}")
    for pair, jac in pairwise_jaccard.items():
        if "QAOA" in pair or "Exact" in pair:
            print(f"    Jaccard {pair}: {jac:.3f}")

    # Guarda no state para persistência
    state["results_by_k"][str(int(k))]["landscape"] = landscape_k

    # --- checkpoint imediato
    json_dump_safe(state, partial_path)

    dt = time.time() - t0
    print(f" ✅ K={k} concluído e salvo em checkpoint | tempo={dt/60:.1f} min")

# =============================================================================
# FINAL SAVE (promove PARTIAL -> FINAL)
# =============================================================================
elapsed = time.time() - global_t0

print("\n" + "="*92)
print(" 🏁 Ranking exploratório (por R² OOF espacial) — auditoria")
print("="*92)

flat = []
for kk, data in state["results_by_k"].items():
    k_int = int(kk)
    for method, entry in data["methods"].items():
        if not isinstance(entry, dict):
            continue
        flat.append({
            "Method": method,
            "Target_K": k_int,
            "R2": float(entry.get("r2", np.nan)) if entry.get("r2") is not None else np.nan,
            "RMSE": float(entry.get("rmse", np.nan)) if entry.get("rmse") is not None else np.nan,
            "Bias": float(entry.get("bias", np.nan)) if entry.get("bias") is not None else np.nan,
            "n_feat": int(entry.get("selected_size", len(entry.get("indices", [])))),
            "QUBO_cost": float(entry.get("qubo_cost", np.nan)) if entry.get("qubo_cost") is not None else np.nan,
            "subset_hash": entry.get("subset_hash", ""),
            "Features": "|".join(entry.get("features", []))
        })

df_rank = pd.DataFrame(flat).sort_values(by="R2", ascending=False)
print(df_rank[["Method","Target_K","R2","RMSE","n_feat","subset_hash"]].head(20).to_string(index=False))

# --- [J] Salvar ranking completo CSV
path_ranking = os.path.join(OUTPUT_DIR, OUT_CSV_RANKING)
df_rank.to_csv(path_ranking, index=False)
print(f"\n📋 Ranking completo salvo: {path_ranking}")

# --- [K] Salvar feature diagnostics CSV
print("\n" + "="*92)
print(" 📊 FEATURE DIAGNOSTICS (importâncias + correlações + R² marginal)")
print("="*92)

feat_diag = []
for i, fname in enumerate(FEATURE_NAMES):
    feat_diag.append({
        "feature": fname,
        "RF_importance": float(rf_imp[i]),
        "ET_importance": float(et_imp[i]),
        "corr_target_abs": float(corr_target[i]),
        "marginal_r2": float(marginal_r2[i]),
        "marginal_r2_norm": float(marginal_r2_norm[i]),
    })

df_feat_diag = pd.DataFrame(feat_diag).sort_values("RF_importance", ascending=False)
print(df_feat_diag.to_string(index=False))

path_feat = os.path.join(OUTPUT_DIR, OUT_CSV_FEATURES)
df_feat_diag.to_csv(path_feat, index=False)
print(f"\n📋 Feature diagnostics salvo: {path_feat}")

# --- [G] Salvar QUBO landscape CSV
print("\n" + "="*92)
print(" 🔍 QUBO LANDSCAPE ANALYSIS (energy gap + convergência heurísticas)")
print("="*92)

landscape_rows = []
for kk, data in state["results_by_k"].items():
    lnd = data.get("landscape", {})
    if not lnd:
        continue
    row = {
        "k_target": lnd["k_target"],
        "energy_gap_abs": lnd["energy_gap_abs"],
        "energy_gap_rel": lnd["energy_gap_rel"],
        "landscape_type": lnd["landscape_type"],
        "convergence_heuristics": lnd["convergence_heuristics"],
        "n_unique_heuristic_subsets": lnd["n_unique_heuristic_subsets"],
        "exact_hash": lnd["heuristic_hashes"].get("Exact_Enumerated_QUBO", ""),
        "sa_hash": lnd["heuristic_hashes"].get("Simulated_Annealing", ""),
        "ga_hash": lnd["heuristic_hashes"].get("Genetic_Algorithm", ""),
    }
    # Jaccard SA vs Exact, GA vs Exact
    pj = lnd.get("pairwise_jaccard", {})
    row["jaccard_SA_vs_Exact"]  = pj.get("Exact_Enumerated_QUBO_vs_Simulated_Annealing",
                                          pj.get("Simulated_Annealing_vs_Exact_Enumerated_QUBO", np.nan))
    row["jaccard_GA_vs_Exact"]  = pj.get("Exact_Enumerated_QUBO_vs_Genetic_Algorithm",
                                          pj.get("Genetic_Algorithm_vs_Exact_Enumerated_QUBO", np.nan))
    row["jaccard_QAOA_vs_Exact"] = pj.get("Exact_Enumerated_QUBO_vs_QAOA_Simulado",
                                           pj.get("QAOA_Simulado_vs_Exact_Enumerated_QUBO", np.nan))
    landscape_rows.append(row)

df_landscape = pd.DataFrame(landscape_rows).sort_values("k_target")
print(df_landscape.to_string(index=False))

path_landscape = os.path.join(OUTPUT_DIR, OUT_CSV_LANDSCAPE)
df_landscape.to_csv(path_landscape, index=False)
print(f"\n📋 QUBO landscape salvo: {path_landscape}")

# Diagnóstico global consolidado
print("\n" + "="*92)
print(" 🧩 DIAGNÓSTICO GLOBAL DO QUBO")
print("="*92)
for _, row in df_landscape.iterrows():
    conv = row["convergence_heuristics"]
    ltype = row["landscape_type"]
    gap = row["energy_gap_abs"]
    k_t = int(row["k_target"])
    same_hash = (row["exact_hash"] == row["sa_hash"] == row["ga_hash"]) if all([row["exact_hash"], row["sa_hash"], row["ga_hash"]]) else False

    if conv == "converged_all_heuristics" and ltype == "unique_minimum":
        verdict = "✅ MÍNIMO ÚNICO BEM DEFINIDO — SA=GA=Exact convergem, gap>0.01"
    elif conv == "converged_all_heuristics" and ltype == "near_degenerate":
        verdict = "⚠️  CONVERGÊNCIA COM GAP PEQUENO — mesmo subset, mas energia próxima de soluções alternativas"
    elif conv == "partial_convergence":
        verdict = "⚠️  CONVERGÊNCIA PARCIAL — apenas 2 dos 3 heurísticos coincidem"
    else:
        verdict = "❌ DIVERGÊNCIA — heurísticos encontraram subsets distintos"
    print(f"  k={k_t}: {verdict} | gap={gap:.4f} | same_hash={same_hash}")

# --- escreve JSON final
json_dump_safe(state, final_path)

print("\n" + "="*92)
print(f"💾 JSON FINAL salvo: {final_path}")
print(f"🧩 Checkpoint mantido: {partial_path}")
print(f"📋 Ranking CSV:        {path_ranking}")
print(f"📋 Features CSV:       {path_feat}")
print(f"📋 Landscape CSV:      {path_landscape}")
print(f"⏱️ Tempo total: {elapsed/60:.1f} min")
print("FINALIZADO ✅")
print("=" * 92)