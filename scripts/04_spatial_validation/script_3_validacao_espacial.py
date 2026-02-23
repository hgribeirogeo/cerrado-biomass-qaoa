#!/usr/bin/env python3
"""
================================================================================
ENSEMBLE BIOMASSA — SCRIPT 3 EMS-SPATIAL v3.0
LO-MRO Geographic Benchmark — consome outputs do Script 2 v2.5
================================================================================

PROPÓSITO:
  Aplicar LO-MRO (Leave-One-MacroRegion-Out) blind test a todos os subsets
  avaliados no Script 2, usando exatamente o mesmo modelo RF direto e as mesmas
  partições espaciais, e comparar via testes estatísticos rigorosos.

COERÊNCIA COM SCRIPTS 1 E 2:
  - Modelo: RandomForestRegressor idêntico ao Script 2 (n_est=300, max_depth=8,
    min_samples_leaf=2, max_features=0.7) — FINAL_MODEL="RF", sem stacking
  - Partições: pick_grid_shape + make_grid_blocks + make_macro_regions_contiguous
    copiados literalmente do Script 2 (mesmo algoritmo de floor-division)
  - Seeds baseline: SEED + 1000 + mid (idêntico ao Script 2 LO-MRO)
  - Seeds benchmark: SEED + 3000 + mid + hash(method) (offset distinto)
  - XGBoost: detecção GPU via subprocess nvidia-smi (sem instanciar modelo)
  - Firewall e BAND_NAMES: idênticos ao Script 2
  - Subsets: mesmo mecanismo de carregamento do JSON (best-k por método)

ESTATÍSTICA:
  - Wilcoxon pareado em |erro| por ponto (não em resíduos crus)
  - Block bootstrap CI para ΔRMSE (por bloco espacial dentro do holdout)
  - Holm–Bonferroni por família = todos os métodos dentro de cada holdout
  - Veredito global: BETTER / WORSE / EQUIVALENT / INCONCLUSIVE
    (baseado em CI bootstrap e banda de equivalência ±EQUIV_DELTA_RMSE)

FIGURAS (EMS):
  - fig_lomro_scatter_baseline.png  — obs vs pred baseline por macro-região
  - fig_delta_rmse_per_holdout.png  — ΔRMSE por método por holdout (barplot)
  - fig_delta_rmse_summary.png      — ΔRMSE ponderado + CI global por método
  - fig_heatmap_delta_rmse.png      — heatmap método × holdout
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, json, re, hashlib
from typing import Dict, List, Tuple, Any
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# Optional: XGBoost (para detecção de GPU — não usado no modelo final)
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

print("=" * 78)
print("   ENSEMBLE BIOMASSA — SCRIPT 3 EMS-SPATIAL v3.0 (LO-MRO Benchmark)")
print("=" * 78)

# ============================================================
# CONFIGURAÇÃO — alinhada ao Script 2 v2.5
# ============================================================
FEATURES_CSV = r"/mnt/e/PROJETOS/biomassa_quantum/results/goias_df_features_buffer50m_v3_2018.csv"
BIOMASSA_CSV = r"/mnt/e/PROJETOS/biomassa_quantum/results/biomassa_por_UA_corrigido.csv"

SCRIPT2_DIR  = r"/mnt/e/PROJETOS/biomassa_quantum/results/ensemble_nestedcv_subsets_ems_spatial_v2_5"
BEST_JSON    = os.path.join(SCRIPT2_DIR, "best_global_subset.json")
TOP3_JSON    = os.path.join(SCRIPT2_DIR, "top_methods_subsets.json")
LOMRO_RAW_CSV = os.path.join(SCRIPT2_DIR, "lomro_predictions_raw.csv")

QAOA_JSON    = r"/mnt/e/PROJETOS/biomassa_quantum/results/qaoa_ibm_real/qaoa_mega_varredura_justa_resultados.json"

OUTPUT_DIR   = r"/mnt/e/PROJETOS/biomassa_quantum/results/script3_lomro_benchmark_v3_0"
FIGS_DIR     = os.path.join(OUTPUT_DIR, "figures")

TARGET_COL    = "Biomassa_Mg_ha"
MAPBIOMAS_COL = "mapbiomas_2018"
SEED          = 42   # mesmo do Script 2
NODATA        = -9999.0

CLASSES_VEGETACAO = {3, 4, 5, 6, 11, 12, 29, 32, 49, 50}

# Idêntico ao Script 2 — não modificar
BAND_NAMES = [
    'B2_seca', 'B3_seca', 'B4_seca', 'B8_seca', 'B11_seca', 'B12_seca',
    'NDVI_seca', 'NDWI_seca', 'NBR_seca', 'NDVI_RE_seca', 'MSI_seca', 'EVI_seca',
    'elevation', 'slope', 'VV_dB', 'VH_dB', 'HV_dB',
    'mapbiomas_2018', 'clay_pct', 'canopy_height', 'canopy_height_sd'
]

# CV — idêntico ao Script 2
N_SPATIAL_BLOCKS_TARGET = 12
N_MACRO_REGIONS          = 5

# Bootstrap
BOOTSTRAP_ITERS  = 3000
BOOTSTRAP_SEED   = 123
EQUIV_DELTA_RMSE = 2.0   # Mg/ha — banda de equivalência prática
ALPHA            = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGS_DIR,   exist_ok=True)


# ============================================================
# [D] XGBoost: detecção GPU via subprocess (sem instanciar modelo)
# Copiado literalmente do Script 2 para coerência
# ============================================================
def _detect_xgb_device() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            gpu_name = r.stdout.decode().strip().splitlines()[0]
            print(f"   [XGB] GPU detectada via nvidia-smi: {gpu_name} → device='cuda'")
            return 'cuda'
    except Exception:
        pass
    print("   [XGB] GPU não detectada → device='cpu'")
    return 'cpu'

_XGB_DEVICE = _detect_xgb_device() if XGB_OK else 'cpu'


# ============================================================
# Prioridade de desempate — idêntica ao Script 2
# ============================================================
_METHOD_PRIORITY = [
    "QAOA_Simulado",
    "Exact_Enumerated_QUBO",
    "Genetic_Algorithm",
    "Simulated_Annealing",
    "RF_topK",
    "ET_topK",
]

def _method_priority(name: str) -> int:
    for i, prefix in enumerate(_METHOD_PRIORITY):
        if name.startswith(prefix):
            return i
    return len(_METHOD_PRIORITY)


# ============================================================
# Particionamento espacial — copiado literalmente do Script 2
# ============================================================
def pick_grid_shape(target_blocks: int, lon_span: float, lat_span: float) -> Tuple[int, int]:
    if lat_span <= 0 or lon_span <= 0:
        nx = int(max(1, round(np.sqrt(target_blocks))))
        ny = int(max(1, round(target_blocks / nx)))
        return nx, ny
    aspect = lon_span / lat_span
    nx_ideal = max(1, int(round(np.sqrt(target_blocks * aspect))))
    best = (nx_ideal, max(1, int(round(target_blocks / nx_ideal))))
    best_score = (abs(best[0] * best[1] - target_blocks) +
                  abs(np.log(max(best[0] / max(best[1], 1), 1e-9) / aspect)))
    for nx_c in range(max(1, nx_ideal - target_blocks), nx_ideal + target_blocks + 1):
        ny_c = max(1, int(round(target_blocks / nx_c)))
        diff_prod = abs(nx_c * ny_c - target_blocks)
        diff_asp  = abs(np.log(max(nx_c / max(ny_c, 1e-9), 1e-9) / aspect))
        score = diff_prod + 0.5 * diff_asp
        if score < best_score:
            best = (nx_c, ny_c)
            best_score = score
    return best

def make_grid_blocks(df: pd.DataFrame, target_blocks: int, seed: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    lon = df["lon"].values.astype(float)
    lat = df["lat"].values.astype(float)
    lon_span = max(1e-12, float(np.nanmax(lon) - np.nanmin(lon)))
    lat_span = max(1e-12, float(np.nanmax(lat) - np.nanmin(lat)))
    nx, ny = pick_grid_shape(target_blocks, lon_span, lat_span)
    dx = lon_span / nx
    dy = lat_span / ny
    ix = np.clip(np.floor((lon - np.nanmin(lon)) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((lat - np.nanmin(lat)) / dy).astype(int), 0, ny - 1)
    return (ix + nx * iy).astype(int), (nx, ny)

def make_macro_regions_contiguous(df: pd.DataFrame, n_macro: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    lon = df["lon"].values.astype(float)
    lat = df["lat"].values.astype(float)
    lon_span = max(1e-12, float(np.nanmax(lon) - np.nanmin(lon)))
    lat_span = max(1e-12, float(np.nanmax(lat) - np.nanmin(lat)))
    aspect = lon_span / lat_span
    if n_macro == 5:
        nx, ny = (5, 1) if aspect >= 1.0 else (1, 5)
    else:
        pairs = [(a, n_macro // a) for a in range(1, n_macro + 1) if n_macro % a == 0]
        best = min(pairs, key=lambda p: abs(np.log((p[0] / p[1]) / aspect)))
        nx, ny = best
    dx = lon_span / nx
    dy = lat_span / ny
    ix = np.clip(np.floor((lon - np.nanmin(lon)) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((lat - np.nanmin(lat)) / dy).astype(int), 0, ny - 1)
    return (ix + nx * iy).astype(int), (nx, ny)


# ============================================================
# Helpers: métricas e subset hash — idênticos ao Script 2
# ============================================================
def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    sl, ic, *_ = stats.linregress(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae, "bias": bias,
            "slope": float(sl), "intercept": float(ic)}

def subset_hash(feats: List[str]) -> str:
    return hashlib.sha256(",".join(sorted(feats)).encode()).hexdigest()[:16]


# ============================================================
# Modelo de inferência: RF direto — idêntico ao FINAL_MODEL="RF" do Script 2
# Hiperparams copiados literalmente de predict_rf_solo() do Script 2
# ============================================================
def predict_rf_solo(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, seed: int) -> np.ndarray:
    """RF direto sem meta-learner, com imputação mediana.
    Idêntico ao Script 2 predict_rf_solo() — não modificar hiperparâmetros."""
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = imputer.fit_transform(np.asarray(X_train, float))
    Xte_imp = imputer.transform(np.asarray(X_test, float))
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=2,
        max_features=0.7, random_state=seed, n_jobs=-1)
    rf.fit(Xtr_imp, y_train)
    return rf.predict(Xte_imp).astype(float)


# ============================================================
# Estatística — preservada do Script 3 original (v2.7)
# ============================================================
def wilcoxon_abs_error(y_true: np.ndarray,
                       pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Wilcoxon pareado em |erro| por ponto. H0: |e_a| = |e_b|."""
    y_true = np.asarray(y_true, float)
    la = np.abs(np.asarray(pred_a, float) - y_true)
    lb = np.abs(np.asarray(pred_b, float) - y_true)
    try:
        _, p = stats.wilcoxon(la, lb)
        return float(p) if np.isfinite(p) else np.nan
    except Exception:
        return np.nan

def block_bootstrap_ci_delta_rmse(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    blocks: np.ndarray, iters: int = 3000, seed: int = 123
) -> Dict[str, float]:
    """Block bootstrap para ΔRMSE = RMSE(B) - RMSE(A)."""
    rng = np.random.default_rng(seed)
    y_true  = np.asarray(y_true, float)
    pred_a  = np.asarray(pred_a, float)
    pred_b  = np.asarray(pred_b, float)
    blocks  = np.asarray(blocks)
    n = len(y_true)

    if n < 3:
        return {"delta_rmse_mean": np.nan, "ci_low_95": np.nan,
                "ci_high_95": np.nan, "n_blocks": 0, "boot_type": "too_small"}

    uniq = np.unique(blocks)
    B = len(uniq)

    if B < 2:   # fallback IID
        deltas = np.array([
            np.sqrt(mean_squared_error(y_true[bs := rng.choice(n, n, replace=True)], pred_b[bs])) -
            np.sqrt(mean_squared_error(y_true[bs], pred_a[bs]))
            for _ in range(iters)
        ])
        return {"delta_rmse_mean": float(np.mean(deltas)),
                "ci_low_95": float(np.percentile(deltas, 2.5)),
                "ci_high_95": float(np.percentile(deltas, 97.5)),
                "n_blocks": int(B), "boot_type": "iid_fallback"}

    block_to_idx = {b: np.where(blocks == b)[0] for b in uniq}
    deltas = np.empty(iters, float)
    for i in range(iters):
        chosen = rng.choice(uniq, size=B, replace=True)
        bs_idx = np.concatenate([block_to_idx[b] for b in chosen])
        if len(bs_idx) > n:
            bs_idx = rng.choice(bs_idx, size=n, replace=False)
        elif len(bs_idx) < n:
            bs_idx = rng.choice(bs_idx, size=n, replace=True)
        rmse_a = np.sqrt(mean_squared_error(y_true[bs_idx], pred_a[bs_idx]))
        rmse_b = np.sqrt(mean_squared_error(y_true[bs_idx], pred_b[bs_idx]))
        deltas[i] = rmse_b - rmse_a

    return {"delta_rmse_mean": float(np.mean(deltas)),
            "ci_low_95": float(np.percentile(deltas, 2.5)),
            "ci_high_95": float(np.percentile(deltas, 97.5)),
            "n_blocks": int(B), "boot_type": "block"}

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, float)
    adj = np.full_like(pvals, np.nan)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return adj
    p = pvals[ok]; m = len(p)
    order = np.argsort(p)
    holm  = np.maximum.accumulate((m - np.arange(m)) * p[order])
    holm  = np.clip(holm, 0.0, 1.0)
    inv   = np.empty_like(order); inv[order] = np.arange(m)
    adj[ok] = holm[inv]
    return adj

def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(w[ok] * x[ok]) / np.sum(w[ok])) if ok.sum() > 0 else np.nan

def bootstrap_ci_global_delta(
    df_sub: pd.DataFrame, value_col: str = "delta_rmse",
    weight_col: str = "n_blind", holdout_col: str = "holdout_region",
    iters: int = 5000, seed: int = 777
) -> Tuple[float, float, float]:
    """Bootstrap em nível de holdout para ΔRMSE global ponderado."""
    rng = np.random.default_rng(seed)
    g = df_sub.groupby(holdout_col, as_index=False).agg(
        delta=(value_col, "mean"), n=(weight_col, "mean"))
    deltas = g["delta"].values.astype(float)
    weights = g["n"].values.astype(float)
    H = len(g)
    if H < 2:
        m = weighted_mean(deltas, weights)
        return m, np.nan, np.nan
    boot = np.array([
        weighted_mean(deltas[bs := rng.choice(H, H, replace=True)], weights[bs])
        for _ in range(iters)
    ])
    m = weighted_mean(deltas, weights)
    return m, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

def verdict_from_ci(ci_low: float, ci_high: float, equiv: float) -> str:
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return "INCONCLUSIVE"
    if ci_high < -equiv:  return "BETTER"
    if ci_low  >  equiv:  return "WORSE"
    if ci_low >= -equiv and ci_high <= equiv: return "EQUIVALENT"
    return "INCONCLUSIVE"


# ============================================================
# Figuras
# ============================================================
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def _add_1to1_line(ax, y_true, y_pred, label_prefix="", color='#1f77b4'):
    ax.scatter(y_true, y_pred, alpha=0.45, s=28, color=color, edgecolors='none')
    lo = min(float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
    hi = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
    mg = (hi - lo) * 0.05
    ax.plot([lo-mg, hi+mg], [lo-mg, hi+mg], 'k--', lw=1.0, label='1:1')
    sl, ic, *_ = stats.linregress(y_true, y_pred)
    xols = np.array([lo-mg, hi+mg])
    ax.plot(xols, sl*xols+ic, '-', color=color, lw=1.5, alpha=0.8,
            label=f'OLS (slope={sl:.2f})')
    met = compute_metrics(y_true, y_pred)
    ax.set_xlabel("Observed AGB (Mg ha⁻¹)", fontsize=9)
    ax.set_ylabel("Predicted AGB (Mg ha⁻¹)", fontsize=9)
    ax.set_title(f"{label_prefix}R²={met['r2']:.3f} | RMSE={met['rmse']:.1f} | "
                 f"Bias={met['bias']:+.1f} | Slope={met['slope']:.2f}", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_xlim(lo-mg, hi+mg); ax.set_ylim(lo-mg, hi+mg)
    ax.set_aspect('equal', adjustable='box')

def plot_lomro_scatter_baseline(rows_raw: List[Dict], out_path: str, method_name: str):
    """Scatter obs vs pred do baseline por macro-região."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows_raw:
        return
    from collections import defaultdict
    grouped = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for r in rows_raw:
        grouped[r["macro_region"]]["y_true"].append(float(r["y_true"]))
        grouped[r["macro_region"]]["y_pred"].append(float(r["y_pred"]))
    mids = sorted(grouped.keys())
    n = len(mids)
    ncols = min(n, 3); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows), squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for idx, mid in enumerate(mids):
        yt = np.array(grouped[mid]["y_true"], float)
        yp = np.array(grouped[mid]["y_pred"], float)
        met = compute_metrics(yt, yp)
        _add_1to1_line(axes_flat[idx], yt, yp, color=PALETTE[idx % len(PALETTE)])
        axes_flat[idx].set_title(
            f"Macro {mid} (n={len(yt)}) | R²={met['r2']:.3f} | "
            f"RMSE={met['rmse']:.1f} | Bias={met['bias']:+.1f}", fontsize=8)
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle(f"LO-MRO Baseline ({method_name}) — Observed vs Predicted AGB",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def plot_delta_rmse_per_holdout(df_bench: pd.DataFrame, out_path: str):
    """Barplot de ΔRMSE por método por holdout."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    holdouts = sorted(df_bench["holdout_region"].unique())
    methods  = sorted(df_bench["method"].unique(), key=_method_priority)
    x = np.arange(len(holdouts))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(holdouts)*2), 5))
    for i, m in enumerate(methods):
        sub = df_bench[df_bench["method"] == m].set_index("holdout_region")
        vals = [sub.loc[h, "delta_rmse"] if h in sub.index else np.nan for h in holdouts]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, vals, width*0.9, label=m.replace("_k", " k="),
               color=PALETTE[i % len(PALETTE)], alpha=0.75)
    ax.axhline(0, color='black', lw=1.0, linestyle='--')
    ax.axhline(-EQUIV_DELTA_RMSE, color='gray', lw=0.8, linestyle=':',
               label=f'±{EQUIV_DELTA_RMSE} Mg/ha equiv.')
    ax.axhline(+EQUIV_DELTA_RMSE, color='gray', lw=0.8, linestyle=':')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Macro {h}" for h in holdouts], fontsize=10)
    ax.set_ylabel("ΔRMSE vs Baseline (Mg ha⁻¹)", fontsize=10)
    ax.set_title("ΔRMSE por Método por Holdout — LO-MRO Geographic Benchmark", fontsize=11)
    ax.legend(fontsize=7, ncol=min(3, len(methods)))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def plot_delta_rmse_summary(df_summary: pd.DataFrame, out_path: str):
    """ΔRMSE ponderado ± CI bootstrap global por método."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_s = df_summary.sort_values("delta_rmse_weighted").copy()
    fig, ax = plt.subplots(figsize=(max(6, len(df_s)*1.2), 5))
    methods = df_s["method"].tolist()
    vals    = df_s["delta_rmse_weighted"].values
    ci_lo   = df_s["ci_low_95"].values
    ci_hi   = df_s["ci_high_95"].values
    colors  = [PALETTE[_method_priority(m) % len(PALETTE)] for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, vals, color=colors, alpha=0.7)
    ax.errorbar(x, vals,
                yerr=[vals - ci_lo, ci_hi - vals],
                fmt='none', color='black', capsize=4, lw=1.5)
    ax.axhline(0, color='black', lw=1.0, linestyle='--')
    ax.axhline(-EQUIV_DELTA_RMSE, color='gray', lw=0.8, linestyle=':')
    ax.axhline(+EQUIV_DELTA_RMSE, color='gray', lw=0.8, linestyle=':',
               label=f'±{EQUIV_DELTA_RMSE} Mg/ha')
    for i, (m, v, verd) in enumerate(zip(methods, vals, df_s["verdict"].tolist())):
        ax.text(i, max(v, ci_hi[i]) + 0.1, verd, ha='center', va='bottom',
                fontsize=7, color='darkred' if verd == 'WORSE' else
                          'darkgreen' if verd == 'BETTER' else 'gray')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_k", "\nk=") for m in methods],
                       rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Weighted ΔRMSE vs Baseline (Mg ha⁻¹)", fontsize=10)
    ax.set_title("Global ΔRMSE Summary — LO-MRO Geographic Benchmark\n"
                 "(error bars = 95% bootstrap CI over holdouts)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def plot_heatmap_delta_rmse(df_bench: pd.DataFrame, out_path: str):
    """Heatmap método × holdout de ΔRMSE."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    methods  = sorted(df_bench["method"].unique(), key=_method_priority)
    holdouts = sorted(df_bench["holdout_region"].unique())
    mat = np.full((len(methods), len(holdouts)), np.nan)
    for i, m in enumerate(methods):
        for j, h in enumerate(holdouts):
            sub = df_bench[(df_bench["method"] == m) & (df_bench["holdout_region"] == h)]
            if not sub.empty:
                mat[i, j] = float(sub["delta_rmse"].iloc[0])
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), EQUIV_DELTA_RMSE)
    fig, ax = plt.subplots(figsize=(max(5, len(holdouts)*1.2), max(4, len(methods)*0.7)))
    im = ax.imshow(mat, cmap='RdYlGn_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label='ΔRMSE (Mg ha⁻¹)')
    ax.set_xticks(range(len(holdouts)))
    ax.set_xticklabels([f"Macro {h}" for h in holdouts], fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.replace("_k", " k=") for m in methods], fontsize=8)
    for i in range(len(methods)):
        for j in range(len(holdouts)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:+.1f}", ha='center', va='center',
                        fontsize=7, color='black')
    ax.set_title("ΔRMSE Heatmap — Method × Holdout (green=better, red=worse than baseline)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# Carregamento de subsets — mesma lógica do Script 2
# ============================================================
def load_subsets_from_json(qaoa_json_path: str) -> Dict[str, List[str]]:
    """
    Carrega melhor subset por método (best-k) do JSON do Script 1.
    Lógica idêntica ao Script 2 para garantir os mesmos 6 subsets.
    """
    if not os.path.exists(qaoa_json_path):
        raise FileNotFoundError(f"JSON não encontrado: {qaoa_json_path}")
    with open(qaoa_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    subsets: Dict[str, List[str]] = {}
    if "results_by_k" in j:
        best_r2_tracker: Dict[str, float] = {}
        for k_val, k_data in j["results_by_k"].items():
            for method_name, metrics in k_data.get("methods", {}).items():
                current_r2 = metrics.get("r2", 0) or 0
                if method_name not in best_r2_tracker or current_r2 > best_r2_tracker[method_name]:
                    best_r2_tracker[method_name] = current_r2
                    for old_key in [k for k in subsets if k.startswith(f"{method_name}_k")]:
                        del subsets[old_key]
                    subsets[f"{method_name}_k{k_val}"] = metrics.get("features", [])
    elif "global_top10" in j:
        for entry in j["global_top10"]:
            subsets[f"{entry['method']}_k{entry['k']}"] = entry["features"]
    else:
        raise KeyError("JSON formato desconhecido.")
    return {m: f for m, f in subsets.items() if len(f) > 0}

def load_best_global(best_json_path: str) -> Tuple[str, List[str], Dict]:
    with open(best_json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return str(j["best_global_method"]), list(j["features"]), j


# ============================================================
# 0) CARREGAR CONFIGURAÇÃO DO SCRIPT 2
# ============================================================
print("\n📥 Consumindo outputs do Script 2 v2.5...")
for req in [BEST_JSON, LOMRO_RAW_CSV]:
    if not os.path.exists(req):
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {req}")

best_method, best_feats, best_json_data = load_best_global(BEST_JSON)
print(f"   best_global_method: {best_method}")
print(f"   features (k={len(best_feats)}): {best_feats}")
print(f"   hash: {subset_hash(best_feats)}")

# Verifica coerência com grid do Script 2
grid_from_json = best_json_data.get("grid_blocks", {})
nx_s2 = grid_from_json.get("nx", None)
ny_s2 = grid_from_json.get("ny", None)
if nx_s2 and ny_s2:
    print(f"   grid_blocks do Script 2: nx={nx_s2}, ny={ny_s2}")

# Baseline predictions do Script 2 (para verificação)
df_lomro_raw = pd.read_csv(LOMRO_RAW_CSV)
print(f"   lomro_predictions_raw.csv: {len(df_lomro_raw)} linhas, "
      f"colunas={list(df_lomro_raw.columns)}")


# ============================================================
# 1) CARREGAR DADOS (mesmo filtro do Script 2)
# ============================================================
print("\n📂 Carregando dados (veg-only)...")
features_df  = pd.read_csv(FEATURES_CSV)
biomassa_df  = pd.read_csv(BIOMASSA_CSV)
df = features_df.merge(biomassa_df[["UA", TARGET_COL]], on="UA", how="inner")

df["lon"]   = pd.to_numeric(df.get("lon_pc", np.nan), errors="coerce")
df["lat"]   = pd.to_numeric(df.get("lat_pc", np.nan), errors="coerce")
df["mb_int"]= pd.to_numeric(df[MAPBIOMAS_COL], errors="coerce").round().fillna(-1).astype(int)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

df = df[
    (df[TARGET_COL] > 0) & (df[TARGET_COL] < 300) &
    df["lon"].notna() & df["lat"].notna() &
    df["mb_int"].isin(CLASSES_VEGETACAO)
].copy().reset_index(drop=True)

print(f"   Base veg: n={len(df)} | y̅={df[TARGET_COL].mean():.1f} ± {df[TARGET_COL].std():.1f} Mg/ha")


# ============================================================
# 2) PARTICIONAMENTO — funções idênticas ao Script 2
# ============================================================
print("\n🧱 Criando GRID blocks contíguos para CV...")
df["spatial_block"], (nx_blk, ny_blk) = make_grid_blocks(df, N_SPATIAL_BLOCKS_TARGET, SEED)
counts_blk = {int(k): int(v) for k, v in
              pd.Series(df["spatial_block"]).value_counts().sort_index().items()}
print(f"   -> Grid: nx={nx_blk}, ny={ny_blk} | blocos: {counts_blk}")

# Verifica coerência com Script 2
if nx_s2 and ny_s2 and (nx_blk != nx_s2 or ny_blk != ny_s2):
    print(f"   ⚠️ Grid diverge do Script 2 (nx={nx_s2},ny={ny_s2}). "
          f"Verificar dados de entrada.")

print("\n🌍 Criando macro-regiões contíguas (LO-MRO)...")
df["macro_region"], (nx_m, ny_m) = make_macro_regions_contiguous(df, N_MACRO_REGIONS)
counts_m = {int(k): int(v) for k, v in
            pd.Series(df["macro_region"]).value_counts().sort_index().items()}
print(f"   -> Macro grid: nx={nx_m}, ny={ny_m} | regiões: {counts_m}")


# ============================================================
# 3) CARREGA SUBSETS (mesma lógica do Script 2)
# ============================================================
print("\n📦 Carregando subsets do JSON (mesma lógica do Script 2)...")
band_cols = set(BAND_NAMES)
ALL_SUBSETS = load_subsets_from_json(QAOA_JSON)

# Filtra: apenas features presentes no df E no BAND_NAMES
SUBSETS_VALID: Dict[str, List[str]] = {}
for method, feats in ALL_SUBSETS.items():
    ok = [f for f in feats if f in df.columns and f in band_cols]
    if len(ok) >= 2:
        SUBSETS_VALID[method] = ok

# Ordenar por prioridade de método para display
print(f"   Subsets carregados: {len(SUBSETS_VALID)}")
for m, feats in sorted(SUBSETS_VALID.items(),
                        key=lambda x: (_method_priority(x[0]), x[0])):
    print(f"   - {m}: k={len(feats)} | hash={subset_hash(feats)}")

# Verifica que best_feats está entre os subsets válidos
if best_feats not in [list(v) for v in SUBSETS_VALID.values()]:
    # Adiciona manualmente se não estiver
    best_ok = [f for f in best_feats if f in df.columns and f in band_cols]
    if len(best_ok) >= 2:
        SUBSETS_VALID[best_method] = best_ok
        print(f"   + best_global adicionado manualmente: {best_method}")


# ============================================================
# 4) LO-MRO BENCHMARK
# Baseline: SEED + 1000 + mid (idêntico ao Script 2 LO-MRO)
# Benchmark: SEED + 3000 + mid + hash(method) % 9973
# ============================================================
print("\n" + "=" * 78)
print("🔒 LO-MRO BENCHMARK — RF direto (FINAL_MODEL='RF', igual ao Script 2)")
print(f"   Baseline: {best_method} | k={len(best_feats)}")
print(f"   Benchmark: {len(SUBSETS_VALID)} subsets")
print("=" * 78)

rows_baseline  = []   # métricas do baseline por holdout
rows_baseline_raw = []  # previsões brutas do baseline (para figura)
rows_benchmark = []   # métricas de cada subset por holdout

macro_ids = sorted(df["macro_region"].unique().tolist())

for mid in macro_ids:
    df_blind = df[df["macro_region"] == mid].copy().reset_index(drop=True)
    df_train = df[df["macro_region"] != mid].copy().reset_index(drop=True)

    if len(df_blind) < 5 or len(df_train) < 30:
        print(f"   ⚠️ macro={mid} ignorado: blind={len(df_blind)}, train={len(df_train)}")
        continue

    print(f"\n{'─' * 78}")
    print(f"🔥 HOLDOUT macro={mid} | blind={len(df_blind)} | train={len(df_train)}")

    y_train = df_train[TARGET_COL].values.astype(float)
    y_blind = df_blind[TARGET_COL].values.astype(float)
    blk_blind = df_blind["spatial_block"].values   # para block bootstrap

    # ── BASELINE (mesma seed do Script 2 LO-MRO) ──
    seed_base = SEED + 1000 + int(mid)
    Xtr_base = df_train[best_feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
    Xbl_base = df_blind[best_feats].apply(pd.to_numeric, errors="coerce").values.astype(float)

    y_pred_base = predict_rf_solo(Xtr_base, y_train, Xbl_base, seed=seed_base)
    met_base = compute_metrics(y_blind, y_pred_base)

    print(f"   [BASELINE {best_method}] "
          f"R²={met_base['r2']:.4f} | RMSE={met_base['rmse']:.2f} | "
          f"Bias={met_base['bias']:+.2f} | Slope={met_base['slope']:.3f}")

    rows_baseline.append({
        "holdout_region": int(mid),
        "n_blind": int(len(df_blind)),
        "method": best_method,
        "k": int(len(best_feats)),
        "subset_hash": subset_hash(best_feats),
        **met_base,
        "seed": int(seed_base),
    })

    for yt_i, yp_i in zip(y_blind.tolist(), y_pred_base.tolist()):
        rows_baseline_raw.append({"macro_region": int(mid),
                                   "y_true": yt_i, "y_pred": yp_i})

    # Verificação vs Script 2 (opcional — identifica divergência de seed/partição)
    s2_raw = df_lomro_raw[df_lomro_raw["macro_region"] == mid]
    if len(s2_raw) == len(df_blind):
        rmse_s2 = float(np.sqrt(mean_squared_error(
            s2_raw["y_true"].values, s2_raw["y_pred"].values)))
        delta_vs_s2 = abs(met_base["rmse"] - rmse_s2)
        if delta_vs_s2 > 0.5:
            print(f"   ⚠️ DIVERGÊNCIA vs Script 2 LO-MRO: ΔRMSE={delta_vs_s2:.3f} "
                  f"(Script2={rmse_s2:.2f}, S3={met_base['rmse']:.2f}) — verificar seed/partição")
        else:
            print(f"   ✅ Baseline coerente com Script 2 (ΔRMSE={delta_vs_s2:.3f} Mg/ha)")

    # ── BENCHMARK: todos os subsets ──
    for method, feats in sorted(SUBSETS_VALID.items(),
                                 key=lambda x: (_method_priority(x[0]), x[0])):
        # Não comparar best_method contra si mesmo (ΔRMSE trivialmente ~0)
        if subset_hash(feats) == subset_hash(best_feats) and method == best_method:
            continue

        seed_bench = SEED + 3000 + int(mid) + (abs(hash(method)) % 9973)
        Xtr_b = df_train[feats].apply(pd.to_numeric, errors="coerce").values.astype(float)
        Xbl_b = df_blind[feats].apply(pd.to_numeric, errors="coerce").values.astype(float)

        y_pred_b = predict_rf_solo(Xtr_b, y_train, Xbl_b, seed=seed_bench)
        met_b    = compute_metrics(y_blind, y_pred_b)

        delta_rmse = met_b["rmse"]  - met_base["rmse"]   # + = pior que baseline
        delta_r2   = met_b["r2"]    - met_base["r2"]

        p_wilcox = wilcoxon_abs_error(y_blind, y_pred_base, y_pred_b)

        ci = block_bootstrap_ci_delta_rmse(
            y_blind, y_pred_base, y_pred_b, blk_blind,
            iters=BOOTSTRAP_ITERS,
            seed=BOOTSTRAP_SEED + int(mid) + (abs(hash(method)) % 1000))

        rows_benchmark.append({
            "holdout_region": int(mid),
            "n_blind": int(len(df_blind)),
            "method": method,
            "k": int(len(feats)),
            "subset_hash": subset_hash(feats),
            **{f"bench_{k}": v for k, v in met_b.items()},
            "delta_rmse": float(delta_rmse),
            "delta_r2":   float(delta_r2),
            "wilcoxon_p_raw": float(p_wilcox) if np.isfinite(p_wilcox) else np.nan,
            "boot_ci_low":    float(ci["ci_low_95"]),
            "boot_ci_high":   float(ci["ci_high_95"]),
            "boot_type":      str(ci["boot_type"]),
            "boot_n_blocks":  int(ci["n_blocks"]),
        })

    # Print resumo benchmark para este holdout
    rows_h = [r for r in rows_benchmark if r["holdout_region"] == mid]
    if rows_h:
        rows_h_sorted = sorted(rows_h, key=lambda r: r["delta_rmse"])
        print(f"\n   Benchmark holdout={mid} (ordenado por ΔRMSE):")
        print(f"   {'Método':<35} {'k':>4} {'ΔRMSE':>8} {'Wilcox-p':>10} {'CI95%'}")
        for r in rows_h_sorted[:8]:  # top 8
            ci_str = f"[{r['boot_ci_low']:+.1f},{r['boot_ci_high']:+.1f}]"
            p_str  = f"{r['wilcoxon_p_raw']:.4f}" if np.isfinite(r['wilcoxon_p_raw']) else "  nan"
            print(f"   {r['method']:<35} {r['k']:>4} {r['delta_rmse']:>+8.2f} {p_str:>10} {ci_str}")


# ============================================================
# 5) CONSOLIDAÇÃO + HOLM-BONFERRONI
# ============================================================
print("\n" + "=" * 78)
print("📊 CONSOLIDAÇÃO — Holm–Bonferroni + veredito global")
print("=" * 78)

df_base = pd.DataFrame(rows_baseline)
df_bench = pd.DataFrame(rows_benchmark)

# Holm–Bonferroni por holdout (família = todos os métodos naquele holdout)
df_bench["p_holm"] = np.nan
for hr, sub in df_bench.groupby("holdout_region"):
    pvals = sub["wilcoxon_p_raw"].values.astype(float)
    adj   = holm_bonferroni(pvals)
    df_bench.loc[sub.index, "p_holm"] = adj

df_bench["sig_holm"] = df_bench["p_holm"] <= ALPHA

# Resumo global por método
summary_rows = []
for method, sub in df_bench.groupby("method"):
    w = sub["n_blind"].values.astype(float)
    delta = sub["delta_rmse"].values.astype(float)
    delta_w = weighted_mean(delta, w)
    mean_b, ci_l, ci_h = bootstrap_ci_global_delta(
        sub, value_col="delta_rmse", weight_col="n_blind",
        holdout_col="holdout_region", iters=5000,
        seed=BOOTSTRAP_SEED + (abs(hash(method)) % 10000))
    verd = verdict_from_ci(ci_l, ci_h, EQUIV_DELTA_RMSE)
    wins     = int((sub["delta_rmse"] < 0.0).sum())
    wins_sig = int(((sub["delta_rmse"] < 0.0) & sub["sig_holm"]).sum()) if "sig_holm" in sub else 0
    summary_rows.append({
        "method": method,
        "k": int(sub["k"].iloc[0]),
        "subset_hash": str(sub["subset_hash"].iloc[0]),
        "delta_rmse_weighted": float(delta_w),
        "ci_low_95": float(ci_l),
        "ci_high_95": float(ci_h),
        "verdict": verd,
        "wins_delta_lt0": wins,
        "wins_sig_holm": wins_sig,
        "n_holdouts": int(len(sub)),
    })

df_summary = pd.DataFrame(summary_rows).sort_values(
    ["delta_rmse_weighted", "verdict"], ascending=[True, True]
).reset_index(drop=True)

# Baseline LO-MRO ponderado
w_base = df_base["n_blind"].values.astype(float)
def wavg(col): return float(np.average(df_base[col].values, weights=w_base))

print(f"\nBaseline ({best_method}) — LO-MRO ponderado por n_blind:")
print(f"  R²={wavg('r2'):.4f} | RMSE={wavg('rmse'):.2f} | "
      f"MAE={wavg('mae'):.2f} | Bias={wavg('bias'):+.2f} | Slope={wavg('slope'):.3f}")

print("\n" + "=" * 78)
print("🏁 RESUMO GLOBAL — ΔRMSE ponderado + CI bootstrap + veredito")
print(f"   Equivalence band: |ΔRMSE| ≤ {EQUIV_DELTA_RMSE:.1f} Mg/ha | "
      f"Wilcoxon loss: |error| | α={ALPHA}")
print("=" * 78)
cols_show = ["method", "k", "delta_rmse_weighted",
             "ci_low_95", "ci_high_95", "verdict",
             "wins_delta_lt0", "wins_sig_holm", "n_holdouts"]
print(df_summary[cols_show].to_string(index=False, justify="left"))


# ============================================================
# 6) SALVAR CSVs
# ============================================================
path_base   = os.path.join(OUTPUT_DIR, "lomro_baseline_metrics.csv")
path_bench  = os.path.join(OUTPUT_DIR, "lomro_benchmark_all.csv")
path_summ   = os.path.join(OUTPUT_DIR, "lomro_benchmark_summary.csv")

df_base.to_csv(path_base, index=False)
df_bench.to_csv(path_bench, index=False)
df_summary.to_csv(path_summ, index=False)


# ============================================================
# 7) FIGURAS
# ============================================================
print("\n📊 Gerando figuras...")

# Fig 1: scatter baseline por macro-região
plot_lomro_scatter_baseline(
    rows_baseline_raw,
    os.path.join(FIGS_DIR, "fig_lomro_scatter_baseline.png"),
    method_name=best_method)
print("   ✅ fig_lomro_scatter_baseline.png")

# Fig 2: ΔRMSE por método por holdout
if not df_bench.empty:
    plot_delta_rmse_per_holdout(
        df_bench,
        os.path.join(FIGS_DIR, "fig_delta_rmse_per_holdout.png"))
    print("   ✅ fig_delta_rmse_per_holdout.png")

# Fig 3: ΔRMSE ponderado global + CI
if not df_summary.empty:
    plot_delta_rmse_summary(
        df_summary,
        os.path.join(FIGS_DIR, "fig_delta_rmse_summary.png"))
    print("   ✅ fig_delta_rmse_summary.png")

# Fig 4: heatmap método × holdout
if not df_bench.empty:
    plot_heatmap_delta_rmse(
        df_bench,
        os.path.join(FIGS_DIR, "fig_heatmap_delta_rmse.png"))
    print("   ✅ fig_heatmap_delta_rmse.png")


# ============================================================
# 8) SUMÁRIO FINAL
# ============================================================
print("\n" + "=" * 78)
print("CONCLUÍDO — SCRIPT 3 EMS-SPATIAL v3.0")
print("=" * 78)
print(f"Script 2 dir:     {SCRIPT2_DIR}")
print(f"Output dir:       {OUTPUT_DIR}")
print(f"CSV baseline:     {path_base}")
print(f"CSV benchmark:    {path_bench}")
print(f"CSV summary:      {path_summ}")
print(f"Figures dir:      {FIGS_DIR}")
print("  fig_lomro_scatter_baseline.png    (obs vs pred baseline por macro-região)")
print("  fig_delta_rmse_per_holdout.png   (ΔRMSE por método por holdout)")
print("  fig_delta_rmse_summary.png       (ΔRMSE global + CI bootstrap)")
print("  fig_heatmap_delta_rmse.png       (heatmap método × holdout)")
print("=" * 78)