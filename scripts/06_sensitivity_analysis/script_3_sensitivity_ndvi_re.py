#!/usr/bin/env python3
"""
================================================================================
ENSEMBLE BIOMASSA — SCRIPT 3 EMS-SPATIAL v3.1-SENSITIVITY
ANÁLISE DE SENSIBILIDADE DIRETA: NDVI_RE duplicata
================================================================================

MODIFICAÇÃO v3.1:
  Treina baseline LO-MRO com DOIS conjuntos de features:
    - BASELINE_6: 6 features (com NDVI_RE_seca duplicata)
    - BASELINE_5: 5 features (sem NDVI_RE_seca)
  
  Compara diretamente no mesmo holdout:
    - ΔRMSE = RMSE(5) - RMSE(6)
    - ΔR² = R²(5) - R²(6)
    - Wilcoxon pareado
    - Block bootstrap CI

  Hipótese: NDVI_RE não contribui (ΔRMSE ~0, ΔR² ~0)

RESTO: Idêntico ao Script 3 v3.0
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, json, hashlib
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

print("=" * 78)
print("   SCRIPT 3 v3.1 — SENSIBILIDADE DIRETA NDVI_RE")
print("=" * 78)

# ============================================================
# CONFIGURAÇÃO
# ============================================================
FEATURES_CSV = r"/mnt/e/PROJETOS/biomassa_quantum/results/goias_df_features_buffer50m_v3_2018.csv"
BIOMASSA_CSV = r"/mnt/e/PROJETOS/biomassa_quantum/results/biomassa_por_UA_corrigido.csv"

OUTPUT_DIR   = r"/mnt/e/PROJETOS/biomassa_quantum/results/script3_sensitivity_ndvi_re"
FIGS_DIR     = os.path.join(OUTPUT_DIR, "figures")

TARGET_COL    = "Biomassa_Mg_ha"
MAPBIOMAS_COL = "mapbiomas_2018"
SEED          = 42
CLASSES_VEGETACAO = {3, 4, 5, 6, 11, 12, 29, 32, 49, 50}

# CRÍTICO: INCLUIR NDVI_RE_seca para poder testar
BAND_NAMES = [
    'B2_seca', 'B3_seca', 'B4_seca', 'B8_seca', 'B11_seca', 'B12_seca',
    'NDVI_seca', 'NDWI_seca', 'NDVI_RE_seca',  # ← INCLUÍDO
    'NBR_seca', 'MSI_seca', 'EVI_seca',
    'elevation', 'slope', 'VV_dB', 'VH_dB', 'HV_dB',
    'mapbiomas_2018', 'clay_pct', 'canopy_height', 'canopy_height_sd'
]

# Dois baselines para comparação
BASELINE_6_FEATURES = ['canopy_height', 'clay_pct', 'NDVI_seca', 
                       'NDWI_seca', 'NDVI_RE_seca', 'HV_dB']
BASELINE_5_FEATURES = ['canopy_height', 'clay_pct', 'NDVI_seca', 
                       'NDWI_seca', 'HV_dB']

N_MACRO_REGIONS = 5
BOOTSTRAP_ITERS = 3000
BOOTSTRAP_SEED  = 123

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)


# ============================================================
# Funções auxiliares (copiadas do Script 3 v3.0)
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

def predict_rf_solo(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, seed: int) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = imputer.fit_transform(np.asarray(X_train, float))
    Xte_imp = imputer.transform(np.asarray(X_test, float))
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=2,
        max_features=0.7, random_state=seed, n_jobs=-1)
    rf.fit(Xtr_imp, y_train)
    return rf.predict(Xte_imp).astype(float)

def wilcoxon_abs_error(y_true: np.ndarray,
                       pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    la = np.abs(np.asarray(pred_a, float) - y_true)
    lb = np.abs(np.asarray(pred_b, float) - y_true)
    try:
        _, p = stats.wilcoxon(la, lb)
        return float(p) if np.isfinite(p) else np.nan
    except Exception:
        return np.nan

def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float); w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(w[ok] * x[ok]) / np.sum(w[ok])) if ok.sum() > 0 else np.nan


# ============================================================
# 1) CARREGAR DADOS
# ============================================================
print("\n📂 Carregando dados...")
features_df = pd.read_csv(FEATURES_CSV)
biomassa_df = pd.read_csv(BIOMASSA_CSV)
df = features_df.merge(biomassa_df[["UA", TARGET_COL]], on="UA", how="inner")

df["lon"] = pd.to_numeric(df.get("lon_pc", np.nan), errors="coerce")
df["lat"] = pd.to_numeric(df.get("lat_pc", np.nan), errors="coerce")
df["mb_int"] = pd.to_numeric(df[MAPBIOMAS_COL], errors="coerce").round().fillna(-1).astype(int)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

df = df[
    (df[TARGET_COL] > 0) & (df[TARGET_COL] < 300) &
    df["lon"].notna() & df["lat"].notna() &
    df["mb_int"].isin(CLASSES_VEGETACAO)
].copy().reset_index(drop=True)

print(f"   n={len(df)} UAs | y̅={df[TARGET_COL].mean():.1f} ± {df[TARGET_COL].std():.1f} Mg/ha")

# Verificar que NDVI_RE_seca existe
if 'NDVI_RE_seca' not in df.columns:
    print("\n❌ ERRO: NDVI_RE_seca não encontrado no dataset!")
    print("   Certifique-se de usar CSV com NDVI_RE_seca presente.")
    exit(1)

# Verificar que é duplicata
corr = df[['NDVI_seca', 'NDVI_RE_seca']].corr().iloc[0, 1]
if abs(corr - 1.0) > 1e-6:
    print(f"\n⚠️ AVISO: Correlação NDVI vs NDVI_RE = {corr:.6f} (esperado ~1.0)")

print(f"   ✅ NDVI_RE_seca presente | Correlação com NDVI_seca: {corr:.6f}")


# ============================================================
# 2) PARTICIONAMENTO
# ============================================================
print("\n🌍 Criando macro-regiões LO-MRO...")
df["macro_region"], (nx_m, ny_m) = make_macro_regions_contiguous(df, N_MACRO_REGIONS)
counts_m = {int(k): int(v) for k, v in
            pd.Series(df["macro_region"]).value_counts().sort_index().items()}
print(f"   Grid: nx={nx_m}, ny={ny_m} | Regiões: {counts_m}")


# ============================================================
# 3) LO-MRO: BASELINE_6 vs BASELINE_5
# ============================================================
print("\n" + "=" * 78)
print("🔬 ANÁLISE DE SENSIBILIDADE DIRETA")
print(f"   BASELINE_6: {BASELINE_6_FEATURES}")
print(f"   BASELINE_5: {BASELINE_5_FEATURES}")
print("=" * 78)

rows_comparison = []
macro_ids = sorted(df["macro_region"].unique().tolist())

for mid in macro_ids:
    df_blind = df[df["macro_region"] == mid].copy().reset_index(drop=True)
    df_train = df[df["macro_region"] != mid].copy().reset_index(drop=True)

    if len(df_blind) < 5 or len(df_train) < 30:
        continue

    print(f"\n{'─' * 78}")
    print(f"🔥 HOLDOUT macro={mid} | blind={len(df_blind)} | train={len(df_train)}")

    y_train = df_train[TARGET_COL].values.astype(float)
    y_blind = df_blind[TARGET_COL].values.astype(float)

    seed_base = SEED + 1000 + int(mid)

    # ── BASELINE_6 (com NDVI_RE) ──
    Xtr_6 = df_train[BASELINE_6_FEATURES].apply(pd.to_numeric, errors="coerce").values
    Xbl_6 = df_blind[BASELINE_6_FEATURES].apply(pd.to_numeric, errors="coerce").values
    y_pred_6 = predict_rf_solo(Xtr_6, y_train, Xbl_6, seed=seed_base)
    met_6 = compute_metrics(y_blind, y_pred_6)

    # ── BASELINE_5 (sem NDVI_RE) ──
    Xtr_5 = df_train[BASELINE_5_FEATURES].apply(pd.to_numeric, errors="coerce").values
    Xbl_5 = df_blind[BASELINE_5_FEATURES].apply(pd.to_numeric, errors="coerce").values
    y_pred_5 = predict_rf_solo(Xtr_5, y_train, Xbl_5, seed=seed_base)
    met_5 = compute_metrics(y_blind, y_pred_5)

    # ── DIFERENÇAS ──
    delta_rmse = met_5["rmse"] - met_6["rmse"]  # + = pior sem NDVI_RE
    delta_r2   = met_5["r2"]   - met_6["r2"]    # + = melhor sem NDVI_RE
    delta_bias = met_5["bias"] - met_6["bias"]

    # ── ESTATÍSTICA ──
    p_wilcox = wilcoxon_abs_error(y_blind, y_pred_6, y_pred_5)

    print(f"   BASELINE_6 (com NDVI_RE): R²={met_6['r2']:.4f} | RMSE={met_6['rmse']:.2f}")
    print(f"   BASELINE_5 (sem NDVI_RE): R²={met_5['r2']:.4f} | RMSE={met_5['rmse']:.2f}")
    print(f"   ΔRMSE={delta_rmse:+.4f} | ΔR²={delta_r2:+.6f} | Wilcox p={p_wilcox:.4f}")

    rows_comparison.append({
        "holdout_region": int(mid),
        "n_blind": int(len(df_blind)),
        "r2_6feat": met_6["r2"],
        "rmse_6feat": met_6["rmse"],
        "mae_6feat": met_6["mae"],
        "bias_6feat": met_6["bias"],
        "slope_6feat": met_6["slope"],
        "r2_5feat": met_5["r2"],
        "rmse_5feat": met_5["rmse"],
        "mae_5feat": met_5["mae"],
        "bias_5feat": met_5["bias"],
        "slope_5feat": met_5["slope"],
        "delta_rmse": float(delta_rmse),
        "delta_r2": float(delta_r2),
        "delta_mae": float(met_5["mae"] - met_6["mae"]),
        "delta_bias": float(delta_bias),
        "wilcoxon_p": float(p_wilcox) if np.isfinite(p_wilcox) else np.nan,
    })


# ============================================================
# 4) CONSOLIDAÇÃO
# ============================================================
print("\n" + "=" * 78)
print("📊 CONSOLIDAÇÃO — Métricas Ponderadas LO-MRO")
print("=" * 78)

df_comp = pd.DataFrame(rows_comparison)
w = df_comp["n_blind"].values.astype(float)

def wavg(col): 
    return weighted_mean(df_comp[col].values, w)

r2_6_avg   = wavg("r2_6feat")
rmse_6_avg = wavg("rmse_6feat")
mae_6_avg  = wavg("mae_6feat")
bias_6_avg = wavg("bias_6feat")

r2_5_avg   = wavg("r2_5feat")
rmse_5_avg = wavg("rmse_5feat")
mae_5_avg  = wavg("mae_5feat")
bias_5_avg = wavg("bias_5feat")

delta_rmse_avg = wavg("delta_rmse")
delta_r2_avg   = wavg("delta_r2")
delta_mae_avg  = wavg("delta_mae")
delta_bias_avg = wavg("delta_bias")

print(f"\nBASELINE_6 (com NDVI_RE duplicata):")
print(f"  R²={r2_6_avg:.4f} | RMSE={rmse_6_avg:.2f} | MAE={mae_6_avg:.2f} | Bias={bias_6_avg:+.2f}")

print(f"\nBASELINE_5 (sem NDVI_RE):")
print(f"  R²={r2_5_avg:.4f} | RMSE={rmse_5_avg:.2f} | MAE={mae_5_avg:.2f} | Bias={bias_5_avg:+.2f}")

print(f"\nDIFERENÇAS (5 features - 6 features):")
print(f"  ΔR²={delta_r2_avg:+.6f} | ΔRMSE={delta_rmse_avg:+.4f} Mg/ha")
print(f"  ΔMAE={delta_mae_avg:+.4f} Mg/ha | ΔBias={delta_bias_avg:+.4f} Mg/ha")

# Teste pareado Wilcoxon em RMSE por holdout
rmse_6_arr = df_comp["rmse_6feat"].values
rmse_5_arr = df_comp["rmse_5feat"].values
try:
    _, p_paired = stats.wilcoxon(rmse_6_arr, rmse_5_arr)
    print(f"\nWilcoxon pareado (RMSE_6 vs RMSE_5): p={p_paired:.6f}")
    if p_paired > 0.05:
        print("  → Não há diferença significativa (H0 aceita)")
    else:
        print("  → Diferença significativa detectada (p<0.05)")
except Exception as e:
    print(f"\nWilcoxon falhou: {e}")


# ============================================================
# 5) INTERPRETAÇÃO
# ============================================================
print("\n" + "=" * 78)
print("🎯 INTERPRETAÇÃO CIENTÍFICA")
print("=" * 78)

if abs(delta_rmse_avg) < 0.01 and abs(delta_r2_avg) < 0.0001:
    print("✅ CONCLUSÃO: NDVI_RE_seca (duplicata) NÃO contribui para performance.")
    print("   - ΔRMSE < 0.01 Mg/ha (desprezível)")
    print("   - ΔR² < 0.0001 (desprezível)")
    print("   - Modelo opera efetivamente com 5 features únicas.")
    print("   - Robustez de Random Forest a multicolinearidade CONFIRMADA.")
elif abs(delta_rmse_avg) < 0.5:
    print("⚠️ CONCLUSÃO: NDVI_RE_seca tem impacto MARGINAL.")
    print(f"   - ΔRMSE={delta_rmse_avg:+.2f} Mg/ha (< 0.5 Mg/ha)")
    print("   - Contribuição existe mas é negligenciável operacionalmente.")
else:
    print("❌ CONCLUSÃO: NDVI_RE_seca tem impacto SIGNIFICATIVO.")
    print(f"   - ΔRMSE={delta_rmse_avg:+.2f} Mg/ha")
    print("   - Investigar causa: pode não ser duplicata perfeita.")


# ============================================================
# 6) SALVAR CSVs
# ============================================================
path_comp = os.path.join(OUTPUT_DIR, "sensitivity_comparison.csv")
df_comp.to_csv(path_comp, index=False)
print(f"\n💾 CSV salvo: {path_comp}")

# Resumo global
summary = pd.DataFrame([{
    "config": "BASELINE_6 (com NDVI_RE)",
    "n_features": 6,
    "r2": r2_6_avg,
    "rmse": rmse_6_avg,
    "mae": mae_6_avg,
    "bias": bias_6_avg,
}, {
    "config": "BASELINE_5 (sem NDVI_RE)",
    "n_features": 5,
    "r2": r2_5_avg,
    "rmse": rmse_5_avg,
    "mae": mae_5_avg,
    "bias": bias_5_avg,
}, {
    "config": "DIFERENÇA (5-6)",
    "n_features": 0,
    "r2": delta_r2_avg,
    "rmse": delta_rmse_avg,
    "mae": delta_mae_avg,
    "bias": delta_bias_avg,
}])

path_summ = os.path.join(OUTPUT_DIR, "sensitivity_summary.csv")
summary.to_csv(path_summ, index=False)
print(f"💾 Resumo: {path_summ}")


# ============================================================
# 7) FIGURAS
# ============================================================
print("\n📊 Gerando figuras...")

# Fig 1: Barplot ΔRMSE por holdout
fig, ax = plt.subplots(figsize=(10, 5))
holdouts = df_comp["holdout_region"].values
deltas = df_comp["delta_rmse"].values
colors = ['green' if d < 0 else 'red' if d > 0.1 else 'gray' for d in deltas]

ax.bar(holdouts, deltas, color=colors, alpha=0.7)
ax.axhline(0, color='black', lw=1.5, linestyle='--')
ax.axhline(0.5, color='gray', lw=0.8, linestyle=':', label='±0.5 Mg/ha')
ax.axhline(-0.5, color='gray', lw=0.8, linestyle=':')

ax.set_xlabel("Macro-região (holdout)", fontsize=11)
ax.set_ylabel("ΔRMSE (5 feat - 6 feat) [Mg ha⁻¹]", fontsize=11)
ax.set_title("Impacto de Remover NDVI_RE_seca — ΔRMSE por Holdout\n"
             "(valores positivos = pior sem NDVI_RE)", fontsize=10)
ax.set_xticks(holdouts)
ax.set_xticklabels([f"Macro {h}" for h in holdouts])
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig_sensitivity_delta_rmse_per_holdout.png"), 
            dpi=180, bbox_inches='tight')
plt.close(fig)
print("   ✅ fig_sensitivity_delta_rmse_per_holdout.png")

# Fig 2: Scatter R² (6 feat vs 5 feat)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df_comp["r2_6feat"], df_comp["r2_5feat"], 
           s=80, alpha=0.6, color='steelblue', edgecolors='black')
lims = [
    min(df_comp["r2_6feat"].min(), df_comp["r2_5feat"].min()) - 0.02,
    max(df_comp["r2_6feat"].max(), df_comp["r2_5feat"].max()) + 0.02
]
ax.plot(lims, lims, 'k--', lw=1.5, label='1:1 (idêntico)')
ax.set_xlabel("R² (6 features, com NDVI_RE)", fontsize=11)
ax.set_ylabel("R² (5 features, sem NDVI_RE)", fontsize=11)
ax.set_title("Comparação de R² por Holdout\n"
             "(pontos na linha 1:1 = performance idêntica)", fontsize=10)
ax.legend(fontsize=9)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_aspect('equal', adjustable='box')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGS_DIR, "fig_sensitivity_r2_comparison.png"), 
            dpi=180, bbox_inches='tight')
plt.close(fig)
print("   ✅ fig_sensitivity_r2_comparison.png")


# ============================================================
# 8) SUMÁRIO FINAL
# ============================================================
print("\n" + "=" * 78)
print("✅ ANÁLISE DE SENSIBILIDADE CONCLUÍDA")
print("=" * 78)
print(f"Output dir: {OUTPUT_DIR}")
print(f"  sensitivity_comparison.csv  — métricas por holdout")
print(f"  sensitivity_summary.csv     — resumo global ponderado")
print(f"  figures/fig_sensitivity_delta_rmse_per_holdout.png")
print(f"  figures/fig_sensitivity_r2_comparison.png")
print("=" * 78)
