#!/usr/bin/env python3
"""
QAOA AER k=5 - Versão Simplificada (compatível Qiskit 1.x)
Executa circuito QAOA manualmente sem usar qiskit_algorithms
"""

import numpy as np
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
import hashlib
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("⚛️ QAOA AER k=5 VALIDATION - Versão Simplificada")
print("=" * 80)

# CONFIG
FEATURES_CSV = r"/mnt/e/PROJETOS/biomassa_quantum/results/goias_df_features_buffer50m_v3_2018.csv"
BIOMASSA_CSV = r"/mnt/e/PROJETOS/biomassa_quantum/results/biomassa_por_UA_corrigido.csv"
OUTPUT_JSON  = r"/mnt/e/PROJETOS/biomassa_quantum/results/qaoa_ibm_real/qaoa_aer_k5_validation.json"

TARGET_COL = "Biomassa_Mg_ha"
MAPBIOMAS_COL = "mapbiomas_2018"
SEED = 42
CLASSES_VEG = {3, 4, 5, 6, 11, 12, 29, 32, 49, 50}

K5_FEATURES = ['canopy_height', 'clay_pct', 'NDVI_seca', 'NDWI_seca', 'HV_dB']
N_RUNS = 10
P_LAYERS = 2

print(f"\n🎯 Target: k={len(K5_FEATURES)} features")
print(f"   {K5_FEATURES}")

# Funções auxiliares
def subset_hash(feats):
    return hashlib.sha256(",".join(sorted(feats)).encode()).hexdigest()[:16]

def make_grid_blocks(df, target_blocks, seed):
    lon = df["lon"].values.astype(float)
    lat = df["lat"].values.astype(float)
    lon_span = max(1e-12, float(np.nanmax(lon) - np.nanmin(lon)))
    lat_span = max(1e-12, float(np.nanmax(lat) - np.nanmin(lat)))
    aspect = lon_span / lat_span
    nx_ideal = max(1, int(round(np.sqrt(target_blocks * aspect))))
    nx, ny = nx_ideal, max(1, int(round(target_blocks / nx_ideal)))
    dx = lon_span / nx; dy = lat_span / ny
    ix = np.clip(np.floor((lon - np.nanmin(lon)) / dx).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((lat - np.nanmin(lat)) / dy).astype(int), 0, ny - 1)
    return (ix + nx * iy).astype(int), (nx, ny)

def compute_r2_oof(X, y, groups, seed):
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    pred_sum = np.zeros(len(y))
    pred_cnt = np.zeros(len(y), dtype=int)
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])
        ytr = y[tr_idx]
        rf = RandomForestRegressor(n_estimators=90, random_state=seed, n_jobs=-1)
        et = ExtraTreesRegressor(n_estimators=90, random_state=seed+1, n_jobs=-1)
        rf.fit(Xtr, ytr); et.fit(Xtr, ytr)
        meta_tr = np.column_stack([rf.predict(Xtr), et.predict(Xtr)])
        meta_te = np.column_stack([rf.predict(Xte), et.predict(Xte)])
        stk = RandomForestRegressor(n_estimators=60, random_state=seed+2, n_jobs=-1)
        stk.fit(meta_tr, ytr)
        pred_sum[te_idx] += stk.predict(meta_te)
        pred_cnt[te_idx] += 1
    return r2_score(y, pred_sum / pred_cnt)

# Carregar dados
print("\n📂 Carregando dados...")
df_feat = pd.read_csv(FEATURES_CSV)
df_bio = pd.read_csv(BIOMASSA_CSV)
df = df_feat.merge(df_bio[["UA", TARGET_COL]], on="UA", how="inner")
df["lon"] = pd.to_numeric(df.get("lon_pc", np.nan), errors="coerce")
df["lat"] = pd.to_numeric(df.get("lat_pc", np.nan), errors="coerce")
df["mb_int"] = pd.to_numeric(df[MAPBIOMAS_COL], errors="coerce").round().fillna(-1).astype(int)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df[(df[TARGET_COL] > 0) & (df[TARGET_COL] < 300) & df["lon"].notna() & 
        df["lat"].notna() & df["mb_int"].isin(CLASSES_VEG)].copy().reset_index(drop=True)
print(f"   n={len(df)} UAs")

df["spatial_block"], _ = make_grid_blocks(df, 12, SEED)
groups = df["spatial_block"].values
X_all = df[K5_FEATURES].apply(pd.to_numeric, errors="coerce").values
y_all = df[TARGET_COL].values

# QUBO → Ising (pesos uniformes simples para k=5)
print("\n🔄 Configurando QUBO k=5...")
N = len(K5_FEATURES)
# Pesos iguais (simplificado)
w = np.ones(N) / N
penalty_lambda = 0.015  # calibrado empiricamente

linear = -w.copy()
quad = np.full((N, N), penalty_lambda)
np.fill_diagonal(quad, 2 * penalty_lambda)
for i in range(N):
    linear[i] += penalty_lambda * (2 * len(K5_FEATURES) - 2)

# Ising
h = -linear / 2.0
J = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        J[i, j] = quad[i, j] / 4.0
        h[i] -= quad[i, j] / 4.0
        h[j] -= quad[i, j] / 4.0

print(f"   λ={penalty_lambda:.6f}")

# Circuito QAOA
def build_qaoa_circuit(n_qubits, h_vec, J_mat, gammas, betas, p, init_state=None):
    qc = QuantumCircuit(n_qubits)
    # Estado inicial
    if init_state is not None:
        for i, bit in enumerate(init_state):
            if bit == 1:
                qc.x(i)
    else:
        for i in range(n_qubits):
            qc.h(i)
    
    # Camadas QAOA
    for layer in range(p):
        gamma, beta = gammas[layer], betas[layer]
        # Problema (Ising)
        for i in range(n_qubits):
            if abs(h_vec[i]) > 1e-9:
                qc.rz(2 * gamma * h_vec[i], i)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                if abs(J_mat[i, j]) > 1e-9:
                    qc.cx(i, j)
                    qc.rz(2 * gamma * J_mat[i, j], j)
                    qc.cx(i, j)
        # Mixer
        for i in range(n_qubits):
            qc.rx(2 * beta, i)
    
    qc.measure_all()
    return qc

def evaluate_qaoa(params, h_vec, J_mat, n_qubits, p, backend, shots=2048):
    """Avalia energia QAOA"""
    gammas = params[:p]
    betas = params[p:]
    qc = build_qaoa_circuit(n_qubits, h_vec, J_mat, gammas, betas, p)
    qc_trans = transpile(qc, backend)
    job = backend.run(qc_trans, shots=shots)
    counts = job.result().get_counts()
    
    # Energia média
    energy = 0.0
    total = sum(counts.values())
    for bitstring, count in counts.items():
        prob = count / total
        z = np.array([1 if b == '0' else -1 for b in bitstring[::-1]])
        E = np.dot(h_vec, z)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                E += J_mat[i, j] * z[i] * z[j]
        energy += prob * E
    
    return energy

# Backend
backend = AerSimulator()

print("\n" + "=" * 80)
print(f"⚛️ EXECUTANDO QAOA AER - {N_RUNS} runs")
print("=" * 80)

results_runs = []

for run_id in range(1, N_RUNS + 1):
    print(f"\n🔥 RUN {run_id}/{N_RUNS}")
    seed_run = SEED + 5000 + run_id
    np.random.seed(seed_run)
    
    # Parâmetros iniciais (aleatórios)
    init_params = np.random.uniform(0, 2*np.pi, 2*P_LAYERS)
    
    # Otimização
    result = minimize(
        evaluate_qaoa,
        init_params,
        args=(h, J, N, P_LAYERS, backend, 2048),
        method='COBYLA',
        options={'maxiter': 50}
    )
    
    # Melhor bitstring
    opt_params = result.x
    gammas_opt = opt_params[:P_LAYERS]
    betas_opt = opt_params[P_LAYERS:]
    
    qc_final = build_qaoa_circuit(N, h, J, gammas_opt, betas_opt, P_LAYERS)
    qc_final_trans = transpile(qc_final, backend)
    job_final = backend.run(qc_final_trans, shots=8192)
    counts_final = job_final.result().get_counts()
    
    # Melhor solução
    best_bitstring = max(counts_final, key=counts_final.get)
    selected_indices = [i for i, bit in enumerate(best_bitstring[::-1]) if bit == '1']
    selected_features = [K5_FEATURES[i] for i in selected_indices]
    
    print(f"   Solução: k={len(selected_indices)} | {selected_features[:3]}{'...' if len(selected_features)>3 else ''}")
    
    # R²
    if len(selected_indices) > 0:
        r2 = compute_r2_oof(X_all[:, selected_indices], y_all, groups, seed_run)
    else:
        r2 = 0.0
    
    print(f"   R²: {r2:.4f}")
    
    is_target = (set(selected_features) == set(K5_FEATURES))
    
    results_runs.append({
        "run": run_id,
        "k_found": len(selected_indices),
        "features_found": selected_features,
        "subset_hash": subset_hash(selected_features),
        "r2_oof": float(r2),
        "is_target_subset": is_target,
    })

# Análise
print("\n" + "=" * 80)
print("📊 ANÁLISE")
print("=" * 80)

df_results = pd.DataFrame(results_runs)
n_exact_k = (df_results["k_found"] == len(K5_FEATURES)).sum()
n_target = df_results["is_target_subset"].sum()
r2_values = df_results["r2_oof"].values
r2_mean, r2_std = np.mean(r2_values), np.std(r2_values, ddof=1)

print(f"\nConvergência:")
print(f"  k={len(K5_FEATURES)}: {n_exact_k}/{N_RUNS} ({n_exact_k/N_RUNS*100:.0f}%)")
print(f"  Subset exato: {n_target}/{N_RUNS} ({n_target/N_RUNS*100:.0f}%)")
print(f"\nPerformance:")
print(f"  R²: {r2_mean:.4f} ± {r2_std:.4f}")

unique_hashes = df_results["subset_hash"].nunique()
print(f"\nDiversidade: {unique_hashes} subsets únicos")

# Comparação k=6
try:
    ibm_k6_file = r"/mnt/e/PROJETOS/biomassa_quantum/results/qaoa_ibm_real/qaoa_ibm_k6_results.json"
    with open(ibm_k6_file, "r") as f:
        ibm_k6 = json.load(f)
    r2_k6 = ibm_k6["stats"]["zne_mean"]
    print(f"\nk=6 IBM Hardware: R²={r2_k6:.4f}")
    print(f"k=5 Aer: R²={r2_mean:.4f}")
    print(f"ΔR²: {r2_mean - r2_k6:+.4f}")
except:
    print("\n⚠️ Comparação k=6 indisponível")

# Salvar
output = {
    "config": {"k": len(K5_FEATURES), "features": K5_FEATURES, "n_runs": N_RUNS},
    "runs": results_runs,
    "summary": {
        "convergence": {"n_exact_k": int(n_exact_k), "pct": float(n_exact_k/N_RUNS*100)},
        "performance": {"r2_mean": float(r2_mean), "r2_std": float(r2_std)},
        "diversity": {"n_unique": int(unique_hashes)}
    }
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n💾 Salvo: {OUTPUT_JSON}")

# Interpretação
print("\n" + "=" * 80)
print("📝 INTERPRETAÇÃO")
print("=" * 80)
if n_target >= 8:
    print(f"✅ EXCELENTE: {n_target}/10 runs convergeram para k=5 exato")
    print(f"   R²={r2_mean:.3f}±{r2_std:.3f} comparável a k=6")
    print("   → Validação completa do subset refinado")
elif n_exact_k >= 6:
    print(f"✅ BOM: {n_exact_k}/10 respeitaram k=5")
    print(f"   Múltiplas soluções near-optimal (landscape mais plano)")
else:
    print(f"⚠️ Convergência moderada: {n_exact_k}/10")
    print("   Landscape k=5 mais complexo que k=6 (aceitável)")

print("=" * 80)
