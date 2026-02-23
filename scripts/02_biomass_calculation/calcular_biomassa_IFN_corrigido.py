#!/usr/bin/env python3
# ============================================================
# calcular_biomassa_IFN_corrigido.py
# ============================================================
# Calcula biomassa (Mg/ha) por UA a partir dos dados brutos do IFN-GO,
# corrigindo os 6 problemas identificados na auditoria:
#
#   1. Fator de expansão por nível (área efetiva por UA, não fixa 0.4 ha)
#   2. Área efetiva calculada por UA (UAs com < 4 subunidades)
#   3. Exclusão de lianas (HAB=5) e bambus (HAB=4)
#   4. Exclusão de árvores mortas em pé (SA=4)
#   5. Tratamento correto de múltiplos fustes (DAP equivalente)
#   6. Fator de correção logarítmica (CF) aplicado
#
# Equações alométricas:
#   - Ribeiro et al. (2011): cerrado sensu stricto, R²=0.898
#     ln(AGB) = -2.977 + 2.119*ln(DAP) + 0.632*ln(rho)
#   - Chave et al. (2014): pan-tropical, para formações florestais
#     AGB = 0.0673 * (rho * DAP² * Ht)^0.976
#
# Entrada:
#   - dap_10_go.xlsx  (DAP >= 10 cm, subunidade inteira 20x50m)
#   - dap_5_go.xlsx   (5 <= DAP < 10 cm, subparcelas 10x10m)
#
# Saída:
#   - biomassa_por_UA_corrigido.csv
#   - biomassa_por_arvore_corrigido.csv
#   - diagnostico_biomassa.csv
# ============================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAÇÃO - EDITAR AQUI
# ============================================================
DATA_DIR = r"E:\PROJETOS\biomassa_quantum\data"
OUTPUT_DIR = DATA_DIR  # Salva no mesmo diretório

DAP10_FILE = os.path.join(DATA_DIR, "dap_10_go.xlsx")
DAP5_FILE  = os.path.join(DATA_DIR, "dap_5_go.xlsx")

# Global Wood Density Database (opcional - se não tiver, usa fallback)
GWDD_FILE = None  # Ex: r"E:\...\GlobalWoodDensityDatabase.xls"

# Densidade média do Cerrado (fallback robusto)
# Fonte: Ribeiro et al. (2011), Miranda et al. (2014)
RHO_CERRADO_MEDIA = 0.60  # g/cm³

# Fator de correção logarítmica (CF)
# Estimado a partir da variância residual típica de equações log no Cerrado
# CF = exp(MSE/2), com MSE ~ 0.20-0.30 para equações bem ajustadas
CF_RIBEIRO = 1.13   # Ribeiro et al. (2011) - cerrado s.s.
CF_CHAVE   = 1.08   # Chave et al. (2014) - pan-tropical

# ============================================================
# TABELA DE DENSIDADES POR FAMÍLIA (Cerrado)
# Fonte: Global Wood Density Database, medianas para famílias
# comuns no Cerrado brasileiro
# ============================================================
RHO_POR_FAMILIA = {
    "Fabaceae":          0.72,
    "Vochysiaceae":      0.52,
    "Anacardiaceae":     0.55,
    "Myrtaceae":         0.75,
    "Malvaceae":         0.48,
    "Dilleniaceae":      0.45,
    "Sapindaceae":       0.58,
    "Annonaceae":        0.50,
    "Combretaceae":      0.68,
    "Arecaceae":         0.40,  # palmeiras - equação diferente, mas rho ref
    "Apocynaceae":       0.52,
    "Chrysobalanaceae":  0.72,
    "Burseraceae":       0.42,
    "Bignoniaceae":      0.58,
    "Malpighiaceae":     0.62,
    "Rubiaceae":         0.58,
    "Melastomataceae":   0.55,
    "Caryocaraceae":     0.72,
    "Calophyllaceae":    0.55,
    "Erythroxylaceae":   0.70,
    "Salicaceae":        0.45,
    "Styracaceae":       0.50,
    "Lauraceae":         0.50,
    "Moraceae":          0.45,
    "Proteaceae":        0.55,
    "Connaraceae":       0.55,
    "Loganiaceae":       0.55,
    "Ochnaceae":         0.60,
    "Nyctaginaceae":     0.62,
    "Lythraceae":        0.58,
}

# ============================================================
# FUNÇÕES
# ============================================================

def atribuir_rho(df):
    """
    Atribui densidade da madeira (rho, g/cm³) usando hierarquia:
      1. Família (tabela acima)
      2. Fallback: média do Cerrado (0.60)
    Registra a fonte em coluna 'rho_fonte'.
    """
    rho = np.full(len(df), np.nan)
    fonte = np.full(len(df), "fallback_cerrado", dtype=object)

    if "family" in df.columns:
        for i, fam in enumerate(df["family"].values):
            if pd.notna(fam) and fam in RHO_POR_FAMILIA:
                rho[i] = RHO_POR_FAMILIA[fam]
                fonte[i] = f"familia_{fam}"

    # Fallback
    mask_nan = np.isnan(rho)
    rho[mask_nan] = RHO_CERRADO_MEDIA

    df = df.copy()
    df["rho"] = rho
    df["rho_fonte"] = fonte
    return df


def calcular_dap_equivalente(group):
    """
    Para árvores com múltiplos fustes (mesmo UA/Subunidade/Subparcela/Narv),
    calcula o DAP equivalente pela soma das áreas basais:
      DAP_eq = sqrt(sum(DAP_i²))
    Retorna UMA linha por árvore com DAP_eq, HT (máximo entre fustes),
    e demais atributos do fuste principal (Nfuste=1).
    """
    if len(group) == 1:
        row = group.iloc[0].copy()
        row["DAP_eq"] = row["DAP"]
        row["n_fustes_real"] = 1
        return row

    # DAP equivalente: sqrt(soma dos DAP²)
    dap_eq = np.sqrt(np.sum(group["DAP"].values ** 2))
    # Altura: máxima entre fustes
    ht_max = group["HT"].max()
    hf_max = group["HF"].max()

    # Pegar atributos do fuste principal (Nfuste==1 se existir)
    fuste1 = group[group["Nfuste"] == 1]
    if len(fuste1) > 0:
        row = fuste1.iloc[0].copy()
    else:
        row = group.iloc[0].copy()

    row["DAP"] = dap_eq  # substituir pelo equivalente
    row["DAP_eq"] = dap_eq
    row["HT"] = ht_max
    row["HF"] = hf_max
    row["n_fustes_real"] = len(group)
    return row


def biomassa_ribeiro(dap, rho, cf=CF_RIBEIRO):
    """
    Ribeiro et al. (2011) - Cerrado sensu stricto
    ln(AGB_kg) = -2.977 + 2.119 * ln(DAP_cm) + 0.632 * ln(rho_g_cm3)
    R² = 0.898
    Retorna AGB em kg.
    """
    ln_agb = -2.977 + 2.119 * np.log(dap) + 0.632 * np.log(rho)
    return np.exp(ln_agb) * cf


def biomassa_chave(dap, ht, rho, cf=CF_CHAVE):
    """
    Chave et al. (2014) - Pan-tropical (formações florestais)
    AGB_kg = 0.0673 * (rho * DAP² * Ht)^0.976
    Retorna AGB em kg.
    """
    return 0.0673 * ((rho * dap**2 * ht) ** 0.976) * cf


# ============================================================
# 1. CARREGAR DADOS BRUTOS
# ============================================================
print("=" * 70)
print("CÁLCULO DE BIOMASSA IFN-GO — VERSÃO CORRIGIDA")
print("=" * 70)

print("\n📂 Carregando dados brutos...")
dap10 = pd.read_excel(DAP10_FILE)
dap5  = pd.read_excel(DAP5_FILE)

print(f"   DAP>=10: {len(dap10):,} registros, {dap10['UA'].nunique()} UAs")
print(f"   DAP 5-10: {len(dap5):,} registros, {dap5['UA'].nunique()} UAs")

# ============================================================
# 2. FILTROS DE QUALIDADE
# ============================================================
print("\n🧹 Aplicando filtros de qualidade...")

# --- 2a. Excluir árvores mortas em pé (SA=4) ---
n_mortas_10 = (dap10["SA"] == 4).sum()
n_mortas_5  = (dap5["SA"] == 4).sum()
dap10 = dap10[dap10["SA"] != 4].copy()
dap5  = dap5[dap5["SA"] != 4].copy()
print(f"   Mortas removidas: {n_mortas_10} (DAP10) + {n_mortas_5} (DAP5)")

# --- 2b. Excluir lianas (HAB=5) e bambus (HAB=4) ---
n_liana_10 = (dap10["HAB"] == 5).sum()
n_bambu_10 = (dap10["HAB"] == 4).sum()
n_liana_5  = (dap5["HAB"] == 5).sum() if "HAB" in dap5.columns else 0
n_bambu_5  = (dap5["HAB"] == 4).sum() if "HAB" in dap5.columns else 0

dap10 = dap10[~dap10["HAB"].isin([4, 5])].copy()
if "HAB" in dap5.columns:
    dap5 = dap5[~dap5["HAB"].isin([4, 5])].copy()

print(f"   Lianas removidas: {n_liana_10} (DAP10) + {n_liana_5} (DAP5)")
print(f"   Bambus removidos: {n_bambu_10} (DAP10) + {n_bambu_5} (DAP5)")

# --- 2c. Excluir cactos (HAB=6) se houver ---
if (dap10["HAB"] == 6).any():
    n_cacto = (dap10["HAB"] == 6).sum()
    dap10 = dap10[dap10["HAB"] != 6].copy()
    print(f"   Cactos removidos: {n_cacto}")

# --- 2d. Filtrar DAP e HT válidos ---
dap10 = dap10[(dap10["DAP"] >= 10) & (dap10["HT"] > 0)].copy()
dap5  = dap5[(dap5["DAP"] >= 5) & (dap5["DAP"] < 10) & (dap5["HT"] > 0)].copy()

print(f"\n   Após filtros: {len(dap10):,} (DAP10) + {len(dap5):,} (DAP5)")

# ============================================================
# 3. TRATAR MÚLTIPLOS FUSTES (DAP equivalente)
# ============================================================
print("\n🌳 Tratando múltiplos fustes (DAP equivalente)...")

# Identificar grupos de fustes: mesmo UA + Subunidade + Subparcela + Narv
grupo_cols = ["UA", "Subunidade", "Subparcela", "Narv"]

# Marcar quantos fustes tem cada árvore
fuste_counts = dap10.groupby(grupo_cols).size().reset_index(name="_n_fustes")
n_multi = (fuste_counts["_n_fustes"] > 1).sum()
print(f"   Árvores com múltiplos fustes: {n_multi}")

# Aplicar consolidação
arvores_10 = []
for name, group in dap10.groupby(grupo_cols):
    arvores_10.append(calcular_dap_equivalente(group))

dap10_consolidado = pd.DataFrame(arvores_10)
print(f"   DAP10 após consolidação: {len(dap10_consolidado):,} árvores únicas")

# DAP5 - mesmo procedimento (embora fustes múltiplos sejam raros)
if "Narv" in dap5.columns:
    grupo_cols_5 = ["UA", "Subunidade", "Subparcela", "Narv"]
    arvores_5 = []
    for name, group in dap5.groupby(grupo_cols_5):
        arvores_5.append(calcular_dap_equivalente(group))
    dap5_consolidado = pd.DataFrame(arvores_5)
else:
    dap5_consolidado = dap5.copy()
    dap5_consolidado["DAP_eq"] = dap5_consolidado["DAP"]
    dap5_consolidado["n_fustes_real"] = 1

print(f"   DAP5 após consolidação: {len(dap5_consolidado):,} árvores únicas")

# ============================================================
# 4. ATRIBUIR DENSIDADE DA MADEIRA (rho)
# ============================================================
print("\n🪵 Atribuindo densidade da madeira...")

dap10_consolidado = atribuir_rho(dap10_consolidado)
dap5_consolidado  = atribuir_rho(dap5_consolidado)

# Diagnóstico
for label, df in [("DAP10", dap10_consolidado), ("DAP5", dap5_consolidado)]:
    fontes = df["rho_fonte"].value_counts()
    n_familia = fontes.drop("fallback_cerrado", errors="ignore").sum()
    n_fallback = fontes.get("fallback_cerrado", 0)
    print(f"   {label}: {n_familia} por família ({n_familia/len(df)*100:.1f}%), "
          f"{n_fallback} fallback ({n_fallback/len(df)*100:.1f}%)")

# ============================================================
# 5. CALCULAR BIOMASSA POR ÁRVORE (kg)
# ============================================================
print("\n📐 Calculando biomassa individual...")

# Para DAP>=10: usar Chave (quando Ht disponível e confiável) como principal,
# Ribeiro como backup. Ambas corrigidas com CF.
d10 = dap10_consolidado
d10["agb_ribeiro_kg"] = biomassa_ribeiro(d10["DAP_eq"].values, d10["rho"].values)
d10["agb_chave_kg"]   = biomassa_chave(d10["DAP_eq"].values, d10["HT"].values, d10["rho"].values)

# Escolha: Chave quando Ht é medida (confiável), Ribeiro como alternativa
# Como o IFN mede Ht para todos, usar Chave como principal
d10["agb_kg"] = d10["agb_chave_kg"]
d10["equacao"] = "Chave2014"

# Sanity check: valores negativos ou absurdos
d10.loc[d10["agb_kg"] <= 0, "agb_kg"] = np.nan
d10.loc[d10["agb_kg"] > 5000, "agb_kg"] = np.nan  # >5 ton por árvore = suspeito

# Para DAP 5-10: Ribeiro (não precisa de Ht, mas temos)
d5 = dap5_consolidado
d5["agb_ribeiro_kg"] = biomassa_ribeiro(d5["DAP_eq"].values, d5["rho"].values)
d5["agb_chave_kg"]   = biomassa_chave(d5["DAP_eq"].values, d5["HT"].values, d5["rho"].values)
d5["agb_kg"] = d5["agb_chave_kg"]
d5["equacao"] = "Chave2014"

d5.loc[d5["agb_kg"] <= 0, "agb_kg"] = np.nan

print(f"   DAP10: agb_kg médio = {d10['agb_kg'].mean():.1f} kg, "
      f"mediana = {d10['agb_kg'].median():.1f} kg")
print(f"   DAP5:  agb_kg médio = {d5['agb_kg'].mean():.1f} kg, "
      f"mediana = {d5['agb_kg'].median():.1f} kg")

# ============================================================
# 6. CALCULAR ÁREA EFETIVA E BIOMASSA POR UA
# ============================================================
print("\n📊 Calculando biomassa por UA com fatores de expansão corretos...")

# --- 6a. COMPONENTE DAP>=10 ---
# Área efetiva = n_subunidades_medidas × 1000 m²
# Cada subunidade = 20m × 50m = 1000 m²

agb10_por_ua = d10.groupby("UA").agg(
    agb_total_kg_10=("agb_kg", "sum"),
    n_arvores_10=("agb_kg", "count"),
    n_subunidades=("Subunidade", "nunique"),
    dap_medio_10=("DAP_eq", "mean"),
    ht_media_10=("HT", "mean"),
    lon_pc=("lon_pc", "first"),
    lat_pc=("lat_pc", "first"),
    mun=("mun", "first"),
    uf=("uf", "first") if "uf" in d10.columns else ("mun", "first"),
).reset_index()

# Área efetiva DAP>=10 (m²)
agb10_por_ua["area_m2_10"] = agb10_por_ua["n_subunidades"] * 1000
agb10_por_ua["area_ha_10"] = agb10_por_ua["area_m2_10"] / 10000

# Biomassa por hectare do componente DAP>=10
agb10_por_ua["bio_Mg_ha_10"] = (
    agb10_por_ua["agb_total_kg_10"] / 1000
) / agb10_por_ua["area_ha_10"]


# --- 6b. COMPONENTE DAP 5-10 ---
# Área efetiva = n_subparcelas_10x10m_únicas × 100 m²
# No Cerrado: subparcelas 1 e 10 (+ eventualmente outras se deslocadas)

# Contar subparcelas únicas por UA
subp5_area = d5.groupby("UA").apply(
    lambda g: g[["Subunidade", "Subparcela"]].drop_duplicates().shape[0]
).reset_index(name="n_subparcelas_5")

agb5_por_ua = d5.groupby("UA").agg(
    agb_total_kg_5=("agb_kg", "sum"),
    n_arvores_5=("agb_kg", "count"),
    dap_medio_5=("DAP_eq", "mean"),
).reset_index()

agb5_por_ua = agb5_por_ua.merge(subp5_area, on="UA", how="left")

# Área efetiva DAP 5-10 (m²)
agb5_por_ua["area_m2_5"] = agb5_por_ua["n_subparcelas_5"] * 100
agb5_por_ua["area_ha_5"] = agb5_por_ua["area_m2_5"] / 10000

# Biomassa por hectare do componente DAP 5-10
agb5_por_ua["bio_Mg_ha_5"] = (
    agb5_por_ua["agb_total_kg_5"] / 1000
) / agb5_por_ua["area_ha_5"]


# --- 6c. JUNTAR COMPONENTES ---
biomassa_ua = agb10_por_ua.merge(
    agb5_por_ua[["UA", "agb_total_kg_5", "n_arvores_5", "n_subparcelas_5",
                  "area_ha_5", "bio_Mg_ha_5", "dap_medio_5"]],
    on="UA", how="left"
)

# Preencher UAs sem DAP5 com zero
for col in ["agb_total_kg_5", "n_arvores_5", "n_subparcelas_5",
            "area_ha_5", "bio_Mg_ha_5", "dap_medio_5"]:
    biomassa_ua[col] = biomassa_ua[col].fillna(0)

# BIOMASSA TOTAL = componente DAP>=10 + componente DAP 5-10
# Cada componente já está em Mg/ha (expandido pela sua área efetiva)
biomassa_ua["Biomassa_Mg_ha"] = (
    biomassa_ua["bio_Mg_ha_10"] + biomassa_ua["bio_Mg_ha_5"]
)

biomassa_ua["n_arvores_total"] = (
    biomassa_ua["n_arvores_10"] + biomassa_ua["n_arvores_5"]
)

# ============================================================
# 7. FLAGS DE QUALIDADE
# ============================================================
print("\n🚩 Gerando flags de qualidade...")

biomassa_ua["flag_subunidades_incompletas"] = biomassa_ua["n_subunidades"] < 4
biomassa_ua["flag_poucas_arvores"] = biomassa_ua["n_arvores_total"] < 10
biomassa_ua["flag_biomassa_extrema"] = (
    (biomassa_ua["Biomassa_Mg_ha"] > 300) |
    (biomassa_ua["Biomassa_Mg_ha"] < 0.5)
)

n_incompletas = biomassa_ua["flag_subunidades_incompletas"].sum()
n_poucas = biomassa_ua["flag_poucas_arvores"].sum()
n_extremas = biomassa_ua["flag_biomassa_extrema"].sum()

print(f"   Subunidades incompletas: {n_incompletas} UAs ({n_incompletas/len(biomassa_ua)*100:.1f}%)")
print(f"   Poucas árvores (<10): {n_poucas} UAs")
print(f"   Biomassa extrema: {n_extremas} UAs")

# ============================================================
# 8. ADICIONAR UAs SEM NENHUMA ÁRVORE (biomassa = 0)
# ============================================================
# UAs que aparecem apenas no DAP5 (sem árvores DAP>=10)
uas_so_dap5 = set(d5["UA"].unique()) - set(d10["UA"].unique())
if uas_so_dap5:
    print(f"\n   UAs apenas com DAP5 (sem DAP>=10): {len(uas_so_dap5)}")
    for ua in uas_so_dap5:
        d5_ua = d5[d5["UA"] == ua]
        subp_info = d5_ua[["Subunidade", "Subparcela"]].drop_duplicates()
        n_subp = len(subp_info)
        area_ha = n_subp * 100 / 10000
        agb_sum = d5_ua["agb_kg"].sum()
        bio_mg_ha = (agb_sum / 1000) / area_ha if area_ha > 0 else 0

        row = {
            "UA": ua,
            "agb_total_kg_10": 0, "n_arvores_10": 0, "n_subunidades": 0,
            "dap_medio_10": 0, "ht_media_10": 0,
            "lon_pc": d5_ua["lon_pc"].iloc[0],
            "lat_pc": d5_ua["lat_pc"].iloc[0],
            "mun": d5_ua["mun"].iloc[0],
            "uf": d5_ua["uf"].iloc[0] if "uf" in d5_ua.columns else "",
            "area_m2_10": 0, "area_ha_10": 0, "bio_Mg_ha_10": 0,
            "agb_total_kg_5": agb_sum, "n_arvores_5": len(d5_ua),
            "n_subparcelas_5": n_subp, "area_ha_5": area_ha,
            "bio_Mg_ha_5": bio_mg_ha, "dap_medio_5": d5_ua["DAP_eq"].mean(),
            "Biomassa_Mg_ha": bio_mg_ha,
            "n_arvores_total": len(d5_ua),
            "flag_subunidades_incompletas": True,
            "flag_poucas_arvores": len(d5_ua) < 10,
            "flag_biomassa_extrema": bio_mg_ha > 300 or bio_mg_ha < 0.5,
        }
        biomassa_ua = pd.concat([biomassa_ua, pd.DataFrame([row])], ignore_index=True)

# ============================================================
# 9. ESTATÍSTICAS FINAIS
# ============================================================
print("\n" + "=" * 70)
print("📊 ESTATÍSTICAS FINAIS")
print("=" * 70)

bio = biomassa_ua["Biomassa_Mg_ha"]
print(f"\n   Total de UAs: {len(biomassa_ua)}")
print(f"   Biomassa (Mg/ha):")
print(f"     Média:   {bio.mean():.1f}")
print(f"     Mediana: {bio.median():.1f}")
print(f"     Min:     {bio.min():.1f}")
print(f"     Max:     {bio.max():.1f}")
print(f"     Desvio:  {bio.std():.1f}")
print(f"     Q25:     {bio.quantile(0.25):.1f}")
print(f"     Q75:     {bio.quantile(0.75):.1f}")

# Componentes
print(f"\n   Componente DAP>=10: média = {biomassa_ua['bio_Mg_ha_10'].mean():.1f} Mg/ha")
print(f"   Componente DAP 5-10: média = {biomassa_ua['bio_Mg_ha_5'].mean():.1f} Mg/ha")
print(f"   Contribuição DAP 5-10: {biomassa_ua['bio_Mg_ha_5'].mean() / bio.mean() * 100:.1f}%")

# Distribuição por município (top 10)
print(f"\n   Top 10 municípios por n° de UAs:")
mun_counts = biomassa_ua.groupby("mun").agg(
    n_uas=("UA", "count"),
    bio_media=("Biomassa_Mg_ha", "mean"),
    bio_mediana=("Biomassa_Mg_ha", "median"),
).sort_values("n_uas", ascending=False).head(10)
for _, row in mun_counts.iterrows():
    print(f"     {row.name}: {int(row['n_uas'])} UAs, "
          f"média={row['bio_media']:.1f}, mediana={row['bio_mediana']:.1f} Mg/ha")

# ============================================================
# 10. EXPORTAR RESULTADOS
# ============================================================
print("\n💾 Exportando resultados...")

# 10a. Biomassa por UA (principal)
cols_ua = [
    "UA", "mun", "lon_pc", "lat_pc",
    "Biomassa_Mg_ha", "bio_Mg_ha_10", "bio_Mg_ha_5",
    "n_arvores_total", "n_arvores_10", "n_arvores_5",
    "n_subunidades", "n_subparcelas_5",
    "area_ha_10", "area_ha_5",
    "dap_medio_10", "ht_media_10",
    "flag_subunidades_incompletas", "flag_poucas_arvores", "flag_biomassa_extrema",
]
# Adicionar uf se existir
if "uf" in biomassa_ua.columns:
    cols_ua.insert(2, "uf")

out_ua = os.path.join(OUTPUT_DIR, "biomassa_por_UA_corrigido.csv")
biomassa_ua[cols_ua].to_csv(out_ua, index=False)
print(f"   ✅ {out_ua}")

# 10b. Biomassa por árvore (para auditoria)
d10_out = d10[["UA", "Subunidade", "Subparcela", "Narv", "Nfuste",
               "DAP_eq", "n_fustes_real", "HT", "HF", "SA",
               "HAB", "Especie_campo",
               "rho", "rho_fonte",
               "agb_ribeiro_kg", "agb_chave_kg", "agb_kg", "equacao"]].copy()
d10_out["DAP_grupo"] = "DAP10"

cols_d5 = ["UA", "Subunidade", "Subparcela"]
if "Narv" in d5.columns:
    cols_d5.append("Narv")
cols_d5 += ["Nfuste", "DAP_eq", "n_fustes_real", "HT", "HF", "SA",
            "HAB", "Especie_campo", "rho", "rho_fonte",
            "agb_ribeiro_kg", "agb_chave_kg", "agb_kg", "equacao"]

d5_out_cols = [c for c in cols_d5 if c in d5.columns]
d5_out = d5[d5_out_cols].copy()
d5_out["DAP_grupo"] = "DAP5"

# Garantir mesmas colunas
for c in d10_out.columns:
    if c not in d5_out.columns:
        d5_out[c] = np.nan

arvores_all = pd.concat([d10_out, d5_out[d10_out.columns]], ignore_index=True)
out_arv = os.path.join(OUTPUT_DIR, "biomassa_por_arvore_corrigido.csv")
arvores_all.to_csv(out_arv, index=False)
print(f"   ✅ {out_arv}")

# 10c. Diagnóstico resumido
diag = pd.DataFrame({
    "Métrica": [
        "Total UAs",
        "UAs com 4 subunidades",
        "UAs com < 4 subunidades",
        "Total árvores (DAP>=10, vivas, sem lianas/bambus)",
        "Total árvores (DAP 5-10, vivas)",
        "Árvores com múltiplos fustes consolidados",
        "Biomassa média (Mg/ha)",
        "Biomassa mediana (Mg/ha)",
        "Componente DAP>=10 médio (Mg/ha)",
        "Componente DAP 5-10 médio (Mg/ha)",
        "CF aplicado (Ribeiro)",
        "CF aplicado (Chave)",
        "Equação principal",
        "Rho média usada",
        "Mortas excluídas (DAP10)",
        "Lianas excluídas (DAP10)",
        "Bambus excluídos (DAP10)",
    ],
    "Valor": [
        len(biomassa_ua),
        (biomassa_ua["n_subunidades"] == 4).sum(),
        (biomassa_ua["n_subunidades"] < 4).sum(),
        len(d10),
        len(d5),
        n_multi,
        f"{bio.mean():.1f}",
        f"{bio.median():.1f}",
        f"{biomassa_ua['bio_Mg_ha_10'].mean():.1f}",
        f"{biomassa_ua['bio_Mg_ha_5'].mean():.1f}",
        CF_RIBEIRO,
        CF_CHAVE,
        "Chave et al. (2014) + CF",
        f"{d10['rho'].mean():.3f}",
        n_mortas_10,
        n_liana_10,
        n_bambu_10,
    ],
})
out_diag = os.path.join(OUTPUT_DIR, "diagnostico_biomassa.csv")
diag.to_csv(out_diag, index=False)
print(f"   ✅ {out_diag}")

print("\n" + "=" * 70)
print("✅ PROCESSAMENTO CONCLUÍDO")
print("=" * 70)
print(f"\nArquivos gerados em: {OUTPUT_DIR}")
print(f"  1. biomassa_por_UA_corrigido.csv     — biomassa por UA (validação)")
print(f"  2. biomassa_por_arvore_corrigido.csv  — biomassa por árvore (auditoria)")
print(f"  3. diagnostico_biomassa.csv           — resumo do processamento")
print(f"\n⚠️  IMPORTANTE: Revise as UAs flagadas antes de usar na modelagem!")
print(f"    - {n_incompletas} UAs com subunidades incompletas")
print(f"    - {n_extremas} UAs com biomassa extrema (>300 ou <0.5 Mg/ha)")
