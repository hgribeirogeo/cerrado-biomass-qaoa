# Data Directory

## Overview
This directory contains processed field inventory and remote sensing feature datasets for 290 permanent forest plots in Goiás, Brazil.

## Files

### biomassa_por_UA_corrigido.csv
Field inventory data with calculated aboveground biomass.

**Columns (20):**
- `UA`: Unique plot identifier
- `mun`: Municipality name
- `uf`: State code (GO = Goiás)
- `lon_pc`, `lat_pc`: Plot center coordinates (decimal degrees, WGS84)
- `Biomassa_Mg_ha`: Total AGB (Mg/ha) - **TARGET VARIABLE**
- `bio_Mg_ha_10`: AGB from trees DAP≥10cm
- `bio_Mg_ha_5`: AGB from trees 5≤DAP<10cm
- `n_arvores_total`, `n_arvores_10`, `n_arvores_5`: Tree counts
- `n_subunidades`, `n_subparcelas_5`: Subplot counts
- `area_ha_10`, `area_ha_5`: Sampled areas (ha)
- `dap_medio_10`, `ht_media_10`: Mean DBH and height (trees ≥10cm)
- `flag_subunidades_incompletas`: Incomplete subplots indicator
- `flag_poucas_arvores`: Few trees (<10) indicator
- `flag_biomassa_extrema`: Extreme biomass (<0.5 or >300 Mg/ha) indicator

**Sample size:** 290 plots  
**Biomass range:** 1.2 - 189.4 Mg/ha  
**Mean ± SD:** 44.2 ± 36.2 Mg/ha

### goias_df_features_buffer50m_v3_2018.csv
Remote sensing features extracted from Google Earth Engine.

**Columns (43):**
- `system:index`: GEE unique identifier
- `UA`: Plot identifier (links to biomassa file)
- `lon_pc`, `lat_pc`: Plot coordinates
- **Spectral indices (5):** NDVI_seca, NDWI_seca, EVI_seca, NBR_seca, MSI_seca
- **Note:** NDVI_RE_seca is mathematical duplicate of NDVI_seca (r=1.0)
- **Native bands (6):** B2_seca, B3_seca, B4_seca, B8_seca, B11_seca, B12_seca
- **SAR (3):** VV_dB, VH_dB, HV_dB
- **Canopy structure (2):** canopy_height, canopy_height_sd
- **Edaphic (1):** clay_pct
- **Topographic (2):** elevation, slope
- **Land cover (1):** mapbiomas_2018
- **Inventory metadata:** area_ha_10, area_ha_5, bio_Mg_h_1, dap_medio_, ht_media_1, etc.
- **Flags:** flag_biomassa, flag_pouca, flag_subun
- `.geo`: GeoJSON geometry

**Temporal window:** Dry season 2018 (May 1 - Sep 30)  
**Buffer:** 50m radius, median aggregation  
**Feature pool:** 20 candidates (excluding inventory-derived variables)

## Data Sources

### Field Inventory
- **Source:** Brazilian National Forest Inventory (Inventário Florestal Nacional - IFN)
- **Agency:** Brazilian Forest Service (SFB) / IBGE
- **Year:** 2018 campaign
- **Protocol:** 1-ha permanent plots, cross-design subplots
- **Measurement:** DBH≥5cm, species ID, tree height (subset)
- **Allometry:** Ribeiro et al. 2011 (cerrado), Chave et al. 2014 (forest)

### Remote Sensing
- **Sentinel-2 MSI:** Surface reflectance, 10-20m, dry season composite
- **Landsat 8 OLI:** Fallback for cloud-heavy tiles, 30m
- **Sentinel-1 GRD:** C-band SAR, 10m, VV+VH polarization
- **Meta/WRI 2023:** Global canopy height, ~10m
- **ETH 2020:** Canopy height standard deviation, 10m
- **OpenLandMap:** Soil clay content 0-5cm, 250m
- **SRTM:** Elevation and slope, 30m
- **MapBiomas Collection 10:** Land cover classes, 30m

## Usage Notes

### Filters Applied
Plots included if:
1. Valid coordinates + native Cerrado vegetation (MapBiomas classes: 3,4,5,6,11,12,29,32,49,50)
2. Biomass > 0 and < 300 Mg/ha
3. Exclude: dead trees (SA=4), lianas (HAB=5), bamboos (HAB=4)

**Quality flags retained** (not exclusion criteria):
- Incomplete subplots
- Few trees (<10)
- Extreme biomass

### Firewall Independence
**CRITICAL:** No field inventory-derived variables (DAP, height, density, observed biomass) were used as predictors. Features are 100% independent remote sensing observations.

Exception: Coordinates used only for spatial extraction, not as predictors.

## Missing Data

- Some plots have missing NDVI/optical values (cloud contamination >30%)
- Canopy height gaps filled by Meta/WRI continuous product
- Clay content at 250m resolution (coarser than other features)

Missing values handled via median imputation in Random Forest.

## Citation

If using this data, cite:

1. **Field inventory:** SFB (2018). Inventário Florestal Nacional. Serviço Florestal Brasileiro.
2. **This dataset:** [Your manuscript citation]
3. **Remote sensing sources:** See individual product citations in manuscript Methods section.

## Contact

For data access questions: [your.email@institution.edu]

**Last updated:** February 2026
