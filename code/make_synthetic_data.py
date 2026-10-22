"""Fully synthetic inputs for running the code end to end.

The real inputs (monthly procurement records of the company and licensed CRU Fertilizer Week prices) are confidential
and are not distributed. This script writes randomly generated files with the same layout, so that the whole pipeline
(prep_prices_v3.py -> run_main3.py -> run_rolling3.py) runs and produces tables with the structure of Tables 3 and 5 of
the paper. Nothing in these files is derived from company or CRU data: price levels are round numbers chosen by hand,
price paths are correlated random walks, and production, capacity and tank figures are invented. The numerical results
therefore differ from those in the paper.

Writes into the target folder:
  company_config.py                        synthetic company inputs (same keys as company_config_TEMPLATE.py)
  cru_monthly_selected.csv                 synthetic reference price series (same columns as the CRU selection)
  raw/purchases_monthly_companywide.csv    synthetic monthly purchases (same columns as the output of 00c_extract_purchases.py)

Usage: python make_synthetic_data.py [target_folder] [seed]
"""
import os, sys
import numpy as np, pandas as pd

A_RATIOS = {  # Table 1 of the paper (consumption ratios are published there)
    'NPK 15-10-12': [0.001, 0.27, 0.25, 0.21, 0.23, 0.12, 0], 'NPK 15-15-15': [0.03, 0.13, 0.20, 0.255, 0.285, 0.137, 0],
    'NPS': [0.04, 0.25, 0.24, 0, 0.38, 0.17, 0], 'DAP': [0, 0.08, 0.06, 0, 0.84, 0.21, 0],
    'SP-36': [0, 0, 0.15, 0, 0.40, 0, 0.53], 'SP-20': [0, 0, 0.03, 0, 0.08, 0, 0.556]}
MATS = ['Urea', 'ZA', 'Asam Sulfat', 'KCl', 'Asam Fosfat', 'Amonia', 'Batuan Fosfat']

# invented production (t/year) and plant data
R_SYN = {
    2021: {'NPK 15-10-12': 1200000, 'NPK 15-15-15': 250000, 'NPS': 60000, 'DAP': 50000, 'SP-36': 120000, 'SP-20': 0},
    2022: {'NPK 15-10-12': 1300000, 'NPK 15-15-15': 150000, 'NPS': 50000, 'DAP': 60000, 'SP-36': 80000, 'SP-20': 0},
    2023: {'NPK 15-10-12': 1250000, 'NPK 15-15-15': 100000, 'NPS': 40000, 'DAP': 70000, 'SP-36': 30000, 'SP-20': 0},
    2024: {'NPK 15-10-12': 1400000, 'NPK 15-15-15': 50000, 'NPS': 40000, 'DAP': 80000, 'SP-36': 10000, 'SP-20': 0}}
CONFIG = '''"""SYNTHETIC company inputs written by make_synthetic_data.py (invented values, not company data)."""
DATA_DIR = None
PURCHASE_WORKBOOKS = None
R_BY_YEAR = {r}
K_DESIGN = 1800000
TANK = {{'Amonia': 30000, 'Asam Fosfat': 60000, 'Asam Sulfat': 40000}}
OWN_AMMONIA_CAPACITY = 300000
OWN_SULFURIC_CAPACITY_EXCL_JV = 600000
SAFE_DAYS = 7
KCL_OTHER = dict(npk_granulation=100000, other_npk=20000, k2so4=10000, r_gran=0.22, r_other_low=0.22, r_other_high=0.45,
                 r_k2so4=0.84)
'''

# reference series: (start level in USD/t, loading on the common factor, own monthly volatility)
CRU_SPEC = {
    'Urea_FOB_Indonesia': (300, 1.0, 0.05), 'AS_CFR_SEA': (150, 0.8, 0.05), 'Ammonia_CFR_SEA': (400, 1.2, 0.06),
    'Potash_CFR_SEA': (300, 0.8, 0.04), 'PhosAcid_CFR_India_100P2O5': (800, 0.7, 0.03), 'SulphAcid_FOB_JpKr': (60, 1.0, 0.08),
    'PRock_FOB_Morocco': (100, 0.5, 0.03), 'NPK161616_CFR_SEA': (520, 0.9, 0.03), 'NPK151515_FOB_Baltic': (500, 0.9, 0.03),
    'NPS2020013_CFR_India': (480, 0.9, 0.04), 'DAP_CFR_India': (620, 0.9, 0.04), 'TSP_CFR_Brazil': (460, 0.8, 0.04),
    'SSP_CPT_Brazil': (280, 0.7, 0.04)}
# realized price = calibration factor x reference (additive premium for sulfuric acid); company-specific factors invented
CAL = {'Amonia': ('Ammonia_CFR_SEA', 1.00), 'Asam Fosfat': ('PhosAcid_CFR_India_100P2O5', 0.55),
       'Asam Sulfat': ('SulphAcid_FOB_JpKr', 20.0), 'KCl': ('Potash_CFR_SEA', 1.02),
       'Batuan Fosfat': ('PRock_FOB_Morocco', 1.05), 'ZA': ('AS_CFR_SEA', 0.95)}
BUY_SHARE = {'Amonia': 0.75, 'Asam Fosfat': 0.70, 'Asam Sulfat': 1.0, 'KCl': 1.0, 'Batuan Fosfat': 1.0, 'ZA': 0.55}


def generate(folder='.', seed=12345):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(folder, 'raw'), exist_ok=True)
    # ---- reference prices 2013-01 .. 2025-12: common log random walk with calm and volatile months, plus a
    #      mean-reverting AR(1) component for each series (keeps relative prices, and hence margins, plausible)
    idx = pd.MultiIndex.from_tuples([(y, m) for y in range(2013, 2026) for m in range(1, 13)], names=['Tahun', 'Bulan'])
    T = len(idx)
    vol = np.where(rng.random(T) < 0.25, 0.07, 0.03)          # calm and volatile months
    f = np.cumsum(rng.normal(0, 1, T)*vol)
    cru = pd.DataFrame(index=idx)
    for col, (p0, beta, s) in CRU_SPEC.items():
        e = np.zeros(T); eps = rng.normal(0, s, T)
        for t in range(1, T): e[t] = 0.9*e[t-1]+eps[t]
        cru[col] = np.round(p0*np.exp(beta*(f-f[0])+e), 1)
    cru.loc[cru.index.get_level_values(0) < 2015, ['NPK151515_FOB_Baltic', 'TSP_CFR_Brazil', 'SSP_CPT_Brazil']] = np.nan
    cru.to_csv(os.path.join(folder, 'cru_monthly_selected.csv'))
    # ---- monthly purchases 2015-01 .. 2025-09
    A = pd.DataFrame(A_RATIOS, index=MATS)
    rows = []
    for y in range(2015, 2026):
        R = R_SYN.get(y, R_SYN[2021] if y < 2021 else R_SYN[2024])
        need = A[list(R)].values@np.array(list(R.values()))/12.0
        for m in range(1, 13):
            if y == 2025 and m > 9: continue
            for mat, (col, k) in CAL.items():
                if rng.random() < 0.08: continue                       # month without a purchase
                ref = cru.loc[(y, m), col]
                P = ref+k if mat == 'Asam Sulfat' else ref*k
                if mat == 'Asam Fosfat':                                # quarterly contract price
                    q0 = (y, 3*((m-1)//3)+1); P = cru.loc[q0, col]*k
                P *= 1+rng.normal(0, 0.02)
                Q = need[MATS.index(mat)]*BUY_SHARE[mat]*rng.uniform(0.4, 1.6)
                rows.append(dict(Tahun=y, Bulan=m, Bahan=mat, Q=round(Q, 1), V=round(P*Q, 2), currency='USD', P=P))
    d = pd.DataFrame(rows)
    # two records that exercise the anomaly rule: an extreme price and a quantity recorded without value
    i = d.index[(d.Tahun == 2019) & (d.Bahan == 'Amonia')][0]; d.loc[i, 'P'] *= 9; d.loc[i, 'V'] = d.loc[i, 'P']*d.loc[i, 'Q']
    j = d.index[(d.Tahun == 2023) & (d.Bahan == 'ZA')][0]; d.loc[j, ['V', 'P']] = 0.0
    d.to_csv(os.path.join(folder, 'raw', 'purchases_monthly_companywide.csv'), index=False)
    with open(os.path.join(folder, 'company_config.py'), 'w') as fh:
        fh.write(CONFIG.format(r=repr(R_SYN)))
    return folder


if __name__ == '__main__':
    tgt = sys.argv[1] if len(sys.argv) > 1 else '.'
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
    generate(tgt, sd)
    print('synthetic inputs written to', os.path.abspath(tgt))
