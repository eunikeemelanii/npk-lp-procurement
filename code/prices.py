"""Raw-material price series (Section 2.1 of the paper), revision 2.

Sources: monthly procurement realization of PT. X (company-wide purchases; quantity, value, weighted price) and the
CRU Fertilizer Week reference series. Steps:
  1. phosphoric acid 2015-2016 converted from a 100% P2O5 basis to the product-acid basis (factor k_PA);
  2. anomalies (zero values, or > 5x / < 0.2x the 13-month rolling median) removed;
  3. calibration of the CRU series to the realized prices: ratio rho_m (eq. 1) or additive beta (sulfuric acid, eq. 2);
  4. months without a usable realized price: gaps of at most two months between realized prices are filled by linear
     interpolation (the contract price in force); longer gaps and gaps at the end of the sample use the calibrated
     CRU proxy.
build_series(..., upto=t) uses only information available at month t (expanding-window calibration, trailing gaps
filled with the CRU proxy) and is used for the rolling-horizon test.
"""
import numpy as np, pandas as pd

MATS_P = ['Amonia', 'Asam Fosfat', 'Asam Sulfat', 'KCl', 'Batuan Fosfat', 'ZA']
CRU_COL = {'Amonia': ('Ammonia_CFR_SEA', 1.0), 'Asam Fosfat': ('PhosAcid_CFR_India_100P2O5', 0.52),
           'Asam Sulfat': ('SulphAcid_FOB_JpKr', 1.0), 'KCl': ('Potash_CFR_SEA', 1.0),
           'Batuan Fosfat': ('PRock_FOB_Morocco', 1.0), 'ZA': ('AS_CFR_SEA', 1.0)}
ADDITIVE = {'Asam Sulfat'}
MAX_INTERP_GAP = 2


def load_inputs(raw='raw/purchases_monthly_companywide.csv', cru='cru_monthly_selected.csv'):
    d = pd.read_csv(raw)
    d = d[(d.Q > 0) & (d.P > 0)]
    real = d.pivot_table(index=['Tahun', 'Bulan'], columns='Bahan', values='P')
    idx = pd.MultiIndex.from_tuples([(y, m) for y in range(2015, 2026) for m in range(1, 13)
                                     if not (y == 2025 and m > 9)], names=['Tahun', 'Bulan'])
    real = real.reindex(idx)[MATS_P]
    zero = pd.read_csv(raw).query('Q > 0 and P <= 0').set_index(['Tahun', 'Bulan', 'Bahan'])
    c = pd.read_csv(cru, index_col=[0, 1]).reindex(idx)
    return real, c, idx, zero


def _cru(c, m):
    col, sc = CRU_COL[m]
    return c[col].interpolate(limit_direction='both')*sc


def build_series(real, c, upto=None, fill_rule='interp', return_meta=False):
    """fill_rule: 'interp' (main), 'cru' (all gaps with calibrated CRU proxy, first-revision rule),
    'cru_only' (calibrated CRU proxy in every month)."""
    real = real.copy(); c = c.copy()
    n = len(real) if upto is None else upto+1
    real = real.iloc[:n]; c = c.iloc[:n]
    # 1. phosphoric-acid basis correction
    pa = real['Asam Fosfat']; pacru = c['PhosAcid_CFR_India_100P2O5']
    yrs = real.index.get_level_values(0)
    rat = (pa/pacru)[yrs >= 2017]
    k_pa = float(rat.median()) if rat.notna().any() else 0.546
    real.loc[yrs <= 2016, 'Asam Fosfat'] = pa[yrs <= 2016]*k_pa
    out = pd.DataFrame(index=real.index); src = pd.DataFrame(index=real.index)
    meta = {'k_PA': k_pa, 'calib': {}, 'anomalies': []}
    for m in MATS_P:
        r = real[m].copy(); q = _cru(c, m)
        rm = r.rolling(13, center=True, min_periods=3).median().bfill().ffill()
        base = np.minimum(rm, 5*r.median())
        bad = (r > 5*base) | (r < 0.2*base)
        for i in r[bad].index: meta['anomalies'].append((m, i[0], i[1], float(r[i])))
        r[bad] = np.nan
        if m in ADDITIVE:
            cal = float((r-q).median()); proxy = q+cal
        else:
            cal = float((r/q).median()); proxy = q*cal
        meta['calib'][m] = cal
        flag = pd.Series(np.where(r.notna(), 'realized', ''), index=r.index, dtype=object)
        s = r.copy()
        if fill_rule == 'cru_only':
            s = proxy.copy(); flag[:] = 'CRU proxy'
        else:
            isna = s.isna().values
            vals = s.values.copy(); fl = flag.values.copy()
            i = 0; N = len(vals)
            while i < N:
                if not isna[i]: i += 1; continue
                j = i
                while j < N and isna[j]: j += 1
                gap = j-i
                if fill_rule == 'interp' and gap <= MAX_INTERP_GAP and i > 0 and j < N:
                    for k_ in range(i, j):
                        w = (k_-(i-1))/(j-(i-1))
                        vals[k_] = vals[i-1]*(1-w)+vals[j]*w; fl[k_] = 'interpolated'
                else:
                    for k_ in range(i, j):
                        vals[k_] = proxy.values[k_]; fl[k_] = 'CRU proxy'
                i = j
            s = pd.Series(vals, index=r.index); flag = pd.Series(fl, index=r.index)
        out[m] = s; src[m] = flag
    out['Urea'] = c['Urea_FOB_Indonesia'].interpolate(limit_direction='both'); src['Urea'] = 'CRU (opportunity cost)'
    if return_meta:
        return out, src, meta
    return out


# ---------------------------------------------------------------- product values (CRU), Table 2 / eq. (3)
def product_values(c, upto=None):
    c = c.copy()
    n = len(c) if upto is None else upto+1
    c = c.iloc[:n]

    def ratio_fill(target, ref):
        t = c[target]; r = c[ref].interpolate(limit_direction='both')
        med = (t/r).median()
        return t.fillna(r*med)
    v = pd.DataFrame(index=c.index)
    npk16 = ratio_fill('NPK161616_CFR_SEA', 'NPK151515_FOB_Baltic')
    tsp = ratio_fill('TSP_CFR_Brazil', 'DAP_CFR_India')
    urea = c['Urea_FOB_Indonesia'].interpolate(limit_direction='both')
    pot = c['Potash_CFR_SEA'].interpolate(limit_direction='both')
    uN, uP, uK = urea/0.46, tsp/0.46, pot/0.60
    NV = lambda n_, p_, k_: n_*uN+p_*uP+k_*uK
    # eq. (3) with B = NPK 16-16-16; for 15-15-15 the ratio is exactly 15/16 (= 45/48)
    v['NPK 15-15-15'] = npk16*NV(15, 15, 15)/NV(16, 16, 16)
    v['NPK 15-10-12'] = npk16*NV(15, 10, 12)/NV(16, 16, 16)
    v['NPS'] = ratio_fill('NPS2020013_CFR_India', 'DAP_CFR_India')
    v['DAP'] = c['DAP_CFR_India'].interpolate(limit_direction='both')
    v['SP-36'] = tsp*36/46
    v['SP-20'] = ratio_fill('SSP_CPT_Brazil', 'TSP_CFR_Brazil')*20/19
    return v[['NPK 15-10-12', 'NPK 15-15-15', 'NPS', 'DAP', 'SP-36', 'SP-20']]
