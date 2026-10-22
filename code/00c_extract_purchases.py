"""Extract monthly purchase quantity (t), value and weighted price from the raw procurement-realization workbooks
(company-wide, all plants). Used for: S0-actual baseline, purchase caps, cargo sizes, internal-supply shares,
and conversion of the Jan-Aug 2025 rupiah records of ammonia and sulfuric acid to USD.
Raw files are confidential and not redistributed."""
import openpyxl, numpy as np, pandas as pd, sys
from company_config import DATA_DIR, PURCHASE_WORKBOOKS
D = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
FILES = [(D+f, sheets) for f, sheets in PURCHASE_WORKBOOKS]
NAMES = {'phosphate rock': 'Batuan Fosfat', 'kcl/mop': 'KCl', 'ammonia': 'Amonia', 'phosphoric acid': 'Asam Fosfat',
         'sulfuric acid': 'Asam Sulfat', 'za': 'ZA', 'za bahan baku': 'ZA'}
TOTALS = {'total', 'total za bb'}
num = lambda x: float(x) if isinstance(x, (int, float)) else np.nan
rows = []
FX2025 = {1: 16259.48, 2: 16337.90, 3: 16468.71, 4: 16768.41, 5: 16444.68, 6: 16309.14, 7: 16276.43, 8: 16299.29,
          9: 16508.18}
for f, sheets in FILES:
    wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
    for sh in sheets:
        R = list(wb[sh].iter_rows(values_only=True))
        cur = None; parts = {}
        def flush():
            # sulfuric acid: use local + import sub-totals when present (the 'Total' row mixes currencies in 2025
            # and is empty in Apr 2025); otherwise the commodity 'Total' row
            use = ['lokal', 'impor'] if ('lokal' in parts or 'impor' in parts) else ['total']
            for m in range(12):
                Q = V = 0.0; cur_ = 'USD'
                for k in use:
                    if k not in parts: continue
                    row = parts[k]
                    q, v = num(row[3+3*m]), num(row[5+3*m])
                    q = 0 if np.isnan(q) else q; v = 0 if np.isnan(v) else v
                    if int(sh) == 2025 and q > 0 and v/q > 20000:   # recorded in rupiah
                        v = v/FX2025[m+1]; cur_ = 'IDR->USD'
                    Q += q; V += v
                rows.append(dict(Tahun=int(sh), Bulan=m+1, Bahan=cur, Q=Q, V=V, currency=cur_))
        for row in R:
            b, c = row[1], row[2]
            if isinstance(b, (int, float)) and b and isinstance(c, str):
                if cur and parts: flush()
                cur = NAMES.get(c.strip().lower()); parts = {}; continue
            if cur and isinstance(c, str):
                cl = c.strip().lower()
                if cl == 'total lokal': parts['lokal'] = row
                elif cl == 'total impor': parts['impor'] = row
                elif cl in TOTALS: parts['total'] = row
        if cur and parts: flush()
df = pd.DataFrame(rows).groupby(['Tahun', 'Bulan', 'Bahan'], as_index=False).agg(Q=('Q', 'sum'), V=('V', 'sum'), currency=('currency', 'max'))
# 2025 local ammonia and sulfuric acid are recorded in rupiah and were converted above with the monthly average
# IDR/USD rate (OECD Main Economic Indicators, series CCUSMA02IDM618N, retrieved from FRED, Oct 2026)
df['P'] = np.where(df.Q > 0, df.V/df.Q.replace(0, np.nan), np.nan)
df = df[df.Tahun >= 2015]
df = df[~((df.Tahun == 2025) & (df.Bulan > 9))]
df.to_csv('raw/purchases_monthly_companywide.csv', index=False)
piv = df.pivot_table(index='Tahun', columns='Bahan', values='Q', aggfunc='sum').round(0)
print(piv)
print(df[df.currency != 'USD'][['Tahun', 'Bulan', 'Bahan', 'Q', 'P']].round(1).to_string())
