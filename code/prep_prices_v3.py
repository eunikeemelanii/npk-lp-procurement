"""Builds the model price inputs (revision 2). Writes harga_bahan_baku_bulanan.csv, sumber_harga_bahan_baku.csv,
nilai_produk_bulanan.csv, anomalies_and_treatment.csv, calibration.csv and the robustness series
harga_cru_fill.csv (first-revision fill rule) and harga_cru_only.csv (calibrated CRU proxy in every month)."""
import pandas as pd, numpy as np
from prices import *
real, c, idx, zero = load_inputs()
out, src, meta = build_series(real, c, return_meta=True)
out.to_csv('harga_bahan_baku_bulanan.csv'); src.to_csv('sumber_harga_bahan_baku.csv')
build_series(real, c, fill_rule='cru').to_csv('harga_cru_fill.csv')
build_series(real, c, fill_rule='cru_only').to_csv('harga_cru_only.csv')
product_values(c).to_csv('nilai_produk_bulanan.csv')
an = pd.DataFrame(meta['anomalies'], columns=['material', 'year', 'month', 'recorded_price_usd_per_t'])
an['reason'] = np.where(an.recorded_price_usd_per_t > 1000, 'more than 5x the 13-month rolling median',
                        'less than 0.2x the 13-month rolling median')
z = zero.reset_index()[['Bahan', 'Tahun', 'Bulan', 'Q']].rename(columns={'Bahan': 'material', 'Tahun': 'year', 'Bulan': 'month'})
z = z[z.year >= 2015]; z['recorded_price_usd_per_t'] = 0.0; z['reason'] = 'quantity recorded without value'
an = pd.concat([an, z.drop(columns='Q')], ignore_index=True)
an['treatment'] = [src.loc[(r.year, r.month), r.material] for r in an.itertuples()]
an = an.sort_values(['material', 'year', 'month'])
an.to_csv('anomalies_and_treatment.csv', index=False)
pd.Series(meta['calib']).to_csv('calibration.csv', header=['rho_or_beta'])
print('k_PA =', round(meta['k_PA'], 3)); print('calibration:', {k: round(v, 3) for k, v in meta['calib'].items()})
print(an.to_string())
print((src == 'realized').mean().round(3)*100)
print(src.apply(pd.Series.value_counts).fillna(0).astype(int))
