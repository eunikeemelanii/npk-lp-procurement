"""Tables 3 and 5 of the paper (Table 5 includes the timing-only row) from results3.json and
results3_rolling.json (run after run_main3.py and run_rolling3.py).
Writes Table3.csv, Table5.csv and Table5_theta.csv and prints them."""
import json
import pandas as pd

R3 = json.load(open('results3.json')); RR = json.load(open('results3_rolling.json'))
YRS = ['2021', '2022', '2023', '2024']
M = 1e6

# ---- Table 3: optimization scenarios in 2024
S = R3['scen2024']
t3 = pd.DataFrame([[n, S[k]['rev']/M, S[k]['proc']/M, None if k == 'S0' else S[k]['dProcpct'], S[k]['hold']/M, S[k]['Z']/M,
                    None if k == 'S0' else S[k]['dZpct']]
                   for n, k in [('S0 Baseline', 'S0'), ('S1 Procurement (perfect foresight)', 'S1'),
                                ('S2 Mix', 'S2'), ('S3 Integrated (perfect foresight)', 'S3')]],
                  columns=['Scenario', 'Product value (million USD)', 'Procurement cost (million USD)',
                           'Procurement cost change (%)', 'Holding cost (million USD)', 'Margin Z (million USD)',
                           'Delta Z vs S0 (%)']).round(2)
t3.to_csv('Table3.csv', index=False)

# ---- Table 5: margin increase vs S0, perfect foresight (PF) and rolling-horizon forecasts, 2021-2024
Y = R3['years']; SH = RR['shrinking']; TH = RR['threshold']
tot = lambda k: sum(SH[y][k+'_dZ'] for y in YRS)
PF3, PF1, S2T = tot('S3_pf'), tot('S1_pf'), sum(Y[y]['dZ2'] for y in YRS)
cols = ['Year', 'Margin S0 (million USD)', 'S3 PF Delta Z (million USD)', 'S1 PF (%)', 'S2 (%)', 'S3 PF (%)',
        'S3 naive (%)', 'S3 MA (%)', 'S3 drift (%)', 'S3 ARIMA (%)', 'Threshold rule, ARIMA, theta = 2% (%)']
rows = [[y, Y[y]['Z0']/M, Y[y]['dZ3']/M, SH[y]['S1'], Y[y]['S2'], SH[y]['S3'], SH[y]['S3_naive'], SH[y]['S3_ma3'],
         SH[y]['S3_drift3'], SH[y]['S3_arima'], TH['arima_0.02']['per_year'][y]] for y in YRS]
tots = [PF1, S2T, PF3, tot('S3_naive'), tot('S3_ma3'), tot('S3_drift3'), tot('S3_arima'), TH['arima_0.02']['total']]
rows.append(['Total (million USD)', sum(Y[y]['Z0'] for y in YRS)/M, PF3/M]+[x/M for x in tots])
rows.append(['Share of S3 PF (%)', None, None]+[100*x/PF3 for x in tots])
# timing part: separate rolling-horizon runs with the mix fixed at S0 (S1 with forecasts); Rule (13) with the actual mix
timing = [PF1, None, None, tot('S1_naive'), tot('S1_ma3'), tot('S1_drift3'), tot('S1_arima'),
          TH['arima_0.02_timing_only']['total']]
rows.append(['Timing only (million USD)', None, None]+[None if x is None else x/M for x in timing])
t5 = pd.DataFrame(rows, columns=cols).round(2)
t5.to_csv('Table5.csv', index=False)

# ---- all safety margins of the threshold rule (theta was selected ex post from {0, 2%, 5%})
th = pd.DataFrame([[meth, th_, TH[f'{meth}_{th_}']['total']/M, 100*TH[f'{meth}_{th_}']['total']/PF3]
                   for meth in ['ma3', 'arima', 'drift3'] for th_ in ['0.0', '0.02', '0.05']],
                  columns=['forecast', 'theta', 'total Delta Z 2021-2024 (million USD)', 'share of S3 bound (%)']).round(2)
th.to_csv('Table5_theta.csv', index=False)

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 20)
print('Table 3\n', t3.to_string(index=False), '\n\nTable 5\n', t5.to_string(index=False), '\n\nThreshold rule, all theta\n',
      th.to_string(index=False))
