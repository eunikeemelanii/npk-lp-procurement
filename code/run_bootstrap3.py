"""Bootstrap simulation of 12-month price paths (2024 plant structure): distribution of the perfect-foresight gains and
of the gains attainable with rolling-horizon forecasts (moving average, drift) and with the threshold rule.
Joint monthly log returns of 2015-2024 (raw materials and product values) resampled with a stationary block bootstrap
(mean block length 3 months); sulfuric acid and urea follow their actual 2024 path (their series approach zero in 2020,
which makes log returns unstable; together they are about 2.4% of procurement cost)."""
import json, numpy as np, pandas as pd, warnings
from setup3 import *
warnings.filterwarnings('ignore')


def fc(s, H, method):
    s = np.asarray(s, float); last = s[-1]
    if method == 'naive': return np.full(H, last)
    if method == 'ma3': return np.full(H, s[-3:].mean())
    if method == 'drift3': return np.maximum(last+(s[-1]-s[-4])/3.0*np.arange(1, H+1), 0.2*last)


fin = year_inputs(2024)
SIMM = [MATS.index(m) for m in ['ZA', 'KCl', 'Asam Fosfat', 'Amonia', 'Batuan Fosfat']]
hist_c = C_ALL.loc[2015:2023].values; hist_v = V_ALL.loc[2015:2023].values
H = pd.concat([C_ALL.iloc[:, SIMM], V_ALL], axis=1).loc[2015:2024].values
LR = np.diff(np.log(H), axis=0)
start = pd.concat([C_ALL.iloc[:, SIMM], V_ALL], axis=1).loc[(2023, 12)].values
rng = np.random.default_rng(20261016)


def rolling(cP, vP, meth):
    I_prev = np.zeros(M); X = np.zeros((12, P)); Y = np.zeros((12, M)); II = np.zeros((12, M))
    for t in range(12):
        Hh = 11-t
        hc = np.vstack([hist_c, cP[:t+1]]); hv = np.vstack([hist_v, vP[:t+1]])
        fcm = np.column_stack([fc(hc[:, j], Hh, meth) for j in range(M)]) if Hh else np.zeros((0, M))
        fvp = np.column_stack([fc(hv[:, k], Hh, meth) for k in range(P)]) if Hh else np.zeros((0, P))
        a = solve(np.vstack([cP[t:t+1], fcm]), np.vstack([vP[t:t+1], fvp]), fin['Rm'][t:], fin['K'], fin['S'], fin['h'], I0=I_prev)
        X[t], Y[t], II[t] = a['x'][0], a['y'][0], a['I'][0]; I_prev = a['I'][0]
    return evaluate(cP, vP, fin['h'], X, Y, II)[0]


def threshold(cP, vP, meth, theta):
    I = np.zeros(M); X = np.zeros((12, P)); Y = np.zeros((12, M)); II = np.zeros((12, M))
    for t in range(12):
        hc = np.vstack([hist_c, cP[:t+1]])
        f1 = np.array([fc(hc[:, j], 1, meth)[0] for j in range(M)])
        s = solve(cP[t:t+1], vP[t:t+1], fin['Rm'][t:t+1], fin['K'], np.zeros(M), np.zeros(M), jit=tuple(range(M)))
        x = s['x'][0]; need = AM@x; nxt = AM@fin['Rm'][min(t+1, 11)]
        y = np.maximum(0, need-I)
        if t < 11:
            buy = (f1/np.maximum(cP[t], 1e-9)-1) > (ETA+theta); buy[IU] = False
            y = np.where(buy, y+np.maximum(0, np.minimum(fin['S'], nxt)-np.maximum(0, I-need)), y)
        I = I+y-need; X[t], Y[t], II[t] = x, y, I
    return evaluate(cP, vP, fin['h'], X, Y, II)[0]


N = 300; recs = []
for i in range(N):
    rows = []; j = rng.integers(len(LR))
    while len(rows) < 12:
        rows.append(LR[j]); j = (j+1) % len(LR) if rng.random() > 1/3 else rng.integers(len(LR))
    path = start*np.exp(np.cumsum(np.array(rows), axis=0))
    cP = fin['c'].copy(); cP[:, SIMM] = path[:, :len(SIMM)]; vP = path[:, len(SIMM):]
    inp = dict(fin); inp['c'] = cP; inp['v'] = vP
    s0, s1, s2, s3 = scenarios(inp)
    r = dict(Z0=s0['Z'], dZ1=s1['Z']-s0['Z'], dZ2=s2['Z']-s0['Z'], dZ3=s3['Z']-s0['Z'],
             vol=float(np.mean(np.abs(np.array(rows)[:, :len(SIMM)]))))
    for meth in ['ma3', 'drift3']: r['d_'+meth] = rolling(cP, vP, meth)-s0['Z']
    r['d_thr_ma3'] = threshold(cP, vP, 'ma3', 0.02)-s0['Z']
    recs.append(r)
sim = pd.DataFrame(recs); sim.to_csv('out_bootstrap_2024.csv', index=False)
q = lambda s: dict(mean=float(s.mean()/1e6), p10=float(s.quantile(.1)/1e6), p50=float(s.median()/1e6),
                   p90=float(s.quantile(.9)/1e6), share_neg=float((s < 0).mean()*100))
out = {k: q(sim[k]) for k in ['dZ1', 'dZ2', 'dZ3', 'd_ma3', 'd_drift3', 'd_thr_ma3']}
out['share_of_pf'] = {k: float(sim[k].sum()/sim.dZ3.sum()*100) for k in ['dZ1', 'dZ2', 'd_ma3', 'd_drift3', 'd_thr_ma3']}
out['timing_share_of_pf_S1'] = {k: float((sim[k]-sim.dZ2).sum()/sim.dZ1.sum()*100) for k in ['d_ma3', 'd_drift3', 'd_thr_ma3']}
out['n'] = N; out['Z0_median'] = float(sim.Z0.median()/1e6)
out['corr_vol_dZ1'] = float(np.corrcoef(sim.vol, sim.dZ1)[0, 1])
json.dump(out, open('results3_bootstrap.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
