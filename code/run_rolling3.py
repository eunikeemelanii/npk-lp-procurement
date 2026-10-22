"""Revision-2 rolling-horizon analyses (Section 3.5):
 (a) shrinking horizon within each year 2021-2024 (decisions use only information available at the decision month:
     price series rebuilt with expanding-window calibration, product values also forecast);
 (b) continuous rolling horizon Jan 2021 - Dec 2024 with a 12-month window and terminal stock valued at the
     forecast price, with and without a one-month lead time for seaborne imports (KCl, phosphate rock, ZA);
 (c) multi-step forecast errors; (d) threshold purchasing rule. The bootstrap simulation (e) is in run_bootstrap3.py.
Writes results3_rolling.json."""
import json, time, warnings, sys
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from setup3 import *
from prices import load_inputs, build_series, product_values
warnings.filterwarnings('ignore')
t0 = time.time()
REAL, CRU, IDX, _ = load_inputs()
IDXL = list(C_ALL.index)
pos = lambda y, m: IDXL.index((y, m))
METHODS = ['naive', 'ma3', 'drift3', 'arima']
ARIMA_ORDERS = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0)]
FAST = '--fast' in sys.argv


def fc_series(s, H, method):
    s = np.asarray(s, float); last = s[-1]
    if H == 0: return np.zeros(0)
    if method == 'naive': return np.full(H, last)
    if method == 'ma3': return np.full(H, s[-3:].mean())
    if method == 'drift3':
        slope = (s[-1]-s[-4])/3.0
        return np.maximum(last+slope*np.arange(1, H+1), 0.2*last)
    if method == 'arima':
        ls = np.log(np.maximum(s, 1e-6))
        if np.std(np.diff(ls[-36:])) < 1e-9: return np.full(H, last)
        best = None
        for o in ARIMA_ORDERS:
            try:
                f = ARIMA(ls, order=o, trend='n').fit()
                if best is None or f.aic < best[0]: best = (f.aic, f)
            except Exception: pass
        return np.exp(best[1].forecast(H))
    raise ValueError(method)


# ---------------------------------------------------------------- vintages and forecast cache
VINT = {}
def vintage(p):
    if p not in VINT:
        c = build_series(REAL, CRU, upto=p)[MATS]; v = product_values(CRU, upto=p)[PRODUCTS]
        VINT[p] = (c, v)
    return VINT[p]


FC = {}
def forecasts(p, method, H=12):
    key = (p, method)
    if key not in FC:
        c, v = vintage(p)
        fcm = np.column_stack([fc_series(c.values[:, j], H, method) for j in range(M)])
        fvp = np.column_stack([fc_series(v.values[:, i], H, method) for i in range(P)])
        FC[key] = (fcm, fvp)
    return FC[key]


def h_of(year, c_df): return ETA*c_df.loc[year-1].values.mean(axis=0)


# ---------------------------------------------------------------- (a) shrinking horizon within each year
def shrinking(year, method, fix_mix):
    fin = year_inputs(year)
    R = R_by_year[year]; Rm = fin['Rm']; K = fin['K']; S = fin['S']
    I_prev = np.zeros(M); X = np.zeros((12, P)); Y = np.zeros((12, M)); II = np.zeros((12, M))
    for t in range(12):
        p = pos(year, t+1); H = 11-t
        if method == 'pf':
            cp, vp, h = fin['c'][t:], fin['v'][t:], fin['h']
        else:
            cv_, vv_ = vintage(p)
            h = h_of(year, cv_)
            fcm, fvp = forecasts(p, method)
            cp = np.vstack([cv_.values[-1:], fcm[:H]]); vp = np.vstack([vv_.values[-1:], fvp[:H]])
        a = solve(cp, vp, Rm[t:], K, S, h, I0=I_prev, fix_mix=fix_mix)
        X[t], Y[t], II[t] = a['x'][0], a['y'][0], a['I'][0]; I_prev = a['I'][0]
    return evaluate(fin['c'], fin['v'], fin['h'], X, Y, II)


res = {'shrinking': {}}
for yr in YEARS:
    fin = year_inputs(yr)
    s0, s1, s2, s3 = scenarios(fin)
    rec = dict(Z0=s0['Z'], proc0=s0['proc'], S1=pct(s1['Z'], s0['Z']), S2=pct(s2['Z'], s0['Z']), S3=pct(s3['Z'], s0['Z']),
               dZ1=s1['Z']-s0['Z'], dZ2=s2['Z']-s0['Z'], dZ3=s3['Z']-s0['Z'])
    for meth in ['pf']+METHODS:
        for nm, fm in [('S3', False), ('S1', True)]:
            Z, rev, proc, hold = shrinking(yr, meth, fm)
            rec[f'{nm}_{meth}'] = pct(Z, s0['Z']); rec[f'{nm}_{meth}_dZ'] = Z-s0['Z']
            rec[f'{nm}_{meth}_dproc'] = pct(proc, s0['proc'])
    res['shrinking'][yr] = rec
    print(yr, {k: round(v, 2) for k, v in rec.items() if not k.endswith(('dZ', 'dproc')) and k not in ('Z0', 'proc0', 'dZ1', 'dZ2', 'dZ3')},
          round(time.time()-t0), 's', flush=True)

# ---------------------------------------------------------------- (b) continuous rolling horizon 2021-2024
MONTHS = [(y, m) for y in YEARS for m in range(1, 13)]
NM = len(MONTHS)
c_fin = np.vstack([year_inputs(y)['c'] for y in YEARS]); v_fin = np.vstack([year_inputs(y)['v'] for y in YEARS])
h_fin = np.vstack([np.tile(year_inputs(y)['h'], (12, 1)) for y in YEARS])
Rm_all = np.vstack([year_inputs(y)['Rm'] for y in YEARS]); K_all = np.concatenate([[year_inputs(y)['K']]*12 for y in YEARS])
S_all = np.vstack([np.tile(year_inputs(y)['S'], (12, 1)) for y in YEARS])
# months after Dec 2024 inside a window use the 2024 structure
Rm_ext = np.vstack([Rm_all, np.tile(Rm_all[-1], (12, 1))]); K_ext = np.concatenate([K_all, [K_all[-1]]*12])
S_ext = np.vstack([S_all, np.tile(S_all[-1], (12, 1))])
c_jan25 = C_ALL.loc[(2025, 1)].values
LEAD = [MATS.index(m) for m in ['KCl', 'Batuan Fosfat', 'ZA']]
W = 12


def continuous(method, fix_mix=False, lead=False):
    I_prev = np.zeros(M); X = np.zeros((NM, P)); Y = np.zeros((NM, M)); II = np.zeros((NM, M))
    commit = {}
    for k_, (yy, mm) in enumerate(MONTHS):
        p = pos(yy, mm)
        if method == 'pf':
            cfull = C_ALL.values; vfull = V_ALL.values
            cp = cfull[p:p+W]; vp = vfull[p:p+W]; tv = cfull[p+W] if p+W < len(cfull) else cfull[-1]
            h = h_fin[k_]
        else:
            cv_, vv_ = vintage(p); fcm, fvp = forecasts(p, method)
            cp = np.vstack([cv_.values[-1:], fcm[:W-1]]); vp = np.vstack([vv_.values[-1:], fvp[:W-1]]); tv = fcm[W-1]
            h = h_of(yy, cv_)
        Tw = cp.shape[0]
        yfix = None
        if lead:
            yfix = {}
            for j in LEAD:
                yfix[(0, j)] = commit.get((k_, j), Rm_ext[k_]@AM[j])     # first month: S0 requirement
        a = solve(cp, vp, Rm_ext[k_:k_+Tw], K_ext[k_:k_+Tw], S_ext[k_:k_+Tw], h, I0=I_prev, fix_mix=fix_mix,
                  end='value', term_val=(1-ETA)*tv, yfix=yfix)
        X[k_], Y[k_], II[k_] = a['x'][0], a['y'][0], a['I'][0]; I_prev = a['I'][0]
        if lead and Tw > 1:
            for j in LEAD: commit[(k_+1, j)] = a['y'][1, j]
    Z, rev, proc, hold = evaluate(c_fin, v_fin, h_fin, X, Y, II)
    Z += float(((1-ETA)*c_jan25*II[-1]).sum())          # closing stock valued at the actual Jan-2025 price
    return Z, X, Y, II


cont = {}
Z0c = sum(res['shrinking'][y]['Z0'] for y in YEARS)
# 48-month perfect-foresight bound with closing stock valued at the Jan-2025 price
pfb = {}
for nm, fm in [('S1', True), ('S3', False)]:
    a = solve(c_fin, v_fin, Rm_all, K_all, S_all, h_fin, fix_mix=fm, end='value', term_val=(1-ETA)*c_jan25)
    pfb[nm] = a['Z']-Z0c
cont['pf48'] = pfb
for meth in ['pf']+METHODS:
    for nm, fm in [('S3', False), ('S1', True)]:
        for ld in [False, True]:
            if ld and meth == 'pf': continue
            Z, *_ = continuous(meth, fm, ld)
            cont[f'{nm}_{meth}' + ('_lead' if ld else '')] = Z-Z0c
    print('continuous', meth, {k: round(v/1e6, 2) for k, v in cont.items() if meth in k and not isinstance(v, dict)}, round(time.time()-t0), 's', flush=True)
cont['Z0'] = Z0c
res['continuous'] = cont

# ---------------------------------------------------------------- (c) multi-step forecast errors
acc = {}
MAIN4 = ['Asam Fosfat', 'KCl', 'Amonia', 'ZA']
for meth in METHODS:
    acc[meth] = {}
    for hh in [1, 3, 6, 11]:
        e = {m: [] for m in MAIN4}
        for (yy, mm) in MONTHS:
            p = pos(yy, mm)
            if p+hh >= len(IDXL): continue
            fcm, _ = forecasts(p, meth)
            for m in MAIN4:
                j = MATS.index(m); a_ = C_ALL.values[p+hh, j]
                e[m].append(abs(fcm[hh-1, j]-a_)/a_*100)
        acc[meth][hh] = {m: float(np.mean(v)) for m, v in e.items()}
res['mape'] = acc
# typical size of month-to-month price changes (for comparison)
chg = {m: float(np.mean(np.abs(np.diff(C_ALL.loc[2021:2024, m].values))/C_ALL.loc[2021:2024, m].values[:-1])*100) for m in MAIN4}
res['abs_monthly_change'] = chg


# ---------------------------------------------------------------- (d) threshold purchasing rule
def mix_month(c1, v1, Rm1, K):
    s = solve(c1[None, :], v1[None, :], Rm1[None, :], K, np.zeros(M), np.zeros(M), jit=tuple(range(M)))
    return s['x'][0]


def threshold(year, method, theta, cpath=None, vpath=None, hist_c=None, hist_v=None, fin=None, fixed_mix=False):
    """Mix re-optimized monthly with current prices; raw material m is bought one month ahead (next month's
    requirement, up to storage) only if the forecast price rise exceeds eta + theta; otherwise just in time."""
    fin = year_inputs(year) if fin is None else fin
    c = fin['c'] if cpath is None else cpath; v = fin['v'] if vpath is None else vpath
    Rm, K, S, h = fin['Rm'], fin['K'], fin['S'], fin['h']
    I = np.zeros(M); X = np.zeros((12, P)); Y = np.zeros((12, M)); II = np.zeros((12, M))
    for t in range(12):
        if hist_c is None:
            p = pos(year, t+1); cv_, vv_ = vintage(p); ccur = cv_.values[-1]; vcur = vv_.values[-1]
            f1 = forecasts(p, method)[0][0] if method != 'naive' else ccur
        else:
            hc = np.vstack([hist_c, c[:t+1]]); ccur = c[t]; vcur = v[t]
            f1 = np.array([fc_series(hc[:, j], 1, method)[0] for j in range(M)])
        x = Rm[t] if fixed_mix else mix_month(ccur, vcur, Rm[t], K)
        need = AM@x; nxt = AM@Rm[min(t+1, 11)]
        y = np.maximum(0, need-I)
        if t < 11:
            buy = (f1/np.maximum(ccur, 1e-9)-1) > (ETA+theta)
            buy[IU] = False
            extra = np.maximum(0, np.minimum(S, nxt)-np.maximum(0, I-need))
            y = np.where(buy, y+extra, y)
        I = I+y-need
        X[t], Y[t], II[t] = x, y, I
    return evaluate(c, v, h, X, Y, II)


thr = {}
for meth in ['ma3', 'arima', 'drift3']:
    for theta in [0.0, 0.02, 0.05]:
        tot = 0; per = {}
        for yr in YEARS:
            Z, *_ = threshold(yr, meth, theta)
            per[yr] = pct(Z, res['shrinking'][yr]['Z0']); tot += Z-res['shrinking'][yr]['Z0']
        thr[f'{meth}_{theta}'] = dict(per_year=per, total=tot)
    print('threshold', meth, {k: round(v['total']/1e6, 2) for k, v in thr.items() if k.startswith(meth)}, round(time.time()-t0), 's', flush=True)
# timing-only part of the rule (actual mix fixed): compare with S1
for meth in ['ma3', 'arima']:
    tot = 0
    for yr in YEARS:
        Z, *_ = threshold(yr, meth, 0.02, fixed_mix=True); tot += Z-res['shrinking'][yr]['Z0']
    thr[f'{meth}_0.02_timing_only'] = dict(total=tot)
res['threshold'] = thr

# (e) the bootstrap simulation is in run_bootstrap3.py
json.dump(res, open('results3_rolling.json', 'w'), indent=1, default=float)
print('done', round(time.time()-t0), 's')
