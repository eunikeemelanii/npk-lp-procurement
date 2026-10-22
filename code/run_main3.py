"""Revision-2 main runs: scenarios S0-S3 (2021-2024), decompositions, duals and ranging, sensitivity,
robustness to operational assumptions, dynamic analysis 2015-2025. Writes results3.json and out_*.csv."""
import json, time, numpy as np, pandas as pd
from setup3 import *
from common import K_DESIGN
t0 = time.time()
res = {}
MAT_P = ['Amonia', 'Asam Fosfat', 'Asam Sulfat', 'KCl', 'Batuan Fosfat', 'ZA']
jm = {m: MATS.index(m) for m in MATS}


def summ(s, b):
    return dict(Z=s['Z'], rev=s['rev'], proc=s['proc'], hold=s['hold'], dZ=s['Z']-b['Z'], dZpct=pct(s['Z'], b['Z']),
                dProc=s['proc']-b['proc'], dProcpct=pct(s['proc'], b['proc']))


# ------------------------------------------------------------------ scenarios by year
SC = {}
for yr in YEARS:
    inp = year_inputs(yr); SC[yr] = (inp,)+scenarios(inp)
inp, S0, S1, S2, S3 = SC[2024]
res['scen2024'] = {n: summ(s, S0) for n, s in zip(['S0', 'S1', 'S2', 'S3'], [S0, S1, S2, S3])}
res['h2024'] = dict(zip(MATS, inp['h']))
res['mix2024'] = {'actual': R2024, 'optimal': dict(zip(PRODUCTS, S3['x'].sum(0)))}
res['T15_2024'] = T15(R2024); res['K_design'] = K_DESIGN
idx12 = range(1, 13)
pd.DataFrame(S3['x'], columns=PRODUCTS, index=idx12).to_csv('out_x_S3_2024.csv')
pd.DataFrame(S3['y'], columns=MATS, index=idx12).to_csv('out_y_S3_2024.csv')
pd.DataFrame(S3['I'], columns=MATS, index=idx12).to_csv('out_I_S3_2024.csv')
pd.DataFrame(S0['y'], columns=MATS, index=idx12).to_csv('out_y_S0_2024.csv')
pd.DataFrame(S1['y'], columns=MATS, index=idx12).to_csv('out_y_S1_2024.csv')
c, v = inp['c'], inp['v']
unit = c@AM; marg = v-unit
res['unit2024'] = dict(zip(PRODUCTS, unit.mean(0))); res['value2024'] = dict(zip(PRODUCTS, v.mean(0)))
res['margin2024'] = dict(zip(PRODUCTS, marg.mean(0)))
res['margin_t15_2024'] = {p: float(marg[:, PRODUCTS.index(p)].mean()*KAPPA[p]) for p in NPK_LINE}
res['margin2024_min'] = dict(zip(PRODUCTS, marg.min(0))); res['margin2024_max'] = dict(zip(PRODUCTS, marg.max(0)))
res['price2024'] = {m: dict(mean=float(c[:, j].mean()), min=float(c[:, j].min()), max=float(c[:, j].max()),
                            cv=float(c[:, j].std()/c[:, j].mean()*100)) for j, m in enumerate(MATS)}
res['cap_dual_S3'] = list(S3['duals']['cap'])
res['rc_S3_mean'] = dict(zip(PRODUCTS, S3['duals']['rc_x'].mean(0)))
res['proc_by_mat_2024'] = {m: dict(S0=float((c[:, j]*S0['y'][:, j]).sum()), S3=float((c[:, j]*S3['y'][:, j]).sum()),
                                   qS0=float(S0['y'][:, j].sum()), qS3=float(S3['y'][:, j].sum()))
                           for j, m in enumerate(MATS)}
res['share_proc_S3'] = {m: res['proc_by_mat_2024'][m]['S3']/S3['proc']*100 for m in MATS}

# ------------------------------------------------------------------ multi-year table + decompositions
res['years'] = {}; res['s1_by_mat'] = {}; res['s2_by_prod'] = {}
for yr in YEARS:
    inp, a0, a1, a2, a3 = SC[yr]
    c, v, h = inp['c'], inp['v'], inp['h']
    res['years'][yr] = dict(Z0=a0['Z'], rev0=a0['rev'], proc0=a0['proc'], S1=pct(a1['Z'], a0['Z']), S2=pct(a2['Z'], a0['Z']),
                            S3=pct(a3['Z'], a0['Z']), dZ1=a1['Z']-a0['Z'], dZ2=a2['Z']-a0['Z'], dZ3=a3['Z']-a0['Z'],
                            dProc1=pct(a1['proc'], a0['proc']), dProc3=pct(a3['proc'], a0['proc']),
                            T15=T15(R_by_year[yr]), K=inp['K'])
    res['s1_by_mat'][yr] = {m: float((c[:, j]*(a0['y'][:, j]-a1['y'][:, j])).sum()-(h[j]*a1['I'][:, j]).sum())
                            for j, m in enumerate(MATS)}
    mu = v-c@AM
    res['s2_by_prod'][yr] = {p: float((mu[:, i]*(a2['x'][:, i]-a0['x'][:, i])).sum()) for i, p in enumerate(PRODUCTS)}
    res['s2_by_prod'][yr]['margin_mean'] = dict(zip(PRODUCTS, mu.mean(0)))
    res['s2_by_prod'][yr]['margin_t15_mean'] = {p: float(mu[:, PRODUCTS.index(p)].mean()*KAPPA[p]) for p in NPK_LINE}

# ------------------------------------------------------------------ alternative price series
alt = {'cru_only': load('harga_cru_only.csv'), 'cru_fill': load('harga_cru_fill.csv')}
res['alt_prices'] = {}
for nm, call in alt.items():
    res['alt_prices'][nm] = {}
    for yr in YEARS:
        inp = year_inputs(yr, c_all=call)
        a0, a1, a2, a3 = scenarios(inp)
        cc, hh = inp['c'], inp['h']
        res['alt_prices'][nm][yr] = dict(S1=pct(a1['Z'], a0['Z']), S2=pct(a2['Z'], a0['Z']), S3=pct(a3['Z'], a0['Z']),
                                         dZ1=a1['Z']-a0['Z'], dZ3=a3['Z']-a0['Z'], Z0=a0['Z'],
                                         by_mat={m: float((cc[:, j]*(a0['y'][:, j]-a1['y'][:, j])).sum()-(hh[j]*a1['I'][:, j]).sum())
                                                 for j, m in enumerate(MATS)})

# ------------------------------------------------------------------ S0-actual (company purchase timing)
d = pd.read_csv('raw/purchases_monthly_companywide.csv')
Qp = d.pivot_table(index=['Tahun', 'Bulan'], columns='Bahan', values='Q').reindex(C_ALL.index).fillna(0)
INT = {}
res['s0_actual'] = {}; res['purchase_profile'] = {}
for yr in YEARS:
    inp, a0, a1, a2, a3 = SC[yr]
    c, v, h = inp['c'], inp['v'], inp['h']
    use = a0['y']                              # monthly consumption at the S0 mix (JIT purchases)
    req = use.sum(0)
    ints = {}
    y = use.copy()
    for m in MAT_P:
        j = jm[m]; q = Qp.loc[yr, m].values
        share_buy = min(1.0, q.sum()/req[j]) if req[j] > 0 else 0.0
        ints[m] = 1-share_buy
        w = q/q.sum() if q.sum() > 0 else np.full(12, 1/12)
        y[:, j] = ints[m]*use[:, j]+share_buy*req[j]*w
        res['purchase_profile'].setdefault(yr, {})[m] = dict(
            wavg_over_flat=float((q*c[:, j]).sum()/q.sum()/c[:, j].mean()) if q.sum() > 0 else None,
            cv=float(q.std()/q.mean()), max_over_mean=float(q.max()/q.mean()), purchases=float(q.sum()), req=float(req[j]),
            corr_q_nextchange=float(np.corrcoef(q[:-1], np.diff(c[:, j]))[0, 1]) if q[:-1].std() > 0 else None)
    INT[yr] = ints
    bal = np.cumsum(y-use, axis=0)
    I0 = np.maximum(0, -bal.min(0))            # minimum opening stock that keeps inventory non-negative
    I = bal+I0
    proc = float((c*y).sum()); hold = float((h*I).sum())
    Zact = a0['rev']-proc-hold
    res['s0_actual'][yr] = dict(Z=Zact, proc=proc, hold=hold, proc_S0=a0['proc'], dProc_vs_S0=pct(proc, a0['proc']),
                                dZ_vs_S0=pct(Zact, a0['Z']), S1_vs_act=pct(a1['Z'], Zact), S3_vs_act=pct(a3['Z'], Zact),
                                S1_vs_act_proc=pct(a1['Z'], a0['rev']-proc), S3_vs_act_proc=pct(a3['Z'], a0['rev']-proc),
                                internal_share=ints)
res['internal_share'] = INT

# ------------------------------------------------------------------ robustness of the 2024 results
inp24 = year_inputs(2024); req24 = req_month(R2024)
from company_config import TANK, OWN_AMMONIA_CAPACITY, OWN_SULFURIC_CAPACITY_EXCL_JV, SAFE_DAYS, KCL_OTHER
# company-wide monthly use (t/month): ammonia = own capacity + 2024 purchases; phosphoric acid = plant requirement;
# sulfuric acid = 2024 purchases + own capacity excluding the joint-venture acid plant (company_config.py)
USE = {'Amonia': (OWN_AMMONIA_CAPACITY+Qp.loc[2024, 'Amonia'].sum())/12, 'Asam Fosfat': req24[jm['Asam Fosfat']],
       'Asam Sulfat': (Qp.loc[2024, 'Asam Sulfat'].sum()+OWN_SULFURIC_CAPACITY_EXCL_JV)/12}


def tank_S(mode):
    S = 1.0*req24.copy()
    for m, cap in TANK.items():
        j = jm[m]; safe = USE[m]*SAFE_DAYS/30
        head = max(0.0, cap-safe)
        share = min(1.0, req24[j]/USE[m])
        S[j] = head*share if mode == 'prop' else head
    return S


def run_variant(inp, label, **kw):
    a0 = solve(**inp, fix_mix=True, jit=tuple(range(M)), **({'Am': kw['Am']} if 'Am' in kw else {}))
    out = {}
    for nm, extra in [('S1', dict(fix_mix=True)), ('S2', dict(jit=tuple(range(M)))), ('S3', {})]:
        try:
            args = {**extra, **kw}
            if 'jit' in extra and 'jit' in kw: args['jit'] = tuple(sorted(set(extra['jit']) | set(kw['jit'])))
            s = solve(**inp, **args)
            out[nm] = pct(s['Z'], a0['Z']); out['d'+nm] = s['Z']-a0['Z']
        except Exception as e:
            out[nm] = None; print(label, nm, e)
    out['Z0'] = a0['Z']
    return out


rob = {}
rob['main'] = run_variant(inp24, 'main')
rob['h_same_year'] = run_variant(year_inputs(2024, h_basis='same'), 'h same')
rob['cru_only'] = res['alt_prices']['cru_only'][2024]
rob['cru_fill'] = res['alt_prices']['cru_fill'][2024]
# purchase cap: historical max/mean monthly company purchases 2021-2024
ucap = {}
for m in MAT_P:
    r_ = [Qp.loc[yr, m].max()/Qp.loc[yr, m].mean() for yr in YEARS]
    ucap[m] = float(max(r_))
res['ucap'] = ucap
yub = np.full((12, M), np.inf)
for m in MAT_P: yub[:, jm[m]] = ucap[m]*req24[jm[m]]
rob['purchase_cap'] = run_variant(inp24, 'cap', yub=yub)
# minimum lot (cargo) size: half a month of requirement for seaborne imports
lot = {jm[m]: (0.5*req24[jm[m]], 10*req24[jm[m]]) for m in ['KCl', 'ZA', 'Batuan Fosfat']}
rob['min_lot'] = run_variant(inp24, 'lot', lot=lot)
# internal supply fixed (flat), purchases flexible
ylb = np.zeros((12, M))
for m, s_ in INT[2024].items(): ylb[:, jm[m]] = s_*req24[jm[m]]
rob['internal_fixed'] = run_variant(inp24, 'internal', ylb=ylb)
# contract-consistent: no phosphoric-acid stock carried across quarter ends; and PA + ammonia just-in-time
rob['pa_quarter'] = run_variant(inp24, 'paq', qend_zero=[(jm['Asam Fosfat'], [2, 5, 8, 11])])
rob['pa_amm_jit'] = run_variant(inp24, 'pajit', jit=(jm['Asam Fosfat'], jm['Amonia']))
# storage from the tank list
for mode in ['prop', 'all']:
    i2 = dict(inp24); i2['S'] = tank_S(mode)
    rob['tank_'+mode] = run_variant(i2, 'tank '+mode)
res['tank_sigma'] = {mode: {m: float(tank_S(mode)[jm[m]]/req24[jm[m]]) for m in TANK} for mode in ['prop', 'all']}
# annual commitment with wider monthly bounds
for dlt in [0.0, 0.05, 0.10]:
    Rv = Rvec(R2024)
    ann = dict(mlo=0.5, mhi=1.5, groups=[(list(range(12)), {i: ((1-dlt)*Rv[i], (1+dlt)*Rv[i]) for i in range(P)})])
    rob[f'annual_{int(dlt*100)}'] = run_variant(inp24, 'annual', annual=ann)
# nutrient-balanced phosphoric-acid ratio for NPK 15-10-12
Am2 = AM.copy(); Am2[jm['Asam Fosfat'], PRODUCTS.index('NPK 15-10-12')] = 0.19
i2 = year_inputs(2024, Am=Am2)
rob['pa_ratio_019'] = run_variant(i2, 'pa ratio', Am=Am2)
a0_ = solve(**i2, fix_mix=True, jit=tuple(range(M)), Am=Am2); res['pa_ratio_019_Z0'] = a0_['Z']
# regulated (fixed) revenue for subsidised NPK 15-10-12: constant value over the year at the 2024 mean
i2 = dict(inp24); vv = inp24['v'].copy(); k = PRODUCTS.index('NPK 15-10-12'); vv[:, k] = vv[:, k].mean(); i2['v'] = vv
rob['regulated_value'] = run_variant(i2, 'regulated')
for lvl in [0.8, 1.2]:
    i3 = dict(inp24); vv = inp24['v'].copy(); vv[:, k] = lvl*inp24['v'][:, k].mean(); i3['v'] = vv
    rob[f'regulated_value_{int(lvl*100)}'] = run_variant(i3, 'regulated lvl')
# combined operational constraints
i2 = dict(inp24); i2['S'] = tank_S('prop')
rob['combined'] = run_variant(i2, 'combined', yub=yub, lot=lot, ylb=ylb, qend_zero=[(jm['Asam Fosfat'], [2, 5, 8, 11])])
res['robust2024'] = rob
# combined also for 2021-2023 (multi-year view)
res['robust_years'] = {}
for yr in YEARS:
    iy = year_inputs(yr); rq = req_month(R_by_year[yr])
    yub_ = np.full((12, M), np.inf); ylb_ = np.zeros((12, M))
    for m in MAT_P: yub_[:, jm[m]] = ucap[m]*rq[jm[m]]
    for m, s_ in INT[yr].items(): ylb_[:, jm[m]] = s_*rq[jm[m]]
    lot_ = {jm[m]: (0.5*rq[jm[m]], 10*rq[jm[m]]) for m in ['KCl', 'ZA', 'Batuan Fosfat'] if rq[jm[m]] > 0}
    S_ = rq.copy()
    for m, cap in TANK.items():
        j = jm[m]; safe = USE[m]*SAFE_DAYS/30; S_[j] = max(0, cap-safe)*min(1.0, rq[j]/USE[m])
    iy['S'] = S_
    res['robust_years'][yr] = dict(internal=run_variant(year_inputs(yr), 'int', ylb=ylb_),
                                   combined=run_variant(iy, 'comb', yub=yub_, lot=lot_, ylb=ylb_,
                                                        qend_zero=[(jm['Asam Fosfat'], [2, 5, 8, 11])]))
print('robustness done', round(time.time()-t0, 1), 's')

# ------------------------------------------------------------------ sensitivity (2024)
inp, S0, S1, S2, S3 = SC[2024]
res['sens_delta'] = {}
for dl in [0, 0.1, 0.2, 0.3, 0.4]:
    a = solve(**inp, delta=dl, jit=tuple(range(M))); b = solve(**inp, delta=dl)
    res['sens_delta'][dl] = dict(S2=pct(a['Z'], S0['Z']), S3=pct(b['Z'], S0['Z']))
res['sens_S'] = {}
for sm in [0.25, 0.5, 1, 2, 3]:
    a = solve(**year_inputs(2024, sigma=sm), fix_mix=True)
    res['sens_S'][sm] = dict(dZpct=pct(a['Z'], S0['Z']), dProcpct=pct(a['proc'], S0['proc']), hold=a['hold'])
res['sens_h'] = {}
for e in [0.01, 0.02, 0.03, 0.05]:
    a0 = solve(**year_inputs(2024, eta=e), fix_mix=True, jit=tuple(range(M)))
    a = solve(**year_inputs(2024, eta=e), fix_mix=True)
    res['sens_h'][e] = dict(dZpct=pct(a['Z'], a0['Z']))
# uniform +10% price per material (eq. 12): Z and S1 gain
res['price10'] = {}
for j, m in enumerate(MATS):
    i2 = dict(inp); cc = inp['c'].copy(); cc[:, j] *= 1.1; i2['c'] = cc
    i2['h'] = inp['h']*np.where(np.arange(M) == j, 1.1, 1.0)
    b0 = solve(**i2, fix_mix=True, jit=tuple(range(M))); b1 = solve(**i2, fix_mix=True); b3 = solve(**i2)
    res['price10'][m] = dict(dZpct=pct(b3['Z'], S3['Z']), share_proc=res['share_proc_S3'][m],
                             S1_gain=pct(b1['Z'], b0['Z']), S1_gain_base=pct(S1['Z'], S0['Z']),
                             S1_gain_usd=b1['Z']-b0['Z'], S1_gain_usd_base=S1['Z']-S0['Z'],
                             same_plan=bool(np.allclose(b3['y'], S3['y'], atol=1) and np.allclose(b3['x'], S3['x'], atol=1)))
# single-month +10% shocks (timing-relevant), S1 gain response in USD million
res['month_shock'] = {}
for m in ['Asam Fosfat', 'KCl', 'Amonia', 'ZA']:
    j = jm[m]; rows = []
    for t in range(12):
        i2 = dict(inp); cc = inp['c'].copy(); cc[t, j] *= 1.1; i2['c'] = cc
        b0 = solve(**i2, fix_mix=True, jit=tuple(range(M))); b1 = solve(**i2, fix_mix=True)
        rows.append((b1['Z']-b0['Z'])-(S1['Z']-S0['Z']))
    res['month_shock'][m] = rows
# volatility scaling around the annual mean
res['volatility'] = {}
for kf in [0, 0.5, 1, 1.5, 2]:
    i2 = dict(inp); cb = inp['c'].mean(0); i2['c'] = np.maximum(cb+kf*(inp['c']-cb), 0)
    b0 = solve(**i2, fix_mix=True, jit=tuple(range(M))); b1 = solve(**i2, fix_mix=True); b3 = solve(**i2)
    res['volatility'][kf] = dict(S1=pct(b1['Z'], b0['Z']), S3=pct(b3['Z'], b0['Z']), dZ1=b1['Z']-b0['Z'])

# ------------------------------------------------------------------ duals: uniqueness and per-month ranging
lam = S3['duals']['cap']; Kb = inp['K']
def Zk(Kvec):
    i2 = dict(inp); i2['K'] = Kvec; s = solve(**i2); return s['Z'], s['duals']['cap']
eps = 100.0; uniq = []
for t in range(12):
    Kp = np.full(12, Kb); Kp[t] += eps; Km = np.full(12, Kb); Km[t] -= eps
    zp, _ = Zk(Kp); zm, _ = Zk(Km)
    uniq.append(dict(lam=float(lam[t]), right=(zp-S3['Z'])/eps, left=(S3['Z']-zm)/eps))
res['dual_uniqueness'] = uniq
rng = []
for t in range(12):
    def lam_at(dk):
        Kv = np.full(12, Kb); Kv[t] = Kb+dk; return Zk(Kv)[1][t]
    # allowable increase
    lo, hi = 0.0, Kb
    if abs(lam_at(hi)-lam[t]) < 1e-6*max(1, abs(lam[t])): inc = np.inf
    else:
        for _ in range(30):
            mid = (lo+hi)/2
            if abs(lam_at(mid)-lam[t]) < 1e-6*max(1, abs(lam[t])): lo = mid
            else: hi = mid
        inc = lo
    Kmin = sum(0.8*R2024[p]/12/KAPPA[p] for p in NPK_LINE)
    lo, hi = 0.0, (Kb-Kmin)*0.999
    if abs(lam_at(-hi)-lam[t]) < 1e-6*max(1, abs(lam[t])): dec = hi
    else:
        for _ in range(30):
            mid = (lo+hi)/2
            if abs(lam_at(-mid)-lam[t]) < 1e-6*max(1, abs(lam[t])): lo = mid
            else: hi = mid
        dec = lo
    rng.append(dict(month=t+1, lam=float(lam[t]), allow_dec=float(dec), allow_inc=float(inc)))
res['ranging_month'] = rng
pd.DataFrame(rng).to_csv('out_capacity_ranging_by_month_2024.csv', index=False)
# aggregate capacity ranging and installed capacity
grid = np.linspace(Kb, Kb*1.30, 121); rows = []
for Kg in grid:
    s = solve(**{**inp, 'K': Kg})
    rows.append(dict(K=Kg, Z=s['Z'], dual_mean=float(s['duals']['cap'].mean()), dual_sum=float(s['duals']['cap'].sum())))
rg = pd.DataFrame(rows); rg.to_csv('out_capacity_ranging_2024.csv', index=False)
des = solve(**year_inputs(2024, cap='design'))
d0 = rg.dual_sum.iloc[0]
res['capacity'] = dict(K_real=Kb, K_design=K_DESIGN/12, Z_S3=S3['Z'], Z_design=des['Z'],
                       pct_design_vs_S3=pct(des['Z'], S3['Z']), pct_design_vs_S0=pct(des['Z'], S0['Z']),
                       gain_design=des['Z']-S3['Z'], gain_linear=float(lam.sum()*(K_DESIGN/12-Kb)),
                       K_first_change=float(rg[(rg.dual_sum-d0).abs() > 0.001*abs(d0)].K.min()),
                       K_dual_halves=float(rg[rg.dual_sum < 0.5*d0].K.min()) if (rg.dual_sum < 0.5*d0).any() else None,
                       dual_mean_design=float(des['duals']['cap'].mean()))
print('duals done', round(time.time()-t0, 1), 's')

# ------------------------------------------------------------------ subsidy-allocation variant (NPK 15-10-12 fixed)
inp = SC[2024][0]
xf_lo = None
def subsidy(jit):
    Rm = inp['Rm']; k = PRODUCTS.index('NPK 15-10-12')
    # fix NPK 15-10-12 via annual-type bounds: monthly band zero width for k
    ann = dict(mlo=0.8, mhi=1.2, groups=[])
    lb_hi = {}
    s = solve(**inp, annual=dict(mlo=0.8, mhi=1.2, groups=[([t], {k: (Rm[t, k], Rm[t, k])}) for t in range(12)]),
              jit=tuple(range(M)) if jit else ())
    return s
s2 = subsidy(True); s3 = subsidy(False)
res['subsidy_fixed'] = dict(S2=pct(s2['Z'], S0['Z']), S3=pct(s3['Z'], S0['Z']))

# ------------------------------------------------------------------ KCl consistency (2024)
kcl_model = float(AM[jm['KCl']]@Rvec(R2024))
k_ = KCL_OTHER
res['kcl_check'] = dict(model=kcl_model, granulation=k_['npk_granulation']*k_['r_gran'], other_npk_low=k_['other_npk']*k_['r_other_low'],
                        other_npk_high=k_['other_npk']*k_['r_other_high'], zk=k_['k2so4']*k_['r_k2so4'],
                        purchases=float(Qp.loc[2024, 'KCl'].sum()))

# ------------------------------------------------------------------ dynamic analysis 2015-2025 (counterfactual, 2024 structure)
dyn = []
Rv = Rvec(R2024)/12; K24 = T15(R2024)/12
for (yy, mm) in C_ALL.index:
    cv_ = C_ALL.loc[(yy, mm)].values[None, :]; vv_ = V_ALL.loc[(yy, mm)].values[None, :]
    s = solve(cv_, vv_, Rv[None, :], K24, np.zeros(M), np.zeros(M), jit=tuple(range(M)))
    s0 = solve(cv_, vv_, Rv[None, :], K24, np.zeros(M), np.zeros(M), jit=tuple(range(M)), fix_mix=True)
    mu = (vv_-cv_@AM)[0]
    row = dict(Tahun=yy, Bulan=mm, Z_opt=s['Z'], Z_base=s0['Z'], gain=s['Z']-s0['Z'], cap_dual=float(s['duals']['cap'][0]))
    for i, p in enumerate(PRODUCTS):
        row['x_'+p] = s['x'][0, i]; row['m_'+p] = mu[i]
        row['at_ub_'+p] = bool(Rv[i] > 0 and abs(s['x'][0, i]-1.2*Rv[i]) < 1e-6*max(1, Rv[i]))
        row['at_lb_'+p] = bool(Rv[i] > 0 and abs(s['x'][0, i]-0.8*Rv[i]) < 1e-6*max(1, Rv[i]))
    dyn.append(row)
dyn = pd.DataFrame(dyn); dyn.to_csv('out_dynamic_2015_2025.csv', index=False)
res['dyn_summary'] = dict(n=len(dyn), gain_sum=float(dyn.gain.sum()), base_sum=float(dyn.Z_base.sum()),
                          at_ub={p: int(dyn['at_ub_'+p].sum()) for p in PRODUCTS}, at_lb={p: int(dyn['at_lb_'+p].sum()) for p in PRODUCTS},
                          dual_zero=[(int(r.Tahun), int(r.Bulan)) for r in dyn.itertuples() if r.cap_dual < 1e-6],
                          dual_max=float(dyn.cap_dual.max()), dual_argmax=[int(x) for x in dyn.loc[dyn.cap_dual.idxmax(), ['Tahun', 'Bulan']]],
                          dap_ub_months=[(int(r.Tahun), int(r.Bulan)) for r in dyn.itertuples() if r._asdict().get('at_ub_DAP', False)])
json.dump(res, open('results3.json', 'w'), indent=1, default=float)
print('done', round(time.time()-t0, 1), 's')
