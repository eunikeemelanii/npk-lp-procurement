import json, numpy as np
from setup3 import *
inp = year_inputs(2024); s3 = solve(**inp); lam = s3['duals']['cap']; Kb = inp['K']
rng = json.load(open('results3.json'))['ranging_month']
out = []
for r in rng:
    t = r['month']-1
    def lam_at(dk):
        Kv = np.full(12, Kb); Kv[t] = Kb+dk; return solve(**{**inp, 'K': Kv})['duals']['cap'][t]
    up = lam_at(r['allow_inc']*1.05+50) if np.isfinite(r['allow_inc']) else None
    dn = lam_at(-(r['allow_dec']*1.05+50)) if r['allow_dec'] < 39000 else None
    out.append(dict(month=r['month'], lam=lam[t], inc=r['allow_inc'], lam_above=up, dec=r['allow_dec'], lam_below=dn))
    print(r['month'], round(lam[t],2), round(r['allow_inc']), None if up is None else round(up,2), round(r['allow_dec']), None if dn is None else round(dn,2))
json.dump(out, open('results3_ranging_detail.json','w'), indent=1, default=float)
