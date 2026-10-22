"""General multi-period LP (eqs. 4-9 of the paper) used for all runs in revision 2.

max  sum_t sum_p v[t,p] x[t,p] - sum_t sum_m c[t,m] y[t,m] - sum_t sum_m h[t,m] I[t,m] (+ terminal value of stock)
s.t. I[t,m] = I[t-1,m] + y[t,m] - sum_p a[m,p] x[t,p]                  (5)
     0 <= I[t,m] <= S[t,m];  I[0] = I0;  I[T] = 0 (or valued)            (6)
     sum_{p in NPK} x[t,p]/kappa_p <= K[t]                               (7)
     (1-delta) R[t,p] <= x[t,p] <= (1+delta) R[t,p]                      (8)   (R = monthly reference volume)
     x, y >= 0                                                           (9)
Options: just-in-time materials, fixed mix, purchase bounds, lot sizes (MILP), annual commitment band,
quarter-end zero stock, fixed purchases (lead-time commitments).
Solver: HiGHS (dual simplex / branch-and-bound) through SciPy.
"""
import numpy as np, pandas as pd
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix, vstack
from common import PRODUCTS, MATS, A, KAPPA, NPK_LINE

P, M = len(PRODUCTS), len(MATS)
AM = A[PRODUCTS].values                    # M x P
IU = MATS.index('Urea')
NPK_IDX = [PRODUCTS.index(p) for p in NPK_LINE]
KAP = np.array([KAPPA.get(p, np.inf) for p in PRODUCTS])


def solve(c, v, Rm, K, S, h, I0=None, delta=0.2, fix_mix=False, jit=(), end='zero', term_val=None,
          ylb=None, yub=None, lot=None, annual=None, qend_zero=None, yfix=None, Am=None, xfix=None):
    c = np.asarray(c, float); v = np.asarray(v, float); Rm = np.asarray(Rm, float)
    T = c.shape[0]
    Am = AM if Am is None else Am
    K = np.broadcast_to(np.asarray(K, float), (T,))
    S = np.broadcast_to(np.asarray(S, float), (T, M)) if np.ndim(S) else np.full((T, M), S)
    h = np.broadcast_to(np.asarray(h, float), (T, M))
    I0 = np.zeros(M) if I0 is None else np.asarray(I0, float)
    nx, ny, nI = T*P, T*M, T*M
    ix = lambda t, p: t*P+p; iy = lambda t, m: nx+t*M+m; iI = lambda t, m: nx+ny+t*M+m
    nz = 0; lots = {}
    if lot:
        for m, q in lot.items():
            for t in range(T): lots[(t, m)] = nx+ny+nI+nz; nz += 1
    N = nx+ny+nI+nz
    obj = np.zeros(N)
    for t in range(T):
        for p in range(P): obj[ix(t, p)] = -v[t, p]
        for m in range(M):
            obj[iy(t, m)] = c[t, m]; obj[iI(t, m)] = h[t, m]
    if end == 'value':
        for m in range(M): obj[iI(T-1, m)] -= term_val[m]
    # equalities: inventory balance
    Aeq = lil_matrix((T*M, N)); beq = np.zeros(T*M)
    for t in range(T):
        for m in range(M):
            r = t*M+m
            Aeq[r, iI(t, m)] = 1; Aeq[r, iy(t, m)] = -1
            for p in range(P):
                if Am[m, p]: Aeq[r, ix(t, p)] = Am[m, p]
            if t == 0: beq[r] = I0[m]
            else: Aeq[r, iI(t-1, m)] = -1
    # inequalities: capacity rows first (duals), then annual commitments, then lot-size links
    rows, b = [], []
    for t in range(T):
        rows.append({ix(t, p): 1.0/KAP[p] for p in NPK_IDX}); b.append(K[t])
    if annual:
        for months, lohi in annual['groups']:
            for p, (lo, hi) in lohi.items():
                rows.append({ix(t, p): 1.0 for t in months}); b.append(hi)
                rows.append({ix(t, p): -1.0 for t in months}); b.append(-lo)
    for (t, m), j in lots.items():
        qmin, U = lot[m]
        rows.append({iy(t, m): -1.0, j: qmin}); b.append(0.0)       # y >= qmin z
        rows.append({iy(t, m): 1.0, j: -U}); b.append(0.0)          # y <= U z
    Aub = lil_matrix((len(rows), N))
    for i, rw in enumerate(rows):
        for j, val in rw.items(): Aub[i, j] = val
    b = np.array(b)
    # bounds
    lb = np.zeros(N); ub = np.full(N, np.inf)
    for t in range(T):
        for p in range(P):
            if xfix is not None:
                lb[ix(t, p)] = ub[ix(t, p)] = xfix[t, p]
            elif fix_mix:
                lb[ix(t, p)] = ub[ix(t, p)] = Rm[t, p]
            elif annual:
                lb[ix(t, p)] = annual['mlo']*Rm[t, p]; ub[ix(t, p)] = annual['mhi']*Rm[t, p]
            else:
                lb[ix(t, p)] = (1-delta)*Rm[t, p]; ub[ix(t, p)] = (1+delta)*Rm[t, p]
        for m in range(M):
            if ylb is not None: lb[iy(t, m)] = ylb[t, m]
            if yub is not None: ub[iy(t, m)] = yub[t, m]
            ub[iI(t, m)] = 0.0 if (m in jit or m == IU) else S[t, m]
    if end == 'zero':
        for m in range(M): ub[iI(T-1, m)] = 0.0
    if qend_zero:
        for m, ts in qend_zero:
            for t in ts: ub[iI(t, m)] = 0.0
    if yfix:
        for (t, m), val in yfix.items(): lb[iy(t, m)] = ub[iy(t, m)] = val
    for j in lots.values(): lb[j], ub[j] = 0, 1
    if not lots:
        res = linprog(obj, A_ub=Aub.tocsr() if len(rows) else None, b_ub=b if len(rows) else None,
                      A_eq=Aeq.tocsr(), b_eq=beq, bounds=np.c_[lb, ub], method='highs')
        if res.status != 0: raise RuntimeError(res.message)
        xsol = res.x; Z = -res.fun
        duals = dict(cap=-res.ineqlin.marginals[:T], eq=res.eqlin.marginals.reshape(T, M),
                     rc_x=(res.lower.marginals+res.upper.marginals)[:nx].reshape(T, P))
    else:
        integ = np.zeros(N); integ[list(lots.values())] = 1
        cons = [LinearConstraint(Aeq.tocsr(), beq, beq), LinearConstraint(Aub.tocsr(), -np.inf, b)]
        res = milp(obj, constraints=cons, bounds=Bounds(lb, ub), integrality=integ,
                   options=dict(mip_rel_gap=1e-6, time_limit=120))
        if res.x is None: raise RuntimeError(res.message)
        xsol = res.x; Z = -res.fun; duals = None
    x = xsol[:nx].reshape(T, P); y = xsol[nx:nx+ny].reshape(T, M); I = xsol[nx+ny:nx+ny+nI].reshape(T, M)
    rev = float((v*x).sum()); proc = float((c*y).sum()); hold = float((h*I).sum())
    return dict(Z=Z, rev=rev, proc=proc, hold=hold, x=x, y=y, I=I, duals=duals,
                tv=float((term_val*I[-1]).sum()) if end == 'value' else 0.0)


def evaluate(c, v, h, x, y, I):
    rev = float((v*x).sum()); proc = float((c*y).sum()); hold = float((np.broadcast_to(h, I.shape)*I).sum())
    return rev-proc-hold, rev, proc, hold
