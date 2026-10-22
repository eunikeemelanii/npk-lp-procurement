"""Common inputs for revision-2 runs."""
import numpy as np, pandas as pd
from common import PRODUCTS, MATS, A, KAPPA, NPK_LINE, R2024, K_DESIGN
from model3 import solve, evaluate, AM, P, M, IU
from company_config import R_BY_YEAR as R_by_year
YEARS = [2021, 2022, 2023, 2024]
ETA = 0.02


def load(fn='harga_bahan_baku_bulanan.csv'):
    c = pd.read_csv(fn, index_col=[0, 1])[MATS]
    return c


C_ALL = load(); V_ALL = pd.read_csv('nilai_produk_bulanan.csv', index_col=[0, 1])[PRODUCTS]


def Rvec(R): return np.array([R[p] for p in PRODUCTS])


def T15(R): return sum(R[p]/KAPPA[p] for p in NPK_LINE)


def req_month(R, Am=AM): return Am@Rvec(R)/12.0       # average monthly requirement per material (S0 mix)


def holding(year, c_all=C_ALL, eta=ETA, basis='prev'):
    if basis == 'prev': return eta*c_all.loc[year-1].values.mean(axis=0)
    if basis == 'same': return eta*c_all.loc[year].values.mean(axis=0)
    raise ValueError(basis)


def year_inputs(year, R=None, sigma=1.0, c_all=C_ALL, v_all=V_ALL, eta=ETA, h_basis='prev', cap='realized', Am=AM):
    R = R_by_year[year] if R is None else R
    c = c_all.loc[year].values; v = v_all.loc[year].values
    Rm = np.tile(Rvec(R)/12.0, (12, 1))
    K = (T15(R) if cap == 'realized' else K_DESIGN)/12.0
    S = sigma*req_month(R, Am)
    h = holding(year, c_all, eta, h_basis)
    return dict(c=c, v=v, Rm=Rm, K=K, S=S, h=h)


def scenarios(inp, **kw):
    S0 = solve(**inp, fix_mix=True, jit=tuple(range(M)), **kw)
    S1 = solve(**inp, fix_mix=True, **kw)
    S2 = solve(**inp, jit=tuple(range(M)), **kw)
    S3 = solve(**inp, **kw)
    return S0, S1, S2, S3


def pct(a, b): return 100.0*(a-b)/b
